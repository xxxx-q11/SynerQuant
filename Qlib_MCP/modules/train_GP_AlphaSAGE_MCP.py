"""
Training MCP Service - AlphaSAGE (Docker Compose version)
Algorithm Type: Genetic Programming for Alpha Factor Mining
Factor Generation Paradigm: train_time_generation
Execution Method: Docker Compose container management
"""

import os
import subprocess
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from types import SimpleNamespace
import sys

# ========== Configure Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/train_GP_AlphaSAGE_docker.log')
    ]
)
logger = logging.getLogger(__name__)

# ========== Configuration Constants ==========
# Qlib_MCP project root directory (contains docker-compose.yml)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# AlphaSAGE workspace path on host machine
WORKSPACE_ROOT = PROJECT_ROOT / "workspace" / "AlphaSAGE"

# qlib data path on host machine
QLIB_DATA_HOST = Path.home() / ".qlib" / "qlib_data"

# Docker Compose service name
SERVICE_NAME = "alphasage"


def _save_seed_factors_file(seed_factors: list, instruments: str, seed: int) -> Optional[str]:
    """
    Save seed factors to temporary file for train_GP.py in Docker container to read
    
    Args:
        seed_factors: Seed factor list, each element can be a string or dictionary
        instruments: Stock pool name
        seed: Random seed
        
    Returns:
        Optional[str]: Relative path of seed factor file in workspace, or None if no seed factors
    """
    if not seed_factors:
        return None
    
    # Normalize seed factor format
    normalized_seeds = []
    for sf in seed_factors:
        if isinstance(sf, dict):
            # Extract expression from dictionary
            expr = sf.get('expression', '')
            if expr:
                normalized_seeds.append({
                    'expression': expr,
                    'category': sf.get('category', 'unknown'),
                    'expected_effect': sf.get('expected_effect', '')
                })
        elif isinstance(sf, str) and sf.strip():
            normalized_seeds.append({
                'expression': sf.strip(),
                'category': 'unknown',
                'expected_effect': ''
            })
    
    if not normalized_seeds:
        return None
    
    # Save to workspace/AlphaSAGE/data/ directory (accessible by Docker container)
    seed_factors_dir = WORKSPACE_ROOT / "data" / "seed_factors"
    seed_factors_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"seed_factors_{instruments}_{seed}_{timestamp}.json"
    filepath = seed_factors_dir / filename
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'seed_factors': normalized_seeds,
            'count': len(normalized_seeds),
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Seed factors saved to: {filepath}, total {len(normalized_seeds)} factors")
    
    # Return relative path to workspace (for use in Docker container)
    return f"data/seed_factors/{filename}"


def _build_docker_compose_cmd(args: SimpleNamespace, seed_factors_file: Optional[str] = None) -> list:
    """
    Build docker compose run command based on MCP input parameters
    
    Args:
        args: SimpleNamespace object containing training parameters
        seed_factors_file: Relative path of seed factor file (relative to workspace)
        
    Returns:
        list: docker compose command list
    """
    # Get parameters from args (if config_params exists, use it preferentially)
    config_params = getattr(args, "config_params", {})
    
    instruments = config_params.get("instruments") or getattr(args, "instruments", "csi300")
    seed = config_params.get("seed") or getattr(args, "seed", 0)
    train_end_year = config_params.get("train_end_year") or getattr(args, "train_end_year", 2020)
    freq = config_params.get("freq") or getattr(args, "freq", "day")
    cuda = config_params.get("cuda") or getattr(args, "cuda", "0")
    
    # Build docker compose run command
    cmd = [
        "docker", "compose",  # Originally "docker compose"
        "-f", str(PROJECT_ROOT / "docker-compose.yml"),
        "run",
        "--rm",  # Automatically remove container after execution
        "--name", f"alphasage-gp-{instruments}-{train_end_year}-{seed}",  # Container name
    ]
    
    # Set environment variables (override default values in docker-compose.yml)
    cmd.extend([
        "-e", f"CUDA_VISIBLE_DEVICES={cuda}",
    ])
    
    # Service name
    cmd.append(SERVICE_NAME)
    
    # Training script and parameters
    cmd.extend([
        "python3", "train_GP.py",
        "--instruments", str(instruments),
        "--seed", str(seed),
        "--train-end-year", str(train_end_year),
        "--freq", str(freq),
        "--cuda", str(cuda),
    ])
    
    # If seed factor file exists, add parameter
    if seed_factors_file:
        cmd.extend(["--seed-factors-file", seed_factors_file])
    
    return cmd


def _find_latest_factor_file(instruments: str, train_end_year: int, freq: str, seed: int) -> Optional[str]:
    """
    Find the latest generated factor file
    
    Args:
        instruments: Stock pool
        train_end_year: Training end year
        freq: Data frequency
        seed: Random seed
        
    Returns:
        Optional[str]: Absolute path of factor file, or None if not found
    """
    # New save directory format: data/gp_{instruments}_{train_end_year}_{freq}_{seed}_{timestamp}
    # Since timestamp is dynamic, need to find all matching directories
    data_dir = WORKSPACE_ROOT / "data"
    
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return None
    
    # Build directory name prefix pattern: gp_{instruments}_{train_end_year}_{freq}_{seed}_
    dir_prefix = f"gp_{instruments}_{train_end_year}_{freq}_{seed}_"
    
    # Find all matching directories
    matching_dirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith(dir_prefix)
    ]
    
    if not matching_dirs:
        logger.warning(f"No matching directories found, prefix: {dir_prefix}, in directory: {data_dir}")
        return None
    
    # Sort by directory modification time, find latest directory (last created)
    matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_dir = matching_dirs[0]
    
    logger.info(f"Found latest directory: {latest_dir}")
    
    # Find all *_top200_factors_heap_merged.json files in latest directory
    factor_files = list(latest_dir.glob("*_factors_heap_merged.json"))
    
    if not factor_files:
        logger.warning(f"No *_factors_heap_merged.json files found in directory: {latest_dir}")
        return None
    
    # Extract number part from filename for sorting (e.g., 10_top200_factors_heap_merged.json -> 10)
    def extract_number(file_path):
        stem = file_path.stem  # e.g., "10_top200_factors_heap_merged"
        try:
            number_str = stem.split('_')[0]
            return int(number_str)
        except (ValueError, IndexError):
            return 0
    
    factor_files.sort(key=extract_number)
    latest_file = str(factor_files[-1].resolve())
    logger.info(f"Found latest generated factor file: {latest_file}")
    
    return latest_file


def run(args: SimpleNamespace) -> str:
    """
    Module interface function, run AlphaSAGE's train_GP.py via Docker Compose
    
    Args:
        args: SimpleNamespace object containing the following attributes:
            - config_file: Configuration file path (optional, not currently supported)
            - config_params: Configuration parameter dictionary (optional), containing:
                - instruments: Stock pool, e.g., "csi300"
                - seed: Random seed
                - train_end_year: Training end year
                - freq: Data frequency, e.g., "day"
                - cuda: CUDA device, e.g., "0"
                - seed_factors: Seed factor list (optional), from mining_feedback["suggested_seeds"]
                  Each element can be a string (factor expression) or dictionary (containing expression, category, expected_effect)
            - task_name: Task name, default "training"
    
    Returns:
        str: Full path of latest generated factor file, or error message
    """
    try:
        # Check if docker-compose.yml exists
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        if not compose_file.exists():
            error_msg = f"docker-compose.yml does not exist: {compose_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check workspace path
        if not WORKSPACE_ROOT.exists():
            error_msg = f"AlphaSAGE workspace does not exist: {WORKSPACE_ROOT}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check qlib data path
        if not QLIB_DATA_HOST.exists():
            error_msg = f"Qlib data directory does not exist: {QLIB_DATA_HOST}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Get training parameters
        config_params = getattr(args, "config_params", {})
        instruments = config_params.get("instruments") or getattr(args, "instruments", "csi300")
        seed = config_params.get("seed") or getattr(args, "seed", 0)
        
        # Process seed factors (from mining_feedback["suggested_seeds"])
        seed_factors = config_params.get("seed_factors") or getattr(args, "seed_factors", [])
        seed_factors_file = None
        if seed_factors:
            logger.info(f"Received {len(seed_factors)} seed factors, preparing to save to file...")
            seed_factors_file = _save_seed_factors_file(seed_factors, instruments, seed)
        
        # Build docker compose command
        cmd = _build_docker_compose_cmd(args, seed_factors_file)
        cmd_str = " ".join(cmd)
        
        logger.info(f"Executing Docker Compose command: {cmd_str}")
        
        # Get training parameters (for finding result files later)
        train_end_year = config_params.get("train_end_year") or getattr(args, "train_end_year", 2020)
        freq = config_params.get("freq") or getattr(args, "freq", "day")
        
        # Run Docker Compose (synchronous execution, wait for completion)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.rstrip("\n")
            # Write to stderr, not stdout
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            logger.info(line)

        proc.wait()

        # Check return code
        if proc.returncode != 0:
            error_msg = (
                f"train_GP_AlphaSAGE Docker Compose execution failed\n"
                f"Exit code: {proc.returncode}\n"
                f"Command: {cmd_str}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Find latest generated factor file
        latest_factor_file = _find_latest_factor_file(
            instruments=instruments,
            train_end_year=train_end_year,
            freq=freq,
            seed=seed
        )
        
        if latest_factor_file:
            logger.info(f"Training completed, factor file: {latest_factor_file}")
            return latest_factor_file
        else:
            logger.warning("Training completed but generated factor file not found")
            return "Training completed but generated factor file not found"
    
    except (FileNotFoundError, RuntimeError) as e:
        # Re-raise known errors
        raise
    
    except Exception as e:
        error_msg = f"Docker Compose execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ========== Test Function ==========
if __name__ == "__main__":
    # Test run (including seed factor examples)
    test_args = SimpleNamespace(
        config_params={
            "instruments": "csi300",
            "seed": 0,
            "train_end_year": 2020,
            "freq": "day",
            "cuda": "0",
            # Seed factor examples (from mining_feedback["suggested_seeds"])
            "seed_factors": [
                {
                    "expression": "Div(Sub(close, Ref(close, 5)), Ref(close, 5))",
                    "category": "momentum",
                    "expected_effect": "5-day momentum factor"
                },
                {
                    "expression": "Div(volume, Ref(volume, 10))",
                    "category": "liquidity",
                    "expected_effect": "10-day volume ratio"
                }
            ]
        },
        task_name="test_training"
    )
    
    result = run(test_args)
    print(f"Training result: {result}")
