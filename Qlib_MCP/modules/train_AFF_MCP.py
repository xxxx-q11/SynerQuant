"""
Training MCP Service - AlphaForge (Docker Compose version)
Algorithm Type: dl
Algorithm Name: AlphaForge - GAN-based Alpha Factor Generation
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
        logging.FileHandler('/tmp/train_AFF_AlphaSAGE_docker.log')
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


def _build_docker_compose_cmd(args: SimpleNamespace) -> list:
    """
    Build docker compose run command based on MCP input parameters
    
    Args:
        args: SimpleNamespace object containing training parameters
        
    Returns:
        list: docker compose command list
    """
    # Get parameters from args (if config_params exists, use it preferentially)
    config_params = getattr(args, "config_params", {})
    
    instruments = config_params.get("instruments") or getattr(args, "instruments", "csi300")
    seed = config_params.get("seed") or getattr(args, "seed", 0)
    train_end_year = config_params.get("train_end_year") or getattr(args, "train_end_year", 2020)
    cuda = config_params.get("cuda") or getattr(args, "cuda", "0")
    save_name = config_params.get("save_name") or getattr(args, "save_name", "test")
    zoo_size = config_params.get("zoo_size") or getattr(args, "zoo_size", 100)
    corr_thresh = config_params.get("corr_thresh") or getattr(args, "corr_thresh", 0.7)
    ic_thresh = config_params.get("ic_thresh") or getattr(args, "ic_thresh", 0.03)
    icir_thresh = config_params.get("icir_thresh") or getattr(args, "icir_thresh", 0.1)
    
    # Build docker compose run command
    cmd = [
        "docker", "compose",
        "-f", str(PROJECT_ROOT / "docker-compose.yml"),
        "run",
        "--rm",  # Automatically remove container after execution
        "--name", f"alphasage-aff-{instruments}-{train_end_year}-{seed}",  # Container name
    ]
    
    # Set environment variables (override default values in docker-compose.yml)
    cmd.extend([
        "-e", f"CUDA_VISIBLE_DEVICES={cuda}",
    ])
    
    # Service name
    cmd.append(SERVICE_NAME)
    
    # Training script and parameters
    cmd.extend([
        "python3", "train_AFF.py",
        "--instruments", str(instruments),
        "--seed", str(seed),
        "--train_end_year", str(train_end_year),  # Changed to underscore
        "--cuda", str(cuda),
        "--save_name", str(save_name),  # Changed to underscore
        "--zoo_size", str(zoo_size),  # Changed to underscore
        "--corr_thresh", str(corr_thresh),  # Changed to underscore
        "--ic_thresh", str(ic_thresh),  # Changed to underscore
        "--icir_thresh", str(icir_thresh),  # Changed to underscore
    ])
    
    return cmd


def _find_latest_factor_file(save_name: str, instruments: str, train_end_year: int, seed: int, zoo_size: int) -> Optional[str]:
    """
    Find the latest generated factor file
    
    Args:
        save_name: Save name prefix
        instruments: Stock pool
        train_end_year: Training end year
        seed: Random seed
        zoo_size: Alpha factor pool size
        
    Returns:
        Optional[str]: Absolute path of factor file, or None if not found
    """
    # New save directory format: data/AFF_logs/zoo_{zoo_size}/{name_prefix}-{timestamp}
    # name_prefix = f"aff_{instruments}_{train_end_year}_{seed}"
    # Since timestamp is dynamic, need to find all matching directories
    data_dir = WORKSPACE_ROOT / "data" / "AFF_logs"
    
    if not data_dir.exists():
        logger.warning(f"AFF_logs directory does not exist: {data_dir}")
        return None
    
    # Find zoo_{zoo_size} directory
    zoo_dir = data_dir / f"zoo_{zoo_size}"
    
    if not zoo_dir.exists():
        logger.warning(f"zoo_{zoo_size} directory does not exist: {zoo_dir}")
        return None
    
    # Build directory name prefix pattern: aff_{instruments}_{train_end_year}_{seed}-
    dir_prefix = f"aff_{instruments}_{train_end_year}_{seed}-"
    
    # Find all matching directories
    matching_dirs = [
        d for d in zoo_dir.iterdir()
        if d.is_dir() and d.name.startswith(dir_prefix)
    ]
    
    if not matching_dirs:
        logger.warning(f"No matching directories found, prefix: {dir_prefix}, in directory: {zoo_dir}")
        return None
    
    # Sort by directory modification time, find latest directory (last created)
    matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_dir = matching_dirs[0]
    
    logger.info(f"Found latest directory: {latest_dir}")
    
    # Find zoo_final_qlib_factors.json file in latest directory
    factor_file = latest_dir / "zoo_final_qlib_factors.json"
    
    if factor_file.exists():
        latest_file = str(factor_file.resolve())
        logger.info(f"Found latest generated factor file: {latest_file}")
        return latest_file
    
    # If JSON file not found, try to find CSV file as fallback
    csv_file = latest_dir / "csv_zoo_final.csv"
    if csv_file.exists():
        logger.info(f"Found CSV factor file: {csv_file}")
        return str(csv_file.resolve())
    
    logger.warning(f"Factor file not found in directory: {latest_dir}")
    return None


def run(args: SimpleNamespace) -> str:
    """
    Module interface function, run AlphaSAGE's train_AFF.py via Docker Compose
    
    Args:
        args: SimpleNamespace object containing the following attributes:
            - config_file: Configuration file path (optional, not currently supported)
            - config_params: Configuration parameter dictionary (optional), containing:
                - instruments: Stock pool, e.g., "csi300"
                - seed: Random seed
                - train_end_year: Training end year
                - cuda: CUDA device, e.g., "0"
                - save_name: Save name prefix
                - zoo_size: Alpha factor pool size
                - corr_thresh: Correlation threshold
                - ic_thresh: IC threshold
                - icir_thresh: ICIR threshold
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
        
        # Build docker compose command
        cmd = _build_docker_compose_cmd(args)
        cmd_str = " ".join(cmd)
        
        logger.info(f"Executing Docker Compose command: {cmd_str}")
        
        # Get training parameters (for finding result files later)
        config_params = getattr(args, "config_params", {})
        instruments = config_params.get("instruments") or getattr(args, "instruments", "csi300")
        seed = config_params.get("seed") or getattr(args, "seed", 0)
        train_end_year = config_params.get("train_end_year") or getattr(args, "train_end_year", 2020)
        save_name = config_params.get("save_name") or getattr(args, "save_name", "test")
        zoo_size = config_params.get("zoo_size") or getattr(args, "zoo_size", 100)
        
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
                f"train_AFF_AlphaSAGE Docker Compose execution failed\n"
                f"Exit code: {proc.returncode}\n"
                f"Command: {cmd_str}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Find latest generated factor file
        latest_factor_file = _find_latest_factor_file(
            save_name=save_name,
            instruments=instruments,
            train_end_year=train_end_year,
            seed=seed,
            zoo_size=zoo_size
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
    # Test run
    test_args = SimpleNamespace(
        config_params={
            "instruments": "csi300",
            "seed": 0,
            "train_end_year": 2020,
            "cuda": "0",
            "save_name": "test",
            "zoo_size": 100,
            "corr_thresh": 0.7,
            "ic_thresh": 0.03,
            "icir_thresh": 0.1
        },
        task_name="test_training"
    )
    
    result = run(test_args)
    print(f"Training result: {result}")
