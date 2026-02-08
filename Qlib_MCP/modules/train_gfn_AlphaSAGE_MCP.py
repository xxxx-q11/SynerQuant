"""
Training MCP Service - AlphaSAGE (Docker Compose version)
Algorithm Type: gfn (GFlowNet for Alpha Factor Discovery)
Factor Generation Paradigm: train_time_generation
Execution Method: Docker Compose container management
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
import sys

# ========== Configure Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/train_gfn_AlphaSAGE_docker.log')
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

# Training script name
TRAIN_SCRIPT = "train_gfn.py"


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
    
    seed = config_params.get("seed") or getattr(args, "seed", 0)
    instrument = config_params.get("instrument") or getattr(args, "instrument", "csi300")
    pool_capacity = config_params.get("pool_capacity") or getattr(args, "pool_capacity", 50)
    cuda = config_params.get("cuda") or getattr(args, "cuda", "0")
    
    # Build docker compose run command
    cmd = [
        "docker", "compose",
        "-f", str(PROJECT_ROOT / "docker-compose.yml"),
        "run",
        "--rm",  # Automatically remove container after execution
        "--name", f"alphasage-gfn-{instrument}-{pool_capacity}-{seed}",  # Container name
    ]
    
    # Set environment variables (override default values in docker-compose.yml)
    cmd.extend([
        "-e", f"CUDA_VISIBLE_DEVICES={cuda}",
    ])
    
    # Service name
    cmd.append(SERVICE_NAME)
    
    # Training script and parameters
    cmd.extend([
        "python3", TRAIN_SCRIPT,
        "--seed", str(seed),
        "--instrument", str(instrument),
        "--pool_capacity", str(pool_capacity),
    ])
    
    return cmd


def _find_latest_log_dir(instrument: str, pool_capacity: int, seed: int) -> Optional[str]:
    """
    Find the latest generated log directory
    
    Args:
        instrument: Trading instrument
        pool_capacity: Alpha pool capacity
        seed: Random seed
        
    Returns:
        Optional[str]: Absolute path of log directory, or None if not found
    """
    # Infer save directory (consistent with log_dir format in train_gfn.py)
    # log_dir format: data/gfn_logs/pool_{pool_capacity}/gfn_{encoder_type}_{instrument}_{pool_capacity}_{seed}-...
    log_base_dir = WORKSPACE_ROOT / "data" / "gfn_logs" / f"pool_{pool_capacity}"
    
    if not log_base_dir.exists():
        logger.warning(f"Log base directory does not exist: {log_base_dir}")
        return None
    
    # Find all matching log directories (starting with gfn_, containing instrument, pool_capacity, and seed)
    pattern = f"gfn_*_{instrument}_{pool_capacity}_{seed}-*"
    log_dirs = list(log_base_dir.glob(pattern))
    
    if not log_dirs:
        logger.warning(f"No matching log directories found, pattern: {pattern}")
        return None
    
    # Sort by modification time, return latest
    log_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_dir = str(log_dirs[0].resolve())
    logger.info(f"Found latest generated log directory: {latest_dir}")
    
    return latest_dir


def _find_latest_factor_file(instrument: str, pool_capacity: int, seed: int) -> Optional[str]:
    """
    Find the latest generated factor file
    
    Args:
        instrument: Trading instrument
        pool_capacity: Alpha pool capacity
        seed: Random seed
        
    Returns:
        Optional[str]: Absolute path of factor file, or None if not found
    """
    # First find the latest log directory
    latest_log_dir = _find_latest_log_dir(instrument, pool_capacity, seed)
    
    if not latest_log_dir:
        logger.warning("Log directory not found, cannot find factor file")
        return None
    
    log_dir_path = Path(latest_log_dir)
    
    # Find all *_pool_factors.json files in log directory
    factor_files = list(log_dir_path.glob("*_pool_factors.json"))
    
    if not factor_files:
        logger.warning(f"No *_pool_factors.json files found in directory: {log_dir_path}")
        return None
    
    # Extract number part from filename for sorting (e.g., 999_pool_factors.json -> 999)
    def extract_number(file_path):
        stem = file_path.stem  # e.g., "999_pool_factors"
        try:
            number_str = stem.split('_')[0]
            return int(number_str)
        except (ValueError, IndexError):
            return 0
    
    # Sort by episode number, return largest (latest)
    factor_files.sort(key=extract_number)
    latest_file = str(factor_files[-1].resolve())
    logger.info(f"Found latest generated factor file: {latest_file}")
    
    return latest_file


def run(args: SimpleNamespace) -> str:
    """
    Module interface function, run AlphaSAGE's train_gfn.py via Docker Compose
    
    Args:
        args: SimpleNamespace object containing the following attributes:
            - config_file: Configuration file path (optional, not currently supported)
            - config_params: Configuration parameter dictionary (optional), containing:
                - seed: Random seed
                - instrument: Trading instrument, e.g., "csi300"
                - pool_capacity: Alpha pool capacity
                - cuda: CUDA device, e.g., "0"
            - task_name: Task name, default "training"
    
    Returns:
        str: Full path of latest generated log directory, or error message
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
        seed = config_params.get("seed") or getattr(args, "seed", 0)
        instrument = config_params.get("instrument") or getattr(args, "instrument", "csi300")
        pool_capacity = config_params.get("pool_capacity") or getattr(args, "pool_capacity", 50)
        
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
                f"train_gfn_AlphaSAGE Docker Compose execution failed\n"
                f"Exit code: {proc.returncode}\n"
                f"Command: {cmd_str}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Find latest generated factor file
        latest_factor_file = _find_latest_factor_file(
            instrument=instrument,
            pool_capacity=pool_capacity,
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
    # Test run
    test_args = SimpleNamespace(
        config_params={
            "seed": 0,
            "instrument": "csi300",
            "pool_capacity": 50,
            "cuda": "0"
        },
        task_name="test_training"
    )
    
    result = run(test_args)
    print(f"Training result: {result}")
