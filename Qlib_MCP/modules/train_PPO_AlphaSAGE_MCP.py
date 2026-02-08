"""
Training MCP Service - AlphaSAGE
Algorithm Type: rl
Algorithm Name: PPO-based Alpha Factor Generation (AlphaSAGE)
Auto-generated Time: 2024
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import uvicorn
import uuid
import logging
import subprocess
import sys
import os
from datetime import datetime
import threading
from collections import deque

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_mcp.log')
    ]
)
logger = logging.getLogger(__name__)

# ========== FastAPI Application ==========
app = FastAPI(
    title="Training MCP - AlphaSAGE",
    description="PPO-based Alpha Factor Generation Training Service",
    version="1.0.0"
)

# ========== Configuration Constants ==========
# Qlib_MCP project root directory (contains docker-compose.yml)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# AlphaSAGE workspace path on host machine
REPO_PATH = PROJECT_ROOT / "workspace" / "AlphaSAGE"

# Docker Compose service name (must match service name in docker-compose.yml)
SERVICE_NAME = "alphasage"

TRAIN_SCRIPT = "train_ppo.py"
MAX_LOG_LINES = 1000  # Maximum log lines

# Task storage (production environment should use Redis or other persistent storage)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# ========== Pydantic Models ==========

class TrainRequest(BaseModel):
    """Training request model"""
    config_file: Optional[str] = Field(None, description="Configuration file path")
    config_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Training parameter dictionary",
        example={
            "seed": 0,
            "instruments": "csi300",
            "pool": 20,
            "steps": 200000
        }
    )
    task_name: str = Field("training", description="Task name")

    class Config:
        schema_extra = {
            "example": {
                "config_params": {
                    "seed": 0,
                    "instruments": "csi300",
                    "pool": 20,
                    "steps": 200000,
                    "train_end_year": 2020
                },
                "task_name": "ppo_training_csi300"
            }
        }


class TrainResponse(BaseModel):
    """Training response model"""
    task_id: str = Field(..., description="Task unique identifier")
    status: str = Field(..., description="Task status: started/running/completed/failed")
    message: str = Field(..., description="Response message")
    command: str = Field(..., description="Executed training command")
    log_preview: str = Field("", description="Log preview (latest few lines)")
    timestamp: str = Field(..., description="Task creation time")


class TaskStatus(BaseModel):
    """Task status model"""
    task_id: str
    status: str
    command: str
    pid: Optional[int] = None
    log_lines: int = 0
    recent_logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ========== Utility Functions ==========

def setup_environment() -> dict:
    """
    Set up training environment variables
    
    Returns:
        dict: Environment variable dictionary
    """
    env = os.environ.copy()
    src_path = str(REPO_PATH / "src")
    pythonpath = str(REPO_PATH) + os.pathsep + src_path
    if "PYTHONPATH" in env:
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    logger.info(f"Environment variables set: PYTHONPATH={env['PYTHONPATH']}")
    env["PYTHONUNBUFFERED"] = "1"  # Disable Python output buffering
    return env



def build_train_command(
    config_file: Optional[str] = None,
    config_params: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Build training command (run via Docker Compose in AlphaSAGE container)
    
    Args:
        config_file: Configuration file path (currently only logged, not directly used in container)
        config_params: Configuration parameter dictionary
        
    Returns:
        List[str]: Command parameter list
    """
    # Parse configuration parameters (with defaults)
    params = config_params or {}
    instruments = str(params.get("instruments", "csi300"))
    seed = int(params.get("seed", 0))
    pool = int(params.get("pool", 10))
    steps = int(params.get("steps", 200_000))
    train_end_year = int(params.get("train_end_year", 2020))
    cuda = str(params.get("cuda", "0"))

    # Build docker compose run command
    cmd: List[str] = [
        "docker", "compose",
        "-f", str(PROJECT_ROOT / "docker-compose.yml"),
        "run",
        "--rm",
        "--name", f"alphasage-ppo-{instruments}-{pool}-{seed}",
        "-e", f"CUDA_VISIBLE_DEVICES={cuda}",
        SERVICE_NAME,
        "python3", TRAIN_SCRIPT,
        "--seed", str(seed),
        "--instruments", instruments,
        "--pool", str(pool),
        "--steps", str(steps),
        "--train-end-year", str(train_end_year),
    ]

    # (Reserved) If future need to pass config file in container, can extend here
    if config_file:
        logger.info(f"Received config_file parameter, but currently not used in Docker container: {config_file}")

    logger.info(f"Configuration parameters: {params}")
    return cmd


def get_log_preview(logs: deque, lines: int = 10) -> str:
    """
    Get log preview
    
    Args:
        logs: Log queue
        lines: Preview line count
        
    Returns:
        str: Log preview text
    """
    if not logs:
        return ""
    
    recent = list(logs)[-lines:]
    return "\n".join(recent)


def _find_latest_factor_file(instruments: str, pool: int, seed: int) -> Optional[str]:
    """
    Find the latest generated factor file
    
    Args:
        instruments: Stock pool
        pool: Alpha pool capacity
        seed: Random seed
        
    Returns:
        Optional[str]: Absolute path of factor file, or None if not found
    """
    # PPO save path format: data/ppo_logs/pool_{pool}/{name_prefix}-{timestamp}/{name_prefix}_{timestamp}_top200_factors_heap_merged.json
    # where name_prefix = f"ppo_{instruments}_{pool}_{seed}"
    data_dir = REPO_PATH / "data" / "ppo_logs" / f"pool_{pool}"
    
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return None
    
    # Build directory name prefix pattern: ppo_{instruments}_{pool}_{seed}-
    dir_prefix = f"ppo_{instruments}_{pool}_{seed}-"
    
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
    
    # Find *_top200_factors_heap_merged.json files in latest directory
    factor_files = list(latest_dir.glob("*_top200_factors_heap_merged.json"))
    
    if not factor_files:
        logger.warning(f"No *_top200_factors_heap_merged.json files found in directory: {latest_dir}")
        return None
    
    # Sort by file modification time, find latest file
    factor_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_file = str(factor_files[0].resolve())
    logger.info(f"Found latest generated factor file: {latest_file}")
    
    return latest_file


# ========== Module Interface Function ==========

def run(args) -> str:
    """
    Module interface function for synchronous calls by MCP tools, etc.

    Consistent with GP version:
    - Directly run train_ppo.py via Docker Compose in AlphaSAGE container
    - Synchronously wait for training completion
    - Real-time forward container output to stderr and logs

    Args:
        args: SimpleNamespace object containing the following attributes:
            - config_file: Configuration file path (optional)
            - config_params: Configuration parameter dictionary (optional), containing:
            - seed: Random seed
            - instruments: Stock pool, e.g., "csi300"
            - pool: Alpha pool capacity
            - steps: Training steps
            - train_end_year: Training end year
            - task_name: Task name, default "training"

    Returns:
        str: Full path of latest generated factor file, or error message
    """
    try:
        # Basic checks: docker-compose.yml and workspace
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        if not compose_file.exists():
            error_msg = f"docker-compose.yml does not exist: {compose_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not REPO_PATH.exists():
            error_msg = f"AlphaSAGE workspace does not exist: {REPO_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Get configuration from args
        config_file = getattr(args, "config_file", None)
        config_params = getattr(args, "config_params", None)

        # Get training parameters (for finding result files later)
        params = config_params or {}
        instruments = str(params.get("instruments", "csi300"))
        seed = int(params.get("seed", 0))
        pool = int(params.get("pool", 10))

        # Build docker compose command
        cmd = build_train_command(config_file=config_file, config_params=config_params)
        cmd_str = " ".join(cmd)

        logger.info(f"Executing Docker Compose PPO command: {cmd_str}")

        # Synchronous execution, real-time forward logs
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            # Same as GP, output to stderr for MCP to capture
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            logger.info(line)

        proc.wait()

        # Check return code
        if proc.returncode != 0:
            error_msg = (
                f"train_PPO_AlphaSAGE Docker Compose execution failed\n"
                f"Exit code: {proc.returncode}\n"
                f"Command: {cmd_str}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Find latest generated factor file
        latest_factor_file = _find_latest_factor_file(
            instruments=instruments,
            pool=pool,
            seed=seed
        )
        
        if latest_factor_file:
            logger.info(f"Training completed, factor file: {latest_factor_file}")
            return latest_factor_file
        else:
            logger.warning("Training completed but generated factor file not found")
            return "Training completed but generated factor file not found"

    except (FileNotFoundError, RuntimeError):
        # Directly raise known errors for MCP to capture
        raise
    except Exception as e:
        error_msg = f"train_PPO_AlphaSAGE execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ========== Background Task Function ==========

def _run_training_task(task_id: str, cmd: List[str]) -> None:
    """
    Execute training task in background
    
    Args:
        task_id: Task ID
        cmd: Command parameter list
    """
    try:
        # Start training process (via Docker Compose, execute in AlphaSAGE container)
        logger.info(f"Starting training process (Docker) [{task_id}]: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Update task status
        with tasks_lock:
            tasks[task_id]["pid"] = proc.pid
            tasks[task_id]["status"] = "running"
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Training process started [{task_id}], PID: {proc.pid}")
        
        # Read output in real-time
        assert proc.stdout is not None
        for line in proc.stdout:
            line_stripped = line.rstrip()
            
            with tasks_lock:
                tasks[task_id]["log"].append(line_stripped)
                tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # Log to file
            logger.info(f"[Training {task_id[:8]}] {line_stripped}")
        
        # Wait for process to end
        return_code = proc.wait()
        
        # Update final status
        with tasks_lock:
            if return_code == 0:
                tasks[task_id]["status"] = "completed"
                logger.info(f"Training task completed [{task_id}]")
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f"Process exit code: {return_code}"
                logger.error(f"Training task failed [{task_id}], exit code: {return_code}")
            
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    except Exception as e:
        error_msg = f"Training task exception: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        
        with tasks_lock:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = error_msg
            tasks[task_id]["updated_at"] = datetime.now().isoformat()


# ========== API Endpoints ==========

@app.get("/", tags=["Basic"])
async def root():
    """Service root path"""
    return {
        "service": "Training MCP - AlphaSAGE",
        "algorithm": "PPO-based Alpha Factor Generation",
        "version": "1.0.0",
        "repository": str(REPO_PATH),
        "endpoints": {
            "docs": "/docs",
            "train_start": "/train/start",
            "train_status": "/train/status/{task_id}",
            "train_logs": "/train/logs/{task_id}",
            "train_list": "/train/list"
        }
    }


@app.get("/health", tags=["Basic"])
async def health_check():
    """Health check"""
    repo_exists = REPO_PATH.exists()
    script_exists = (REPO_PATH / TRAIN_SCRIPT).exists()
    
    return {
        "status": "healthy" if (repo_exists and script_exists) else "unhealthy",
        "repository_exists": repo_exists,
        "train_script_exists": script_exists,
        "active_tasks": len([t for t in tasks.values() if t["status"] == "running"])
    }


@app.post("/train/start", response_model=TrainResponse, tags=["Training"])
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    """
    Start training task
    
    Start PPO training process based on provided configuration.
    
    Args:
        request: Training request parameters
        background_tasks: FastAPI background task manager
        
    Returns:
        TrainResponse: Training response, containing task ID and status
        
    Raises:
        HTTPException: When startup fails
    """
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        # Verify repository path
        if not REPO_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Repository path does not exist: {REPO_PATH}"
            )
        
        # Verify training script
        train_script_path = REPO_PATH / TRAIN_SCRIPT
        if not train_script_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Training script does not exist: {train_script_path}"
            )
        
        # Build training command
        cmd = build_train_command(
            config_file=request.config_file,
            config_params=request.config_params
        )
        command_str = " ".join(cmd)
        
        logger.info(f"Creating training task [{task_id}]: {request.task_name}")
        logger.info(f"Training command: {command_str}")
        
        # Initialize task status
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "error": None,
                "created_at": timestamp,
                "updated_at": timestamp
            }
        
        # Add background training task
        background_tasks.add_task(_run_training_task, task_id, cmd)
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message=f"Training task started: {request.task_name}",
            command=command_str,
            log_preview="Training will start soon...",
            timestamp=timestamp
        )
    
    except ValueError as e:
        logger.error(f"Parameter validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Failed to start training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@app.get("/train/status/{task_id}", response_model=TaskStatus, tags=["Training"])
async def get_task_status(task_id: str) -> TaskStatus:
    """
    Get training task status
    
    Args:
        task_id: Task ID
        
    Returns:
        TaskStatus: Task status information
        
    Raises:
        HTTPException: When task does not exist
    """
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = tasks[task_id]
        
        return TaskStatus(
            task_id=task["task_id"],
            status=task["status"],
            command=task["command"],
            pid=task.get("pid"),
            log_lines=len(task["log"]),
            recent_logs=list(task["log"])[-20:],  # Latest 20 lines
            error=task.get("error"),
            created_at=task["created_at"],
            updated_at=task["updated_at"]
        )


@app.get("/train/logs/{task_id}", tags=["Training"])
async def get_task_logs(
    task_id: str,
    lines: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get training task logs
    
    Args:
        task_id: Task ID
        lines: Number of log lines to return
        offset: Log offset (from end)
        
    Returns:
        dict: Dictionary containing logs
        
    Raises:
        HTTPException: When task does not exist
    """
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = tasks[task_id]
        all_logs = list(task["log"])
        total_lines = len(all_logs)
        
        # Calculate log slice
        start = max(0, total_lines - offset - lines)
        end = total_lines - offset
        
        return {
            "task_id": task_id,
            "total_lines": total_lines,
            "offset": offset,
            "lines": lines,
            "logs": all_logs[start:end]
        }


@app.get("/train/list", tags=["Training"])
async def list_tasks() -> Dict[str, Any]:
    """
    List all training tasks
    
    Returns:
        dict: Task list
    """
    with tasks_lock:
        task_list = []
        for task_id, task in tasks.items():
            task_list.append({
                "task_id": task_id,
                "task_name": task.get("task_name", "unknown"),
                "status": task["status"],
                "pid": task.get("pid"),
                "created_at": task["created_at"],
                "updated_at": task["updated_at"]
            })
        
        return {
            "total": len(task_list),
            "tasks": sorted(task_list, key=lambda x: x["created_at"], reverse=True)
        }


@app.delete("/train/{task_id}", tags=["Training"])
async def delete_task(task_id: str) -> Dict[str, str]:
    """
    Delete training task record
    
    Note: Will not terminate running processes
    
    Args:
        task_id: Task ID
        
    Returns:
        dict: Deletion result
        
    Raises:
        HTTPException: When task does not exist or is running
    """
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = tasks[task_id]
        if task["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running task, please stop task first"
            )
        
        del tasks[task_id]
        logger.info(f"Task record deleted: {task_id}")
        
        return {"message": f"Task deleted: {task_id}"}


# ========== Main Program Entry ==========

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Training MCP Service - PPO from AlphaSAGE repository")
    logger.info(f"Repository path: {REPO_PATH}")
    logger.info(f"Training script: {TRAIN_SCRIPT}")
    logger.info(f"Python version: {sys.version}")
    logger.info("=" * 60)
    
    # Verify environment
    if not REPO_PATH.exists():
        logger.error(f"Error: Repository path does not exist: {REPO_PATH}")
        sys.exit(1)
    
    if not (REPO_PATH / TRAIN_SCRIPT).exists():
        logger.error(f"Error: Training script does not exist: {REPO_PATH / TRAIN_SCRIPT}")
        sys.exit(1)
    
    # Start service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info",
        access_log=True
    )