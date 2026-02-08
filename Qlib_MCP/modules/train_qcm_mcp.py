"""
Training MCP Service - AlphaQCM
Algorithm Type: rl
Auto-generated Time: 2024
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import uvicorn
import uuid
import logging
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import threading
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_mcp.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Training MCP - AlphaQCM",
    description="Training Service - AlphaQCM (Quantile-based Distributional RL for Alpha Mining)",
    version="1.0.0"
)

# ========== Configuration Constants ==========

REPO_PATH = Path(r"./Qlib_MCP/workspace/AlphaSAGE").resolve()
TRAIN_SCRIPT = "train_qcm.py"
MAX_LOG_LINES = 1000  # Maximum retained log lines

# Task storage (production environment should use Redis or other persistent storage)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()


# ========== Pydantic Models ==========

class TrainRequest(BaseModel):
    """Training request model"""
    
    model: str = Field(
        default="iqn",
        description="Model type: qrdqn, iqn, fqf"
    )
    seed: int = Field(
        default=0,
        description="Random seed",
        ge=0
    )
    pool: int = Field(
        default=100,
        description="Alpha pool capacity",
        ge=1
    )
    std_lam: float = Field(
        default=1.0,
        description="Standard deviation lambda parameter",
        gt=0
    )
    instruments: str = Field(
        default="csi300",
        description="Stock pool name: csi300, csi500, csi800, csi1000, csiall, sp500, etc."
    )
    train_end_year: int = Field(
        default=2020,
        description="Training end year",
        ge=2000,
        le=2030
    )
    config_file: Optional[str] = Field(
        None,
        description="Custom configuration file path (relative to repository root)"
    )
    task_name: str = Field(
        default="training",
        description="Task name",
        min_length=1,
        max_length=100
    )
    
    @validator('model')
    def validate_model(cls, v: str) -> str:
        """Validate model type"""
        allowed_models = ['qrdqn', 'iqn', 'fqf']
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {allowed_models}")
        return v
    
    @validator('pool')
    def validate_pool(cls, v: int) -> int:
        """Validate pool size"""
        recommended_pools = [10, 20, 50, 100]
        if v not in recommended_pools:
            logger.warning(f"Pool size {v} is not in recommended values {recommended_pools}")
        return v
    
    @validator('std_lam')
    def validate_std_lam(cls, v: float) -> float:
        """Validate std_lam parameter"""
        recommended_lams = [0.5, 1.0, 2.0]
        if v not in recommended_lams:
            logger.warning(f"std_lam {v} is not in recommended values {recommended_lams}")
        return v
    
    @validator('instruments')
    def validate_instruments(cls, v: str) -> str:
        """Validate instruments parameter"""
        common_instruments = ['csi300', 'csi500', 'csi800', 'csi1000', 'csiall', 'sp500']
        if v not in common_instruments:
            logger.warning(f"instruments {v} is not in common values {common_instruments}, but will continue to use")
        return v


class TrainResponse(BaseModel):
    """Training response model"""
    
    task_id: str = Field(..., description="Task unique identifier")
    status: str = Field(..., description="Task status: started, running, completed, failed")
    message: str = Field(..., description="Status message")
    command: str = Field(..., description="Executed training command")
    log_preview: str = Field(default="", description="Log preview (latest few lines)")
    task_name: str = Field(..., description="Task name")
    created_at: str = Field(..., description="Creation time")


class TaskStatus(BaseModel):
    """Task status model"""
    
    task_id: str
    status: str
    command: str
    task_name: str
    created_at: str
    pid: Optional[int] = None
    log_lines: int = 0
    last_log: str = ""
    error: Optional[str] = None


class LogResponse(BaseModel):
    """Log response model"""
    
    task_id: str
    total_lines: int
    logs: List[str]
    status: str


# ========== Utility Functions ==========

def setup_environment() -> Dict[str, str]:
    """
    Set up environment variables required for training
    
    Returns:
        Environment variable dictionary
    """
    env = os.environ.copy()
    
    # Add repository path to PYTHONPATH
    env["PYTHONPATH"] = str(REPO_PATH) + os.pathsep + env.get("PYTHONPATH", "")
    
    # Disable Python output buffering to ensure real-time log retrieval
    env["PYTHONUNBUFFERED"] = "1"
    
    # Set CUDA-related environment variables (if needed)
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    
    logger.info(f"Environment variables set: PYTHONPATH={env['PYTHONPATH'][:100]}...")
    return env


def validate_repository() -> bool:
    """
    Verify that repository path and training script exist
    
    Returns:
        Whether validation passed
    
    Raises:
        FileNotFoundError: When path or file does not exist
    """
    # Repository/workspace path check (host machine)
    if not REPO_PATH.exists():
        raise FileNotFoundError(f"Repository path does not exist: {REPO_PATH}")
    
    train_script_path = REPO_PATH / TRAIN_SCRIPT
    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script does not exist: {train_script_path}")

    # Additional checks for docker-compose and qlib data directory (reference GP_AlphaSAGE)
    project_root = Path(__file__).parent.parent.resolve()
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        raise FileNotFoundError(f"docker-compose.yml does not exist: {compose_file}")

    qlib_data_host = Path.home() / ".qlib" / "qlib_data"
    if not qlib_data_host.exists():
        raise FileNotFoundError(f"Qlib data directory does not exist: {qlib_data_host}")
    
    logger.info(f"Repository and Docker environment validation passed: REPO_PATH={REPO_PATH}, compose={compose_file}")
    return True


def build_train_command(request: TrainRequest) -> List[str]:
    """
    Build training command based on request
    
    Args:
        request: Training request object
    
    Returns:
        Command list (call AlphaSAGE container via docker compose run)
    """
    # Reference train_GP_AlphaSAGE_MCP.py, run container via docker compose
    project_root = Path(__file__).parent.parent.resolve()
    compose_file = project_root / "docker-compose.yml"

    cmd: List[str] = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        "--name",
        f"alphasage-qcm-{request.model}-{request.pool}-{request.seed}",
    ]

    # Pass necessary environment variables to container (can be simplified if not needed)
    cmd.extend(
        [
            "-e",
            "PYTHONUNBUFFERED=1",
        ]
    )

    # If host machine has CUDA_VISIBLE_DEVICES set, pass it to container
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        cmd.extend(["-e", f"CUDA_VISIBLE_DEVICES={cuda_visible}"])

    # Docker Compose service name (should match docker-compose.yml)
    service_name = "alphasage"
    cmd.append(service_name)

    # Actual command executed in container
    cmd.extend(
        [
            "python3",
            TRAIN_SCRIPT,
            "--model",
            request.model,
            "--seed",
            str(request.seed),
            "--pool",
            str(request.pool),
            "--std-lam",
            str(request.std_lam),
            "--instruments",
            request.instruments,
            "--train-end-year",
            str(request.train_end_year),
        ]
    )

    return cmd


def get_log_preview(logs: deque, lines: int = 10) -> str:
    """
    Get log preview
    
    Args:
        logs: Log queue
        lines: Preview line count
    
    Returns:
        Log preview string
    """
    if not logs:
        return ""
    
    preview_lines = list(logs)[-lines:]
    return "\n".join(preview_lines)


# ========== Background Task Function ==========

def _run_training(task_id: str, cmd: List[str]) -> None:
    """
    Execute training task in background
    
    Args:
        task_id: Task ID
        cmd: Training command list
    """
    try:
        env = setup_environment()
        
        logger.info(f"Starting training process [{task_id}]: {' '.join(cmd)}")
        
        # Start subprocess
        # Use project root directory as working directory to ensure docker compose can correctly find docker-compose.yml
        project_root = Path(__file__).parent.parent.resolve()

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Update task status
        with tasks_lock:
            tasks[task_id]["pid"] = proc.pid
            tasks[task_id]["status"] = "running"
        
        logger.info(f"Training process started [{task_id}], PID: {proc.pid}")
        
        # Get original stderr (for real-time output to terminal)
        original_stderr = sys.stderr if hasattr(sys, "_original_stderr") else sys.__stderr__
        
        # Read and store output in real-time, also print to terminal
        error_lines: List[str] = []  # Collect possible error information
        
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            
            line_stripped = line.rstrip()
            
            # Simple error keyword detection
            line_lower = line_stripped.lower()
            if any(keyword in line_lower for keyword in ["error", "exception", "traceback", "failed"]):
                error_lines.append(line_stripped)
                if len(error_lines) > 20:
                    error_lines.pop(0)
            
            # Write to terminal (stderr), so you can see qcm training output in real-time when calling MCP
            print(f"[Training {task_id}] {line_stripped}", file=original_stderr, flush=True)
            
            with tasks_lock:
                tasks[task_id]["log"].append(line_stripped)
                # Limit log size
                if len(tasks[task_id]["log"]) > MAX_LOG_LINES:
                    tasks[task_id]["log"].popleft()
            
            logger.info(f"[Training {task_id[:8]}] {line_stripped}")
        
        # Wait for process to complete
        return_code = proc.wait()
        
        # Get last few log lines for error diagnosis
        with tasks_lock:
            log_list = list(tasks[task_id]["log"])
            last_logs = log_list[-30:] if len(log_list) >= 30 else log_list
        
        # Update final status
        with tasks_lock:
            if return_code == 0:
                tasks[task_id]["status"] = "completed"
                logger.info(f"Training task completed [{task_id}]")
                print(f"[Training {task_id}] Training task completed successfully", file=original_stderr, flush=True)
            else:
                tasks[task_id]["status"] = "failed"
                # Build detailed error information
                error_msg_parts = [f"Process exit code: {return_code}"]
                
                if error_lines:
                    error_msg_parts.append("\nDetected error information:")
                    error_msg_parts.extend(error_lines[-10:])
                
                if last_logs:
                    error_msg_parts.append("\nLast few log lines:")
                    error_msg_parts.extend(last_logs[-10:])
                
                error_msg = "\n".join(error_msg_parts)
                tasks[task_id]["error"] = error_msg
                
                logger.error(f"Training task failed [{task_id}], exit code: {return_code}")
                logger.error(f"Error details:\n{error_msg}")
                print(f"[Training {task_id}] Training task failed, exit code: {return_code}", file=original_stderr, flush=True)
                print(f"[Training {task_id}] Error details:\n{error_msg}", file=original_stderr, flush=True)
    
    except subprocess.SubprocessError as e:
        logger.error(f"Training task subprocess exception [{task_id}]: {e}", exc_info=True)
        original_stderr = sys.stderr if hasattr(sys, "_original_stderr") else sys.__stderr__
        print(f"[Training {task_id}] Subprocess exception: {e}", file=original_stderr, flush=True)
        
        with tasks_lock:
            log_list = list(tasks[task_id]["log"])
            last_logs = log_list[-20:] if len(log_list) >= 20 else log_list
            
            error_msg = f"Subprocess exception: {str(e)}"
            if last_logs:
                error_msg += "\n\nCollected logs:\n" + "\n".join(last_logs)
            
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = error_msg
    
    except Exception as e:
        logger.error(f"Training task exception [{task_id}]: {e}", exc_info=True)
        original_stderr = sys.stderr if hasattr(sys, "_original_stderr") else sys.__stderr__
        print(f"[Training {task_id}] Execution exception: {e}", file=original_stderr, flush=True)
        
        with tasks_lock:
            log_list = list(tasks[task_id]["log"])
            last_logs = log_list[-20:] if len(log_list) >= 20 else log_list
            
            error_msg = f"Execution exception: {str(e)}\nException type: {type(e).__name__}"
            if last_logs:
                error_msg += "\n\nCollected logs:\n" + "\n".join(last_logs)
            
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = error_msg


# ========== Utility Functions ==========

def _find_latest_factor_file(
    model: str,
    seed: int,
    pool: int,
    std_lam: float,
    instruments: str
) -> Optional[str]:
    """
    Find the latest generated factor file (valid_best_factors.json)
    
    Args:
        model: Model type (qrdqn/iqn/fqf)
        seed: Random seed
        pool: Alpha pool capacity
        std_lam: Standard deviation lambda parameter
        instruments: Stock pool name
        
    Returns:
        Optional[str]: Absolute path of factor file, or None if not found
    """
    # QCM save directory format: data/{instruments}_logs/pool_{pool}_QCM_{std_lam}/{model}-seed{seed}-{timestamp}-.../
    data_dir = REPO_PATH / "data" / f"{instruments}_logs"
    
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return None
    
    # Build second-level directory name: pool_{pool}_QCM_{std_lam}
    pool_dir_name = f"pool_{pool}_QCM_{std_lam}"
    pool_dir = data_dir / pool_dir_name
    
    if not pool_dir.exists():
        logger.warning(f"Pool directory does not exist: {pool_dir}")
        return None
    
    # Build directory name prefix pattern: {model}-seed{seed}-
    dir_prefix = f"{model}-seed{seed}-"
    
    # Find all matching directories
    matching_dirs = [
        d for d in pool_dir.iterdir()
        if d.is_dir() and d.name.startswith(dir_prefix)
    ]
    
    if not matching_dirs:
        logger.warning(f"No matching directories found, prefix: {dir_prefix}, in directory: {pool_dir}")
        return None
    
    # Sort by directory modification time, find latest directory (last created)
    matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_dir = matching_dirs[0]
    
    logger.info(f"Found latest directory: {latest_dir}")
    
    # Find valid_best_factors.json file in latest directory
    factor_file = latest_dir / "valid_best_factors.json"
    
    if not factor_file.exists():
        logger.warning(f"valid_best_factors.json file not found in directory: {latest_dir}")
        return None
    
    factor_file_path = str(factor_file.resolve())
    logger.info(f"Found latest generated factor file: {factor_file_path}")
    
    return factor_file_path


# ========== Module Interface Function ==========

def run(args) -> str:
    """
    Module interface function, run AlphaSAGE's train_qcm.py via Docker Compose
    Synchronously wait for training completion, reference train_GP_AlphaSAGE_MCP.py implementation
    
    Args:
        args: SimpleNamespace object containing the following attributes:
            - model: Model type (qrdqn/iqn/fqf), default "qrdqn"
            - seed: Random seed, default 0
            - pool: Alpha pool capacity, default 20
            - std_lam: Standard deviation lambda parameter, default 1.0
            - instruments: Stock pool name, default "csi300"
            - train_end_year: Training end year, default 2020
            - task_name: Task name, default "training"
    
    Returns:
        str: Full path of latest generated factor file (valid_best_factors.json), or error message
    """
    try:
        # Check if docker-compose.yml exists
        project_root = Path(__file__).parent.parent.resolve()
        compose_file = project_root / "docker-compose.yml"
        if not compose_file.exists():
            error_msg = f"docker-compose.yml does not exist: {compose_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check workspace path
        if not REPO_PATH.exists():
            error_msg = f"AlphaSAGE workspace does not exist: {REPO_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check qlib data path
        qlib_data_host = Path.home() / ".qlib" / "qlib_data"
        if not qlib_data_host.exists():
            error_msg = f"Qlib data directory does not exist: {qlib_data_host}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Build TrainRequest from args
        request = TrainRequest(
            model=getattr(args, "model", "iqn"),
            seed=getattr(args, "seed", 0),
            pool=getattr(args, "pool", 100),
            std_lam=getattr(args, "std_lam", 1.0),
            instruments=getattr(args, "instruments", "csi300"),
            train_end_year=getattr(args, "train_end_year", 2020),
            task_name=getattr(args, "task_name", "training")
        )
        
        # Build docker compose command
        cmd = build_train_command(request)
        cmd_str = " ".join(cmd)
        
        logger.info(f"Executing Docker Compose command: {cmd_str}")
        
        # Run Docker Compose (synchronous execution, wait for completion)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(project_root),
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
                f"train_qcm_AlphaSAGE Docker Compose execution failed\n"
                f"Exit code: {proc.returncode}\n"
                f"Command: {cmd_str}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Find latest generated factor file
        latest_factor_file = _find_latest_factor_file(
            model=request.model,
            seed=request.seed,
            pool=request.pool,
            std_lam=request.std_lam,
            instruments=request.instruments
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


# ========== API Endpoints ==========

@app.on_event("startup")
async def startup_event() -> None:
    """Service initialization on startup"""
    try:
        validate_repository()
        logger.info("Training MCP service started successfully")
    except Exception as e:
        logger.error(f"Service startup failed: {e}", exc_info=True)
        raise


@app.get("/", tags=["Basic"])
async def root() -> Dict[str, Any]:
    """
    Service root path, returns service information
    
    Returns:
        Basic service information
    """
    return {
        "service": "Training MCP - AlphaQCM",
        "algorithm": "AlphaQCM (Quantile-based Distributional RL for Alpha Mining)",
        "algorithm_type": "rl",
        "version": "1.0.0",
        "repository": str(REPO_PATH),
        "train_script": TRAIN_SCRIPT,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "train_start": "POST /train/start",
            "train_status": "GET /train/status/{task_id}",
            "train_logs": "GET /train/logs/{task_id}",
            "train_list": "GET /train/list"
        },
        "supported_models": ["qrdqn", "iqn", "fqf"],
        "recommended_pools": [10, 20, 50, 100],
        "recommended_std_lam": [0.5, 1.0, 2.0]
    }


@app.get("/health", tags=["Basic"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    try:
        validate_repository()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/train/start", response_model=TrainResponse, tags=["Training"])
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    """
    Start training task
    
    Args:
        request: Training request parameters
        background_tasks: FastAPI background tasks
    
    Returns:
        Training task information
    
    Raises:
        HTTPException: When startup fails
    """
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    try:
        # Validate repository
        validate_repository()
        
        # Build training command
        cmd = build_train_command(request)
        command_str = " ".join(cmd)
        
        logger.info(f"Preparing to start training task [{task_id}]: {command_str}")
        
        # Initialize task record
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "error": None
            }
        
        # Add background task
        background_tasks.add_task(_run_training, task_id, cmd)
        
        # Get log preview
        log_preview = f"Training task submitted, waiting to start...\nCommand: {command_str}"
        
        logger.info(f"Training task submitted [{task_id}]")
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message="Training task started successfully",
            command=command_str,
            log_preview=log_preview,
            task_name=request.task_name,
            created_at=created_at
        )
    
    except FileNotFoundError as e:
        logger.error(f"File does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
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
        Task status information
    
    Raises:
        HTTPException: When task does not exist
    """
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = tasks[task_id]
        last_log = list(task["log"])[-1] if task["log"] else ""
        
        return TaskStatus(
            task_id=task_id,
            status=task["status"],
            command=task["command"],
            task_name=task["task_name"],
            created_at=task["created_at"],
            pid=task["pid"],
            log_lines=len(task["log"]),
            last_log=last_log,
            error=task.get("error")
        )


@app.get("/train/logs/{task_id}", response_model=LogResponse, tags=["Training"])
async def get_task_logs(
    task_id: str,
    lines: int = 100,
    offset: int = 0
) -> LogResponse:
    """
    Get training task logs
    
    Args:
        task_id: Task ID
        lines: Number of log lines to return
        offset: Offset (from end)
    
    Returns:
        Log information
    
    Raises:
        HTTPException: When task does not exist
    """
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = tasks[task_id]
        all_logs = list(task["log"])
        total_lines = len(all_logs)
        
        # Calculate log range
        start_idx = max(0, total_lines - offset - lines)
        end_idx = max(0, total_lines - offset)
        
        logs = all_logs[start_idx:end_idx]
        
        return LogResponse(
            task_id=task_id,
            total_lines=total_lines,
            logs=logs,
            status=task["status"]
        )


@app.get("/train/list", tags=["Training"])
async def list_tasks() -> Dict[str, Any]:
    """
    List all training tasks
    
    Returns:
        Task list
    """
    with tasks_lock:
        task_list = [
            {
                "task_id": task_id,
                "task_name": task["task_name"],
                "status": task["status"],
                "created_at": task["created_at"],
                "pid": task["pid"]
            }
            for task_id, task in tasks.items()
        ]
        
        return {
            "total": len(task_list),
            "tasks": task_list
        }


@app.delete("/train/{task_id}", tags=["Training"])
async def delete_task(task_id: str) -> Dict[str, str]:
    """
    Delete training task record (will not terminate running processes)
    
    Args:
        task_id: Task ID
    
    Returns:
        Deletion confirmation information
    
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
                detail="Cannot delete running task, please wait for task to complete or fail"
            )
        
        del tasks[task_id]
        logger.info(f"Task deleted: {task_id}")
        
        return {"message": f"Task {task_id} deleted"}


# ========== Main Program Entry ==========

if __name__ == "__main__":
    """
    Main program entry
    
    Start FastAPI server
    """
    logger.info("=" * 60)
    logger.info("Starting Training MCP Service - AlphaQCM")
    logger.info(f"Repository path: {REPO_PATH}")
    logger.info(f"Training script: {TRAIN_SCRIPT}")
    logger.info("=" * 60)
    
    try:
        # Pre-startup validation
        validate_repository()
        
        # Start service
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8010,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Service startup failed: {e}", exc_info=True)
        sys.exit(1)