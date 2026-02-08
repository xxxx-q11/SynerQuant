"""
Qlib Benchmark Runner - FastAPI Service
Provides Qlib benchmark model management and execution functionality using FastAPI framework
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
import yaml
import re
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qlib_benchmark_runner.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qlib Benchmark Runner",
    description="Qlib Benchmark Model Management and Execution Service",
    version="1.0.0"
)

# ========== Configuration Constants ==========

# Qlib_MCP root directory: .../trading_agent/Qlib_MCP
BASE_DIR = Path(__file__).resolve().parents[1]

# benchmarks directory: Qlib_MCP/workspace/qlib_benchmark/benchmarks
BENCHMARKS_DIR = (BASE_DIR / "workspace" / "qlib_benchmark" / "benchmarks").resolve()

# provider_uri can also be changed to not depend on hardcoded home directory (optional)
DEFAULT_PROVIDER_URI = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")

MAX_LOG_LINES = 2000  # Maximum retained log lines

# Task storage (production environment should use Redis or other persistent storage)
run_tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()


# ========== Pydantic Models ==========

class ModelInfo(BaseModel):
    """Model information model"""
    path: str = Field(..., description="YAML file path")
    filename: str = Field(..., description="Filename")
    model_class: str = Field(..., description="Model class")
    dataset: str = Field(..., description="Dataset")
    market: str = Field(..., description="Market")
    benchmark: str = Field(..., description="Benchmark index")
    provider_uri: str = Field(..., description="Current provider_uri")
    start_time: Optional[str] = Field(None, description="Data start time")
    end_time: Optional[str] = Field(None, description="Data end time")
    train_period: Optional[List] = Field(None, description="Training period")
    valid_period: Optional[List] = Field(None, description="Validation period")
    test_period: Optional[List] = Field(None, description="Test period")


class ModelListResponse(BaseModel):
    """Model list response"""
    total_models: int = Field(..., description="Number of models")
    total_configs: int = Field(..., description="Total number of configuration files")
    models: Dict[str, List[str]] = Field(..., description="Model dictionary")
    benchmarks_dir: str = Field(..., description="Benchmarks directory path")


class UpdateProviderRequest(BaseModel):
    """Update Provider URI request"""
    yaml_path: str = Field(..., description="YAML configuration file path")
    provider_uri: str = Field(..., description="New provider_uri path")


class UpdateProviderResponse(BaseModel):
    """Update Provider URI response"""
    success: bool = Field(..., description="Whether successful")
    yaml_path: str = Field(..., description="YAML file path")
    old_uri: str = Field(..., description="Old provider_uri")
    new_uri: str = Field(..., description="New provider_uri")
    message: str = Field(..., description="Message")


class RunModelRequest(BaseModel):
    """Run model request"""
    yaml_path: str = Field(..., description="YAML configuration file path")
    provider_uri: Optional[str] = Field("~/.qlib/qlib_data/cn_data", description="Optional: provider_uri (will be updated before running)")
    experiment_name: Optional[str] = Field(None, description="Optional: experiment name")
    task_name: str = Field(default="qlib_training", description="Task name")


class RunModelResponse(BaseModel):
    """Run model response"""
    task_id: str = Field(..., description="Task unique identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")
    command: str = Field(..., description="Executed command")
    yaml_path: str = Field(..., description="YAML configuration path")
    provider_uri: Optional[str] = Field(None, description="Provider URI used")
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    task_name: str = Field(..., description="Task name")
    created_at: str = Field(..., description="Creation time")


class TaskStatus(BaseModel):
    """Task status model"""
    task_id: str
    status: str
    command: str
    task_name: str
    yaml_path: str
    created_at: str
    pid: Optional[int] = None
    log_lines: int = 0
    last_log: str = ""
    error: Optional[str] = None
    exit_code: Optional[int] = None


class LogResponse(BaseModel):
    """Log response model"""
    task_id: str
    total_lines: int
    logs: List[str]
    status: str


class QrunStatusResponse(BaseModel):
    """Qrun status response"""
    available: bool = Field(..., description="Whether qrun is available")
    message: str = Field(..., description="Status message")
    version_info: Optional[str] = Field(None, description="Version information")


# ========== Utility Functions ==========

def scan_benchmark_models() -> Dict[str, List[str]]:
    """
    Scan benchmarks directory and return all available models
    
    Returns:
        Model dictionary, keys are model names, values are YAML file path lists
    """
    models = {}
    
    if not BENCHMARKS_DIR.exists():
        logger.warning(f"Benchmarks directory does not exist: {BENCHMARKS_DIR}")
        return models
    
    # Iterate through each model directory
    for model_dir in BENCHMARKS_DIR.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith('.'):
            # Find all yaml files
            yaml_files = list(model_dir.glob("workflow_config_*.yaml"))
            if yaml_files:
                models[model_dir.name] = [str(f) for f in yaml_files]
    
    logger.info(f"Scanned {len(models)} model types")
    return models


def get_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """
    Read YAML configuration file
    
    Args:
        yaml_path: YAML file path
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: File does not exist
        yaml.YAMLError: YAML parsing error
    """
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_model_info(yaml_path: str) -> ModelInfo:
    """
    Extract model information from YAML configuration file
    
    Args:
        yaml_path: YAML file path
        
    Returns:
        Model information object
    """
    from datetime import date, datetime
    
    def convert_to_str(value):
        """Convert date objects to strings"""
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, list):
            return [convert_to_str(item) for item in value]
        return str(value) if value else None
    
    config = get_yaml_config(yaml_path)
    
    # Extract basic information
    task_config = config.get('task', {})
    model_config = task_config.get('model', {})
    dataset_config = task_config.get('dataset', {})
    dataset_kwargs = dataset_config.get('kwargs', {})
    handler_config = dataset_kwargs.get('handler', {})
    handler_kwargs = handler_config.get('kwargs', {})
    
    # Extract and convert date fields
    start_time = convert_to_str(handler_kwargs.get('start_time'))
    end_time = convert_to_str(handler_kwargs.get('end_time'))
    train_period = convert_to_str(dataset_kwargs.get('segments', {}).get('train'))
    valid_period = convert_to_str(dataset_kwargs.get('segments', {}).get('valid'))
    test_period = convert_to_str(dataset_kwargs.get('segments', {}).get('test'))
    
    info = ModelInfo(
        path=yaml_path,
        filename=os.path.basename(yaml_path),
        model_class=model_config.get('class', 'Unknown'),
        dataset=handler_config.get('class', 'Unknown'),
        market=config.get('market', 'Unknown'),
        benchmark=config.get('benchmark', 'Unknown'),
        provider_uri=config.get('qlib_init', {}).get('provider_uri', 'Not specified'),
        start_time=start_time,
        end_time=end_time,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    return info


def update_yaml_provider_uri(yaml_path: str, provider_uri: str) -> tuple[str, str]:
    """
    Update provider_uri in YAML configuration file
    
    Args:
        yaml_path: YAML file path
        provider_uri: New provider_uri
        
    Returns:
        (old URI, new URI) tuple
    """
    config = get_yaml_config(yaml_path)
    
    # Get old URI
    old_uri = config.get('qlib_init', {}).get('provider_uri', 'Not set')
    
    # Update provider_uri
    if 'qlib_init' not in config:
        config['qlib_init'] = {}
    config['qlib_init']['provider_uri'] = provider_uri
    
    # Write back to file
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Updated provider_uri: {yaml_path}")
    logger.info(f"  Old URI: {old_uri}")
    logger.info(f"  New URI: {provider_uri}")
    
    return old_uri, provider_uri


def check_qrun_available() -> tuple[bool, str, Optional[str]]:
    """
    Check if qrun command is available
    
    Returns:
        (whether available, message, version info) tuple
    """
    try:
        result = subprocess.run(
            ["qrun", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True, "qrun command available", result.stdout[:200]
        else:
            return False, "qrun command returned error", None
            
    except FileNotFoundError:
        return False, "qrun command not found, please install pyqlib", None
    except Exception as e:
        return False, f"Error checking qrun: {str(e)}", None


def build_qrun_command(yaml_path: str, experiment_name: Optional[str] = None) -> List[str]:
    """
    Build qrun command (run qlib-benchmark container via Docker Compose)
    
    Args:
        yaml_path: YAML configuration file path (host path, will be mounted into container)
        experiment_name: Optional experiment name
        
    Returns:
        Command list (call qlib-benchmark container via docker compose run)
    """
    # Reference train_qcm_mcp.py, run container via docker compose
    project_root = Path(__file__).parent.parent.resolve()
    compose_file = project_root / "docker-compose.yml"
    
    # Convert host path to container path
    # Host path: ./workspace/qlib_benchmark/benchmarks/.../workflow_config_*.yaml
    # Container path: /workspace/qlib_benchmark/benchmarks/.../workflow_config_*.yaml
    # Note: docker-compose.yml mounts ./workspace:/workspace
    yaml_path_obj = Path(yaml_path).resolve()
    workspace_root = project_root / "workspace"
    
    try:
        # Try to convert to path relative to workspace
        relative_path = yaml_path_obj.relative_to(workspace_root)
        container_yaml_path = f"/workspace/{relative_path}"
    except ValueError:
        # If cannot convert, may be absolute path but not under workspace, try to use directly
        # Or assume path is already relative to benchmarks
        if "benchmarks" in str(yaml_path_obj):
            # Extract part after benchmarks
            parts = str(yaml_path_obj).split("benchmarks")
            if len(parts) > 1:
                # Need to include qlib_benchmark layer
                container_yaml_path = f"/workspace/qlib_benchmark/benchmarks{parts[1]}"
            else:
                container_yaml_path = str(yaml_path_obj)
        else:
            container_yaml_path = str(yaml_path_obj)
    
    cmd: List[str] = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        "--name",
        f"qlib-benchmark-{Path(yaml_path).stem}-{uuid.uuid4().hex[:8]}",
    ]
    
    # Pass necessary environment variables to container
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
    service_name = "qlib-benchmark"
    cmd.append(service_name)
    
    # Actual command executed in container
    cmd.extend(
        [
            "python3",
            "-m",
            "qlib.cli.run",
            container_yaml_path,
        ]
    )
    
    if experiment_name:
        cmd.extend(["--experiment_name", experiment_name])
    
    return cmd


# ========== Module Interface Functions ==========

def list_models(args=None) -> None:
    """
    Module interface function for listing all available benchmark models
    
    Args:
        args: SimpleNamespace object (optional, this function does not need parameters)
    """
    models = scan_benchmark_models()
    total_configs = sum(len(files) for files in models.values())
    
    logger.info(f"Listing models: {len(models)} model types, {total_configs} configurations")
    
    # Print results
    print("=" * 80)
    print("Qlib Benchmark Model List")
    print("=" * 80)
    print(f"Number of model types: {len(models)}")
    print(f"Total configuration files: {total_configs}")
    print(f"Benchmarks directory: {BENCHMARKS_DIR}")
    print()
    
    for model_name, yaml_files in models.items():
        print(f"Model: {model_name} ({len(yaml_files)} configurations)")
        for yaml_file in yaml_files:
            print(f"  - {yaml_file}")
        print()
    
    print("=" * 80)


def run(args) -> str:
    """
    Module interface function for calling by other modules
    Run qlib-benchmark container via Docker Compose, synchronously wait for training completion
    Reference train_qcm_mcp.py implementation
    
    Args:
        args: SimpleNamespace object containing the following attributes:
            - yaml_path: YAML configuration file path (required)
            - provider_uri: provider_uri path (optional), will be updated to YAML file before running
            - experiment_name: Experiment name (optional)
            - task_name: Task name, default "qlib_training"
    
  Returns:
        dict: Dictionary containing training status, experiment_id, recorder_id, and data path information
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
        workspace_path = project_root / "workspace" / "qlib_benchmark"
        if not workspace_path.exists():
            error_msg = f"qlib_benchmark workspace does not exist: {workspace_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check qlib data path
        qlib_data_host = Path.home() / ".qlib" / "qlib_data"
        if not qlib_data_host.exists():
            error_msg = f"Qlib data directory does not exist: {qlib_data_host}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Get parameters from args
        yaml_path = getattr(args, "yaml_path", None)
        if not yaml_path:
            raise ValueError("yaml_path parameter is required")
        
        provider_uri = getattr(args, "provider_uri", None)
        experiment_name = getattr(args, "experiment_name", None)
        task_name = getattr(args, "task_name", "qlib_training")
        
        # If provider_uri is provided, update YAML file
        if provider_uri:
            old_uri, new_uri = update_yaml_provider_uri(yaml_path, provider_uri)
            logger.info(f"Updated provider_uri: {old_uri} -> {new_uri}")
        
        # Build docker compose command
        cmd = build_qrun_command(yaml_path, experiment_name)
        cmd_str = " ".join(cmd)
        
        logger.info(f"Executing Docker Compose command: {cmd_str}")
        
        # Variables for extracting experiment_id and recorder_id
        experiment_id = None
        recorder_id = None
        
        # Regular expressions to match experiment_id and recorder_id
        exp_pattern = re.compile(r'Experiment (\d+) starts running')
        recorder_pattern = re.compile(r'Recorder ([a-f0-9]+) starts running under Experiment (\d+)')
        
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
            
            # Try to extract experiment_id
            exp_match = exp_pattern.search(line)
            if exp_match:
                experiment_id = exp_match.group(1)
                logger.info(f"Detected Experiment ID: {experiment_id}")
            
            # Try to extract recorder_id (usually recorder_id appears after experiment_id)
            recorder_match = recorder_pattern.search(line)
            if recorder_match:
                recorder_id = recorder_match.group(1)
                if not experiment_id:
                    experiment_id = recorder_match.group(2)
                logger.info(f"Detected Recorder ID: {recorder_id}, Experiment ID: {experiment_id}")

        proc.wait()

        # Check return code
        if proc.returncode != 0:
            error_msg = (
                f"qlib_benchmark Docker Compose execution failed\n"
                f"Exit code: {proc.returncode}\n"
                f"Command: {cmd_str}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("qlib_benchmark Docker Compose execution completed")
        
        # Build return result
        result = {
            "status": "success",
            "message": "qlib_benchmark Docker Compose execution completed",
            "experiment_id": experiment_id,
            "recorder_id": recorder_id,
            "yaml_path": yaml_path,
            "experiment_name": experiment_name
        }
        
        # If experiment_id is found, build data paths
        if experiment_id:
            # mlruns directory is under workspace
            mlruns_base = project_root / "workspace" / "mlruns"
            experiment_path = mlruns_base / str(experiment_id)
            
            result["experiment_path"] = str(experiment_path)
            
            # If recorder_id is found, build complete artifacts paths
            if recorder_id:
                artifacts_path = experiment_path / recorder_id / "artifacts"
                portfolio_analysis_path = artifacts_path / "portfolio_analysis"
                metrics_path = experiment_path / recorder_id /  "metrics"
                
                result["artifacts_path"] = str(artifacts_path)
                result["portfolio_analysis_path"] = str(portfolio_analysis_path)
                result["port_analysis_file"] = str(portfolio_analysis_path / "port_analysis_1day.pkl")
                result["indicator_analysis_file"] = str(portfolio_analysis_path / "indicator_analysis_1day.pkl")
                result["pred_file"] = str(artifacts_path / "pred.pkl")
                result["label_file"] = str(artifacts_path / "label.pkl")
                result["report_file"] = str(portfolio_analysis_path / "report_normal_1day.pkl")
                result["positions_file"] = str(portfolio_analysis_path / "positions_normal_1day.pkl")
                result["ic"] = str(artifacts_path / "sig_analysis/ic.pkl")
                result["rank_ic"] = str(artifacts_path / "sig_analysis/ric.pkl")
                result[ "1day.excess_return_with_cost.annualized_return"] =str(metrics_path / "1day.excess_return_with_cost.annualized_return")
                result[ "1day.excess_return_with_cost.max_drawdown"] =str(metrics_path / "1day.excess_return_with_cost.max_drawdown")

        # Return dictionary
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        # Re-raise known errors
        raise
    
    except Exception as e:
        error_msg = f"Docker Compose execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ========== Background Task Function ==========

def _run_qlib_model(task_id: str, cmd: List[str], yaml_path: str) -> None:
    """
    Execute qlib model training task in background (via Docker Compose)
    
    Args:
        task_id: Task ID
        cmd: Training command list (docker compose command)
        yaml_path: YAML configuration file path (host path)
    """
    try:
        env = os.environ.copy()
        
        logger.info(f"Starting training process [{task_id}]: {' '.join(cmd)}")
        
        # Use project root directory as working directory to ensure docker compose can correctly find docker-compose.yml
        project_root = Path(__file__).parent.parent.resolve()
        
        # Disable Python output buffering
        env["PYTHONUNBUFFERED"] = "1"
        
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
            run_tasks[task_id]["pid"] = proc.pid
            run_tasks[task_id]["status"] = "running"
        
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
            
            # Write to terminal (stderr), so you can see training output in real-time when calling MCP
            print(f"[Training {task_id}] {line_stripped}", file=original_stderr, flush=True)
            
            with tasks_lock:
                run_tasks[task_id]["log"].append(line_stripped)
                # Limit log size
                if len(run_tasks[task_id]["log"]) > MAX_LOG_LINES:
                    run_tasks[task_id]["log"].popleft()
            
            logger.info(f"[Training {task_id[:8]}] {line_stripped}")
        
        # Wait for process to complete
        return_code = proc.wait()
        
        # Get last few log lines for error diagnosis
        with tasks_lock:
            log_list = list(run_tasks[task_id]["log"])
            last_logs = log_list[-30:] if len(log_list) >= 30 else log_list
        
        # Update final status
        with tasks_lock:
            run_tasks[task_id]["exit_code"] = return_code
            if return_code == 0:
                run_tasks[task_id]["status"] = "completed"
                logger.info(f"Training task completed [{task_id}]")
                print(f"[Training {task_id}] Training task completed successfully", file=original_stderr, flush=True)
            else:
                run_tasks[task_id]["status"] = "failed"
                # Build detailed error information
                error_msg_parts = [f"Process exit code: {return_code}"]
                
                if error_lines:
                    error_msg_parts.append("\nDetected error information:")
                    error_msg_parts.extend(error_lines[-10:])
                
                if last_logs:
                    error_msg_parts.append("\nLast few log lines:")
                    error_msg_parts.extend(last_logs[-10:])
                
                error_msg = "\n".join(error_msg_parts)
                run_tasks[task_id]["error"] = error_msg
                
                logger.error(f"Training task failed [{task_id}], exit code: {return_code}")
                logger.error(f"Error details:\n{error_msg}")
                print(f"[Training {task_id}] Training task failed, exit code: {return_code}", file=original_stderr, flush=True)
                print(f"[Training {task_id}] Error details:\n{error_msg}", file=original_stderr, flush=True)
    
    except subprocess.SubprocessError as e:
        logger.error(f"Training task subprocess exception [{task_id}]: {e}", exc_info=True)
        original_stderr = sys.stderr if hasattr(sys, "_original_stderr") else sys.__stderr__
        print(f"[Training {task_id}] Subprocess exception: {e}", file=original_stderr, flush=True)
        
        with tasks_lock:
            log_list = list(run_tasks[task_id]["log"])
            last_logs = log_list[-20:] if len(log_list) >= 20 else log_list
            
            error_msg = f"Subprocess exception: {str(e)}"
            if last_logs:
                error_msg += "\n\nCollected logs:\n" + "\n".join(last_logs)
            
            run_tasks[task_id]["status"] = "failed"
            run_tasks[task_id]["error"] = error_msg
    
    except Exception as e:
        logger.error(f"Training task exception [{task_id}]: {e}", exc_info=True)
        original_stderr = sys.stderr if hasattr(sys, "_original_stderr") else sys.__stderr__
        print(f"[Training {task_id}] Execution exception: {e}", file=original_stderr, flush=True)
        
        with tasks_lock:
            log_list = list(run_tasks[task_id]["log"])
            last_logs = log_list[-20:] if len(log_list) >= 20 else log_list
            
            error_msg = f"Execution exception: {str(e)}\nException type: {type(e).__name__}"
            if last_logs:
                error_msg += "\n\nCollected logs:\n" + "\n".join(last_logs)
            
            run_tasks[task_id]["status"] = "failed"
            run_tasks[task_id]["error"] = error_msg


# ========== API Endpoints ==========

@app.on_event("startup")
async def startup_event() -> None:
    """Service initialization on startup"""
    logger.info("=" * 60)
    logger.info("Qlib Benchmark Runner service starting")
    logger.info(f"Benchmarks directory: {BENCHMARKS_DIR}")
    logger.info(f"Default Provider URI: {DEFAULT_PROVIDER_URI}")
    logger.info("=" * 60)
    
    if not BENCHMARKS_DIR.exists():
        logger.warning(f"Warning: Benchmarks directory does not exist: {BENCHMARKS_DIR}")
    
    # Check qrun availability
    available, message, _ = check_qrun_available()
    if available:
        logger.info(f"[OK] {message}")
    else:
        logger.warning(f"[WARNING] {message}")


@app.get("/", tags=["Basic"])
async def root() -> Dict[str, Any]:
    """
    Service root path, returns service information
    
    Returns:
        Basic service information
    """
    return {
        "service": "Qlib Benchmark Runner",
        "description": "Qlib Benchmark Model Management and Execution Service",
        "version": "1.0.0",
        "benchmarks_dir": str(BENCHMARKS_DIR),
        "default_provider_uri": DEFAULT_PROVIDER_URI,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "list_models": "GET /models/list",
            "get_model_info": "GET /models/info",
            "update_provider_uri": "POST /models/update_provider",
            "run_model": "POST /models/run",
            "check_qrun": "GET /qrun/status",
            "task_status": "GET /tasks/status/{task_id}",
            "task_logs": "GET /tasks/logs/{task_id}",
            "task_list": "GET /tasks/list"
        }
    }


@app.get("/health", tags=["Basic"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    status = "healthy"
    details = []
    
    # Check benchmarks directory
    if not BENCHMARKS_DIR.exists():
        status = "degraded"
        details.append(f"Benchmarks directory does not exist: {BENCHMARKS_DIR}")
    
    # Check qrun
    available, message, _ = check_qrun_available()
    if not available:
        status = "degraded"
        details.append(message)
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "details": "; ".join(details) if details else "All OK"
    }


@app.get("/models/list", response_model=ModelListResponse, tags=["Model Management"])
async def list_models() -> ModelListResponse:
    """
    List all available benchmark models
    
    Returns:
        Model list information
    """
    models = scan_benchmark_models()
    total_configs = sum(len(files) for files in models.values())
    
    logger.info(f"Listing models: {len(models)} model types, {total_configs} configurations")
    
    return ModelListResponse(
        total_models=len(models),
        total_configs=total_configs,
        models=models,
        benchmarks_dir=str(BENCHMARKS_DIR)
    )


@app.get("/models/info", response_model=ModelInfo, tags=["Model Management"])
async def get_model_info(yaml_path: str) -> ModelInfo:
    """
    Get detailed model configuration information
    
    Args:
        yaml_path: YAML configuration file path
        
    Returns:
        Detailed model information
        
    Raises:
        HTTPException: File does not exist or parsing error
    """
    try:
        info = extract_model_info(yaml_path)
        logger.info(f"Getting model information: {info.filename}")
        return info
        
    except FileNotFoundError as e:
        logger.error(f"File does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to get model information: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model information: {str(e)}")


@app.post("/models/update_provider", response_model=UpdateProviderResponse, tags=["Model Management"])
async def update_provider_uri(request: UpdateProviderRequest) -> UpdateProviderResponse:
    """
    Update provider_uri in YAML configuration file
    
    Args:
        request: Update request
        
    Returns:
        Update result
        
    Raises:
        HTTPException: Update failed
    """
    try:
        old_uri, new_uri = update_yaml_provider_uri(
            request.yaml_path,
            request.provider_uri
        )
        
        return UpdateProviderResponse(
            success=True,
            yaml_path=request.yaml_path,
            old_uri=old_uri,
            new_uri=new_uri,
            message="Provider URI updated successfully"
        )
        
    except FileNotFoundError as e:
        logger.error(f"File does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to update provider_uri: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.post("/models/run", response_model=RunModelResponse, tags=["Model Execution"])
async def run_model(
    request: RunModelRequest,
    background_tasks: BackgroundTasks
) -> RunModelResponse:
    """
    Run qlib benchmark model
    
    Args:
        request: Run request
        background_tasks: FastAPI background tasks
        
    Returns:
        Run task information
        
    Raises:
        HTTPException: Run failed
    """
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    try:
        # Check if YAML file exists
        if not Path(request.yaml_path).exists():
            raise FileNotFoundError(f"YAML file does not exist: {request.yaml_path}")
        
        # If provider_uri is provided, update first
        if request.provider_uri:
            logger.info(f"Updating provider_uri: {request.provider_uri}")
            update_yaml_provider_uri(request.yaml_path, request.provider_uri)
        
        # Check qrun availability
        available, message, _ = check_qrun_available()
        if not available:
            raise RuntimeError(f"qrun not available: {message}")
        
        # Build command
        cmd = build_qrun_command(request.yaml_path, request.experiment_name)
        command_str = " ".join(cmd)
        
        logger.info(f"Preparing to start training task [{task_id}]: {command_str}")
        
        # Initialize task record
        with tasks_lock:
            run_tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "yaml_path": request.yaml_path,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "error": None,
                "exit_code": None
            }
        
        # Add background task
        background_tasks.add_task(_run_qlib_model, task_id, cmd, request.yaml_path)
        
        logger.info(f"Training task submitted [{task_id}]")
        
        return RunModelResponse(
            task_id=task_id,
            status="started",
            message="Training task started successfully",
            command=command_str,
            yaml_path=request.yaml_path,
            provider_uri=request.provider_uri,
            experiment_name=request.experiment_name,
            task_name=request.task_name,
            created_at=created_at
        )
        
    except FileNotFoundError as e:
        logger.error(f"File does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Failed to start training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@app.get("/qrun/status", response_model=QrunStatusResponse, tags=["Tool Check"])
async def check_qrun_status() -> QrunStatusResponse:
    """
    Check if qrun command is available
    
    Returns:
        qrun status information
    """
    available, message, version_info = check_qrun_available()
    
    logger.info(f"qrun status check: {message}")
    
    return QrunStatusResponse(
        available=available,
        message=message,
        version_info=version_info
    )


@app.get("/tasks/status/{task_id}", response_model=TaskStatus, tags=["Task Management"])
async def get_task_status(task_id: str) -> TaskStatus:
    """
    Get training task status
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
        
    Raises:
        HTTPException: Task does not exist
    """
    with tasks_lock:
        if task_id not in run_tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = run_tasks[task_id]
        last_log = list(task["log"])[-1] if task["log"] else ""
        
        return TaskStatus(
            task_id=task_id,
            status=task["status"],
            command=task["command"],
            task_name=task["task_name"],
            yaml_path=task["yaml_path"],
            created_at=task["created_at"],
            pid=task["pid"],
            log_lines=len(task["log"]),
            last_log=last_log,
            error=task.get("error"),
            exit_code=task.get("exit_code")
        )


@app.get("/tasks/logs/{task_id}", response_model=LogResponse, tags=["Task Management"])
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
        HTTPException: Task does not exist
    """
    with tasks_lock:
        if task_id not in run_tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = run_tasks[task_id]
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


@app.get("/tasks/list", tags=["Task Management"])
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
                "yaml_path": task["yaml_path"],
                "created_at": task["created_at"],
                "pid": task["pid"]
            }
            for task_id, task in run_tasks.items()
        ]
        
        # Sort by creation time in descending order
        task_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "total": len(task_list),
            "tasks": task_list
        }


@app.delete("/tasks/{task_id}", tags=["Task Management"])
async def delete_task(task_id: str) -> Dict[str, str]:
    """
    Delete training task record (will not terminate running processes)
    
    Args:
        task_id: Task ID
        
    Returns:
        Deletion confirmation information
        
    Raises:
        HTTPException: Task does not exist or is running
    """
    with tasks_lock:
        if task_id not in run_tasks:
            raise HTTPException(status_code=404, detail=f"Task does not exist: {task_id}")
        
        task = run_tasks[task_id]
        if task["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running task, please wait for task to complete or fail"
            )
        
        del run_tasks[task_id]
        logger.info(f"Task deleted: {task_id}")
        
        return {"message": f"Task {task_id} deleted"}


# ========== Main Program Entry ==========

if __name__ == "__main__":
    """
    Main program entry
    
    Start FastAPI server
    """
    logger.info("=" * 60)
    logger.info("Starting Qlib Benchmark Runner service")
    logger.info(f"Benchmarks directory: {BENCHMARKS_DIR}")
    logger.info(f"Default Provider URI: {DEFAULT_PROVIDER_URI}")
    logger.info("=" * 60)
    
    # Start service
    uvicorn.run(
        app,
        host="localhost",
        port=8010,
        log_level="info",
        access_log=True
    )

