import os
import sys
import json
import warnings
from typing import Optional, Dict, Any
from types import SimpleNamespace
from pydantic import BaseModel, Field

# Ignore pkg_resources deprecation warnings (from third-party library py_mini_racer)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from fastapi import FastAPI, HTTPException
import uvicorn

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from modules import train_qcm_mcp as mod_train_qcm_mcp
#from modules import calculate_ic as mod_calculate_ic
from modules import train_AFF_MCP as mod_train_AFF
from modules import train_gfn_AlphaSAGE_MCP as mod_train_gfn_AlphaSAGE
from modules import train_GP_AlphaSAGE_MCP as mod_train_GP_AlphaSAGE
from modules import train_PPO_AlphaSAGE_MCP as mod_train_PPO_AlphaSAGE
from modules import qlib_benchmark_runner_fastapi as mod_qlib_benchmark_runner


app = FastAPI(title="Qlib MCP API", description="API for Qlib MCP tools", version="1.0.0")

def to_ns(model: BaseModel) -> SimpleNamespace:
    """Convert Pydantic model to SimpleNamespace for compatibility with existing modules."""
    return SimpleNamespace(**model.model_dump(exclude_none=True))

# --- Models ---

class TrainQCMRequest(BaseModel):
    model: str = Field("iqn", description="Model type: qrdqn, iqn, fqf")
    seed: int = Field(0, description="Random seed", ge=0)
    pool: int = Field(20, description="Alpha pool capacity", ge=1)
    std_lam: float = Field(1.0, description="Standard deviation lambda parameter", gt=0)
    instruments: str = Field("csi300", description="Stock pool name: csi300, csi500, csi800, csi1000, csiall, sp500, etc.")
    train_end_year: int = Field(2020, description="Training end year", ge=2000, le=2030)
    task_name: str = Field("training", description="Task name", min_length=1, max_length=100)

class TrainAFFRequest(BaseModel):
    config_file: Optional[str] = Field(None, description="Configuration file path (optional)")
    config_params: Optional[Dict[str, Any]] = Field(None, description="Configuration parameter dictionary, containing instruments, train_end_year, seed, cuda, save_name, zoo_size, corr_thresh, ic_thresh, icir_thresh, etc.")
    task_name: str = Field("training", description="Task name")

class TrainGFNAlphaSAGERequest(BaseModel):
    config_file: Optional[str] = Field(None, description="Configuration file path (optional)")
    config_params: Optional[Dict[str, Any]] = Field(None, description="Configuration parameter dictionary, MCP interface only supports modifying seed, instrument, pool_capacity three parameters, other parameters use default values")
    task_name: str = Field("training", description="Task name")

class TrainGPAlphaSAGERequest(BaseModel):
    config_file: Optional[str] = Field(None, description="Configuration file path (optional)")
    config_params: Optional[Dict[str, Any]] = Field(None, description="Configuration parameter dictionary, containing instruments, seed, train_end_year, freq, cuda, seed_factors, etc. seed_factors comes from mining_feedback['suggested_seeds'], used to guide GP initial population")
    task_name: str = Field("training", description="Task name")

class TrainPPOAlphaSAGERequest(BaseModel):
    config_file: Optional[str] = Field(None, description="Configuration file path (optional)")
    config_params: Optional[Dict[str, Any]] = Field(None, description="Configuration parameter dictionary, containing seed, instruments, pool, steps, train_end_year, etc.")
    task_name: str = Field("training", description="Task name")

class QlibBenchmarkRunnerRequest(BaseModel):
    yaml_path: str = Field(..., description="YAML configuration file path (required)")
    provider_uri: Optional[str] = Field(None, description="provider_uri path (optional), will be updated to YAML file before running")
    experiment_name: Optional[str] = Field(None, description="Experiment name (optional)")
    task_name: str = Field("qlib_training", description="Task name")

class CalculateICRequest(BaseModel):
    formula: str = Field(..., description="Qlib format factor expression, e.g. 'Ts_Mean($close, 10) / Ts_Mean($close, 30) - 1'")
    instruments: str = Field("csi300", description="Stock pool name: csi300, csi500, csi800, csi1000, csiall, all")
    start_date: str = Field("2020-01-01", description="Start date, format YYYY-MM-DD")
    end_date: str = Field("2023-12-31", description="End date, format YYYY-MM-DD")
    label_expr: str = Field("Ref($close, -2)/Ref($close, -1) - 1", description="Label expression, used to calculate future return rate")
    provider_uri: Optional[str] = Field(None, description="Qlib data path, default auto-detect")


# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to Qlib MCP API"}

def _get_endpoint_for_tool(tool_name: str) -> str:
    """Map tool name to API endpoint."""
    return f"/{tool_name}"

@app.get("/tools", tags=["Meta"])
async def list_tools():
    """Return a list of all available tools with descriptions."""
    tools_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "configs", 
        "tools.json"
    )
    
    if not os.path.exists(tools_config_path):
         raise HTTPException(status_code=404, detail="Tools config not found")

    with open(tools_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 从 categories 中提取所有工具
    all_tools = []
    if "categories" in config:
        for category in config["categories"]:
            if "tools" in category:
                all_tools.extend(category["tools"])
    
    formatted_tools = []
    for tool in all_tools:
        formatted_tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "category": tool.get("category", ""),
            "endpoint": _get_endpoint_for_tool(tool["name"]),
            "parameters": tool.get("inputSchema", {}).get("properties", {}),
            "required": tool.get("inputSchema", {}).get("required", []),
            "example": tool.get("example", {})
        })
    
    return {
        "tools": formatted_tools,
        "total": len(formatted_tools)
    }

@app.get("/tools/openai", tags=["Meta"])
async def list_tools_openai_format():
    """Return tools in OpenAI Function Calling format."""
    tools_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "configs", 
        "tools.json"
    )
    
    if not os.path.exists(tools_config_path):
         raise HTTPException(status_code=404, detail="Tools config not found")
    
    with open(tools_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 从 categories 中提取所有工具
    all_tools = []
    if "categories" in config:
        for category in config["categories"]:
            if "tools" in category:
                all_tools.extend(category["tools"])
    
    functions = []
    for tool in all_tools:
        function_def = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("inputSchema", {})
            }
        }
        functions.append(function_def)
    
    return {"tools": functions}



@app.post("/train_qcm", tags=["RL Training"])
async def train_qcm(request: TrainQCMRequest):
    """
    训练AlphaQCM模型（基于强化学习的Alpha挖掘：QRDQN/IQN/FQF）
    """
    try:
        mod_train_qcm_mcp.run(to_ns(request))
        return {"status": "success", "message": "train_qcm done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_AFF", tags=["DL Training"])
async def train_AFF(request: TrainAFFRequest):
    """
    训练AlphaForge模型（基于GAN的Alpha因子生成：GAN-based Alpha Factor Generation）
    通过Docker Compose在AlphaSAGE容器中运行，同步执行并返回因子文件路径
    """
    try:
        result = mod_train_AFF.run(to_ns(request))
        return {"status": "success", "message": "train_AFF done", "factor_file": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_gfn_AlphaSAGE", tags=["GFN Training"])
async def train_gfn_AlphaSAGE(request: TrainGFNAlphaSAGERequest):
    """
    训练AlphaSAGE模型（基于GFlowNet的Alpha因子生成：GFlowNet-based Alpha Factor Generation）
    """
    try:
        mod_train_gfn_AlphaSAGE.run(to_ns(request))
        return {"status": "success", "message": "train_gfn_AlphaSAGE done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_GP_AlphaSAGE", tags=["GP Training"])
async def train_GP_AlphaSAGE(request: TrainGPAlphaSAGERequest):
    """
    训练AlphaSAGE仓库中的GP模型（基于遗传编程的Alpha因子挖掘：Genetic Programming for Alpha Factor Mining）
    支持通过 config_params.seed_factors 传入种子因子（来自 mining_feedback["suggested_seeds"]）指导 GP 初始种群
    """
    try:
        result = mod_train_GP_AlphaSAGE.run(to_ns(request))
        # result 应该是最后生成的因子文件路径
        if result:
            return {"status": "success", "message": "train_GP_AlphaSAGE done", "factor_file": result}
        return {"status": "success", "message": "train_GP_AlphaSAGE done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_PPO_AlphaSAGE", tags=["RL Training"])
async def train_PPO_AlphaSAGE(request: TrainPPOAlphaSAGERequest):
    """
    训练AlphaSAGE仓库中的PPO模型（基于PPO的Alpha因子生成：PPO-based Alpha Factor Generation）
    """
    try:
        result = mod_train_PPO_AlphaSAGE.run(to_ns(request))
        # result 应该是最后生成的因子文件路径
        if result:
            return {"status": "success", "message": "train_PPO_AlphaSAGE done", "factor_file": result}
        return {"status": "success", "message": "train_PPO_AlphaSAGE done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qlib_benchmark_runner", tags=["Benchmark"])
async def qlib_benchmark_runner(request: QlibBenchmarkRunnerRequest):
    """
    运行Qlib Benchmark模型（支持XGBoost、LightGBM、GRU等标准模型，通过YAML配置文件运行）
    """
    try:
        mod_qlib_benchmark_runner.run(to_ns(request))
        return {"status": "success", "message": "qlib_benchmark_runner done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qlib_benchmark_list_models", tags=["Benchmark"])
async def qlib_benchmark_list_models():
    """
    列出所有可用的Qlib Benchmark模型（扫描benchmarks目录，返回模型类型和配置文件路径列表）
    """
    try:
        result = mod_qlib_benchmark_runner.list_models(SimpleNamespace())
        return result if result else {"status": "success", "message": "qlib_benchmark_list_models done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_ic", tags=["因子评估"])
async def calculate_ic(request: CalculateICRequest):
    """
    计算因子IC指标（信息系数：IC均值、IC标准差、IR、Rank IC、IC胜率等）
    """
    try:
        mod_calculate_ic.run(to_ns(request))
        return {"status": "success", "message": "calculate_ic done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8008, reload=True)

