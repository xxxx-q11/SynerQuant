# FactorMiningAgent MCP 工具调用指南

## 概述

`FactorMiningAgent` 已集成 MCP (Model Context Protocol) 客户端，可以直接调用 Qlib_MCP 服务器提供的各种量化工具。

## 安装依赖

确保已安装 MCP 库：

```bash
pip install mcp
```

## 初始化 Agent

```python
from Agent.FactorMiningAgent import FactorMiningAgent
from Agent.base_agent import BaseAgent

# 创建 LLM 服务实例
llm_service = YourLLMService(...)

# 初始化 Agent（会自动连接 MCP 服务器）
agent = FactorMiningAgent(llm_service)

# 或者指定自定义的 MCP 服务器路径
agent = FactorMiningAgent(
    llm_service, 
    mcp_server_path="/path/to/mcp_server_inline.py"
)
```

## 可用方法

### 1. 列出所有可用工具

```python
tools = agent.list_available_tools()
for tool in tools:
    print(f"工具: {tool['name']}")
    print(f"描述: {tool['description']}")
    print(f"参数: {tool['inputSchema']}")
```

### 2. 计算因子 IC 指标

```python
result = agent.calculate_ic(
    formula="Ts_Mean($close, 10) / Ts_Mean($close, 30) - 1",
    instruments="csi300",
    start_date="2020-01-01",
    end_date="2023-12-31",
    label_expr="Ref($close, -2)/Ref($close, -1) - 1"
)
print(result)
```

### 3. 训练模型

```python
# 训练 QCM 模型
result = agent.train_model(
    model_type="train_qcm",
    model="qrdqn",
    seed=42,
    pool=20,
    std_lam=1.0,
    task_name="my_training"
)

# 训练 GP AlphaSAGE 模型
result = agent.train_model(
    model_type="train_GP_AlphaSAGE",
    config_params={
        "instruments": "csi300",
        "seed": 0,
        "train_end_year": 2020
    },
    task_name="gp_training"
)
```

### 4. 调用任意 MCP 工具

```python
# 调用任意工具
result = agent.call_mcp_tool(
    tool_name="qlib_benchmark_list_models",
    arguments={}
)

# 调用 calculate_ic
result = agent.call_mcp_tool(
    tool_name="calculate_ic",
    arguments={
        "formula": "Ts_Mean($close, 10) / Ts_Mean($close, 30) - 1",
        "instruments": "csi300",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31"
    }
)
```

### 5. 通过 process 方法调用

```python
result = agent.process({
    "mcp_tool": "calculate_ic",
    "tool_arguments": {
        "formula": "Ts_Mean($close, 5) / Ts_Mean($close, 20) - 1",
        "instruments": "csi300",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31"
    }
})
```

## 可用的 MCP 工具

根据 `Qlib_MCP/configs/tools.json` 配置，以下工具可用：

1. **calculate_ic** - 计算因子 IC 指标
2. **train_qcm** - 训练 AlphaQCM 模型
3. **train_AFF** - 训练 AlphaForge 模型
4. **train_gfn_AlphaSAGE** - 训练 GFlowNet AlphaSAGE 模型
5. **train_GP_AlphaSAGE** - 训练 GP AlphaSAGE 模型
6. **train_PPO_AlphaSAGE** - 训练 PPO AlphaSAGE 模型
7. **qlib_benchmark_runner** - 运行 Qlib Benchmark 模型
8. **qlib_benchmark_list_models** - 列出所有可用的 Benchmark 模型

## 工具参数说明

每个工具的参数定义在 `Qlib_MCP/configs/tools.json` 中。可以通过以下方式查看：

```python
# 获取工具信息
tool_info = agent.mcp_client.get_tool_info("calculate_ic")
print(tool_info)
```

## 错误处理

所有方法都会抛出异常，建议使用 try-except：

```python
try:
    result = agent.calculate_ic(formula="...")
except RuntimeError as e:
    print(f"调用失败: {e}")
```

## 完整示例

参考 `Agent/example_mcp_usage.py` 查看完整的使用示例。

## 注意事项

1. MCP 服务器需要通过 stdio 协议通信，确保 Python 环境正确配置
2. 某些工具（如训练模型）可能需要较长时间执行
3. 确保 MCP 服务器脚本路径正确，默认路径为 `Qlib_MCP/mcp_server_inline.py`
4. 如果 MCP 客户端初始化失败，相关方法会返回空列表或抛出异常

