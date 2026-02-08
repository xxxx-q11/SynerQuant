# FactorMiningAgent 自主工具发现功能

## 📋 功能概述

为 `FactorMiningAgent` 新增了智能的自主工具发现和调用能力，让 Agent 能够：

1. 🔍 **自动查找** 所有可用的 MCP 工具
2. 🤖 **智能选择** 最适合当前任务的训练工具（通过 LLM 推理）
3. 🚀 **自动调用** 选择的工具进行模型训练
4. 📊 **返回详细结果** 包括工具选择理由、参数、执行结果等

## 🎯 核心方法

### `auto_discover_and_train(task_description)`

这是核心方法，实现了完整的自主工具发现和训练流程。

**特点：**
- 无需手动指定工具名称
- LLM 自动分析任务需求并选择最合适的工具
- LLM 建议合理的参数配置
- 完整的错误处理和日志记录

**使用示例：**
```python
result = agent.auto_discover_and_train(
    task_description="训练一个用于股票价格预测的量化交易模型"
)
```

**返回结构：**
```python
{
    "tools": [...],              # 所有可用工具列表
    "train_tools": [...],        # 训练相关工具列表
    "selected_tool": "...",      # LLM 选择的工具名称
    "selection_reason": "...",   # 选择理由
    "parameters": {...},         # 建议的参数
    "result": "...",            # 训练结果（成功时）
    "error": "...",             # 错误信息（失败时）
    "logs": [...],              # 执行日志
    "success": True/False       # 是否成功
}
```

## 🔄 工作流程

```
┌─────────────────────────────────────┐
│  1. 查找所有可用的 MCP 工具         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. 筛选训练相关的工具              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. LLM 分析任务描述                │
│     选择最合适的工具                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. LLM 建议合理的参数配置          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. 调用选择的工具进行训练          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. 返回详细的执行结果              │
└─────────────────────────────────────┘
```

## 💻 完整使用示例

```python
from Agent.FactorMiningAgent import FactorMiningAgent
from Agent.base_agent import BaseAgent, LLMProvider
import os

# 1. 初始化 LLM 服务
llm_service = BaseAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    provider=LLMProvider.OPENAI
)

# 2. 初始化 FactorMiningAgent
agent = FactorMiningAgent(llm_service)

# 3. 查看所有可用的 MCP 工具（可选）
tools = agent.list_available_tools()
print(f"找到 {len(tools)} 个工具:")
for tool in tools:
    print(f"  - {tool['name']}: {tool['description']}")

# 4. 自主发现并训练
result = agent.auto_discover_and_train(
    task_description="训练一个用于股票价格预测的量化交易模型"
)

# 5. 处理结果
if result.get("success"):
    print("✅ 训练成功!")
    print(f"选择的工具: {result['selected_tool']}")
    print(f"选择理由: {result['selection_reason']}")
    print(f"使用参数: {result['parameters']}")
    print(f"训练结果: {result['result']}")
else:
    print("❌ 训练失败!")
    print(f"错误: {result.get('error')}")

# 6. 查看日志
print("\n执行日志:")
for log in result.get('logs', []):
    print(f"  - {log}")
```

## 📦 新增和修改的方法

### 新增方法

#### `auto_discover_and_train(task_description)`
- **功能**：自主发现并训练
- **参数**：`task_description` - 任务描述字符串
- **返回**：包含完整执行信息的字典

### 已有方法（保持兼容）

#### `list_available_tools()`
- **功能**：获取所有可用的 MCP 工具
- **返回**：工具列表

#### `call_mcp_tool(tool_name, arguments)`
- **功能**：直接调用指定的 MCP 工具
- **参数**：工具名称和参数字典
- **返回**：执行结果字符串

#### `calculate_ic(...)`
- **功能**：计算因子 IC 指标
- **保持不变**

#### `train_model(model_type, **kwargs)`
- **功能**：训练指定类型的模型
- **保持不变**

## 🌟 主要特性

### 1. 智能工具选择
- LLM 会根据任务描述分析并选择最合适的工具
- 提供详细的选择理由
- 建议合理的参数配置

### 2. 错误容错
- 如果 LLM 解析失败，自动使用第一个训练工具作为默认选择
- 完整的异常处理和错误信息返回

### 3. 详细日志
- 记录每一步操作
- 便于调试和追踪执行过程

### 4. 向后兼容
- 保留所有原有方法
- 不影响现有代码的使用

## 📝 实现细节

实现位于 `Agent/FactorMiningAgent.py` 文件中，主要修改：

1. **新增 `auto_discover_and_train` 方法**（第 148-270 行）
   - 查找和筛选工具
   - 构造 LLM 提示词
   - 解析 LLM 响应
   - 调用工具并处理结果

2. **LLM 提示词设计**
   - 清晰的任务描述
   - 结构化的工具列表
   - JSON 格式的响应要求

3. **响应解析**
   - 支持多种 JSON 格式
   - 处理代码块包裹的 JSON
   - 容错处理

## 🔧 配置要求

### 必需
- Python 3.7+
- MCP 库：`pip install mcp`
- 有效的 LLM API Key（OpenAI、Claude 等）

### 可选
- `Qlib_MCP/mcp_server_inline.py` - MCP 服务器脚本

## 📚 文档和示例

- **详细使用指南**：`Agent/AUTO_DISCOVER_USAGE.md`
- **示例代码**：`Agent/test_auto_discover.py`
- **演示脚本**：`test_auto_discover_simple.py`
- **源代码**：`Agent/FactorMiningAgent.py`

## ⚠️ 注意事项

1. **LLM 依赖**：功能依赖 LLM 的选择能力，确保 LLM 配置正确
2. **参数验证**：LLM 建议的参数可能需要手动验证和调整
3. **超时处理**：训练任务可能耗时较长，注意设置合理的超时时间
4. **成本考虑**：每次调用会消耗 LLM tokens

## 🎉 总结

新增的 `auto_discover_and_train` 方法让 `FactorMiningAgent` 具备了真正的自主性，能够：
- 自动发现可用工具
- 智能选择合适工具
- 自动配置参数
- 执行训练任务

这大大简化了使用流程，提高了 Agent 的智能化程度！

