"""
FactorMiningAgent MCP 工具调用示例
演示如何通过 FactorMiningAgent 调用 MCP 服务提供的工具
"""
import sys
import os
# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Agent.FactorMiningAgent import FactorMiningAgent
from Agent.base_agent import BaseAgent, LLMProvider

# 示例：创建一个简单的 LLM 服务（实际使用时需要真实的 LLM 服务）
class DummyLLMService(BaseAgent):
    def chat(self, messages, stream=False, **kwargs):
        pass
    
    def get_provider(self):
        return LLMProvider.OPENAI

# 初始化 Agent
llm_service = DummyLLMService(api_key="dummy", model="dummy")
agent = FactorMiningAgent(llm_service)

# 示例 1: 列出所有可用的 MCP 工具
print("=" * 60)
print("示例 1: 列出所有可用的 MCP 工具")
print("=" * 60)
tools = agent.list_available_tools()
print(f"找到 {len(tools)} 个工具:")
for tool in tools:
    print(f"  - {tool['name']}: {tool['description']}")
print()

# 示例 2: 计算因子 IC 指标
print("=" * 60)
print("示例 2: 计算因子 IC 指标")
print("=" * 60)
try:
    result = agent.calculate_ic(
        formula="Ts_Mean($close, 10) / Ts_Mean($close, 30) - 1",
        instruments="csi300",
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    print("IC 计算结果:")
    print(result)
except Exception as e:
    print(f"错误: {e}")
print()

# 示例 3: 调用任意 MCP 工具
print("=" * 60)
print("示例 3: 调用任意 MCP 工具")
print("=" * 60)
try:
    # 调用 qlib_benchmark_list_models 工具
    result = agent.call_mcp_tool("qlib_benchmark_list_models", {})
    print("可用模型列表:")
    print(result)
except Exception as e:
    print(f"错误: {e}")
print()

# 示例 4: 训练模型
print("=" * 60)
print("示例 4: 训练模型")
print("=" * 60)
try:
    # 训练 QCM 模型
    result = agent.train_model(
        model_type="train_qcm",
        model="qrdqn",
        seed=42,
        pool=20,
        std_lam=1.0,
        task_name="test_training"
    )
    print("训练结果:")
    print(result)
except Exception as e:
    print(f"错误: {e}")
print()

# 示例 5: 通过 process 方法调用
print("=" * 60)
print("示例 5: 通过 process 方法调用")
print("=" * 60)
result = agent.process({
    "mcp_tool": "calculate_ic",
    "tool_arguments": {
        "formula": "Ts_Mean($close, 5) / Ts_Mean($close, 20) - 1",
        "instruments": "csi300",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31"
    }
})
print("处理结果:")
print(result)

