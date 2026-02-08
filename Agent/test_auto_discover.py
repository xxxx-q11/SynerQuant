"""
测试 FactorMiningAgent 的自主工具发现和调用功能
演示如何让 Agent 自主查找 MCP 工具并选择合适的工具进行训练
"""
import sys
import os
# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 直接导入，避免经过 __init__.py
import sys
from pathlib import Path
agent_path = Path(__file__).parent
sys.path.insert(0, str(agent_path))

from FactorMiningAgent import FactorMiningAgent
from base_agent import BaseAgent, LLMProvider


def main():
    """主函数"""
    print("=" * 80)
    print("FactorMiningAgent 自主工具发现和训练测试")
    print("=" * 80)
    print()
    
    # 初始化 LLM 服务
    # 请根据实际情况配置 API Key 和模型
    try:
        llm_service = BaseAgent(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
            model="gpt-4",  # 或其他模型
            provider=LLMProvider.OPENAI
        )
        print("✅ LLM 服务初始化成功")
    except Exception as e:
        print(f"⚠️  LLM 服务初始化失败: {e}")
        print("将使用简化版本继续测试...")
        # 创建一个简单的虚拟服务
        class SimpleLLM(BaseAgent):
            def chat(self, messages, stream=False, **kwargs):
                # 返回一个简单的选择
                return '''```json
{
    "selected_tool": "train_qcm",
    "reason": "QCM (Quantitative Communication Model) 是一个适合量化交易的模型，支持强化学习方法",
    "suggested_parameters": {
        "model": "qrdqn",
        "seed": 42,
        "pool": 20,
        "std_lam": 1.0
    }
}
```'''
            
            def get_provider(self):
                return LLMProvider.OPENAI
        
        llm_service = SimpleLLM(api_key="dummy", model="dummy")
    
    print()
    
    # 初始化 FactorMiningAgent
    try:
        agent = FactorMiningAgent(llm_service)
        print("✅ FactorMiningAgent 初始化成功")
    except Exception as e:
        print(f"❌ FactorMiningAgent 初始化失败: {e}")
        return
    
    print()
    print("=" * 80)
    print()
    
    # 测试 1: 手动列出工具
    print("【测试 1】手动列出所有可用工具")
    print("-" * 80)
    tools = agent.list_available_tools()
    print(f"找到 {len(tools)} 个工具:")
    for tool in tools:
        print(f"  📦 {tool['name']}")
        print(f"     {tool['description']}")
    print()
    
    # 测试 2: 自主发现并训练
    print("=" * 80)
    print("【测试 2】自主发现工具并选择训练")
    print("-" * 80)
    print()
    
    result = agent.auto_discover_and_train(
        task_description="训练一个用于股票价格预测的量化交易模型"
    )
    
    print()
    print("=" * 80)
    print("📊 最终结果")
    print("=" * 80)
    
    if result.get("success"):
        print(f"✅ 训练成功!")
        print(f"选择的工具: {result.get('selected_tool')}")
        print(f"选择理由: {result.get('selection_reason')}")
        print(f"使用参数: {result.get('parameters')}")
        print(f"\n训练结果预览:")
        result_text = result.get('result', '')
        if len(result_text) > 1000:
            print(result_text[:1000] + "\n... (结果已截断) ...")
        else:
            print(result_text)
    else:
        print(f"❌ 训练失败!")
        print(f"选择的工具: {result.get('selected_tool')}")
        print(f"错误信息: {result.get('error')}")
    
    print()
    print("📋 执行日志:")
    for log in result.get('logs', []):
        print(f"  - {log}")
    
    print()
    print("=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

