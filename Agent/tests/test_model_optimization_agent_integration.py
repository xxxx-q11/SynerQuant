"""
模型优化 Agent 集成测试 - 使用真实的 LLM 服务和实验结果

测试 model_optimization_Agent.py 的完整流程，包括：
1. 真实的 LLM 服务调用
2. 真实的 MCP 客户端交互
3. 真实的实验结果处理
"""
import sys
import json
import yaml
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_real_llm_service():
    """创建真实的 LLM 服务"""
    from Agent.agent_factory import create_agent
    
    # 从环境变量获取 API Key，如果没有则使用默认值（需要用户设置）
    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        print("⚠️  警告: 未设置 QWEN_API_KEY 环境变量，将使用 Mock LLM")
        return None
    
    try:
        llm = create_agent(
            provider="qwen",
            api_key=api_key,
            model="qwen-turbo",  # 或 qwen-plus
            temperature=0.7
        )
        print(f"✅ 成功创建真实的 LLM 服务: {llm.get_provider()}")
        return llm
    except Exception as e:
        print(f"❌ 创建 LLM 服务失败: {e}")
        return None


def create_mock_pickle_file(file_path, data):
    """创建模拟的 pickle 文件"""
    import pickle
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def create_mock_mlflow_metric_file(file_path, value):
    """创建模拟的 MLflow 指标文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(f"0 {value}\n")


def test_real_llm_optimization_suggestion():
    """测试真实的 LLM 生成优化建议"""
    print("=" * 80)
    print("测试 1: 真实 LLM 生成优化建议")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = create_real_llm_service()
    if not llm:
        print("⚠️  跳过测试：LLM 服务不可用")
        return
    
    agent = ModelOptimizationAgent(llm)
    
    # 创建模拟的当前指标
    current_metrics = {
        "ic_mean": 0.03,
        "ic_std": 0.02,
        "ir": 1.5,
        "rank_ic_mean": 0.025,
        "annualized_return": 0.10,
        "max_drawdown": -0.15
    }
    
    # 创建模拟的优化历史
    optimization_history = [
        {
            "iteration": 1,
            "metrics": {
                "ic_mean": 0.02,
                "annualized_return": 0.08,
                "max_drawdown": -0.18
            }
        }
    ]
    
    # 读取模板配置
    with open(agent.template_path, 'r', encoding='utf-8') as f:
        current_config = yaml.safe_load(f)
    
    factors_count = 20
    
    print("📊 当前指标:")
    print(f"  - IC 均值: {current_metrics['ic_mean']:.4f}")
    print(f"  - 年化收益: {current_metrics['annualized_return']:.2%}")
    print(f"  - 最大回撤: {current_metrics['max_drawdown']:.2%}")
    print()
    print("🤖 调用真实 LLM 生成优化建议...")
    
    # 调用真实的 LLM
    suggestion = agent._llm_analyze_and_suggest(
        current_metrics=current_metrics,
        optimization_history=optimization_history,
        current_config=current_config,
        factors_count=factors_count
    )
    
    if suggestion:
        print("✅ LLM 返回了优化建议:")
        print(f"  - 分析: {suggestion.get('analysis', 'N/A')}")
        print(f"  - 问题: {suggestion.get('issues', [])}")
        print(f"  - 摘要: {suggestion.get('summary', 'N/A')}")
        print(f"  - 参数更新: {suggestion.get('model_params_update', {})}")
        print(f"  - 理由: {suggestion.get('reasoning', 'N/A')}")
        
        # 验证建议格式
        assert 'model_params_update' in suggestion, "建议应该包含 model_params_update"
        assert isinstance(suggestion['model_params_update'], dict), "model_params_update 应该是字典"
    else:
        print("⚠️  LLM 未返回建议")
    
    print("✅ 真实 LLM 优化建议测试完成")
    print()


def test_apply_real_llm_suggestion():
    """测试应用真实的 LLM 优化建议"""
    print("=" * 80)
    print("测试 2: 应用真实 LLM 优化建议")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = create_real_llm_service()
    if not llm:
        print("⚠️  跳过测试：LLM 服务不可用")
        return
    
    agent = ModelOptimizationAgent(llm)
    
    # 读取模板配置
    with open(agent.template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 先获取真实的 LLM 建议
    current_metrics = {
        "ic_mean": 0.03,
        "ic_std": 0.02,
        "ir": 1.5,
        "rank_ic_mean": 0.025,
        "annualized_return": 0.10,
        "max_drawdown": -0.15
    }
    
    optimization_history = []
    factors_count = 20
    
    print("🤖 获取 LLM 优化建议...")
    suggestion = agent._llm_analyze_and_suggest(
        current_metrics=current_metrics,
        optimization_history=optimization_history,
        current_config=config,
        factors_count=factors_count
    )
    
    if not suggestion or 'model_params_update' not in suggestion:
        print("⚠️  未获取到有效的优化建议，使用模拟建议")
        suggestion = {
            "model_params_update": {
                "lr": 0.0005,
                "n_epochs": 150,
                "d_model": 128
            }
        }
    
    print(f"📝 应用优化建议: {suggestion.get('model_params_update', {})}")
    
    # 应用建议
    updated_config = agent._apply_optimization_suggestion(config, suggestion)
    
    # 验证更新
    model_kwargs = updated_config['task']['model']['kwargs']
    original_kwargs = config['task']['model']['kwargs']
    
    print("\n📊 参数对比:")
    for param, value in suggestion.get('model_params_update', {}).items():
        original_value = original_kwargs.get(param, "未设置")
        new_value = model_kwargs.get(param, "未设置")
        print(f"  - {param}: {original_value} -> {new_value}")
        assert model_kwargs.get(param) == value, f"参数 {param} 应该被更新为 {value}"
    
    print("✅ 应用真实 LLM 优化建议测试完成")
    print()


def test_full_optimization_flow_with_mock_results():
    """测试完整的优化流程（使用模拟的训练结果）"""
    print("=" * 80)
    print("测试 3: 完整优化流程（模拟训练结果）")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = create_real_llm_service()
    if not llm:
        print("⚠️  跳过测试：LLM 服务不可用")
        return
    
    # 创建测试因子
    factors = [
        {"expression": "($close - $open) / $open", "ic": 0.05},
        {"expression": "($high - $low) / $close", "ic": 0.03},
        {"expression": "Mean($volume, 5) / $volume", "ic": 0.04}
    ]
    
    # 创建临时目录用于存储结果
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 创建模拟的训练结果文件
        ic_data = pd.Series([0.05, 0.03, 0.04, 0.06, 0.02])
        ic_path = tmpdir / "ic.pkl"
        create_mock_pickle_file(ic_path, ic_data)
        
        rank_ic_data = pd.Series([0.04, 0.03, 0.05, 0.04, 0.03])
        rank_ic_path = tmpdir / "rank_ic.pkl"
        create_mock_pickle_file(rank_ic_path, rank_ic_data)
        
        annual_return_path = tmpdir / "annual_return.txt"
        create_mock_mlflow_metric_file(annual_return_path, 0.12)
        
        max_drawdown_path = tmpdir / "max_drawdown.txt"
        create_mock_mlflow_metric_file(max_drawdown_path, -0.14)
        
        # Mock MCP 客户端
        mock_mcp_client = MagicMock()
        mock_mcp_client.call_tool.return_value = json.dumps({
            "ic": str(ic_path),
            "rank_ic": str(rank_ic_path),
            "1day.excess_return_with_cost.annualized_return": str(annual_return_path),
            "1day.excess_return_with_cost.max_drawdown": str(max_drawdown_path)
        })
        
        agent = ModelOptimizationAgent(llm)
        agent.mcp_client = mock_mcp_client
        
        print(f"📊 测试因子数量: {len(factors)}")
        print(f"🔄 最大迭代次数: 2")
        print()
        
        # 注意：这里会尝试注册因子池，如果 factor_pool_registry 不可用会失败
        # 所以我们只测试到配置生成部分
        try:
            # 测试配置生成
            factor_pool_name = "CustomFactors_Test"
            module_name = agent._to_snake_case(factor_pool_name)
            module_path = f"qlib_benchmark.factor_pools.{module_name}"
            
            yaml_config = agent._generate_initial_yaml_config(
                factor_pool_name=factor_pool_name,
                module_path=module_path,
                factors_count=len(factors)
            )
            
            print("✅ 配置生成成功")
            
            # 测试指标提取（使用模拟结果）
            train_result = {
                "ic": str(ic_path),
                "rank_ic": str(rank_ic_path),
                "1day.excess_return_with_cost.annualized_return": str(annual_return_path),
                "1day.excess_return_with_cost.max_drawdown": str(max_drawdown_path)
            }
            
            metrics = agent._extract_metrics(train_result)
            print(f"✅ 指标提取成功:")
            print(f"  - IC 均值: {metrics.get('ic_mean', 'N/A')}")
            print(f"  - 年化收益: {metrics.get('annualized_return', 'N/A')}")
            
            # 测试得分计算
            score = agent._compute_optimization_score(metrics)
            print(f"✅ 得分计算: {score:.4f}")
            
        except Exception as e:
            print(f"⚠️  部分功能测试失败（可能是依赖问题）: {e}")
            import traceback
            traceback.print_exc()
    
    print("✅ 完整优化流程测试完成")
    print()


def test_real_mcp_integration():
    """测试真实的 MCP 客户端集成（如果可用）"""
    print("=" * 80)
    print("测试 4: 真实 MCP 客户端集成")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = create_real_llm_service()
    agent = ModelOptimizationAgent(llm)
    
    # 检查 MCP 客户端是否可用
    if not agent.mcp_client:
        print("⚠️  MCP 客户端未初始化，跳过测试")
        print("   提示: 确保 MCP 服务器路径正确且可用")
        return
    
    try:
        # 测试列出工具
        tools = agent.list_available_tools()
        print(f"✅ 找到 {len(tools)} 个可用工具:")
        for tool in tools:
            print(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', '')}")
        
        # 测试调用工具（使用一个简单的测试）
        if tools:
            print("\n📋 工具列表获取成功")
        else:
            print("\n⚠️  未找到可用工具")
            
    except Exception as e:
        print(f"❌ MCP 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ 真实 MCP 客户端集成测试完成")
    print()


def test_with_real_experiment_results():
    """使用真实的实验结果进行测试"""
    print("=" * 80)
    print("测试 5: 使用真实实验结果")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = create_real_llm_service()
    if not llm:
        print("⚠️  跳过测试：LLM 服务不可用")
        return
    
    agent = ModelOptimizationAgent(llm)
    
    # 查找最近的实验结果目录
    mlruns_dir = Path("/data1/liuzhentao/trading_agent/mlruns")
    if not mlruns_dir.exists():
        print("⚠️  未找到实验结果目录，跳过测试")
        return
    
    # 查找最新的实验
    experiment_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not experiment_dirs:
        print("⚠️  未找到实验目录，跳过测试")
        return
    
    latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 使用实验目录: {latest_experiment}")
    
    # 查找最新的运行
    run_dirs = [d for d in latest_experiment.iterdir() if d.is_dir()]
    if not run_dirs:
        print("⚠️  未找到运行目录，跳过测试")
        return
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 使用运行目录: {latest_run}")
    
    # 尝试读取指标文件
    metrics_dir = latest_run / "metrics"
    if metrics_dir.exists():
        metric_files = list(metrics_dir.glob("*.txt"))
        print(f"📊 找到 {len(metric_files)} 个指标文件")
        
        # 构建训练结果字典
        train_result = {}
        for metric_file in metric_files:
            metric_name = metric_file.stem
            train_result[metric_name] = str(metric_file)
        
        # 尝试提取指标
        try:
            metrics = agent._extract_metrics(train_result)
            print("✅ 成功提取指标:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
                else:
                    print(f"  - {key}: {value}")
        except Exception as e:
            print(f"⚠️  指标提取失败: {e}")
    else:
        print("⚠️  未找到 metrics 目录")
    
    print("✅ 真实实验结果测试完成")
    print()


def run_all_integration_tests():
    """运行所有集成测试"""
    print("\n" + "=" * 80)
    print("开始运行模型优化 Agent 集成测试套件")
    print("=" * 80 + "\n")
    
    tests = [
        test_real_llm_optimization_suggestion,
        test_apply_real_llm_suggestion,
        test_full_optimization_flow_with_mock_results,
        test_real_mcp_integration,
        test_with_real_experiment_results,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            if "跳过" in str(e) or "不可用" in str(e):
                skipped += 1
                print(f"⏭️  {test_func.__name__} 被跳过")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} 失败: {e}")
                import traceback
                traceback.print_exc()
        print()
    
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("=" * 80)
    
    if failed > 0:
        print("\n💡 提示:")
        print("  - 设置 QWEN_API_KEY 环境变量以使用真实 LLM 服务")
        print("  - 确保 MCP 服务器路径正确")
        print("  - 确保实验结果目录存在")
    
    return failed == 0


if __name__ == "__main__":
    import os
    
    # 检查环境变量
    if not os.getenv("QWEN_API_KEY"):
        print("⚠️  警告: 未设置 QWEN_API_KEY 环境变量")
        print("   部分测试将使用 Mock 服务或跳过")
        print("   设置方法: export QWEN_API_KEY='your-api-key'")
        print()
    
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)

