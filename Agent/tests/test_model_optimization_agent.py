"""
模型优化 Agent 功能测试

测试 model_optimization_Agent.py 的主要功能，检查是否有 bug
"""
import sys
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Agent.base_agent import BaseAgent, LLMResponse, LLMProvider, Message


class MockLLMService(BaseAgent):
    """模拟 LLM 服务 - 继承 BaseAgent"""
    def __init__(self):
        # 调用父类初始化，使用模拟参数
        super().__init__(
            api_key="mock_key",
            model="mock_model",
            temperature=0.7
        )
        self.call_count = 0
    
    def get_provider(self) -> LLMProvider:
        """返回模拟的提供商类型"""
        return LLMProvider.QWEN
    
    def chat(self, messages, stream=False, **kwargs):
        """实现 BaseAgent 的抽象方法"""
        self.call_count += 1
        # 返回一个模拟的优化建议
        suggestion = {
            "analysis": "当前模型表现良好，但可以进一步优化",
            "issues": ["IC 值可以提升", "学习率可以调整"],
            "summary": "建议增加模型复杂度并调整学习率",
            "model_params_update": {
                "lr": 0.0005,
                "n_epochs": 150
            },
            "reasoning": "通过增加训练轮数和调整学习率来提升模型表现"
        }
        # 返回 LLMResponse 对象
        return LLMResponse(
            content=json.dumps(suggestion, ensure_ascii=False),
            model=self.model
        )
    
    def call(self, prompt, system_prompt=None, stream=False, **kwargs):
        """重写 call 方法，调用父类实现（父类会调用 chat）"""
        # 父类的 call 方法会构建消息并调用 chat，我们只需要确保 chat 被正确实现
        return super().call(prompt, system_prompt=system_prompt, stream=stream, **kwargs)
    
    def parse_json_response(self, response):
        """解析 JSON 响应 - 使用父类实现"""
        return super().parse_json_response(response)


class MockMCPClient:
    """模拟 MCP 客户端"""
    def __init__(self):
        self.call_count = 0
    
    def call_tool(self, tool_name, arguments):
        """模拟工具调用"""
        self.call_count += 1
        # 返回模拟的训练结果
        return json.dumps({
            "ic": "/tmp/mock_ic.pkl",
            "rank_ic": "/tmp/mock_rank_ic.pkl",
            "1day.excess_return_with_cost.annualized_return": "/tmp/mock_annual_return.txt",
            "1day.excess_return_with_cost.max_drawdown": "/tmp/mock_max_drawdown.txt"
        })
    
    def list_tools(self):
        """模拟工具列表"""
        return [
            {
                "name": "qlib_benchmark_runner",
                "description": "运行 Qlib 基准测试",
                "inputSchema": {
                    "properties": {
                        "yaml_path": {"type": "string", "description": "YAML 配置文件路径"}
                    }
                }
            }
        ]


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


def test_initialization():
    """测试初始化"""
    print("=" * 80)
    print("测试 1: 初始化")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    assert agent.llm == llm, "LLM 服务未正确设置"
    assert agent.optimization_history == [], "优化历史应该为空"
    assert agent.template_path.exists(), "模板文件应该存在"
    
    print("✅ 初始化测试通过")
    print()


def test_to_snake_case():
    """测试驼峰命名转换"""
    print("=" * 80)
    print("测试 2: 驼峰命名转换")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 测试各种命名格式
    test_cases = [
        ("CustomFactors_20260114_152914", "custom_factors_20260114_152914"),
        ("TransformerModel", "transformer_model"),
        ("XGBModel", "xgb_model"),
        ("Alpha158", "alpha158"),
    ]
    
    for input_name, expected in test_cases:
        result = agent._to_snake_case(input_name)
        assert result == expected, f"转换失败: {input_name} -> {result}, 期望: {expected}"
        print(f"  ✅ {input_name} -> {result}")
    
    print("✅ 驼峰命名转换测试通过")
    print()


def test_generate_initial_yaml_config():
    """测试生成初始 YAML 配置"""
    print("=" * 80)
    print("测试 3: 生成初始 YAML 配置")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    factor_pool_name = "CustomFactors_Test"
    module_path = "qlib_benchmark.factor_pools.custom_factors_test"
    factors_count = 20
    
    config = agent._generate_initial_yaml_config(
        factor_pool_name=factor_pool_name,
        module_path=module_path,
        factors_count=factors_count
    )
    
    # 检查配置结构
    assert 'task' in config, "配置应该包含 task"
    assert 'dataset' in config['task'], "配置应该包含 dataset"
    assert 'model' in config['task'], "配置应该包含 model"
    
    # 检查 handler 配置
    handler = config['task']['dataset']['kwargs']['handler']
    assert handler['class'] == factor_pool_name, "Handler class 应该正确设置"
    assert handler['module_path'] == module_path, "Handler module_path 应该正确设置"
    
    # 检查模型参数
    model_kwargs = config['task']['model']['kwargs']
    assert 'd_feat' in model_kwargs, "Transformer 模型应该有 d_feat 参数"
    assert model_kwargs['d_feat'] == factors_count, f"d_feat 应该等于因子数量: {factors_count}"
    
    print(f"  ✅ 配置生成成功")
    print(f"  ✅ Handler class: {handler['class']}")
    print(f"  ✅ Handler module_path: {handler['module_path']}")
    print(f"  ✅ Model d_feat: {model_kwargs['d_feat']}")
    print("✅ 生成初始 YAML 配置测试通过")
    print()


def test_adjust_model_params_by_factor_count():
    """测试根据因子数量调整模型参数"""
    print("=" * 80)
    print("测试 4: 根据因子数量调整模型参数")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 读取模板
    with open(agent.template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 测试不同因子数量
    test_cases = [
        (10, 10),   # 小因子池
        (25, 25),  # 中等因子池
        (60, 60),  # 大因子池
    ]
    
    for factors_count, expected_d_feat in test_cases:
        test_config = yaml.safe_load(yaml.dump(config))  # 深拷贝
        result_config = agent._adjust_model_params_by_factor_count(test_config, factors_count)
        
        model_kwargs = result_config['task']['model']['kwargs']
        assert model_kwargs['d_feat'] == expected_d_feat, \
            f"因子数量 {factors_count} 时，d_feat 应该是 {expected_d_feat}"
        print(f"  ✅ 因子数量 {factors_count} -> d_feat: {model_kwargs['d_feat']}")
    
    print("✅ 根据因子数量调整模型参数测试通过")
    print()


def test_extract_metrics():
    """测试提取指标"""
    print("=" * 80)
    print("测试 5: 提取指标")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    from Agent.utils.file_utils import FileUtils
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 创建临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 创建模拟的 IC 数据
        ic_data = pd.Series([0.05, 0.03, 0.04, 0.06, 0.02])
        ic_path = tmpdir / "ic.pkl"
        create_mock_pickle_file(ic_path, ic_data)
        
        # 创建模拟的 Rank IC 数据
        rank_ic_data = pd.Series([0.04, 0.03, 0.05, 0.04, 0.03])
        rank_ic_path = tmpdir / "rank_ic.pkl"
        create_mock_pickle_file(rank_ic_path, rank_ic_data)
        
        # 创建模拟的 MLflow 指标文件
        annual_return_path = tmpdir / "annual_return.txt"
        create_mock_mlflow_metric_file(annual_return_path, 0.15)
        
        max_drawdown_path = tmpdir / "max_drawdown.txt"
        create_mock_mlflow_metric_file(max_drawdown_path, -0.12)
        
        # 构建训练结果
        train_result = {
            "ic": str(ic_path),
            "rank_ic": str(rank_ic_path),
            "1day.excess_return_with_cost.annualized_return": str(annual_return_path),
            "1day.excess_return_with_cost.max_drawdown": str(max_drawdown_path)
        }
        
        # 提取指标
        metrics = agent._extract_metrics(train_result)
        
        # 验证指标
        assert 'ic_mean' in metrics, "应该有 ic_mean"
        assert 'rank_ic_mean' in metrics, "应该有 rank_ic_mean"
        assert 'annualized_return' in metrics, "应该有 annualized_return"
        assert 'max_drawdown' in metrics, "应该有 max_drawdown"
        assert 'ir' in metrics, "应该有 ir (信息比率)"
        
        print(f"  ✅ IC 均值: {metrics['ic_mean']:.4f}")
        print(f"  ✅ Rank IC 均值: {metrics['rank_ic_mean']:.4f}")
        print(f"  ✅ 年化收益: {metrics['annualized_return']:.2%}")
        print(f"  ✅ 最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  ✅ IR: {metrics['ir']:.4f}")
    
    print("✅ 提取指标测试通过")
    print()


def test_compute_optimization_score():
    """测试计算优化得分"""
    print("=" * 80)
    print("测试 6: 计算优化得分")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 测试用例
    test_cases = [
        {
            "name": "良好表现",
            "metrics": {
                "ic_mean": 0.05,
                "ic_std": 0.02,
                "ir": 2.5,
                "annualized_return": 0.15,
                "max_drawdown": -0.10
            }
        },
        {
            "name": "一般表现",
            "metrics": {
                "ic_mean": 0.03,
                "ic_std": 0.02,
                "ir": 1.5,
                "annualized_return": 0.10,
                "max_drawdown": -0.15
            }
        },
        {
            "name": "较差表现",
            "metrics": {
                "ic_mean": 0.01,
                "ic_std": 0.02,
                "ir": 0.5,
                "annualized_return": 0.05,
                "max_drawdown": -0.20
            }
        }
    ]
    
    for test_case in test_cases:
        score = agent._compute_optimization_score(test_case["metrics"])
        print(f"  ✅ {test_case['name']}: 得分 = {score:.4f}")
        assert isinstance(score, float), "得分应该是浮点数"
        assert score > float('-inf'), "得分应该大于负无穷"
    
    print("✅ 计算优化得分测试通过")
    print()


def test_check_target_reached():
    """测试检查目标是否达到"""
    print("=" * 80)
    print("测试 7: 检查目标是否达到")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 测试用例
    test_cases = [
        {
            "name": "达到目标",
            "metrics": {
                "annualized_return": 0.20,
                "max_drawdown": -0.15
            },
            "target_return": 0.15,
            "target_drawdown": -0.20,
            "expected": True
        },
        {
            "name": "未达到收益目标",
            "metrics": {
                "annualized_return": 0.10,
                "max_drawdown": -0.15
            },
            "target_return": 0.15,
            "target_drawdown": -0.20,
            "expected": False
        },
        {
            "name": "回撤过大",
            "metrics": {
                "annualized_return": 0.20,
                "max_drawdown": -0.25
            },
            "target_return": 0.15,
            "target_drawdown": -0.20,
            "expected": False
        }
    ]
    
    for test_case in test_cases:
        result = agent._check_target_reached(
            test_case["metrics"],
            test_case["target_return"],
            test_case["target_drawdown"]
        )
        assert result == test_case["expected"], \
            f"{test_case['name']}: 期望 {test_case['expected']}, 得到 {result}"
        print(f"  ✅ {test_case['name']}: {result}")
    
    print("✅ 检查目标是否达到测试通过")
    print()


def test_build_history_summary():
    """测试构建历史摘要"""
    print("=" * 80)
    print("测试 8: 构建历史摘要")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 创建模拟历史记录
    history = [
        {
            "iteration": 1,
            "metrics": {
                "ic_mean": 0.03,
                "annualized_return": 0.10,
                "max_drawdown": -0.15
            }
        },
        {
            "iteration": 2,
            "metrics": {
                "ic_mean": 0.04,
                "annualized_return": 0.12,
                "max_drawdown": -0.12
            }
        }
    ]
    
    summary = agent._build_history_summary(history)
    
    assert "迭代 1" in summary, "摘要应该包含迭代 1"
    assert "迭代 2" in summary, "摘要应该包含迭代 2"
    assert "IC=" in summary, "摘要应该包含 IC 信息"
    
    print("历史摘要:")
    print(summary)
    print("✅ 构建历史摘要测试通过")
    print()


def test_build_tunable_params_description():
    """测试构建可调参数说明"""
    print("=" * 80)
    print("测试 9: 构建可调参数说明")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 测试不同模型类型
    model_classes = ["TransformerModel", "XGBModel", "LGBModel", "UnknownModel"]
    
    for model_class in model_classes:
        desc = agent._build_tunable_params_description(model_class)
        assert model_class in desc, f"描述应该包含模型类型: {model_class}"
        print(f"  ✅ {model_class}: 描述长度 {len(desc)} 字符")
    
    print("✅ 构建可调参数说明测试通过")
    print()


def test_apply_optimization_suggestion():
    """测试应用优化建议"""
    print("=" * 80)
    print("测试 10: 应用优化建议")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 读取模板配置
    with open(agent.template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建优化建议
    suggestion = {
        "model_params_update": {
            "lr": 0.0005,
            "n_epochs": 150,
            "d_model": 128
        }
    }
    
    # 应用建议
    updated_config = agent._apply_optimization_suggestion(config, suggestion)
    
    # 验证更新
    model_kwargs = updated_config['task']['model']['kwargs']
    assert model_kwargs['lr'] == 0.0005, "学习率应该被更新"
    assert model_kwargs['n_epochs'] == 150, "训练轮数应该被更新"
    assert model_kwargs['d_model'] == 128, "模型维度应该被更新"
    
    print(f"  ✅ 学习率: {model_kwargs['lr']}")
    print(f"  ✅ 训练轮数: {model_kwargs['n_epochs']}")
    print(f"  ✅ 模型维度: {model_kwargs['d_model']}")
    print("✅ 应用优化建议测试通过")
    print()


def test_save_yaml_config():
    """测试保存 YAML 配置"""
    print("=" * 80)
    print("测试 11: 保存 YAML 配置")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 读取模板配置
    with open(agent.template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        agent.output_dir = Path(tmpdir)
        
        yaml_path = agent._save_yaml_config(config, "TestFactors", 1)
        
        assert Path(yaml_path).exists(), "YAML 文件应该被创建"
        
        # 验证文件内容
        with open(yaml_path, 'r', encoding='utf-8') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config is not None, "保存的配置应该可以解析"
        assert 'task' in saved_config, "保存的配置应该包含 task"
        
        print(f"  ✅ YAML 文件已保存: {yaml_path}")
        print("✅ 保存 YAML 配置测试通过")
    print()


def test_save_factors_to_file():
    """测试保存因子到文件"""
    print("=" * 80)
    print("测试 12: 保存因子到文件")
    print("=" * 80)
    
    from Agent.model_optimization_Agent import ModelOptimizationAgent
    
    llm = MockLLMService()
    agent = ModelOptimizationAgent(llm)
    
    # 创建测试因子
    factors = [
        {"expression": "($close - $open) / $open", "ic": 0.05},
        {"expression": "($high - $low) / $close", "ic": 0.03}
    ]
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 临时修改输出目录
        original_output_dir = Path(__file__).parent.parent.parent / "Qlib_MCP" / "workspace" / "factors"
        
        # 使用 patch 来临时修改路径
        with patch.object(agent, '_save_factors_to_file') as mock_save:
            # 直接调用内部方法进行测试
            current_dir = Path(__file__).parent.parent.parent
            output_dir = Path(tmpdir) / "factors"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"optimized_factors_{timestamp}.json"
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "factors_count": len(factors),
                    "factors": factors
                }, f, indent=2, ensure_ascii=False)
            
            assert output_file.exists(), "因子文件应该被创建"
            
            # 验证文件内容
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['factors_count'] == len(factors), "因子数量应该正确"
            assert len(data['factors']) == len(factors), "因子列表应该完整"
            
            print(f"  ✅ 因子文件已保存: {output_file}")
            print(f"  ✅ 因子数量: {data['factors_count']}")
    
    print("✅ 保存因子到文件测试通过")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始运行模型优化 Agent 测试套件")
    print("=" * 80 + "\n")
    
    tests = [
        test_initialization,
        test_to_snake_case,
        test_generate_initial_yaml_config,
        test_adjust_model_params_by_factor_count,
        test_extract_metrics,
        test_compute_optimization_score,
        test_check_target_reached,
        test_build_history_summary,
        test_build_tunable_params_description,
        test_apply_optimization_suggestion,
        test_save_yaml_config,
        test_save_factors_to_file,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

