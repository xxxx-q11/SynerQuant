"""
因子评估功能测试

从原始 factor_eval_Agent.py 的 if __name__ == "__main__" 部分提取
"""
import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_correlation_filter():
    """测试因子相关性过滤功能"""
    print("=" * 80)
    print("测试因子相关性过滤功能")
    print("=" * 80)
    
    # 导入评估器
    from Agent.evaluators.correlation_evaluator import CorrelationEvaluator
    
    # 创建测试因子
    test_factors = [{
        "qlib_expression": "(Less($open, $close)-$low)/$open",
        "ic": 0.029146308079361916,
        "needs_cs_rank": False,
        "is_valid": True,
        "original_expression": "Rank(TsWMA(TsIr(TsWMA(volume,10),20),20))",
        "rank_ic_valid": 0.0042740413919091225
    }]
    
    # 创建评估器实例
    evaluator = CorrelationEvaluator()
    
    print(f"\n测试因子:")
    print(f"  表达式: {test_factors[0]['qlib_expression']}")
    print(f"  IC: {test_factors[0]['ic']}")
    print(f"  需要 CSRank: {test_factors[0]['needs_cs_rank']}")
    print(f"\n开始测试相关性过滤...")
    
    # 测试相关性检查（不传入 SOTA 因子池，应该返回 True）
    should_keep = evaluator.check_correlation(
        test_factors[0],
        sota_factors=[],  # 空的 SOTA 因子池
        threshold=0.99,
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    print(f"\n结果: should_keep = {should_keep}")
    assert should_keep == True, "空 SOTA 池时应该返回 True"
    print("✓ 测试通过")


def test_file_utils():
    """测试文件工具类"""
    print("=" * 80)
    print("测试文件工具类")
    print("=" * 80)
    
    from Agent.utils.file_utils import FileUtils, ConfigLoader
    
    # 测试生成哈希文件名
    filename = FileUtils.generate_hash_filename(
        content="test_factor_expression",
        prefix="eval_result_",
        suffix=".json"
    )
    print(f"生成的文件名: {filename}")
    assert filename.startswith("eval_result_")
    assert filename.endswith(".json")
    print("✓ 测试通过")
    
    # 测试加载 Qlib 操作符配置
    try:
        ops = ConfigLoader.load_qlib_operators()
        print(f"加载的操作符配置包含 {len(ops.get('qlib_operators', {}).get('operators', {}))} 类操作符")
        print("✓ 配置加载成功")
    except FileNotFoundError:
        print("! 配置文件不存在（可能需要先创建）")


def test_factor_filter():
    """测试因子筛选功能"""
    print("=" * 80)
    print("测试因子筛选功能")
    print("=" * 80)
    
    from Agent.factor_eval_agent_new import FactorEvalAgent
    
    # 创建 Agent（不需要 LLM 服务来测试筛选功能）
    agent = FactorEvalAgent(llm_service=None)
    
    # 测试数据
    test_factors = [
        {"expression": "factor1", "ic": 0.05, "rank_ic_valid": 0.03},
        {"expression": "factor2", "ic": 0.02, "rank_ic_valid": 0.01},  # IC 太低
        {"expression": "factor3", "ic": 0.04, "rank_ic_valid": -0.01},  # rank_ic 为负
        {"expression": "factor4", "ic": 0.06, "rank_ic_valid": 0.05},
        {"expression": "factor5", "ic": None, "rank_ic_valid": 0.02},  # IC 为 None
    ]
    
    filtered = agent.filter_and_sort_factors(test_factors, ic_threshold=0.03)
    
    print(f"原始因子数: {len(test_factors)}")
    print(f"筛选后因子数: {len(filtered)}")
    
    # 验证结果
    assert len(filtered) == 2, f"预期 2 个因子，实际 {len(filtered)}"
    assert filtered[0]["expression"] == "factor4", "应该按 rank_ic_valid 降序排列"
    assert filtered[1]["expression"] == "factor1"
    
    print("筛选结果:")
    for f in filtered:
        print(f"  - {f['expression']}: IC={f['ic']}, rank_ic={f['rank_ic_valid']}")
    
    print("✓ 测试通过")


def test_workflow_generator():
    """测试 Workflow 配置生成器"""
    print("=" * 80)
    print("测试 Workflow 配置生成器")
    print("=" * 80)
    
    from Agent.generators.workflow_config_generator import WorkflowConfigGenerator
    
    generator = WorkflowConfigGenerator()
    
    # 检查模型参数配置
    print("模型参数配置:")
    for size, configs in generator.MODEL_PARAMS.items():
        print(f"  {size}:")
        for model, params in configs.items():
            print(f"    {model}: {len(params)} 个参数")
    
    print("✓ 配置加载成功")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("运行因子评估模块测试")
    print("=" * 80 + "\n")
    
    tests = [
        ("文件工具类", test_file_utils),
        ("因子筛选", test_factor_filter),
        ("Workflow 生成器", test_workflow_generator),
        # ("相关性过滤", test_correlation_filter),  # 需要 Qlib 环境
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()

