#!/usr/bin/env python3
"""
策略生成 Agent 测试脚本

测试 StrategyGenerationAgent 的双版本回测功能（baseline 和 LLM 增强）

用法:
    python test_strategy_generation.py <yaml_file_path> [--use-llm] [--skip-backtest]

示例:
    # 测试 baseline 版本（不使用 LLM）
    python test_strategy_generation.py /path/to/workflow_config.yaml
    
    # 测试 LLM 增强版本
    python test_strategy_generation.py /path/to/workflow_config.yaml --use-llm
    
    # 测试双版本对比（baseline + LLM 增强）
    python test_strategy_generation.py /path/to/workflow_config.yaml --use-llm --run-both
    
    # 跳过回测，只测试配置生成
    python test_strategy_generation.py /path/to/workflow_config.yaml --skip-backtest
"""
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Agent.strategy_generation_agent import StrategyGenerationAgent, StrategyConfig
from Agent.agent_factory import load_env_config, create_agent


def extract_factors_from_yaml(yaml_path: str) -> List[str]:
    """
    从 yaml 配置文件中提取因子列表（仅因子名称）
    
    Args:
        yaml_path: yaml 配置文件路径
        
    Returns:
        因子名称列表
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")
    
    # 读取 yaml 配置
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取 handler 配置
    handler_config = None
    if 'task' in config and 'dataset' in config['task']:
        dataset_kwargs = config['task']['dataset'].get('kwargs', {})
        handler_config = dataset_kwargs.get('handler', {})
    
    if not handler_config:
        raise ValueError("YAML 配置中未找到 handler 配置")
    
    class_name = handler_config.get('class')
    module_path = handler_config.get('module_path')
    
    if not class_name or not module_path:
        raise ValueError(f"Handler 配置不完整: class={class_name}, module_path={module_path}")
    
    print(f"[提取因子] 从因子池提取: {class_name} ({module_path})")
    
    # 从模块路径推断 JSON 元数据文件路径
    module_parts = module_path.split('.')
    if len(module_parts) < 3 or module_parts[0] != 'qlib_benchmark' or module_parts[1] != 'factor_pools':
        raise ValueError(f"不支持的模块路径格式: {module_path}")
    
    module_name = module_parts[2]
    
    # 查找因子池目录
    factors_root = project_root / "Qlib_MCP" / "workspace" / "qlib_benchmark"
    metadata_file = factors_root / "factor_pools" / f"{module_name}.json"
    
    if not metadata_file.exists():
        # 尝试从 yaml 文件所在目录查找
        yaml_dir = yaml_path.parent
        possible_paths = [
            factors_root / "factor_pools" / f"{module_name}.json",
            yaml_dir.parent / "factor_pools" / f"{module_name}.json",
            project_root / "Qlib_MCP" / "workspace" / "qlib_benchmark" / "factor_pools" / f"{module_name}.json",
        ]
        
        found = False
        for path in possible_paths:
            if path.exists():
                metadata_file = path
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"未找到因子池元数据文件: {module_name}.json\n"
                f"已尝试路径:\n" + "\n".join(f"  - {p}" for p in possible_paths)
            )
    
    # 读取因子元数据
    print(f"[提取因子] 读取元数据文件: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    factors = metadata.get('factors', [])
    if not factors:
        raise ValueError(f"因子池 {class_name} 中没有因子数据")
    
    # 提取因子名称列表
    factor_names = [f.get('name', f.get('expression', '')) for f in factors if f]
    factor_names = [name for name in factor_names if name]  # 过滤空值
    
    print(f"[提取因子] 成功提取 {len(factor_names)} 个因子")
    return factor_names


def extract_model_info_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    从 YAML 配置文件中提取模型信息
    
    Args:
        yaml_path: YAML 配置文件路径
        
    Returns:
        模型信息字典
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取模型配置
    model_info = {
        "yaml_config_path": str(yaml_path),
        "factor_pool_name": "test_pool",
        "module_path": None,
        "model_class": "TransformerModel",
        "model_kwargs": {},
        "best_metrics": {}
    }
    
    # 尝试从配置中提取更多信息
    if 'task' in config and 'dataset' in config['task']:
        dataset_kwargs = config['task']['dataset'].get('kwargs', {})
        handler_config = dataset_kwargs.get('handler', {})
        if handler_config:
            model_info["module_path"] = handler_config.get('module_path')
            model_info["factor_pool_name"] = handler_config.get('class', 'test_pool')
    
    return model_info


def get_llm_service():
    """获取 LLM 服务实例"""
    try:
        config = load_env_config()
        if not config:
            print("[警告] 无法加载 LLM 配置，将使用规则决策")
            return None
        
        llm_service = create_agent(
            provider=config.get("provider", "qwen"),
            api_key=config.get("api_key"),
            model=config.get("model"),
            base_url=config.get("base_url"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
        )
        print(f"[LLM] 服务创建成功: provider={config.get('provider')}, model={config.get('model')}")
        return llm_service
    except Exception as e:
        print(f"[警告] LLM 服务创建失败: {e}，将使用规则决策")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="测试策略生成 Agent 的双版本回测功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试 baseline 版本（不使用 LLM）
  %(prog)s workflow_config.yaml
  
  # 测试 LLM 增强版本
  %(prog)s workflow_config.yaml --use-llm
  
  # 测试双版本对比（baseline + LLM 增强）
  %(prog)s workflow_config.yaml --use-llm --run-both
  
  # 跳过回测，只测试配置生成
  %(prog)s workflow_config.yaml --skip-backtest
        """
    )
    
    parser.add_argument(
        'yaml_file',
        type=str,
        help='YAML 配置文件路径'
    )
    
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='使用 LLM 增强决策（需要配置 LLM 服务）'
    )
    
    parser.add_argument(
        '--run-both',
        action='store_true',
        help='运行两个版本进行对比（baseline 和 LLM 增强）'
    )
    
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='跳过回测，只测试配置生成'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        default=50,
        help='Top K 股票数量 (默认: 50)'
    )
    
    parser.add_argument(
        '--n-drop',
        type=int,
        default=10,
        help='每次调仓丢弃数量 (默认: 10)'
    )
    
    parser.add_argument(
        '--news-data-path',
        type=str,
        default=None,
        help='新闻数据文件路径（JSON 格式）'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 策略生成 Agent 测试")
    print("=" * 80)
    print(f"YAML 文件: {args.yaml_file}")
    print(f"使用 LLM: {args.use_llm}")
    print(f"运行双版本: {args.run_both}")
    print(f"跳过回测: {args.skip_backtest}")
    print(f"TopK: {args.topk}, N_Drop: {args.n_drop}")
    print("=" * 80)
    
    try:
        # Step 1: 从 yaml 文件中提取因子和模型信息
        print("\n📋 Step 1: 提取配置信息...")
        factor_list = extract_factors_from_yaml(args.yaml_file)
        model_info = extract_model_info_from_yaml(args.yaml_file)
        print(f"✅ 提取完成: {len(factor_list)} 个因子")
        
        # Step 2: 初始化 LLM 服务（如果需要）
        llm_service = None
        if args.use_llm:
            print("\n🤖 Step 2: 初始化 LLM 服务...")
            llm_service = get_llm_service()
            if not llm_service:
                print("⚠️  LLM 服务不可用，将使用规则决策")
                args.use_llm = True
        else:
            print("\n🤖 Step 2: 跳过 LLM 服务初始化（使用规则决策）")
        
        # Step 3: 创建策略配置
        print("\n🔧 Step 3: 创建策略配置...")
        strategy_config = StrategyConfig(
            topk=args.topk,
            n_drop=args.n_drop,
            max_turnover=0.3,
            min_trade_value=10000,
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5,
            limit_threshold=0.095,
        )
        print(f"✅ 策略配置: TopK={strategy_config.topk}, N_Drop={strategy_config.n_drop}")
        
        # Step 4: 创建策略生成 Agent
        print("\n🚀 Step 4: 创建策略生成 Agent...")
        agent = StrategyGenerationAgent(
            llm_service=llm_service,
            config=strategy_config,
            use_llm_decision=args.use_llm,
            news_data_path=args.news_data_path,
            news_batch_size=10
        )
        print("✅ Agent 创建成功")
        
        # Step 5: 执行策略生成流程
        print("\n🎯 Step 5: 执行策略生成流程...")
        run_both = args.run_both and args.use_llm  # 只有启用 LLM 时才运行双版本
        
        result = agent.process(
            model_info=model_info,
            sota_pool_list=factor_list,
            run_backtest=not args.skip_backtest,
            run_both_versions=run_both
        )
        
        # Step 6: 输出结果
        print("\n" + "=" * 80)
        print("📊 测试结果汇总")
        print("=" * 80)
        
        if result.get("status") == "success":
            print("✅ 策略生成完成!")
            
            # 输出策略配置信息
            strategy_config_dict = result.get("strategy_config", {})
            print(f"\n📋 策略配置:")
            print(f"  - TopK: {strategy_config_dict.get('topk', 'N/A')}")
            print(f"  - N_Drop: {strategy_config_dict.get('n_drop', 'N/A')}")
            print(f"  - 使用 Agent 决策: {strategy_config_dict.get('use_agent_decision', 'N/A')}")
            print(f"  - 使用 LLM 决策: {strategy_config_dict.get('use_llm_decision', 'N/A')}")
            
            # 输出回测结果
            if not args.skip_backtest:
                if run_both:
                    # 双版本对比结果
                    print(f"\n📈 回测结果对比:")
                    
                    baseline_metrics = result.get("baseline_metrics")
                    llm_metrics = result.get("llm_enhanced_metrics")
                    
                    if baseline_metrics:
                        print(f"\n  Baseline 版本:")
                        print(f"    - IC 均值: {baseline_metrics.get('ic_mean', 'N/A'):.4f}" if baseline_metrics.get('ic_mean') is not None else "    - IC 均值: N/A")
                        print(f"    - 年化收益: {baseline_metrics.get('annualized_return', 'N/A'):.2%}" if baseline_metrics.get('annualized_return') is not None else "    - 年化收益: N/A")
                        print(f"    - 最大回撤: {baseline_metrics.get('max_drawdown', 'N/A'):.2%}" if baseline_metrics.get('max_drawdown') is not None else "    - 最大回撤: N/A")
                    
                    if llm_metrics:
                        print(f"\n  LLM 增强版本:")
                        print(f"    - IC 均值: {llm_metrics.get('ic_mean', 'N/A'):.4f}" if llm_metrics.get('ic_mean') is not None else "    - IC 均值: N/A")
                        print(f"    - 年化收益: {llm_metrics.get('annualized_return', 'N/A'):.2%}" if llm_metrics.get('annualized_return') is not None else "    - 年化收益: N/A")
                        print(f"    - 最大回撤: {llm_metrics.get('max_drawdown', 'N/A'):.2%}" if llm_metrics.get('max_drawdown') is not None else "    - 最大回撤: N/A")
                    
                    # 输出对比信息
                    comparison = result.get("comparison")
                    if comparison and comparison.get("differences"):
                        print(f"\n  📊 对比分析:")
                        for key, diff_info in comparison["differences"].items():
                            if diff_info.get("improved"):
                                sign = "↑"
                            else:
                                sign = "↓"
                            print(f"    - {key}: {sign} {diff_info.get('absolute', 0):.4f} ({diff_info.get('percentage', 0):+.2f}%)")
                else:
                    # 单版本结果
                    backtest_metrics = result.get("backtest_metrics")
                    if backtest_metrics:
                        print(f"\n📈 回测结果:")
                        print(f"  - IC 均值: {backtest_metrics.get('ic_mean', 'N/A'):.4f}" if backtest_metrics.get('ic_mean') is not None else "  - IC 均值: N/A")
                        print(f"  - 年化收益: {backtest_metrics.get('annualized_return', 'N/A'):.2%}" if backtest_metrics.get('annualized_return') is not None else "  - 年化收益: N/A")
                        print(f"  - 最大回撤: {backtest_metrics.get('max_drawdown', 'N/A'):.2%}" if backtest_metrics.get('max_drawdown') is not None else "  - 最大回撤: N/A")
            
            # 输出配置文件路径
            strategy = result.get("strategy", {})
            if strategy.get("yaml_config_path"):
                print(f"\n💾 生成的配置文件: {strategy['yaml_config_path']}")
            
            # 输出日志摘要
            logs = result.get("logs", [])
            if logs:
                print(f"\n📝 执行日志 (最后10条):")
                for log in logs[-10:]:
                    print(f"  {log}")
        else:
            print("❌ 策略生成失败!")
            error = result.get("error", "未知错误")
            print(f"错误信息: {error}")
            
            # 输出日志
            if "logs" in result:
                print("\n📋 执行日志:")
                for log in result["logs"][-20:]:  # 只显示最后20条
                    print(f"  {log}")
        
        print("=" * 80)
        
        # 返回状态码
        return 0 if result.get("status") == "success" else 1
        
    except Exception as e:
        import traceback
        print(f"\n❌ 执行出错: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

