#!/usr/bin/env python3
"""
模型优化独立启动脚本

用法:
    python optimize_yaml_main.py <yaml_file_path> [--max_iterations N] [--target_return R] [--target_drawdown D]

示例:
    python optimize_yaml_main.py /path/to/workflow_config.yaml
    python optimize_yaml_main.py /path/to/workflow_config.yaml --max_iterations 5 --target_return 0.12 --target_drawdown -0.15
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

from Agent.model_optimization_Agent import ModelOptimizationAgent
from Agent.agent_factory import load_env_config, create_agent


def extract_factors_from_yaml(yaml_path: str) -> List[Dict[str, Any]]:
    """
    从 yaml 配置文件中提取因子列表
    
    Args:
        yaml_path: yaml 配置文件路径
        
    Returns:
        因子列表，格式为 [{"expression": "因子表达式", "ic": IC值}, ...]
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
    # module_path 格式: qlib_benchmark.factor_pools.custom_factors_xxx
    # 需要转换为文件路径: Qlib_MCP/workspace/qlib_benchmark/factor_pools/custom_factors_xxx.json
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
        # 检查是否有 factor_pools 目录
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
    
    print(f"[提取因子] 成功提取 {len(factors)} 个因子")
    return factors


def get_llm_service():
    """获取 LLM 服务实例"""
    config = load_env_config()
    return create_agent(
        provider=config.get("provider", "qwen"),
        api_key=config.get("api_key"),
        model=config.get("model"),
        base_url=config.get("base_url"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens"),
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于 YAML 配置文件进行模型优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s workflow_config.yaml
  %(prog)s workflow_config.yaml --max_iterations 5
  %(prog)s workflow_config.yaml --max_iterations 10 --target_return 0.15 --target_drawdown -0.20
        """
    )
    
    parser.add_argument(
        'yaml_file',
        type=str,
        help='要优化的 YAML 配置文件路径'
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=10,
        help='最大迭代次数 (默认: 10)'
    )
    
    parser.add_argument(
        '--target_return',
        type=float,
        default=0.15,
        help='目标年化收益率 (默认: 0.15)'
    )
    
    parser.add_argument(
        '--target_drawdown',
        type=float,
        default=-0.20,
        help='目标最大回撤 (默认: -0.20)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 模型优化启动")
    print("=" * 80)
    print(f"YAML 文件: {args.yaml_file}")
    print(f"最大迭代次数: {args.max_iterations}")
    print(f"目标年化收益: {args.target_return:.2%}")
    print(f"目标最大回撤: {args.target_drawdown:.2%}")
    print("=" * 80)
    
    try:
        # Step 1: 从 yaml 文件中提取因子
        print("\n📋 Step 1: 提取因子信息...")
        factors = extract_factors_from_yaml(args.yaml_file)
        
        # Step 2: 初始化 LLM 服务
        print("\n🤖 Step 2: 初始化 LLM 服务...")
        llm_service = get_llm_service()
        print("✅ LLM 服务初始化成功")
        
        # Step 3: 创建优化 Agent
        print("\n🔧 Step 3: 创建模型优化 Agent...")
        agent = ModelOptimizationAgent(llm_service)
        print("✅ Agent 创建成功")
        
        # Step 4: 执行优化
        print("\n🎯 Step 4: 开始模型优化...")
        result = agent.process(
            factors=factors,
            max_iterations=args.max_iterations,
            target_annualized_return=args.target_return,
            target_max_drawdown=args.target_drawdown
        )
        
        # Step 5: 输出结果
        print("\n" + "=" * 80)
        print("📊 优化结果汇总")
        print("=" * 80)
        
        if result.get("status") == "success":
            print("✅ 优化完成!")
            
            best_result = result.get("best_result")
            if best_result:
                print(f"\n🏆 最佳配置 (迭代 {best_result['iteration']}):")
                metrics = best_result.get("metrics", {})
                print(f"  - IC 均值: {metrics.get('ic_mean', 'N/A'):.4f}" if metrics.get('ic_mean') else "  - IC 均值: N/A")
                print(f"  - Rank IC 均值: {metrics.get('rank_ic_mean', 'N/A'):.4f}" if metrics.get('rank_ic_mean') else "  - Rank IC 均值: N/A")
                print(f"  - IR: {metrics.get('ir', 'N/A'):.4f}" if metrics.get('ir') else "  - IR: N/A")
                print(f"  - 年化收益: {metrics.get('annualized_return', 'N/A'):.2%}" if metrics.get('annualized_return') else "  - 年化收益: N/A")
                print(f"  - 最大回撤: {metrics.get('max_drawdown', 'N/A'):.2%}" if metrics.get('max_drawdown') else "  - 最大回撤: N/A")
                print(f"  - 综合得分: {result.get('best_score', 'N/A'):.4f}" if result.get('best_score') else "  - 综合得分: N/A")
                print(f"\n📁 最优配置文件: {best_result.get('yaml_path', 'N/A')}")
            
            print(f"\n📈 总迭代次数: {result.get('total_iterations', 0)}")
            print(f"📝 因子数量: {result.get('factors_count', 0)}")
            
            if result.get("yaml_config_path"):
                print(f"\n💾 最终配置路径: {result['yaml_config_path']}")
        else:
            print("❌ 优化失败!")
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

