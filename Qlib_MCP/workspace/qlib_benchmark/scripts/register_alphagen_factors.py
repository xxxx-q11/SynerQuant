#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaSAGE 因子注册脚本

这个脚本用于将 AlphaSAGE 挖掘出的因子注册为 qlib 可用的 Handler 类。

使用方法:
    # 方式1: 注册单个因子文件
    python register_alphagen_factors.py --file out_gp/csi300_2020_day_0/4_qlib_factors.json --name GPFactorsCsi300
    
    # 方式2: 注册目录下所有因子文件
    python register_alphagen_factors.py --dir out_gp/csi300_2020_day_0 --name GPFactorsCsi300
    
    # 方式3: 只注册 IC 最高的 N 个因子
    python register_alphagen_factors.py --file factors.json --name MyFactors --top-n 50 --min-ic 0.02

注册后可以在 yaml 配置中使用:
    handler:
      class: GPFactorsCsi300
      module_path: qlib_benchmark.factor_pools.gp_factors_csi300
      kwargs:
        start_time: 2008-01-01
        end_time: 2020-08-01
        ...
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加 AlphaSAGE 路径
alphagen_path = project_root.parent / "AlphaSAGE" / "src"
if alphagen_path.exists():
    sys.path.insert(0, str(alphagen_path))


def register_from_file(
    factor_file: str,
    pool_name: str,
    top_n: int = None,
    min_ic: float = None,
    description: str = ""
):
    """从单个文件注册因子"""
    from factor_pool_registry import register_from_alphagen_file
    
    print(f"[Register] 正在从文件注册因子: {factor_file}")
    print(f"[Register] 因子池名称: {pool_name}")
    
    module_path = register_from_alphagen_file(
        factor_file=factor_file,
        pool_name=pool_name,
        description=description,
        top_n=top_n,
        min_ic=min_ic
    )
    
    print(f"\n[Register] 注册完成!")
    print(f"[Register] 模块路径: {module_path}")
    print(f"\n在 yaml 配置中使用:")
    print(f"  handler:")
    print(f"    class: {pool_name}")
    print(f"    module_path: {module_path}")
    print(f"    kwargs:")
    print(f"      start_time: 2008-01-01")
    print(f"      end_time: 2020-08-01")
    print(f"      ...")
    
    return module_path


def register_from_directory(
    factor_dir: str,
    pool_name: str,
    top_n: int = None,
    min_ic: float = None
):
    """从目录注册因子（合并所有因子文件）"""
    from factor_pool_registry import get_registry
    from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
    
    factor_path = Path(factor_dir)
    if not factor_path.exists():
        raise FileNotFoundError(f"目录不存在: {factor_dir}")
    
    converter = AlphaSAGEToQlibConverter()
    all_factors = {}  # {qlib_expression: ic}
    
    # 遍历所有 JSON 文件
    for json_file in sorted(factor_path.glob("*.json")):
        try:
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否已转换
            if 'qlib_fields' in data and data['qlib_fields']:
                for i, field in enumerate(data['qlib_fields']):
                    ic = 0.0
                    if 'factors' in data and i < len(data['factors']):
                        ic = data['factors'][i].get('ic', 0.0)
                    if field not in all_factors or all_factors[field] < ic:
                        all_factors[field] = ic
            elif 'factors' in data:
                # 转换并过滤无效表达式（如对常量做 Rolling 操作的表达式）
                converted = converter.convert_batch(data['factors'], filter_invalid=True)
                for factor in converted:
                    # 只处理有效的表达式
                    if factor.get('qlib_expression') and factor.get('is_valid', True):
                        qlib_expr = factor['qlib_expression']
                        ic = factor.get('ic', 0.0)
                        if qlib_expr not in all_factors or all_factors[qlib_expr] < ic:
                            all_factors[qlib_expr] = ic
            elif 'cache' in data:
                # 原始 cache 格式
                for expr, ic in data['cache'].items():
                    if ic > 0:
                        try:
                            qlib_expr = converter.convert(expr)
                            if qlib_expr not in all_factors or all_factors[qlib_expr] < ic:
                                all_factors[qlib_expr] = ic
                        except Exception:
                            pass
            
            print(f"[Register] 处理文件: {json_file.name}, 累计因子数: {len(all_factors)}")
        except Exception as e:
            print(f"[Warning] 跳过文件 {json_file}: {e}")
    
    # 转换为列表并排序
    factors_list = [
        {'expression': expr, 'ic': ic}
        for expr, ic in sorted(all_factors.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # 应用过滤
    if min_ic is not None:
        factors_list = [f for f in factors_list if f['ic'] >= min_ic]
    
    if top_n is not None:
        factors_list = factors_list[:top_n]
    
    print(f"\n[Register] 合并后因子总数: {len(factors_list)}")
    
    # 注册
    registry = get_registry()
    description = f"从目录 {factor_dir} 合并的因子池\n因子数量: {len(factors_list)}"
    module_path = registry.register_factor_pool(pool_name, factors_list, description)
    
    print(f"\n[Register] 注册完成!")
    print(f"[Register] 模块路径: {module_path}")
    
    return module_path


def main():
    parser = argparse.ArgumentParser(
        description="AlphaSAGE 因子注册脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='因子文件路径 (JSON)'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='因子目录路径 (将合并所有 JSON 文件)'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        required=True,
        help='因子池名称 (将作为 Handler 类名)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='只取 IC 最高的前 N 个因子'
    )
    parser.add_argument(
        '--min-ic',
        type=float,
        default=None,
        help='最小 IC 阈值'
    )
    parser.add_argument(
        '--description',
        type=str,
        default="",
        help='因子池描述'
    )
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.error("必须指定 --file 或 --dir")
    
    if args.file and args.dir:
        parser.error("不能同时指定 --file 和 --dir")
    
    if args.file:
        register_from_file(
            factor_file=args.file,
            pool_name=args.name,
            top_n=args.top_n,
            min_ic=args.min_ic,
            description=args.description
        )
    else:
        register_from_directory(
            factor_dir=args.dir,
            pool_name=args.name,
            top_n=args.top_n,
            min_ic=args.min_ic
        )


if __name__ == '__main__':
    main()

