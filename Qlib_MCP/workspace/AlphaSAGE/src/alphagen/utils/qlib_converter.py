"""
AlphaSAGE 表达式 → Qlib 表达式 转换器

将 AlphaSAGE 基于 PyTorch 的因子表达式（expression.py）转换为 
Qlib 可识别的表达式字符串格式。

使用方法:
    from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
    
    converter = AlphaSAGEToQlibConverter()
    
    # 单个表达式转换
    qlib_expr = converter.convert("TsIr(TsMean(volume,30),50)")
    # 输出: "Div(Mean($volume, 30), Std($volume, 30))"
    
    # 批量转换
    qlib_exprs = converter.convert_batch([
        {"expression": "TsIr(TsMean(volume,30),50)", "ic": 0.044},
        {"expression": "TsMin(close,20)", "ic": 0.035}
    ])
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path


class AlphaSAGEToQlibConverter:
    """AlphaSAGE 表达式到 Qlib 表达式的转换器"""
    
    # 特征名映射: AlphaSAGE -> Qlib
    FEATURE_MAP = {
        'close': '$close',
        'Close': '$close',
        'CLOSE': '$close',
        'open_': '$open',
        'Open': '$open',
        'OPEN': '$open',
        'high': '$high',
        'High': '$high',
        'HIGH': '$high',
        'low': '$low',
        'Low': '$low',
        'LOW': '$low',
        'volume': '$volume',
        'Volume': '$volume',
        'VOLUME': '$volume',
        'vwap': '$vwap',
        'Vwap': '$vwap',
        'VWAP': '$vwap',
    }
    
    # 一元算子映射 (直接对应)
    UNARY_OP_MAP = {
        'Abs': 'Abs',
        'Sign': 'Sign',
        'Log': 'Log',
    }
    
    # 需要展开的一元算子
    UNARY_OP_EXPAND = {
        'SLog1p': lambda x: f'Mul(Sign({x}), Log(Add(Abs({x}), 1)))',
        'Inv': lambda x: f'Div(1, {x})',
    }
    
    # 截面算子 - 这些算子在 qlib 表达式层无法直接实现
    # 需要在数据处理层通过处理器（如 CSRankNorm）实现
    # 转换时直接返回内部表达式，并标记需要截面处理
    CROSS_SECTIONAL_OPS = {'Rank'}
    
    # 二元算子映射
    BINARY_OP_MAP = {
        'Add': 'Add',
        'Sub': 'Sub',
        'Mul': 'Mul',
        'Div': 'Div',
        'Pow': 'Power',
        'Greater': 'Greater',
        'Less': 'Less',
    }
    
    # Rolling 算子映射 (直接对应)
    ROLLING_OP_MAP = {
        'Ref': 'Ref',
        'TsMean': 'Mean',
        'TsSum': 'Sum',
        'TsStd': 'Std',
        'TsVar': 'Var',
        'TsSkew': 'Skew',
        'TsKurt': 'Kurt',
        'TsMax': 'Max',
        'TsMin': 'Min',
        'TsMed': 'Med',
        'TsMad': 'Mad',
        'TsRank': 'Rank',
        'TsDelta': 'Delta',
        'TsWMA': 'WMA',
        'TsEMA': 'EMA',
    }
    
    # 需要展开的 Rolling 算子 (返回一个函数，接收表达式和窗口大小)
    ROLLING_OP_EXPAND = {
        # TsIr = mean / std
        'TsIr': lambda x, n: f'Div(Mean({x}, {n}), Std({x}, {n}))',
        # TsMinMaxDiff = max - min
        'TsMinMaxDiff': lambda x, n: f'Sub(Max({x}, {n}), Min({x}, {n}))',
        # TsMaxDiff = current - max (当前值减去窗口最大值)
        'TsMaxDiff': lambda x, n: f'Sub({x}, Max({x}, {n}))',
        # TsMinDiff = current - min (当前值减去窗口最小值)
        'TsMinDiff': lambda x, n: f'Sub({x}, Min({x}, {n}))',
        # TsDiv = current / mean (当前值除以窗口均值)
        'TsDiv': lambda x, n: f'Div({x}, Mean({x}, {n}))',
        # TsPctChange = (current - oldest) / oldest
        # 注意: Ref(x, N-1) 是 N-1 天前的值
        'TsPctChange': lambda x, n: f'Div(Sub({x}, Ref({x}, {int(n)-1})), Ref({x}, {int(n)-1}))',
    }
    
    # Pair Rolling 算子映射
    PAIR_ROLLING_OP_MAP = {
        'TsCov': 'Cov',
        'TsCorr': 'Corr',
    }
    
    # Qlib Rolling 算子列表（第一个参数必须是表达式，不能是常量）
    QLIB_ROLLING_OPS = {
        'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min',
        'Med', 'Mad', 'Rank', 'Delta', 'WMA', 'EMA', 'Ref'
    }
    
    # Qlib Pair Rolling 算子列表
    QLIB_PAIR_ROLLING_OPS = {'Cov', 'Corr'}
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化转换器
        
        Args:
            strict_mode: 严格模式下，遇到无法转换的算子会抛出异常
        """
        self.strict_mode = strict_mode
        # 用于追踪当前表达式是否包含截面算子
        self._has_cross_sectional_op = False
        # 用于标记表达式是否有效
        self._is_valid_expression = True
        self._invalid_reason = None
        
    def convert(self, alphagen_expr: str) -> str:
        """
        将 AlphaSAGE 表达式转换为 Qlib 表达式
        
        Args:
            alphagen_expr: AlphaSAGE 格式的表达式字符串
            
        Returns:
            Qlib 格式的表达式字符串
            
        Example:
            >>> converter = AlphaSAGEToQlibConverter()
            >>> converter.convert("TsIr(TsMean(volume,30),50)")
            'Div(Mean(Mean($volume, 30), 50), Std(Mean($volume, 30), 50))'
        """
        # 重置状态标记
        self._has_cross_sectional_op = False
        self._is_valid_expression = True
        self._invalid_reason = None
        
        # 去除空格
        expr = alphagen_expr.strip()
        
        # 递归转换（返回表达式和常量性标记）
        result, _ = self._convert_recursive(expr)
        
        # 验证已在递归转换中完成，不再需要单独调用 _validate_qlib_expression
        
        return result
    
    def _validate_qlib_expression_regex(self, qlib_expr: str):
        """
        [备用方法] 使用正则表达式验证转换后的 Qlib 表达式是否有效
        
        注意：此方法只能检测直接的常量参数，无法检测嵌套的常量表达式。
        主要验证逻辑已移至 _convert_recursive 中的常量性追踪。
        
        检测无效模式：
        - Rolling 算子的第一个参数是直接常量（如 Med(-0.01, 10)）
        - Pair Rolling 算子的参数是直接常量
        
        Args:
            qlib_expr: Qlib 格式的表达式字符串
        """
        # 构建 Rolling 算子的正则模式
        # 匹配：OpName(纯数字, 数字) - 如 Med(-0.01, 10)
        rolling_ops_pattern = '|'.join(self.QLIB_ROLLING_OPS)
        # 匹配负数、小数、整数：-0.01, 5.0, 10, -5 等
        number_pattern = r'-?\d+\.?\d*'
        
        # 检测 Rolling 算子第一个参数是常量的情况
        invalid_rolling_pattern = rf'({rolling_ops_pattern})\(\s*({number_pattern})\s*,\s*\d+\s*\)'
        match = re.search(invalid_rolling_pattern, qlib_expr)
        if match:
            self._is_valid_expression = False
            self._invalid_reason = f"Rolling 算子 {match.group(1)} 的第一个参数是常量 {match.group(2)}，qlib 不支持对常量做滚动操作"
            return
        
        # 检测 Pair Rolling 算子参数是常量的情况
        pair_rolling_ops_pattern = '|'.join(self.QLIB_PAIR_ROLLING_OPS)
        
        # 第一个参数是常量：Cov(-0.01, expr, n)
        invalid_pair_first = rf'({pair_rolling_ops_pattern})\(\s*({number_pattern})\s*,'
        match = re.search(invalid_pair_first, qlib_expr)
        if match:
            self._is_valid_expression = False
            self._invalid_reason = f"Pair Rolling 算子 {match.group(1)} 的第一个参数是常量 {match.group(2)}，qlib 不支持"
            return
        
        # 第二个参数是常量：Cov(expr, -0.01, n)
        # 匹配模式更复杂，需要跳过第一个参数
        invalid_pair_second = rf'({pair_rolling_ops_pattern})\([^,]+,\s*({number_pattern})\s*,\s*\d+\s*\)'
        match = re.search(invalid_pair_second, qlib_expr)
        if match:
            self._is_valid_expression = False
            self._invalid_reason = f"Pair Rolling 算子 {match.group(1)} 的第二个参数是常量 {match.group(2)}，qlib 不支持"
            return
    
    def is_valid(self) -> bool:
        """
        检查最近一次转换的表达式是否有效
        
        Returns:
            表达式是否有效
        """
        return self._is_valid_expression
    
    def get_invalid_reason(self) -> Optional[str]:
        """
        获取表达式无效的原因
        
        Returns:
            无效原因，如果表达式有效则返回 None
        """
        return self._invalid_reason
    
    def convert_with_metadata(self, alphagen_expr: str) -> Dict[str, Any]:
        """
        将 AlphaSAGE 表达式转换为 Qlib 表达式，并返回元数据
        
        Args:
            alphagen_expr: AlphaSAGE 格式的表达式字符串
            
        Returns:
            包含 qlib_expression、needs_cs_rank、is_valid 等字段的字典
        """
        qlib_expr = self.convert(alphagen_expr)
        result = {
            'qlib_expression': qlib_expr if self._is_valid_expression else None,
            'needs_cs_rank': self._has_cross_sectional_op,
            'is_valid': self._is_valid_expression
        }
        if not self._is_valid_expression:
            result['invalid_reason'] = self._invalid_reason
        return result
    
    def check_needs_cs_rank(self, alphagen_expr: str) -> bool:
        """
        检查表达式是否包含截面 Rank 算子
        
        Args:
            alphagen_expr: AlphaSAGE 格式的表达式字符串
            
        Returns:
            是否需要截面 Rank 处理
        """
        self.convert(alphagen_expr)
        return self._has_cross_sectional_op
    
    def _convert_recursive(self, expr: str) -> Tuple[str, bool]:
        """
        递归转换表达式
        
        Returns:
            (转换后的表达式, 是否是纯常量表达式)
            
        纯常量表达式是指不包含任何时序数据（如 $close, $volume 等）的表达式。
        对常量表达式做 Rolling 操作在 qlib 中是无效的。
        """
        expr = expr.strip()
        
        # 检查是否是 Constant
        if expr.startswith('Constant('):
            # Constant(1.0) -> 1.0
            match = re.match(r'Constant\(([^)]+)\)', expr)
            if match:
                return match.group(1), True  # 常量
        
        # 检查是否是纯特征名 -> 不是常量（是时序数据）
        if expr in self.FEATURE_MAP:
            return self.FEATURE_MAP[expr], False
        
        # 检查是否是纯数字 -> 是常量
        try:
            float(expr)
            return expr, True
        except ValueError:
            pass
        
        # 检查是否被括号完全包裹，如果是则去掉括号后重新转换
        # 这样可以正确处理 (Sub(TsRank(...))) 这种被额外括号包裹的情况
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            fully_wrapped = True
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    # 如果在最后一个字符前 depth 变为 0，说明不是完全包裹
                    if depth == 0 and i < len(expr) - 1:
                        fully_wrapped = False
                        break
            if fully_wrapped and depth == 0:
                # 被括号完全包裹，去掉括号后重新转换
                return self._convert_recursive(expr[1:-1])
        
        # 解析算子和参数
        parsed = self._parse_operator(expr)
        if parsed is None:
            # 无法解析，可能是纯特征名或常量
            if expr in self.FEATURE_MAP:
                return self.FEATURE_MAP[expr], False
            # 无法识别的表达式，保守地认为是常量
            return expr, True
        
        op_name, args = parsed
        
        # 递归转换参数，同时获取每个参数的常量性
        converted_args = []
        args_is_constant = []
        for arg in args:
            conv_arg, is_const = self._convert_recursive(arg)
            converted_args.append(conv_arg)
            args_is_constant.append(is_const)
        
        # 转换算子，同时验证常量性
        return self._convert_operator_with_validation(op_name, converted_args, args_is_constant)
    
    def _parse_operator(self, expr: str) -> Optional[Tuple[str, List[str]]]:
        """
        解析表达式，提取算子名和参数
        
        Args:
            expr: 表达式字符串，如 "TsMean(volume,30)"
            
        Returns:
            (算子名, [参数列表]) 或 None
        """
        # 匹配算子名
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\((.+)\)$', expr)
        if not match:
            return None
        
        op_name = match.group(1)
        args_str = match.group(2)
        
        # 解析参数（需要处理嵌套括号）
        args = self._split_args(args_str)
        
        return op_name, args
    
    def _split_args(self, args_str: str) -> List[str]:
        """
        分割参数字符串，处理嵌套括号
        
        Args:
            args_str: 参数字符串，如 "TsMean(volume,30),50"
            
        Returns:
            参数列表，如 ["TsMean(volume,30)", "50"]
        """
        args = []
        current_arg = []
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
                current_arg.append(char)
            elif char == ')':
                depth -= 1
                current_arg.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)
        
        if current_arg:
            args.append(''.join(current_arg).strip())
        
        return args
    
    def _convert_operator_with_validation(
        self, 
        op_name: str, 
        args: List[str], 
        args_is_constant: List[bool]
    ) -> Tuple[str, bool]:
        """
        转换单个算子，同时验证 Rolling 算子的参数是否有效
        
        核心逻辑：追踪每个子表达式的"常量性"，当 Rolling 算子的第一个参数
        是常量表达式时，标记整个表达式为无效。
        
        Args:
            op_name: 算子名
            args: 已转换的参数列表
            args_is_constant: 每个参数是否是常量表达式
            
        Returns:
            (转换后的表达式, 该表达式是否是常量)
        """
        # 截面算子 - 保持子表达式的常量性
        if op_name in self.CROSS_SECTIONAL_OPS:
            self._has_cross_sectional_op = True
            return args[0], args_is_constant[0]
        
        # 一元算子 - 保持常量性
        if op_name in self.UNARY_OP_MAP:
            qlib_op = self.UNARY_OP_MAP[op_name]
            return f'{qlib_op}({args[0]})', args_is_constant[0]
        
        # 一元算子展开 - 保持常量性
        if op_name in self.UNARY_OP_EXPAND:
            expand_func = self.UNARY_OP_EXPAND[op_name]
            return expand_func(args[0]), args_is_constant[0]
        
        # 二元算子 - 两个都是常量时结果才是常量
        if op_name in self.BINARY_OP_MAP:
            qlib_op = self.BINARY_OP_MAP[op_name]
            result_is_constant = args_is_constant[0] and args_is_constant[1]
            return f'{qlib_op}({args[0]}, {args[1]})', result_is_constant
        
        # Rolling 算子 - 验证第一个参数不能是常量
        if op_name in self.ROLLING_OP_MAP:
            qlib_op = self.ROLLING_OP_MAP[op_name]
            if args_is_constant[0]:  # 第一个参数是常量，无效！
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Rolling 算子 {qlib_op} 的第一个参数 '{args[0]}' "
                    f"是常量表达式，qlib 不支持对常量做滚动操作"
                )
            # Rolling 算子的结果不是常量（它依赖时间序列，即使对常量操作也产生序列）
            return f'{qlib_op}({args[0]}, {args[1]})', False
        
        # Rolling 算子展开 - 同样需要验证
        if op_name in self.ROLLING_OP_EXPAND:
            if args_is_constant[0]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Rolling 算子 {op_name} 的第一个参数 '{args[0]}' "
                    f"是常量表达式，qlib 不支持对常量做滚动操作"
                )
            expand_func = self.ROLLING_OP_EXPAND[op_name]
            return expand_func(args[0], args[1]), False
        
        # Pair Rolling 算子 - 验证前两个参数
        if op_name in self.PAIR_ROLLING_OP_MAP:
            qlib_op = self.PAIR_ROLLING_OP_MAP[op_name]
            # 检查两个时序参数是否有常量
            if args_is_constant[0] and args_is_constant[1]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Pair Rolling 算子 {qlib_op} 的两个参数都是常量表达式，qlib 不支持"
                )
            elif args_is_constant[0]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Pair Rolling 算子 {qlib_op} 的第一个参数 '{args[0]}' "
                    f"是常量表达式，qlib 不支持"
                )
            elif args_is_constant[1]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Pair Rolling 算子 {qlib_op} 的第二个参数 '{args[1]}' "
                    f"是常量表达式，qlib 不支持"
                )
            return f'{qlib_op}({args[0]}, {args[1]}, {args[2]})', False
        
        # 未知算子
        if self.strict_mode:
            raise ValueError(f"无法转换的算子: {op_name}")
        
        # 非严格模式下，保持原样，保守地认为结果不是常量
        return f'{op_name}({", ".join(args)})', False
    
    def _convert_operator(self, op_name: str, args: List[str]) -> str:
        """
        转换单个算子（不带常量性验证，保留用于兼容）
        
        Args:
            op_name: 算子名
            args: 已转换的参数列表
            
        Returns:
            转换后的表达式
        """
        # 截面算子 - 需要在数据处理层实现
        # 这里直接返回内部表达式，并标记需要截面处理
        if op_name in self.CROSS_SECTIONAL_OPS:
            self._has_cross_sectional_op = True
            # Rank(x) -> 直接返回 x，截面 Rank 会在数据处理层通过 CSRankNorm 实现
            # 注意：这意味着因子的语义是"对内部表达式先计算，然后在处理层做截面排名"
            return args[0]
        
        # 一元算子 - 直接对应
        if op_name in self.UNARY_OP_MAP:
            qlib_op = self.UNARY_OP_MAP[op_name]
            return f'{qlib_op}({args[0]})'
        
        # 一元算子 - 需要展开
        if op_name in self.UNARY_OP_EXPAND:
            expand_func = self.UNARY_OP_EXPAND[op_name]
            return expand_func(args[0])
        
        # 二元算子
        if op_name in self.BINARY_OP_MAP:
            qlib_op = self.BINARY_OP_MAP[op_name]
            return f'{qlib_op}({args[0]}, {args[1]})'
        
        # Rolling 算子 - 直接对应
        if op_name in self.ROLLING_OP_MAP:
            qlib_op = self.ROLLING_OP_MAP[op_name]
            return f'{qlib_op}({args[0]}, {args[1]})'
        
        # Rolling 算子 - 需要展开
        if op_name in self.ROLLING_OP_EXPAND:
            expand_func = self.ROLLING_OP_EXPAND[op_name]
            return expand_func(args[0], args[1])
        
        # Pair Rolling 算子
        if op_name in self.PAIR_ROLLING_OP_MAP:
            qlib_op = self.PAIR_ROLLING_OP_MAP[op_name]
            return f'{qlib_op}({args[0]}, {args[1]}, {args[2]})'
        
        # 未知算子
        if self.strict_mode:
            raise ValueError(f"无法转换的算子: {op_name}")
        
        # 非严格模式下，保持原样
        return f'{op_name}({", ".join(args)})'
    
    def convert_batch(
        self, 
        factors: List[Dict[str, Any]],
        keep_original: bool = True,
        filter_invalid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量转换因子表达式
        
        Args:
            factors: 因子列表，格式为 [{"expression": "...", "ic": ...}, ...]
            keep_original: 是否保留原始表达式
            filter_invalid: 是否过滤无效表达式（如对常量做 Rolling 操作的表达式）
            
        Returns:
            转换后的因子列表，包含 qlib_expression 和 needs_cs_rank 字段
        """
        result = []
        invalid_count = 0
        
        for factor in factors:
            if isinstance(factor, dict):
                expr = factor.get('expression', '')
                ic = factor.get('ic', 0.0)
                # 保留输入因子中的所有额外字段（除了 expression 和 ic，这些会被处理）
                extra_fields = {k: v for k, v in factor.items() 
                               if k not in ['expression', 'ic']}
            else:
                expr = str(factor)
                ic = 0.0
                extra_fields = {}
            
            try:
                qlib_expr = self.convert(expr)
                
                # 检查表达式是否有效
                if not self._is_valid_expression:
                    invalid_count += 1
                    if filter_invalid:
                        # 过滤无效表达式，不添加到结果中
                        continue
                    else:
                        # 不过滤，但标记为无效
                        converted = {
                            'qlib_expression': None,
                            'original_expression': expr,
                            'ic': ic,
                            'needs_cs_rank': False,
                            'is_valid': False,
                            'invalid_reason': self._invalid_reason
                        }
                        # 保留额外字段
                        converted.update(extra_fields)
                        result.append(converted)
                        continue
                
                converted = {
                    'qlib_expression': qlib_expr,
                    'ic': ic,
                    'needs_cs_rank': self._has_cross_sectional_op,
                    'is_valid': True
                }
                if keep_original:
                    converted['original_expression'] = expr
                # 保留额外字段（如 rank_ic_valid 等）
                converted.update(extra_fields)
                result.append(converted)
                
            except Exception as e:
                print(f"[Warning] 转换失败: {expr}, 错误: {e}")
                if not filter_invalid:
                    # 保留原始表达式
                    converted = {
                        'qlib_expression': None,
                        'original_expression': expr,
                        'ic': ic,
                        'needs_cs_rank': False,
                        'is_valid': False,
                        'error': str(e)
                    }
                    # 保留额外字段
                    converted.update(extra_fields)
                    result.append(converted)
        
        if invalid_count > 0:
            action = "已过滤" if filter_invalid else "已标记"
            print(f"[Converter] {action} {invalid_count} 个无效表达式（如对常量做 Rolling 操作）")
        
        return result
    
    def batch_needs_cs_rank(self, converted_factors: List[Dict[str, Any]]) -> bool:
        """
        检查批量转换的因子中是否有任何一个需要截面 Rank 处理
        
        Args:
            converted_factors: convert_batch 返回的因子列表
            
        Returns:
            是否有因子需要截面 Rank 处理
        """
        return any(f.get('needs_cs_rank', False) for f in converted_factors)
    
    def convert_and_save(
        self,
        input_file: str,
        output_file: str = None,
        format_type: str = 'json'
    ) -> str:
        """
        读取 AlphaSAGE 因子文件，转换后保存
        
        Args:
            input_file: 输入文件路径 (AlphaSAGE 因子 JSON)
            output_file: 输出文件路径，默认为 input_file 加 _qlib 后缀
            format_type: 输出格式 ('json' 或 'qlib_fields')
            
        Returns:
            输出文件路径
        """
        input_path = Path(input_file)
        
        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        factors = data.get('factors', [])
        
        # 批量转换
        converted_factors = self.convert_batch(factors)
        
        # 确定输出路径
        if output_file is None:
            output_path = input_path.parent / f"{input_path.stem}_qlib{input_path.suffix}"
        else:
            output_path = Path(output_file)
        
        # 保存结果
        if format_type == 'json':
            output_data = {
                'timestamp': data.get('timestamp', ''),
                'factors_count': len(converted_factors),
                'source': str(input_path),
                'factors': converted_factors
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'qlib_fields':
            # 输出为 qlib 可直接使用的 fields 格式
            fields = []
            names = []
            for i, factor in enumerate(converted_factors):
                if factor.get('qlib_expression'):
                    fields.append(factor['qlib_expression'])
                    ic = factor.get('ic', 0)
                    names.append(f'ALPHA_{i+1:03d}_IC{ic:.4f}')
            
            output_data = {
                'fields': fields,
                'names': names,
                'factors_detail': converted_factors
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Converter] 转换完成，保存至: {output_path}")
        print(f"[Converter] 成功转换 {sum(1 for f in converted_factors if f.get('qlib_expression'))} / {len(converted_factors)} 个因子")
        
        return str(output_path)


def convert_alphagen_to_qlib(expr: str) -> str:
    """
    便捷函数：将 AlphaSAGE 表达式转换为 Qlib 表达式
    
    Args:
        expr: AlphaSAGE 表达式
        
    Returns:
        Qlib 表达式
    """
    converter = AlphaSAGEToQlibConverter()
    return converter.convert(expr)


def convert_factor_file(input_file: str, output_file: str = None) -> str:
    """
    便捷函数：转换因子文件
    
    Args:
        input_file: 输入的 AlphaSAGE 因子 JSON 文件
        output_file: 输出文件路径
        
    Returns:
        输出文件路径
    """
    converter = AlphaSAGEToQlibConverter()
    return converter.convert_and_save(input_file, output_file)


# 测试代码
if __name__ == '__main__':
    converter = AlphaSAGEToQlibConverter()
    
    # 测试用例
    test_cases = [
        # 简单特征
        ("close", "$close"),
        ("volume", "$volume"),
        
        # 一元算子
        ("Abs(close)", "Abs($close)"),
        ("Log(close)", "Log($close)"),
        ("SLog1p(close)", "Mul(Sign($close), Log(Add(Abs($close), 1)))"),
        ("Inv(close)", "Div(1, $close)"),
        
        # 二元算子
        ("Add(close,volume)", "Add($close, $volume)"),
        ("Div(close,volume)", "Div($close, $volume)"),
        ("Pow(close,2)", "Power($close, 2)"),
        
        # Rolling 算子
        ("TsMean(close,20)", "Mean($close, 20)"),
        ("TsStd(volume,30)", "Std($volume, 30)"),
        ("Ref(close,5)", "Ref($close, 5)"),
        
        # 复杂 Rolling 算子
        ("TsIr(close,20)", "Div(Mean($close, 20), Std($close, 20))"),
        ("TsMinMaxDiff(high,10)", "Sub(Max($high, 10), Min($high, 10))"),
        
        # 嵌套表达式
        ("TsIr(TsMean(volume,30),50)", None),  # 复杂嵌套
        ("TsMin(TsIr(TsMean(volume,30),20),40)", None),  # 深度嵌套
        
        # Pair Rolling
        ("TsCorr(close,volume,20)", "Corr($close, $volume, 20)"),
    ]
    
    print("=" * 60)
    print("AlphaSAGE -> Qlib 表达式转换测试")
    print("=" * 60)
    
    for alphagen_expr, expected in test_cases:
        qlib_expr = converter.convert(alphagen_expr)
        status = "✓" if (expected is None or qlib_expr == expected) else "✗"
        print(f"\n{status} {alphagen_expr}")
        print(f"  -> {qlib_expr}")
        if expected and qlib_expr != expected:
            print(f"  期望: {expected}")
    
    # 测试无效表达式检测
    print("\n" + "=" * 60)
    print("无效表达式检测测试")
    print("=" * 60)
    
    invalid_test_cases = [
        # 对直接常量做 Rolling 操作（无效）
        "Sub(TsMed(Constant(-0.01),10), TsMean(volume,10))",  # Med(-0.01, 10)
        "Sub(TsMad(Constant(-5.0),40), volume)",  # Mad(-5.0, 40)
        "TsMean(Constant(1.0),20)",  # Mean(1.0, 20)
        "TsMax(Constant(0),10)",  # Max(0, 10)
        
        # 对嵌套常量表达式做 Rolling 操作（无效）- 新增测试
        "TsWMA(Div(1, 2.0), 50)",  # WMA(Div(1, 2.0), 50) - Div(1,2)=0.5 是常量
        "TsMean(Add(1, 2), 10)",  # Mean(Add(1, 2), 10) - Add(1,2)=3 是常量
        "TsStd(Mul(3, 4), 20)",  # Std(Mul(3, 4), 20) - Mul(3,4)=12 是常量
        "Mul(TsCorr(high, Sub(TsWMA(Div(1, 2.0), 50), volume), 40), TsCorr(high, volume, 20))",  # 复杂嵌套
        
        # 对 Pair Rolling 算子的常量参数测试（无效）
        "TsCorr(Div(1, 2), volume, 20)",  # 第一个参数是常量
        "TsCov(close, Add(1, 1), 30)",  # 第二个参数是常量
        
        # 有效的表达式
        "TsMean(close,20)",
        "Sub(TsMed(close,10), TsMean(volume,10))",
        "TsWMA(Div(close, volume), 50)",  # Div(close, volume) 不是常量
        "Mul(TsCorr(high, Sub(volume, close), 40), TsCorr(high, volume, 20))",  # 都是非常量
    ]
    
    for expr in invalid_test_cases:
        result = converter.convert_with_metadata(expr)
        status = "✓ 有效" if result['is_valid'] else "✗ 无效"
        print(f"\n{status}: {expr}")
        print(f"  -> {result['qlib_expression']}")
        if not result['is_valid']:
            print(f"  原因: {result.get('invalid_reason', '未知')}")
    
    print("\n" + "=" * 60)
    print("测试完成")

