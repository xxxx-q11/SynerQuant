"""
Factor pool registry - dynamically manage custom factor pools

This module allows dynamically registering mined factors as Handler classes similar to Alpha158,
which can be used directly in yaml configuration files.

Supports two factor sources:
1. Directly pass qlib format factor expressions
2. Import from AlphaSAGE factor files (automatic conversion)
"""
from pathlib import Path
from typing import List, Dict, Any, Union
import json
import sys
from datetime import datetime

# Note: Lazy import qlib here to avoid requiring full qlib environment when only doing factor conversion
_qlib_imported = False
DataHandlerLP = None
check_transform_proc = None

def _ensure_qlib_imported():
    """Ensure qlib is imported (lazy import)"""
    global _qlib_imported, DataHandlerLP, check_transform_proc
    if not _qlib_imported:
        from qlib.data.dataset.handler import DataHandlerLP as _DataHandlerLP
        from qlib.contrib.data.handler import check_transform_proc as _check_transform_proc
        DataHandlerLP = _DataHandlerLP
        check_transform_proc = _check_transform_proc
        _qlib_imported = True


# Default processor configuration
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class FactorPoolRegistry:
    """Factor pool registry"""
    
    def __init__(self, registry_dir: str = None):
        """
        Initialize factor pool registry
        
        Args:
            registry_dir: Factor pool registry directory
        """
        # Ensure qlib is imported
        _ensure_qlib_imported()
        
        if registry_dir is None:
            # Default: create factor_pools directory in current directory
            self.registry_dir = Path(__file__).parent / "factor_pools"
        else:
            self.registry_dir = Path(registry_dir)
        
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py to make it a Python package
        init_file = self.registry_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Auto-generated factor pool module"""\n', encoding='utf-8')
    
    def register_factor_pool(
        self, 
        pool_name: str, 
        factors: List[Dict[str, Any]],
        description: str = "",
        needs_cs_rank: bool = False
    ) -> str:
        """
        Register a factor pool
        
        Args:
            pool_name: Factor pool name (will be used as class name)
            factors: Factor list, format [{"expression": "factor expression", "ic": IC value}, ...]
            description: Factor pool description
            needs_cs_rank: Whether cross-sectional Rank processing is needed (automatically add CSRankNorm processor)
            
        Returns:
            Full path of generated Handler class (for yaml configuration)
            
        Example:
            >>> registry = FactorPoolRegistry()
            >>> factors = [{"expression": "$close/$open", "ic": 0.05}, ...]
            >>> class_path = registry.register_factor_pool("MyFactors", factors)
            >>> # Use in yaml: 
            >>> # handler:
            >>> #   class: MyFactors
            >>> #   module_path: qlib_benchmark.factor_pools.my_factors
        """
        # Generate class name and module name
        class_name = pool_name
        module_name = self._to_snake_case(pool_name)
        module_file = self.registry_dir / f"{module_name}.py"
        
        # Generate Handler class code
        class_code = self._generate_handler_class(
            class_name=class_name,
            factors=factors,
            description=description,
            needs_cs_rank=needs_cs_rank
        )
        
        # Save to file
        module_file.write_text(class_code, encoding='utf-8')
        
        # Save factor metadata
        metadata = {
            "pool_name": pool_name,
            "class_name": class_name,
            "module_name": module_name,
            "factors_count": len(factors),
            "description": description,
            "needs_cs_rank": needs_cs_rank,
            "created_at": datetime.now().isoformat(),
            "factors": factors
        }
        
        metadata_file = self.registry_dir / f"{module_name}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Return full module path
        module_path = f"qlib_benchmark.factor_pools.{module_name}"
        
        print(f"[FactorPoolRegistry] Factor pool registered: {class_name}")
        print(f"[FactorPoolRegistry] Module path: {module_path}")
        print(f"[FactorPoolRegistry] Factor count: {len(factors)}")
        if needs_cs_rank:
            print(f"[FactorPoolRegistry] CSRankNorm processor added (cross-sectional Rank support)")
        
        return module_path
    
    def _generate_handler_class(
        self, 
        class_name: str, 
        factors: List[Dict[str, Any]],
        description: str,
        needs_cs_rank: bool = False
    ) -> str:
        """
        Generate Handler class Python code
        
        Args:
            class_name: Class name
            factors: Factor list
            description: Description
            needs_cs_rank: Whether cross-sectional Rank processing is needed
            
        Returns:
            Python code string
        """
        # Parse factor expressions
        fields = []
        names = []
        
        for i, factor in enumerate(factors):
            if isinstance(factor, dict):
                if "expression" in factor:
                    expression = factor["expression"]
                    ic = factor.get("ic", 0.0)
                elif "qlib_expression" in factor:
                    # Support qlib_expression key
                    expression = factor["qlib_expression"]
                    ic = factor.get("ic", 0.0)
                else:
                    # Compatible format: {"factor expression": ic value}
                    expression = list(factor.keys())[0]
                    ic = factor[expression]
                
                fields.append(expression)
                
                #fields.append(expression)
                # Generate factor name (including IC information)
                if isinstance(ic, (int, float)):
                    names.append(f"FACTOR_{i+1:03d}_IC{ic:.4f}")
                else:
                    names.append(f"FACTOR_{i+1:03d}")
        
        # Generate different processor configurations based on whether cross-sectional Rank processing is needed
        if needs_cs_rank:
            # Add CSRankNorm processor to implement cross-sectional ranking
            infer_processors_code = '''_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "CSRankNorm", "kwargs": {}},  # Cross-sectional Rank normalization (for factors containing Rank operators)
    {"class": "Fillna", "kwargs": {}},
]'''
            cs_rank_note = "\nNote: This factor pool contains cross-sectional Rank operators, CSRankNorm processor has been automatically added."
        else:
            infer_processors_code = '''_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]'''
            cs_rank_note = ""
        
        # Generate code
        code = f'''"""
{class_name} - Auto-generated factor pool
{description}{cs_rank_note}

Factor count: {len(factors)}
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This class is auto-generated by FactorPoolRegistry,
can be used in yaml configuration files like Alpha158.
"""

from qlib.data.dataset.handler import DataHandlerLP

# 默认处理器配置
_DEFAULT_LEARN_PROCESSORS = [
    {{"class": "DropnaLabel"}},
    {{"class": "CSZScoreNorm", "kwargs": {{"fields_group": "label"}}}},
]

{infer_processors_code}


class {class_name}(DataHandlerLP):
    """
    {class_name} - Custom factor pool
    
    {description}
    
    Usage (in yaml):
        handler:
            class: {class_name}
            module_path: qlib_benchmark.factor_pools.{self._to_snake_case(class_name)}
            kwargs:
                start_time: 2008-01-01
                end_time: 2020-08-01
                fit_start_time: 2008-01-01
                fit_end_time: 2014-12-31
                instruments: csi300
    """
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        # Process processor configuration
        from qlib.contrib.data.handler import check_transform_proc
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Configure data loader
        data_loader = {{
            "class": "QlibDataLoader",
            "kwargs": {{
                "config": {{
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                }},
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            }},
        }}
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )
    
    def get_feature_config(self):
        """
        Get factor configuration
        
        Contains {len(factors)} custom factors
        """
        fields = {fields!r}
        
        names = {names!r}
        
        return fields, names
    
    def get_label_config(self):
        """Define label (prediction target)"""
        # Default label: future 2-day return rate
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
'''
        
        return code
    
    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        import re
        # Insert underscore before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore between lowercase and uppercase letters
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def list_factor_pools(self) -> List[Dict[str, Any]]:
        """
        List all registered factor pools
        
        Returns:
            Factor pool list, each containing name, description, factor count, etc.
        """
        pools = []
        
        for metadata_file in self.registry_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    pools.append({
                        "name": metadata["pool_name"],
                        "class_name": metadata["class_name"],
                        "module_path": f"qlib_benchmark.factor_pools.{metadata['module_name']}",
                        "factors_count": metadata["factors_count"],
                        "description": metadata.get("description", ""),
                        "created_at": metadata.get("created_at", ""),
                    })
            except Exception as e:
                print(f"读取因子池元数据失败 {metadata_file}: {e}")
        
        return pools
    
    def get_factor_pool_info(self, pool_name: str) -> Dict[str, Any]:
        """
        获取因子池详细信息
        
        Args:
            pool_name: 因子池名称
            
        Returns:
            因子池详细信息（包含所有因子）
        """
        module_name = self._to_snake_case(pool_name)
        metadata_file = self.registry_dir / f"{module_name}.json"
        
        if not metadata_file.exists():
            raise ValueError(f"因子池不存在: {pool_name}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)


# 全局注册器实例
_global_registry = None

def get_registry() -> FactorPoolRegistry:
    """获取全局注册器实例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactorPoolRegistry()
    return _global_registry


def register_factor_pool(pool_name: str, factors: List[Dict[str, Any]], description: str = "") -> str:
    """
    注册因子池的便捷函数
    
    Args:
        pool_name: 因子池名称
        factors: 因子列表
        description: 描述
        
    Returns:
        模块路径
    """
    registry = get_registry()
    return registry.register_factor_pool(pool_name, factors, description)


def register_from_alphagen_file(
    factor_file: str,
    pool_name: str = None,
    description: str = "",
    top_n: int = None,
    min_ic: float = None
) -> str:
    """
    从 AlphaSAGE 因子文件导入并注册因子池
    
    支持两种文件格式:
    1. AlphaSAGE 原始格式 (包含 'factors' 字段，每个因子有 'expression' 和 'ic')
    2. 已转换的 qlib 格式 (包含 'qlib_fields' 字段)
    
    Args:
        factor_file: AlphaSAGE 因子文件路径（JSON格式）
        pool_name: 因子池名称，默认从文件名生成
        description: 因子池描述
        top_n: 只取 IC 最高的前 N 个因子
        min_ic: 最小 IC 阈值
        
    Returns:
        模块路径
        
    Example:
        >>> module_path = register_from_alphagen_file(
        ...     "out_gp/csi300_2020/4_qlib_factors.json",
        ...     pool_name="GPFactors_CSI300",
        ...     top_n=50
        ... )
        >>> # 在 yaml 中使用:
        >>> # handler:
        >>> #   class: GPFactors_CSI300
        >>> #   module_path: qlib_benchmark.factor_pools.gp_factors_csi300
    """
    factor_path = Path(factor_file)
    
    if not factor_path.exists():
        raise FileNotFoundError(f"因子文件不存在: {factor_file}")
    
    # 读取因子文件
    with open(factor_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检测文件格式并转换
    factors_for_registry = []
    
    if 'qlib_fields' in data and data['qlib_fields']:
        # 已经是 qlib 格式
        qlib_fields = data['qlib_fields']
        qlib_names = data.get('qlib_names', [f'ALPHA_{i+1:03d}' for i in range(len(qlib_fields))])
        factors_detail = data.get('factors', [])
        
        for i, field in enumerate(qlib_fields):
            ic = 0.0
            if i < len(factors_detail):
                ic = factors_detail[i].get('ic', 0.0)
            factors_for_registry.append({
                'expression': field,
                'ic': ic
            })
    
    elif 'factors' in data:
        # AlphaSAGE 原始格式，需要转换
        # 动态导入转换器
        try:
            from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
            converter = AlphaSAGEToQlibConverter()
        except ImportError:
            # 如果无法导入，尝试添加路径
            alphagen_path = Path(__file__).parent.parent / "AlphaSAGE" / "src"
            if alphagen_path.exists():
                sys.path.insert(0, str(alphagen_path))
                from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
                converter = AlphaSAGEToQlibConverter()
            else:
                raise ImportError(
                    "无法导入 AlphaSAGEToQlibConverter。"
                    "请确保 AlphaSAGE 项目已正确安装，或使用已转换的 qlib 格式文件。"
                )
        
        raw_factors = data['factors']
        # 转换并过滤无效表达式（如对常量做 Rolling 操作的表达式）
        converted = converter.convert_batch(raw_factors, filter_invalid=True)
        
        for factor in converted:
            # 只添加有效的表达式（filter_invalid=True 已经过滤了大部分，这里双重检查）
            if factor.get('qlib_expression') and factor.get('is_valid', True):
                factors_for_registry.append({
                    'expression': factor['qlib_expression'],
                    'ic': factor.get('ic', 0.0),
                    'needs_cs_rank': factor.get('needs_cs_rank', False)
                })
    else:
        raise ValueError(f"无法识别的因子文件格式: {factor_file}")
    
    # 按 IC 排序
    factors_for_registry.sort(key=lambda x: x.get('ic', 0), reverse=True)
    
    # 应用过滤条件
    if min_ic is not None:
        factors_for_registry = [f for f in factors_for_registry if f.get('ic', 0) >= min_ic]
    
    if top_n is not None:
        factors_for_registry = factors_for_registry[:top_n]
    
    # 生成因子池名称
    if pool_name is None:
        # 从文件名生成
        pool_name = factor_path.stem.replace('_qlib_factors', '').replace('-', '_')
        pool_name = ''.join(word.capitalize() for word in pool_name.split('_'))
        pool_name = f"GP{pool_name}"
    
    # 检查是否有因子需要截面 Rank 处理
    needs_cs_rank = any(f.get('needs_cs_rank', False) for f in factors_for_registry)
    
    # 生成描述
    if not description:
        description = f"从 AlphaSAGE 导入的因子池\n来源: {factor_file}\n因子数量: {len(factors_for_registry)}"
    
    # 注册因子池
    registry = get_registry()
    return registry.register_factor_pool(pool_name, factors_for_registry, description, needs_cs_rank=needs_cs_rank)

