"""因子挖掘 Agent"""
import sys
import os
import json
import copy
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime
import random
#from utils.file_process import explore_repo_structure,find_training_scripts,select_training_script,read_file_for_llm,find_readme_files,get_top_factors_from_gp_json
#from Agent.prompts import FACTOR_MINING_SYSTEM_PROMPT, FACTOR_MINING_ANALYSIS_PROMPT

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("警告: MCP 客户端不可用")


# 导入因子池注册器
try:
    # 尝试从相对路径导入
    qlib_benchmark_path = Path(__file__).parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
    if str(qlib_benchmark_path) not in sys.path:
        sys.path.insert(0, str(qlib_benchmark_path))
    from factor_pool_registry import FactorPoolRegistry
    FACTOR_REGISTRY_AVAILABLE = True
except ImportError:
    FACTOR_REGISTRY_AVAILABLE = False
    print("警告: 因子池注册器不可用")


class FactorEvalAgent:
    """因子评估 Agent"""
    
    def __init__(self, llm_service, mcp_server_path: Optional[str] = None):
        """
        初始化因子评估 Agent
        
        Args:
            llm_service: LLM 服务实例 (BaseAgent)
            mcp_server_path: MCP 服务器脚本路径，默认为 Qlib_MCP/mcp_server_inline.py
        """
        self.llm = llm_service
        # from utils.mcp_client import SyncMCPClient
        # MCP_AVAILABLE = True
        # 初始化 MCP 客户端
        if MCP_AVAILABLE:
            if mcp_server_path is None:
                # 默认路径：相对于当前文件找到 Qlib_MCP/mcp_server_inline.py
                current_dir = Path(__file__).parent.parent
                mcp_server_path = current_dir / "Qlib_MCP" / "mcp_server_inline.py"
                if not mcp_server_path.exists():
                    print(f"警告: MCP 服务器脚本不存在: {mcp_server_path}")
                    self.mcp_client = None
                else:
                    try:
                        self.mcp_client = SyncMCPClient(str(mcp_server_path))
                        print(f"MCP 客户端初始化成功: {mcp_server_path}")
                    except Exception as e:
                        print(f"警告: MCP 客户端初始化失败: {e}")
                        self.mcp_client = None
            else:
                try:
                    self.mcp_client = SyncMCPClient(mcp_server_path)
                    print(f"MCP 客户端初始化成功: {mcp_server_path}")
                except Exception as e:
                    print(f"警告: MCP 客户端初始化失败: {e}")
                    self.mcp_client = None
        else:
            self.mcp_client = None

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        获取可用的 MCP 工具列表
        
        Returns:
            工具列表，每个工具包含 name, description, inputSchema
        """
        if not self.mcp_client:
            return []
        
        try:
            tools = self.mcp_client.list_tools()
            return tools
        except Exception as e:
            print(f"获取工具列表失败: {e}")
            return []
    
    def call_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数字典
            
        Returns:
            工具执行结果（文本格式）
        """
        if not self.mcp_client:
            raise RuntimeError("MCP 客户端未初始化")
        
        try:
            result = self.mcp_client.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            error_msg = f"调用工具 {tool_name} 失败: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def filter_and_sort_factors(self, factors: List[Dict[str, Any]], ic_threshold: float = 0.03) -> List[Dict[str, Any]]:
        """
        筛选和排序因子
        
        Args:
            factors: 因子列表，格式为 [{"expression": "因子表达式", "ic": IC值, "rank_ic_valid": Rank_IC值}, ...]
            ic_threshold: IC阈值，默认0.03，只保留IC大于此值的因子
            
        Returns:
            筛选和排序后的因子列表，按照rank_ic_valid降序排列
        """
        if not factors:
            return []
        
        filtered_factors = []
        
        for factor in factors:
            if not isinstance(factor, dict):
                continue
            
            # 获取IC值，处理可能为None或字符串的情况
            ic = factor.get("ic")
            if ic is None:
                continue
            
            # 尝试转换为浮点数
            try:
                if isinstance(ic, str):
                    # 如果是字符串，尝试转换
                    if ic == "未知" or ic.strip() == "":
                        continue
                    ic_value = float(ic)
                else:
                    ic_value = float(ic)
            except (ValueError, TypeError):
                continue
            
            # 筛选IC大于阈值的因子
            if ic_value > ic_threshold:
                filtered_factors.append(factor)
        
        # 按照rank_ic_valid降序排列
        def get_rank_ic_valid(factor: Dict[str, Any]) -> float:
            """获取rank_ic_valid值，用于排序"""
            rank_ic_valid = factor.get("rank_ic_valid")
            if rank_ic_valid is None:
                return float('-inf')  # 没有rank_ic_valid的因子排在最后
            
            try:
                if isinstance(rank_ic_valid, str):
                    if rank_ic_valid == "未知" or rank_ic_valid.strip() == "":
                        return float('-inf')
                    return float(rank_ic_valid)
                else:
                    return float(rank_ic_valid)
            except (ValueError, TypeError):
                return float('-inf')
        # 剔除rank_ic_valid小于0的因子
        def is_valid_rank_ic(factor: Dict[str, Any]) -> bool:
            """检查rank_ic_valid是否有效（>= 0）"""
            rank_ic_valid = factor.get("rank_ic_valid")
            if rank_ic_valid is None:
                return False  # 没有rank_ic_valid的因子被剔除
            
            try:
                if isinstance(rank_ic_valid, str):
                    if rank_ic_valid == "未知" or rank_ic_valid.strip() == "":
                        return False
                    rank_ic_value = float(rank_ic_valid)
                else:
                    rank_ic_value = float(rank_ic_valid)
                
                # 只保留rank_ic_valid >= 0的因子
                return rank_ic_value >= 0
            except (ValueError, TypeError):
                return False
        # 先过滤掉rank_ic_valid小于0的因子
        #print(f"filtered_factors: {filtered_factors}")
        filtered_factors = [f for f in filtered_factors if is_valid_rank_ic(f)]
        #print(f"filtered_factors: {filtered_factors}")
        # 按rank_ic_valid降序排序
        sorted_factors = sorted(filtered_factors, key=get_rank_ic_valid, reverse=True)
        
        return sorted_factors

    def filter_high_correlation_with_sota(
        self,
        factor: Dict[str, Any],
        sota_pool_list: List[str] = None,
        instruments: str = "csi300",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        correlation_threshold: float = 0.99
    ) -> bool:
        """
        检查新因子是否与 SOTA 因子高相关
        
        计算新因子与 SOTA 因子的 Pearson 相关系数（按日期分组，取平均），
        如果最大相关性 >= correlation_threshold，返回 False；否则返回 True。
        
        Args:
            factor: 新因子，格式为 {"qlib_expression": "表达式", "ic": IC值, "needs_cs_rank": bool}
            sota_pool_list: SOTA 因子池列表
            instruments: 股票池名称
            start_date: 训练集开始日期
            end_date: 训练集结束日期
            correlation_threshold: 相关性阈值（>=此值返回 False），默认0.99
            
        Returns:
            True: 因子与 SOTA 因子相关性不高，可以保留
            False: 因子与 SOTA 因子高相关，应该移除
        """
        if not factor:
            return True
        
        # 确定 SOTA 因子池路径
        if sota_pool_list is None:
            print("警告: SOTA 因子池为空，跳过相关性检查")
            return True
        
        try:
            # 加载 SOTA 因子
            sota_factors = sota_pool_list
            print(f"\n[检查因子相关性] 开始检查因子，SOTA 因子数量: {len(sota_factors)}")
            # 初始化 Qlib（如果需要）
            try:
                # 添加模块路径
                modules_path = Path(__file__).parent.parent / "Qlib_MCP" / "modules"
                if str(modules_path) not in sys.path:
                    sys.path.insert(0, str(modules_path))
                
                # 动态导入
                from calculate_ic import init_qlib, preprocess_formula  # type: ignore
                success, _ = init_qlib()
                if not success:
                    print("警告: Qlib 初始化失败，跳过相关性检查")
                    return True
            except ImportError as e:
                print(f"警告: 无法导入 calculate_ic 模块: {e}，跳过相关性检查")
                return True
            except Exception as e:
                print(f"警告: 无法初始化 Qlib: {e}，跳过相关性检查")
                return True
            
            try:
                from qlib.data import D
                from qlib.data.dataset.processor import CSRankNorm
            except ImportError as e:
                print(f"警告: 无法导入 qlib 模块: {e}，跳过相关性检查")
                return True
            
            try:
                # 获取因子表达式
                expression = factor.get('qlib_expression')
                if not expression:
                    print("警告: 因子表达式为空，跳过相关性检查")
                    return True
                
                needs_cs_rank = factor.get('needs_cs_rank', False)
                
                # 计算新因子值
                formula = preprocess_formula(expression)
                instrument_list = D.list_instruments(
                    instruments=D.instruments(instruments),
                    start_time=start_date,
                    end_time=end_date,
                    as_list=True
                )
                
                if len(instrument_list) == 0:
                    print("警告: 无法获取股票列表，跳过相关性检查")
                    return True
                    
                new_factor_df = D.features(
                    instrument_list,
                    [formula],
                    start_time=start_date,
                    end_time=end_date
                )
                
                if isinstance(new_factor_df, pd.DataFrame):
                    new_factor_values = new_factor_df.iloc[:, 0]
                else:
                    new_factor_values = new_factor_df
                
                if isinstance(new_factor_values.index, pd.MultiIndex):
                    new_factor_values.index.names = ['datetime', 'instrument']
                
                # 如果需要截面 Rank 处理
                if needs_cs_rank:
                    processor = CSRankNorm()
                    factor_df = new_factor_values.to_frame('factor')
                    factor_df = processor(factor_df)
                    new_factor_values = factor_df['factor']
                    
                # 使用矩阵并行计算与所有 SOTA 因子的相关性
                # 一次性计算所有 SOTA 因子值
                sota_formulas = [preprocess_formula(sota_expr) for sota_expr in sota_factors]
                
                try:
                    # 批量计算所有 SOTA 因子值
                    sota_factors_df = D.features(
                        instrument_list,
                        sota_formulas,
                        start_time=start_date,
                        end_time=end_date
                    )
                    
                    # 确保索引名称一致
                    if isinstance(sota_factors_df.index, pd.MultiIndex):
                        sota_factors_df.index.names = ['datetime', 'instrument']
                    
                    # 对齐新因子和所有 SOTA 因子
                    # 将新因子转换为 DataFrame（单列）
                    new_factor_df = new_factor_values.to_frame('new_factor')
                    
                    # 合并所有因子
                    all_factors_df = pd.concat([new_factor_df, sota_factors_df], axis=1).dropna()
                    
                    if len(all_factors_df) < 10:
                        max_corr = -np.inf
                    else:
                        # 按日期分组，使用矩阵运算批量计算相关性
                        # 存储每个日期每个 SOTA 因子的相关性
                        # 形状: (n_dates, n_sota_factors)
                        date_correlations = []
                        
                        for date, group in all_factors_df.groupby(level=0):
                            if len(group) < 3:
                                continue
                            
                            # 提取新因子列和所有 SOTA 因子列
                            new_factor_col = group['new_factor'].values.astype(float)
                            sota_factor_cols = group.drop(columns=['new_factor']).values.astype(float)
                            
                            if sota_factor_cols.shape[1] == 0:
                                continue
                            
                            # 使用矩阵运算计算 Pearson 相关系数
                            # 标准化：减去均值，除以标准差
                            new_factor_mean = np.mean(new_factor_col)
                            new_factor_std = np.std(new_factor_col)
                            if new_factor_std < 1e-10:
                                continue
                            
                            new_factor_norm = (new_factor_col - new_factor_mean) / new_factor_std
                            
                            # 对每个 SOTA 因子标准化并计算相关性
                            sota_means = np.mean(sota_factor_cols, axis=0)
                            sota_stds = np.std(sota_factor_cols, axis=0)
                            
                            # 过滤掉标准差为0的列（常数因子）
                            valid_mask = sota_stds > 1e-10
                            if not np.any(valid_mask):
                                continue
                            
                            # 只处理有效的 SOTA 因子
                            sota_factor_cols_valid = sota_factor_cols[:, valid_mask]
                            sota_means_valid = sota_means[valid_mask]
                            sota_stds_valid = sota_stds[valid_mask]
                            
                            # 标准化 SOTA 因子（广播）
                            sota_factors_norm = (sota_factor_cols_valid - sota_means_valid) / sota_stds_valid
                            
                            # 计算 Pearson 相关系数：E[(X - E[X])(Y - E[Y])] / (std(X) * std(Y))
                            # 由于已经标准化，相关系数 = E[X * Y] = mean(X * Y)
                            correlations = np.mean(new_factor_norm[:, np.newaxis] * sota_factors_norm, axis=0)
                            
                            # 创建完整的相关性数组（包括无效的 SOTA 因子，设为 NaN）
                            full_correlations = np.full(len(sota_factors), np.nan)
                            full_correlations[valid_mask] = correlations
                            
                            date_correlations.append(full_correlations)
                        
                        if len(date_correlations) > 0:
                            # 转换为矩阵：每行是一个日期，每列是一个 SOTA 因子
                            correlation_matrix = np.array(date_correlations)  # shape: (n_dates, n_sota_factors)
                            
                            # 计算每个 SOTA 因子的平均相关性（跨日期）
                            avg_correlations = np.nanmean(correlation_matrix, axis=0)  # shape: (n_sota_factors,)
                            
                            # 取最大相关性
                            max_corr = np.nanmax(avg_correlations) if len(avg_correlations) > 0 else -np.inf
                        else:
                            max_corr = -np.inf
                            
                except Exception as e:
                    # 如果批量计算失败，回退到逐个计算
                    print(f"警告: 矩阵并行计算失败 ({str(e)[:50]})，回退到逐个计算")
                    max_corr = -np.inf
                    
                    for sota_expr in sota_factors:
                        try:
                            sota_formula = preprocess_formula(sota_expr)
                            sota_factor_df = D.features(
                                instrument_list,
                                [sota_formula],
                                start_time=start_date,
                                end_time=end_date
                            )
                            
                            if isinstance(sota_factor_df, pd.DataFrame):
                                sota_factor_values = sota_factor_df.iloc[:, 0]
                            else:
                                sota_factor_values = sota_factor_df
                            
                            if isinstance(sota_factor_values.index, pd.MultiIndex):
                                sota_factor_values.index.names = ['datetime', 'instrument']
                            
                            aligned = pd.concat(
                                [new_factor_values, sota_factor_values],
                                axis=1,
                                keys=['factor1', 'factor2']
                            ).dropna()
                            
                            if len(aligned) < 10:
                                continue
                            
                            correlations = []
                            for date, group in aligned.groupby(level=0):
                                if len(group) < 3:
                                    continue
                                try:
                                    corr, _ = pearsonr(group['factor1'], group['factor2'])
                                    if not np.isnan(corr):
                                        correlations.append(corr)
                                except Exception:
                                    continue
                            
                            if len(correlations) > 0:
                                avg_corr = np.mean(correlations)
                                if avg_corr > max_corr:
                                    max_corr = avg_corr
                        except Exception:
                            continue
                
                # 判断是否需要移除
                print(f"最大相关性: {max_corr:.4f}, 阈值: {correlation_threshold}")
                if max_corr >= correlation_threshold:
                    print(f"因子与 SOTA 因子高相关 (最大相关性: {max_corr:.4f} >= {correlation_threshold})，返回 False")
                    return False
                else:
                    print(f"因子与 SOTA 因子相关性不高 (最大相关性: {max_corr:.4f} < {correlation_threshold})，返回 True")
                    return True
                    
            except Exception as e:
                # 计算失败的因子也保留（可能是数据问题）
                print(f"警告: 计算相关性失败 ({str(e)[:50]})，返回 True")
                return True
            
        except Exception as e:
            print(f"警告: 检查因子相关性时出错: {e}，返回 True")
            return True

    def _register_merged_factor_pool(
        self,
        new_factor: Dict[str, Any],
        sota_pool_list: Optional[str] = None
    ) -> Optional[str]:
        """
        将新因子和 SOTA 因子池合并注册成新的因子池
        
        Args:
            new_factor: 新因子，格式为 {"qlib_expression": "表达式", "ic": IC值, "needs_cs_rank": bool}
            sota_pool_path: SOTA 因子池文件路径，默认为 data/sota_pool.json
            
        Returns:
            注册后的模块路径，如果失败返回 None
        """
        if not FACTOR_REGISTRY_AVAILABLE:
            print("警告: 因子池注册器不可用，无法注册因子池")
            return None
        
        try:
            sota_factors = sota_pool_list
            print(f"sota_factors: {sota_factors}")
            # 准备因子列表
            factors_for_registry = []
            
            # 添加新因子（放在第一位）
            if new_factor:
                new_expression = new_factor.get('qlib_expression')
                if new_expression:
                    factors_for_registry.append({
                        'expression': new_expression,
                        'ic': new_factor.get('ic', 0.0)
                    })
            
            # 添加 SOTA 因子（sota_pool 是字符串列表）
            if isinstance(sota_factors, list):
                for sota_expr in sota_factors:
                    if isinstance(sota_expr, str) and sota_expr.strip():
                        factors_for_registry.append({
                            'expression': sota_expr,
                            'ic': 0.0  # SOTA 因子没有 IC 值，设为 0.0
                        })
            
            if not factors_for_registry:
                print("计算原始因子池的ic和rank_ic")
            
            # 检查是否需要截面 Rank 处理
            #needs_cs_rank = new_factor.get('needs_cs_rank', False)
            
            # 生成因子池名称（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pool_name = f"CustomFactors_{timestamp}"
            if new_factor:
            # 生成描述
                description = f"合并因子池\n新因子: {new_expression}\nSOTA 因子数量: {len(sota_factors)}\n总因子数量: {len(factors_for_registry)}"
            else:
                description = f"原始因子池\nSOTA 因子数量: {len(sota_factors)}\n总因子数量: {len(factors_for_registry)}"
            
            # 创建注册器实例
            qlib_benchmark_path = Path(__file__).parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
            registry = FactorPoolRegistry(registry_dir=str(qlib_benchmark_path / "factor_pools"))
            
            # 注册因子池
            module_path = registry.register_factor_pool(
                pool_name=pool_name,
                factors=factors_for_registry,
                description=description,
                #needs_cs_rank=needs_cs_rank
            )
            
            print(f"[因子池注册] 成功注册因子池: {pool_name}")
            print(f"[因子池注册] 模块路径: {module_path}")
            print(f"[因子池注册] 因子数量: {len(factors_for_registry)}")
            
            # 自动生成 workflow 配置文件
            try:
                workflow_config_path = self.generate_workflow_config(
                    module_path=module_path,
                    model_type="xgboost"  # 默认使用 xgboost，可以通过参数调整
                )
                print(f"[因子池注册] 已自动生成 workflow 配置文件: {workflow_config_path}")
            except Exception as e:
                print(f"警告: 生成 workflow 配置文件失败: {e}")
                import traceback
                traceback.print_exc()
            
            return module_path,workflow_config_path
            
        except Exception as e:
            print(f"警告: 注册因子池时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_workflow_config(
        self,
        module_path: str,
        output_path: Optional[str] = None,
        base_config_path: Optional[str] = None,
        model_type: str = "xgboost",
        segments: Optional[Dict[str, List[str]]] = None,
        train_end_year: Optional[int] = 2020,
        train_start_year: int = 2010,
        instruments: str = "csi300"
    ) -> str:
        """
        根据注册的因子池路径生成 workflow 配置文件
        
        Args:
            module_path: 因子池的模块路径（如 "qlib_benchmark.factor_pools.custom_factors_20251215_210903"）
            output_path: 输出配置文件路径，如果为 None 则自动生成
            base_config_path: 基础配置文件路径，默认为 workflow_base_xgboost_model.yaml
            model_type: 模型类型，可选 "xgboost" 或 "xgboost"，默认 "xgboost"
            segments: 时间段字典，格式 {"train": [start, end], "valid": [start, end], "test": [start, end]}
                     如果提供此参数，会覆盖 train_end_year 自动计算。如果为 None 且 train_end_year 也为 None，则使用基础配置中的值
            train_end_year: 训练集结束年份（参考 GP 的时间设置方式），如果提供此参数，会自动计算时间段：
                           - 训练集: {train_start_year}-01-01 至 {train_end_year}-12-31
                           - 验证集: {train_end_year + 1}-01-01 至 {train_end_year + 1}-12-31
                           - 测试集: {train_end_year + 2}-01-01 至 {train_end_year + 4}-12-31
                           优先级低于 segments，如果 segments 和 train_end_year 都为 None，则使用基础配置中的值
            train_start_year: 训练集开始年份，默认 2010（参考 GP 的设置）
            
        Returns:
            生成的配置文件路径
        """
        import yaml
        
        # 确定基础配置文件路径
        if base_config_path is None:
            qlib_benchmark_path = Path(__file__).parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
            base_config_path = qlib_benchmark_path / "benchmarks" / "train_temp" / "workflow_base_lightgbm_model.yaml"
        
        if not Path(base_config_path).exists():
            raise FileNotFoundError(f"基础配置文件不存在: {base_config_path}")
        
        # 从 module_path 提取类名
        # module_path 格式: qlib_benchmark.factor_pools.custom_factors_20251215_210903
        # 需要提取类名: CustomFactors_20251215_210903
        module_parts = module_path.split('.')
        if len(module_parts) < 3 or module_parts[-2] != 'factor_pools':
            raise ValueError(f"无效的模块路径格式: {module_path}")
        
        module_name = module_parts[-1]  # custom_factors_20251215_210903
        
        # 尝试从 metadata 文件读取类名和因子数量
        qlib_benchmark_path = Path(__file__).parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
        metadata_file = qlib_benchmark_path / "factor_pools" / f"{module_name}.json"
        
        factors_count = 20  # 默认因子数量
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    class_name = metadata.get('class_name', None)
                    factors_count = metadata.get('factors_count', 20)
            except Exception as e:
                print(f"警告: 读取元数据文件失败: {e}")
                class_name = None
        else:
            class_name = None
        
        # 如果无法从元数据获取，则从模块名推导（将下划线命名转为驼峰命名）
        if class_name is None:
            # 将 snake_case 转为 PascalCase
            parts = module_name.split('_')
            class_name = ''.join(word.capitalize() for word in parts)
        
        # 读取基础配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取 data_handler_config（YAML 锚点会被 yaml.safe_load 自动展开）
        data_handler_config = {}
        # 首先尝试从顶层获取
        if 'data_handler_config' in config:
            data_handler_config = config['data_handler_config']
        # 如果不存在，尝试从当前 handler 配置中获取（锚点引用会被展开）
        else:
            handler_config = config.get('task', {}).get('dataset', {}).get('kwargs', {}).get('handler', {})
            if isinstance(handler_config, dict) and 'kwargs' in handler_config:
                data_handler_config = handler_config['kwargs']
        data_handler_config.update({
            'start_time': f'{train_start_year}-01-01',
            'end_time': f'{train_end_year + 4}-12-31',
            'fit_start_time': f'{train_start_year}-01-01',
            'fit_end_time': f'{train_end_year}-12-31',
            'instruments': instruments
        })
        
        # 修改 handler 配置
        if 'task' in config and 'dataset' in config['task'] and 'kwargs' in config['task']['dataset']:
            config['task']['dataset']['kwargs']['handler'] = {
                'class': class_name,
                'module_path': module_path,
                'kwargs': data_handler_config
            }
            
            # 设置 segments（训练/验证/测试时间段）
            # 优先级：segments > train_end_year自动计算 > 基础配置
            if segments is not None:
                # 使用提供的 segments 字典（最高优先级）
                config['task']['dataset']['kwargs']['segments'] = segments
            elif train_end_year is not None:
                # 使用 train_end_year 自动计算时间段（参考 GP 的时间设置方式）
                segments_config = {
                    'train': [f'{train_start_year}-01-01', f'{train_end_year}-12-31'],
                    'valid': [f'{train_end_year + 1}-01-01', f'{train_end_year + 1}-12-31'],
                    'test': [f'{train_end_year + 2}-01-01', f'{train_end_year + 4}-12-31']
                }   
                config['port_analysis_config']['backtest']['start_time'] = segments_config['test'][0]
                config['port_analysis_config']['backtest']['end_time'] = segments_config['test'][1]
                print(f"[生成配置文件] 已同步 backtest 时间到测试集时间段: {segments_config['test'][0]} 至 {segments_config['test'][1]}")
                config['task']['dataset']['kwargs']['segments'] = segments_config
            # 如果 segments 和 train_end_year 都为 None，则使用基础配置中的值（不需要额外处理）
        
        # 根据因子数量和 model_type 调整模型配置
        if 'task' in config and 'model' in config['task']:
            # 根据因子数量确定参数配置
            if factors_count < 20:
                # 因子少（<20）：降低正则化，增加模型容量
                lgbm_params = {
                    'loss': 'mse',
                    'colsample_bytree': 0.95,
                    'learning_rate': 0.2,
                    'subsample': 0.8789,
                    'lambda_l1': 50.0,
                    'lambda_l2': 150.0,
                    'max_depth': 9,
                    'num_leaves': 255,
                    'num_threads': 20
                }
                xgb_params = {
                    'colsample_bytree': 0.95,
                    'eta': 0.0421,
                    'eval_metric': 'rmse',
                    'max_depth': 9,
                    'n_estimators': 647,
                    'nthread': 20,
                    'subsample': 0.8789,
                    'alpha': 50.0,  # L1正则化
                    'lambda': 150.0  # L2正则化
                }
            elif factors_count < 50:
                # 中等因子数（20-50）：中等正则化（当前默认配置）
                lgbm_params = {
                    'loss': 'mse',
                    'colsample_bytree': 0.8879,
                    'learning_rate': 0.2,
                    'subsample': 0.8789,
                    'lambda_l1': 205.6999,
                    'lambda_l2': 580.9768,
                    'max_depth': 8,
                    'num_leaves': 210,
                    'num_threads': 20
                }
                xgb_params = {
                    'colsample_bytree': 0.8879,
                    'eta': 0.0421,
                    'eval_metric': 'rmse',
                    'max_depth': 8,
                    'n_estimators': 647,
                    'nthread': 20,
                    'subsample': 0.8789,
                    # 'colsample_bytree': 0.8879,
                    # 'eta': 0.0421,
                    # 'eval_metric': 'rmse',
                    # 'max_depth': 8,
                    # 'n_estimators': 647,
                    # 'nthread': 20,
                    # 'subsample': 0.8789,
                    # 'alpha': 205.6999,
                    # 'lambda': 580.9768
                }
            else:
                # 因子多（>=50）：强正则化，降低复杂度
                lgbm_params = {
                    'loss': 'mse',
                    'colsample_bytree': 0.7,
                    'learning_rate': 0.15,
                    'subsample': 0.8,
                    'lambda_l1': 300.0,
                    'lambda_l2': 800.0,
                    'max_depth': 6,
                    'num_leaves': 127,
                    'num_threads': 20
                }
                xgb_params = {
                    'colsample_bytree': 0.7,
                    'eta': 0.03,
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'n_estimators': 800,
                    'nthread': 20,
                    'subsample': 0.8,
                    'alpha': 300.0,
                    'lambda': 800.0
                }
            
            # 根据 model_type 设置模型配置
            if model_type == "xgboost":
                config['task']['model'] = {
                    'class': 'XGBModel',
                    'module_path': 'qlib.contrib.model.xgboost',
                    'kwargs': xgb_params
                }
            else:
                # xgboost 默认
                if 'kwargs' not in config['task']['model']:
                    config['task']['model']['kwargs'] = {}
                # 更新参数，保留基础配置中可能存在的其他参数
                config['task']['model']['kwargs'].update(lgbm_params)
            
            print(f"[生成配置文件] 根据因子数量 ({factors_count}) 调整模型参数:")
            if factors_count < 20:
                print(f"  - 使用小因子池配置（降低正则化，增加模型容量）")
            elif factors_count < 50:
                print(f"  - 使用中等因子池配置（默认配置）")
            else:
                print(f"  - 使用大因子池配置（强正则化，降低复杂度）")
        
        # 确定输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(base_config_path).parent
            output_path = output_dir / f"workflow_config_{class_name}_{timestamp}.yaml"
        
        # 保存配置文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"[生成配置文件] 已生成配置文件: {output_path}")
        print(f"[生成配置文件] 因子池类名: {class_name}")
        print(f"[生成配置文件] 模块路径: {module_path}")
        
        # 打印时间段信息
        segments_info = config.get('task', {}).get('dataset', {}).get('kwargs', {}).get('segments', {})
        if segments_info:
            if train_end_year is not None:
                print(f"[生成配置文件] 时间段配置 (基于 train_end_year={train_end_year} 自动计算):")
            else:
                print(f"[生成配置文件] 时间段配置:")
            if 'train' in segments_info:
                print(f"  - 训练集: {segments_info['train'][0]} 至 {segments_info['train'][1]}")
            if 'valid' in segments_info:
                print(f"  - 验证集: {segments_info['valid'][0]} 至 {segments_info['valid'][1]}")
            if 'test' in segments_info:
                print(f"  - 测试集: {segments_info['test'][0]} 至 {segments_info['test'][1]}")
        
        return str(output_path)
    def append_factor_eval_result(
        self, 
        factor: str, 
        eval_result: Dict[str, Any], 
        eval_results_dir: Optional[str] = None,
        sota_pool_list: Optional[List[str]] = None
    ) -> str:
        """
        保存因子评估结果到文件
        
        Args:
            factor: 因子表达式（字符串）
            eval_result: 评估结果字典
            eval_results_dir: 评估结果保存目录，默认为 data/eval_results/
            
        Returns:
            str: 保存的文件路径
        """
        import hashlib
        from datetime import datetime
        
        # 确定保存目录
        if eval_results_dir is None:
            current_dir = Path(__file__).parent.parent
            eval_results_dir = current_dir / "data" / "eval_results"
        else:
            eval_results_dir = Path(eval_results_dir)
        
        # 确保目录存在
        eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名：使用因子表达式的哈希值 + 时间戳
        factor_hash = hashlib.md5(factor.encode('utf-8')).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_result_{factor_hash}_{timestamp}.json"
        file_path = eval_results_dir / filename
        
        # 构建要保存的数据
        save_data = {
            "factor": factor,
            "timestamp": timestamp,
            "eval_result": eval_result,
            "SOTA_pool": sota_pool_list

        }
        
        try:
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功保存评估结果到: {file_path}")
            return str(file_path)
            
        except Exception as e:
            print(f"错误: 保存评估结果时发生异常: {e}")
            return ""

    def read_mlflow_metric_latest(self, file_path: str) -> float:
        """
        读取 MLflow 指标文件，返回最新的指标值
        
        Args:
            file_path: 指标文件路径
            
        Returns:
            最新的指标值，如果文件为空则返回 None
        """
        from pathlib import Path
        
        file_path = Path(file_path)
        print(f"file_path: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"指标文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
        
        return None 

    def get_pickle_statistics(self, file_path):
        """
        读取 pickle 文件并返回统计信息字典
        
        参数:
            file_path (str): pickle 文件路径
            
        返回:
            dict: 包含统计信息的字典，键包括:
                - count: 数据点数量
                - mean: 均值
                - std: 标准差
                - min: 最小值
                - 25%: 25%分位数
                - 50%: 中位数（50%分位数）
                - 75%: 75%分位数
                - max: 最大值
                
        异常:
            如果文件不存在或无法读取，返回 None
        """
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return None
        
        if not os.path.isfile(file_path):
            print(f"错误: 路径不是文件: {file_path}")
            return None
        
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 如果是 pandas Series，直接使用 describe()
            if isinstance(data, pd.Series):
                stats = data.describe()
                return {
                    'count': float(stats.get('count', 0)),
                    'mean': float(stats.get('mean', 0)),
                    'std': float(stats.get('std', 0)),
                    'min': float(stats.get('min', 0)),
                    '25%': float(stats.get('25%', 0)),
                    '50%': float(stats.get('50%', 0)),
                    '75%': float(stats.get('75%', 0)),
                    'max': float(stats.get('max', 0))
                }
            
            # 如果是 pandas DataFrame，返回第一列的统计信息
            elif isinstance(data, pd.DataFrame):
                if data.empty:
                    print("警告: DataFrame 为空")
                    return None
                # 选择第一列进行统计
                first_col = data.iloc[:, 0]
                stats = first_col.describe()
                return {
                    'count': float(stats.get('count', 0)),
                    'mean': float(stats.get('mean', 0)),
                    'std': float(stats.get('std', 0)),
                    'min': float(stats.get('min', 0)),
                    '25%': float(stats.get('25%', 0)),
                    '50%': float(stats.get('50%', 0)),
                    '75%': float(stats.get('75%', 0)),
                    'max': float(stats.get('max', 0))
                }
            
            # 如果是 numpy 数组，转换为 Series 再统计
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    print("警告: 数组为空")
                    return None
                # 展平数组
                flat_data = data.flatten()
                series = pd.Series(flat_data)
                stats = series.describe()
                return {
                    'count': float(stats.get('count', 0)),
                    'mean': float(stats.get('mean', 0)),
                    'std': float(stats.get('std', 0)),
                    'min': float(stats.get('min', 0)),
                    '25%': float(stats.get('25%', 0)),
                    '50%': float(stats.get('50%', 0)),
                    '75%': float(stats.get('75%', 0)),
                    'max': float(stats.get('max', 0))
                }
            
            else:
                print(f"警告: 不支持的数据类型: {type(data).__name__}")
                return None
                
        except Exception as e:
            print(f"错误: 无法读取 pickle 文件")
            print(f"错误信息: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            return None

    def append_factor_to_sota_pool(self, factor: str, sota_pool_path: Optional[str] = None) -> bool:
        """
        将因子追加到 SOTA 因子池
        
        Args:
            factor: 要追加的因子表达式（字符串）
            sota_pool_path: SOTA 因子池文件路径，默认为 data/sota_pool.json
            
        Returns:
            bool: 如果成功追加返回 True，如果因子已存在返回 False
        """
        # 确定 SOTA 因子池路径
        if sota_pool_path is None:
            current_dir = Path(__file__).parent.parent
            sota_pool_path = current_dir / "data" / "sota_pool.json"
        
        # 确保因子是字符串且非空
        if not isinstance(factor, str) or not factor.strip():
            print(f"错误: 因子必须是非空字符串")
            return False
        
        factor = factor.strip()
        
        try:
            # 读取现有的因子列表
            if Path(sota_pool_path).exists():
                with open(sota_pool_path, 'r', encoding='utf-8') as f:
                    sota_factors = json.load(f)
            else:
                # 如果文件不存在，创建新的列表
                sota_factors = []
                # 确保目录存在
                Path(sota_pool_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 检查因子是否已存在（避免重复）
            if factor in sota_factors:
                print(f"因子已存在于 SOTA 因子池中: {factor}")
                return False
            
            # 追加新因子
            sota_factors.append(factor)
            
            # 写回文件
            with open(sota_pool_path, 'w', encoding='utf-8') as f:
                json.dump(sota_factors, f, ensure_ascii=False, indent=0)
            
            print(f"成功将因子追加到 SOTA 因子池: {factor}")
            print(f"SOTA 因子池当前包含 {len(sota_factors)} 个因子")
            return True
            
        except Exception as e:
            print(f"错误: 追加因子到 SOTA 因子池时发生异常: {e}")
            return False

    def get_base_pool(self, base_pool_path: Optional[str] = None) -> List[str]:
        """
        读取基础因子池文件，返回因子列表
        
        Args:
            base_pool_path: 基础因子池文件路径，默认为 data/base_pool.json
            
        Returns:
            List[str]: 因子列表，如果文件不存在或读取失败则返回空列表
        """
        # 确定基础因子池路径
        if base_pool_path is None:
            current_dir = Path(__file__).parent.parent
            base_pool_path = current_dir / "data" / "base_pool.json"
        else:
            base_pool_path = Path(base_pool_path)
        
        try:
            # 读取因子列表
            if base_pool_path.exists():
                with open(base_pool_path, 'r', encoding='utf-8') as f:
                    base_factors = json.load(f)
                
                # 确保返回的是列表
                if isinstance(base_factors, list):
                    print(f"成功读取基础因子池，包含 {len(base_factors)} 个因子")
                    return base_factors
                else:
                    print(f"警告: 基础因子池文件格式不正确，期望列表，实际为 {type(base_factors)}")
                    return []
            else:
                print(f"警告: 基础因子池文件不存在: {base_pool_path}")
                return []
                
        except json.JSONDecodeError as e:
            print(f"错误: 解析基础因子池 JSON 文件失败: {e}")
            return []
        except Exception as e:
            print(f"错误: 读取基础因子池时发生异常: {e}")
            return []

    def _convert_to_bool(self, value):
        """
        将字符串或布尔值转换为布尔值
        
        Args:
            value: 可能是字符串 "True"/"False" 或布尔值 True/False
            
        Returns:
            bool: 转换后的布尔值
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # 明确处理 true 和 false 的字符串表示
            value_lower = value.lower().strip()
            if value_lower in ("true", "1", "yes", "是", "t"):
                return True
            elif value_lower in ("false", "0", "no", "否", "f", ""):
                return False
            else:
                # 未知字符串，默认返回 False（更安全）
                return False
        # 其他类型，使用 bool() 转换
        return bool(value)

    def evaluate_and_analyze_factor(
        self,
        factor: Dict[str, Any],
        logs: List[str],
        sota_pool_list: List[str] = None,
        origin_factor_pool_analysis_result: Dict[str, Any] = None,
        correlation_threshold: float = 0.99
    ) -> Dict[str, Any]:
        """
        评估单个因子：检查相关性、注册因子池、运行基准测试、LLM评估
        
        Args:
            factor: 要评估的因子字典
            logs: 日志列表（会被修改）
            origin_factor_pool_ic_analysis_result: 原始因子池IC统计结果，如果为None则使用默认值
            origin_factor_pool_rank_ic_analysis_result: 原始因子池rank_IC统计结果，如果为None则使用默认值
            correlation_threshold: 相关性阈值，默认0.99
            
        Returns:
            包含评估结果的字典
        """
        # 默认的原始因子池统计结果
        if origin_factor_pool_analysis_result:
            #origin_factor_pool_analysis_result = self.get_origin_factor_pool_analysis_result(base_pool_list=sota_pool_list)
            origin_factor_pool_ic_analysis_result = origin_factor_pool_analysis_result["ic"]
            origin_factor_pool_rank_ic_analysis_result = origin_factor_pool_analysis_result["rank_ic"]
            origin_factor_pool_annualized_return_test_result = origin_factor_pool_analysis_result["annualized_return_test_result"]
            origin_factor_pool_max_drawdown_test_result = origin_factor_pool_analysis_result["max_drawdown_test_result"]
        else:
            print("没有原始因子池的ic和rank_ic评估结果")
        
        # 检查与SOTA因子的相关性
        try:
            should_keep = self.filter_high_correlation_with_sota(
                factor,
                correlation_threshold=correlation_threshold,
                sota_pool_list=sota_pool_list
            )
        except Exception as e:
            print(f"错误: 检查与SOTA因子的相关性时发生异常: {e}")
            return {"should_keep": False}
        # 如果与SOTA因子高相关，移除它
        if not should_keep:
            logs.append(f"因子与 SOTA 因子高相关，已移除")
            return {"should_keep": False}
        
        # 因子通过相关性检查，保留它
        logs.append(f"因子通过相关性检查，已保留")
        
        # 合并新因子和 sota_pool，注册成新的因子池
        registered_module_path,workflow_config_path = self._register_merged_factor_pool(
            new_factor=factor,
            sota_pool_list=sota_pool_list
        )
        
        if not registered_module_path:
            logs.append(f"警告: 因子池注册失败")
            return {"should_keep": True, "registered_module_path": None}
        
        logs.append(f"已将新因子和 SOTA 因子池合并注册: {registered_module_path}, {workflow_config_path}")
        # 调用MCP工具运行基准测试
        result = self.call_mcp_tool(
            "qlib_benchmark_runner",
            {
                "yaml_path": str(workflow_config_path),
                "experiment_name": None
            }
        )
        
        if isinstance(result, str):
            result_dict = json.loads(result)
        else:
            result_dict = result
        
        # 分析IC和rank_IC结果
        ic_analysis_result = self.get_pickle_statistics(result_dict["ic"])
        rank_ic_analysis_result = self.get_pickle_statistics(result_dict["rank_ic"])
        annualized_return_test_result = self.read_mlflow_metric_latest(result_dict["1day.excess_return_with_cost.annualized_return"])
        max_drawdown_test_result = self.read_mlflow_metric_latest(result_dict["1day.excess_return_with_cost.max_drawdown"])
        print(f"factor: {factor}")
        print(f"ic_analysis_result: {ic_analysis_result}")
        print(f"rank_ic_analysis_result: {rank_ic_analysis_result}")
        print(f"origin_factor_pool_ic_analysis_result: {origin_factor_pool_ic_analysis_result}")
        print(f"origin_factor_pool_rank_ic_analysis_result: {origin_factor_pool_rank_ic_analysis_result}")
        print(f"origin_factor_pool_annualized_return_test_result: {origin_factor_pool_annualized_return_test_result}")
        print(f"origin_factor_pool_max_drawdown_test_result: {origin_factor_pool_max_drawdown_test_result}")
        print(f"annualized_return_test_result: {annualized_return_test_result}")
        print(f"max_drawdown_test_result: {max_drawdown_test_result}")
        # 调用 LLM 进行因子分析
        response = self.llm.call(
            prompt=f"""
            如果有一堆因子评估专家，如下信息是一个新因子加入到因子池后使用xgboost优化权重，在测试集上ic的评估结果，请根据以下因子评估结果，给出因子评估的结论:
            原始因子池的ic评估结果:
            {origin_factor_pool_ic_analysis_result}     
            原始因子池的排名ic评估结果:
            {origin_factor_pool_rank_ic_analysis_result}
            原始因子池使用xgboost优化权重后在测试集上的有交易成本的年化回报和最大回撤:
            origin_factor_pool_annualized_return_test_result: {origin_factor_pool_annualized_return_test_result}
            origin_factor_pool_max_drawdown_test_result: {origin_factor_pool_max_drawdown_test_result}
            新因子加入到因子池后使用xgboost优化权重，在测试集上ic的评估结果和排名ic的评估结果,以及在测试集上的有交易成本的年化回报和最大回撤:
            ic_analysis_result: {ic_analysis_result}
            rank_ic_analysis_result: {rank_ic_analysis_result}
            annualized_return_test_result: {annualized_return_test_result}
            max_drawdown_test_result: {max_drawdown_test_result}
            这些因子专家们根据上述信息给出新因子评估的结论，并给出理由。
            严格遵循json格式，以json格式返回，不要使用换行等转义字符，键为conclusion和reason。
            """,
            stream=False
        )
        factor_analysis_response = self.llm.parse_json_response(response)
        #print(f"factor_analysis_response: {factor_analysis_response}")
        
        # 调用 LLM 进行经济学解释
        factor_economic_explanation_response = self.llm.call(
            prompt=f"""
            如果有一堆因子评估专家，如下是一个因子，这些因子专家们根据上述因子，尝试给出经济学原理解释，因子公式中的rank是在同一只股票的时间序列上:
            {factor}
            ic是训练集上的ic均值，rank_ic是验证集上的rank_ic均值。
            严格遵循json格式，以json格式返回，不要使用换行等转义字符，键为factor，conclusion和reason。
            """,
            stream=False
        )
        factor_economic_explanation_response = self.llm.parse_json_response(factor_economic_explanation_response)
        #print(f"factor_economic_explanation_response: {factor_economic_explanation_response}")
        
        # 调用 LLM 进行综合评估
        factor_evaluation_response = self.llm.call(
            prompt=f"""
            如果有一堆因子评估专家，如下是新因子加入到因子池后使用xgboost优化权重，在测试集上ic的评估结果，以及根据因子本身做的经济学解释，请根据这些信息，给出因子评估的结论:
            原始的因子池，原始因子池中的因子ic被默认赋值为0，但是实际上这些因子ic值并不为0：
            {sota_pool_list}
            原始因子池使用xgboost优化权重后ic评估结果:
            {origin_factor_pool_ic_analysis_result}
            原始因子池的排名ic评估结果:
            {origin_factor_pool_rank_ic_analysis_result}
            原始因子池使用xgboost优化权重后在测试集上的有交易成本的年化回报和最大回测:
            annualized_return_test_result: {origin_factor_pool_annualized_return_test_result}
            max_drawdown_test_result: {origin_factor_pool_max_drawdown_test_result}
            新因子{factor['qlib_expression']}加入到因子池后使用xgboost优化权重，在测试集上ic的评估结果和排名ic的评估结果,以及在测试集上的有交易成本的年化回报和最大回测:
            annualized_return_test_result: {annualized_return_test_result}
            max_drawdown_test_result: {max_drawdown_test_result}
            ic_analysis_result: {ic_analysis_result}
            rank_ic_analysis_result: {rank_ic_analysis_result}
            对于因子加入因子池之后的评估：
            {factor_analysis_response}
            根据因子本身做的经济学解释：
            {factor_economic_explanation_response} 
            在评估结果时：1. 任何微小的改进都应予以考虑并纳入其中（将“if_keep”选项设为“True”）。
            2. 如果新的因素（或因素组合）能提高年化收益率，就建议将其作为替代方案来取代当前的最佳结果。
            3. 其他指标的细微变化是可以接受的，只要年化收益率能够提高就行。

            严格遵循json格式，以json格式返回，不要使用换行等转义字符，键为if_keep和conclusion和reason。
            """,
            stream=False
        )
        factor_evaluation_response = self.llm.parse_json_response(factor_evaluation_response)
        #print(f"factor_evaluation_response: {factor_evaluation_response}")
        
        return {
            "should_keep": self._convert_to_bool(factor_evaluation_response["if_keep"]),
            "factor_expression": factor["qlib_expression"],
            "ic_analysis_result": ic_analysis_result,
            "rank_ic_analysis_result": rank_ic_analysis_result,
            "annualized_return_test_result": annualized_return_test_result,
            "max_drawdown_test_result": max_drawdown_test_result,
            "factor_analysis_response": factor_analysis_response,
            "factor_economic_explanation_response": factor_economic_explanation_response,
            "factor_evaluation_response": factor_evaluation_response,
        }

    def get_origin_factor_pool_analysis_result(self,base_pool_list: List[str] = None) -> Dict[str, Any]:
        """
        获取原始因子池的ic评估结果
        """
        registered_module_path,workflow_config_path = self._register_merged_factor_pool(
            new_factor=None,
            sota_pool_list=base_pool_list
        )
        if not registered_module_path:
            return None
        
        result = self.call_mcp_tool(
            "qlib_benchmark_runner",
            {
                "yaml_path": str(workflow_config_path),
                "experiment_name": None
            }
        )
        if isinstance(result, str):
            result_dict = json.loads(result)
        else:
            result_dict = result
        
        # 分析IC和rank_IC结果
        print(f"result_dict: {result_dict}")
        original_ic_analysis_result = self.get_pickle_statistics(result_dict["ic"])
        original_rank_ic_analysis_result = self.get_pickle_statistics(result_dict["rank_ic"])
        original_annualized_return_test_result = self.read_mlflow_metric_latest(result_dict["1day.excess_return_with_cost.annualized_return"])
        original_max_drawdown_test_result = self.read_mlflow_metric_latest(result_dict["1day.excess_return_with_cost.max_drawdown"])
        return {
            "original_ic_analysis_result": original_ic_analysis_result,
            "original_rank_ic_analysis_result": original_rank_ic_analysis_result,
            "original_annualized_return_test_result": original_annualized_return_test_result,
            "original_max_drawdown_test_result": original_max_drawdown_test_result
        }

    def revise_factor(self,eval_result: Dict[str, Any],sota_pool_list: List[str] = None,) -> Dict[str, Any]:
        ops = {
                "qlib_operators": {
                    "source_file": "qlib-main/qlib/data/ops.py",
                    "additional_files": [
                    "qlib-main/qlib/data/pit.py",
                    "qlib-main/qlib/data/base.py"
                    ],
                    "operators": {
                    "element_wise": [
                        {
                        "name": "Abs",
                        "description": "绝对值",
                        "class": "NpElemOperator",
                        "parameters": ["feature"]
                        },
                        {
                        "name": "Sign",
                        "description": "符号函数",
                        "class": "NpElemOperator",
                        "parameters": ["feature"]
                        },
                        {
                        "name": "Log",
                        "description": "对数",
                        "class": "NpElemOperator",
                        "parameters": ["feature"]
                        },
                        {
                        "name": "Not",
                        "description": "逻辑非",
                        "class": "NpElemOperator",
                        "parameters": ["feature"]
                        },
                        {
                        "name": "Mask",
                        "description": "掩码操作",
                        "class": "NpElemOperator",
                        "parameters": ["feature", "instrument"]
                        },
                        {
                        "name": "ChangeInstrument",
                        "description": "改变标的",
                        "class": "ElemOperator",
                        "parameters": ["instrument", "feature"]
                        }
                    ],
                    "binary": [
                        {
                        "name": "Add",
                        "description": "加法",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Sub",
                        "description": "减法",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Mul",
                        "description": "乘法",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Div",
                        "description": "除法",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Power",
                        "description": "幂运算",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Greater",
                        "description": "取较大值",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Less",
                        "description": "取较小值",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Gt",
                        "description": "大于比较",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Ge",
                        "description": "大于等于比较",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Lt",
                        "description": "小于比较",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Le",
                        "description": "小于等于比较",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Eq",
                        "description": "等于比较",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Ne",
                        "description": "不等于比较",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "And",
                        "description": "逻辑与",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        },
                        {
                        "name": "Or",
                        "description": "逻辑或",
                        "class": "NpPairOperator",
                        "parameters": ["feature_left", "feature_right"]
                        }
                    ],
                    "ternary": [
                        {
                        "name": "If",
                        "description": "条件选择",
                        "class": "ExpressionOps",
                        "parameters": ["condition", "feature_left", "feature_right"]
                        }
                    ],
                    "rolling": [
                        {
                        "name": "Ref",
                        "description": "引用（延迟/提前N期）",
                        "class": "Rolling",
                        "parameters": ["feature", "N"],
                        "note": "N=0返回首日数据，N>0返回N期前数据，N<0返回未来数据"
                        },
                        {
                        "name": "Mean",
                        "description": "滚动均值",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Sum",
                        "description": "滚动求和",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Std",
                        "description": "滚动标准差",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Var",
                        "description": "滚动方差",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Skew",
                        "description": "滚动偏度",
                        "class": "Rolling",
                        "parameters": ["feature", "N"],
                        "constraint": "N >= 3"
                        },
                        {
                        "name": "Kurt",
                        "description": "滚动峰度",
                        "class": "Rolling",
                        "parameters": ["feature", "N"],
                        "constraint": "N >= 4"
                        },
                        {
                        "name": "Max",
                        "description": "滚动最大值",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Min",
                        "description": "滚动最小值",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Med",
                        "description": "滚动中位数",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Mad",
                        "description": "滚动平均绝对偏差",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Rank",
                        "description": "滚动排名（百分位）",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Quantile",
                        "description": "滚动分位数",
                        "class": "Rolling",
                        "parameters": ["feature", "N", "qscore"]
                        },
                        {
                        "name": "Count",
                        "description": "滚动非NaN元素计数",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Delta",
                        "description": "滚动窗口内首尾差值",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Slope",
                        "description": "滚动线性回归斜率",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Rsquare",
                        "description": "滚动线性回归R方",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "Resi",
                        "description": "滚动回归残差",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "WMA",
                        "description": "加权移动平均",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "EMA",
                        "description": "指数移动平均",
                        "class": "Rolling",
                        "parameters": ["feature", "N"],
                        "note": "支持N为整数或0-1之间的浮点数（alpha）"
                        },
                        {
                        "name": "IdxMax",
                        "description": "滚动窗口内最大值索引",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        },
                        {
                        "name": "IdxMin",
                        "description": "滚动窗口内最小值索引",
                        "class": "Rolling",
                        "parameters": ["feature", "N"]
                        }
                    ],
                    "pair_rolling": [
                        {
                        "name": "Corr",
                        "description": "滚动相关系数",
                        "class": "PairRolling",
                        "parameters": ["feature_left", "feature_right", "N"]
                        },
                        {
                        "name": "Cov",
                        "description": "滚动协方差",
                        "class": "PairRolling",
                        "parameters": ["feature_left", "feature_right", "N"]
                        }
                    ],
                    "time_resample": [
                        {
                        "name": "TResample",
                        "description": "时间重采样",
                        "class": "TResample",
                        "parameters": ["feature", "freq", "func"]
                        }
                    ],
                    "pit_operators": [
                        {
                        "name": "P",
                        "description": "点对时数据聚合",
                        "file": "qlib-main/qlib/data/pit.py",
                        "parameters": ["feature"]
                        },
                        {
                        "name": "PRef",
                        "description": "点对时数据引用",
                        "file": "qlib-main/qlib/data/pit.py",
                        "parameters": ["feature", "period"]
                        }
                    ],
                    "base_features": [
                        {
                        "name": "Feature",
                        "description": "基础特征",
                        "class": "Feature"
                        },
                        {
                        "name": "PFeature",
                        "description": "点对时基础特征",
                        "class": "PFeature"
                        }
                    ]
                    },
                    "total_count": 50,
                    "notes": {
                    "rolling_window": "N=0表示expanding（扩展窗口），0<N<1表示EWM（指数加权），N>=1表示rolling（滚动窗口）",
                    "registration": "所有操作符通过Operators对象统一管理，可通过register_all_ops函数注册",
                    "custom_ops": "支持通过配置custom_ops注册自定义操作符"
                    }
                }
                }
        revised_factor_expression_response =self.llm.call(
            prompt=f"""
            如果有一堆因子评估专家，如下是新因子加入当前因子池后，在测试集上的ic和rank_ic评估结果，以及基于因子公式本身的解释，:
            {eval_result}
            原始因子池如下：{sota_pool_list}，
            这些因子专家们根据这些信息，结合经济学知识，在充分考虑已有因子池中的因子的基础上，分别尝试修改因子公式本身，
            修改后的因子应当与因子池已有的因子不同，同时增加其在测试集上的ic和rank_ic上的表现，
            以及在测试集上的有交易成本的年化回报和最大回撤上的表现，
            并给出修改后的因子表达式。修改后的因子表达式需要支持qlib的表达式语法，qlib支持的操作符如下：{ops}，
            类似于“.rolling(20).mean()”这样的表达式是错误的，不要使用qlib的api，直接使用qlib的表达式语法。
            严格遵循json格式，以json格式返回，不要使用换行等转义字符，，键为revised_factor_expression和reason。
            revised_factor_expression: 修改后的因子表达式
            reason: 修改理由
            """,
            stream=False
        )
        revised_factor_expression_response = self.llm.parse_json_response(revised_factor_expression_response)
        return revised_factor_expression_response

    def add_factor_to_pool(
        self,
        top_factor: Dict[str, Any],
        eval_result: Dict[str, Any],
        sota_pool_list: List[str],
        origin_factor_pool_analysis_result: Dict[str, Any],
        added_factor_count: int
    ) -> int:
        """
        将评估通过的因子添加到因子池，并更新相关记录
        
        Args:
            top_factor: 要添加的因子字典，包含 'qlib_expression' 键
            eval_result: 因子评估结果字典
            sota_pool_list: 当前因子池列表
            origin_factor_pool_analysis_result: 因子池分析结果字典
            added_factor_count: 当前已添加因子计数
            
        Returns:
            int: 更新后的已添加因子计数
        """
        print(f"保留因子: {top_factor['qlib_expression']}")
        sota_pool_list.append(top_factor['qlib_expression'])
        self.append_factor_eval_result(top_factor['qlib_expression'], eval_result, sota_pool_list=sota_pool_list)
        
        # 更新计数器并保存新加入因子的ic和rank_ic
        added_factor_count += 1
        origin_factor_pool_analysis_result[f"新加入第{added_factor_count}个因子后因子池的ic和rank_ic"] = {
            "ic": eval_result["ic_analysis_result"],
            "rank_ic": eval_result["rank_ic_analysis_result"],
            "annualized_return_test_result": eval_result["annualized_return_test_result"],
            "max_drawdown_test_result": eval_result["max_drawdown_test_result"]
        }
        
        return added_factor_count

    def evaluate_and_add_factor(
        self,
        top_factor: Dict[str, Any],
        logs: Any,
        sota_pool_list: List[str],
        origin_factor_pool_analysis_result: Dict[str, Any],
        added_factor_count: int,
        correlation_threshold: float = 0.99
    ) -> tuple[bool, int]:
        """
        评估因子并尝试添加到因子池
        
        Args:
            top_factor: 要评估的因子字典
            logs: 日志对象
            sota_pool_list: 当前因子池列表
            origin_factor_pool_analysis_result: 因子池分析结果字典
            added_factor_count: 当前已添加因子计数
            correlation_threshold: 相关性阈值
            
        Returns:
            tuple[bool, int]: (是否成功添加, 更新后的计数器)
        """
        # 调用评估函数
        eval_result = self.evaluate_and_analyze_factor(
            factor=top_factor,
            logs=logs,
            correlation_threshold=correlation_threshold,
            sota_pool_list=sota_pool_list,
            origin_factor_pool_analysis_result=list(origin_factor_pool_analysis_result.values())[-1] if origin_factor_pool_analysis_result else None
        )
        
        if eval_result["should_keep"]:
            added_factor_count = self.add_factor_to_pool(
                top_factor=top_factor,
                eval_result=eval_result,
                sota_pool_list=sota_pool_list,
                origin_factor_pool_analysis_result=origin_factor_pool_analysis_result,
                added_factor_count=added_factor_count
            )

            return True, added_factor_count, eval_result
        
        return False, added_factor_count, eval_result

    def process(self, factors: list, sota_pool_list: List[str] = None, origin_factor_pool_analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        基于已有的因子池进行因子评估
        
        Args:
            factors: 因子列表，格式为 [{"expression": "因子表达式", "ic": IC值}, ...]
            sota_pool_list: 当前因子池列表，如果为None则从基础因子池初始化
            origin_factor_pool_analysis_result: 原始因子池分析结果，如果为None则重新获取
            
        Returns:
            包含因子评估结果的字典
        """
        print("开始因子评估")
        logs = []
        print(f"factors: {factors[0:5]}")
        filter_factors = self.filter_and_sort_factors(factors)
        print(f"filter_factors: {filter_factors[0:5]}")
        base_factor_pool_list = self.get_base_pool()
        print(f"base_factor_pool_list: {base_factor_pool_list}")
        
        # 如果 sota_pool_list 为空，则从基础因子池初始化
        if sota_pool_list is None:
            sota_pool_list = copy.deepcopy(base_factor_pool_list)
            logs.append(f"[FactorEval] 初始化 sota_pool_list，包含 {len(sota_pool_list)} 个因子")
            print(f"[FactorEval] 初始化 sota_pool_list，包含 {len(sota_pool_list)} 个因子")
        else:
            logs.append(f"[FactorEval] 使用已有的 sota_pool_list，包含 {len(sota_pool_list)} 个因子")
            print(f"[FactorEval] 使用已有的 sota_pool_list，包含 {len(sota_pool_list)} 个因子")
        
        # 如果 origin_factor_pool_analysis_result 为空，则初始化并获取
        if origin_factor_pool_analysis_result is None:
            origin_factor_pool_analysis_result = {}
            # 获取原始因子池的ic和rank_ic
            origin_result = self.get_origin_factor_pool_analysis_result(base_pool_list=base_factor_pool_list)
            print(f"origin_result: {origin_result}")
            if origin_result:
                origin_factor_pool_analysis_result["原始因子池的ic和rank_ic"] = {
                    "ic": origin_result.get("original_ic_analysis_result", {}),
                    "rank_ic": origin_result.get("original_rank_ic_analysis_result", {}),
                    "annualized_return_test_result": origin_result.get("original_annualized_return_test_result", {}),
                    "max_drawdown_test_result": origin_result.get("original_max_drawdown_test_result", {})
                }
            logs.append(f"[FactorEval] 初始化 origin_factor_pool_analysis_result")
        else:
            logs.append(f"[FactorEval] 使用已有的 origin_factor_pool_analysis_result，包含 {len(origin_factor_pool_analysis_result)} 个历史记录")

        # 添加计数器跟踪新加入的因子数量
        added_factor_count = 0
        filter_factors = filter_factors[0:5]
        # 对前4个因子进行评估
        if filter_factors:  
            # 创建因子索引列表并打乱，用于随机抽取
            factor_indices = list(range(len(filter_factors)))
            random.shuffle(factor_indices)
            idx = 0
            print(f"factor_indices: {factor_indices}")
            while idx < len(factor_indices) and len(sota_pool_list) <= 40:
                #break
                #random_index = factor_indices[idx]
                top_factor = filter_factors[idx]
                idx += 1
                if len(filter_factors) == idx and len(sota_pool_list) <= 40:
                    #logs.append(f"[FactorEval] 过滤后的因子数量({len(filter_factors)})少于阈值({MIN_FACTOR_COUNT})，返回因子挖掘节点继续挖掘")
                    return {
                        "status": "need_more_factors",
                        "logs": logs,
                        "filtered_factors": filter_factors,
                        "current_node": "factor_mining",  # 返回因子挖掘节点
                        "sota_pool_list": sota_pool_list,  # 保存状态
                        "origin_factor_pool_analysis_result": origin_factor_pool_analysis_result,  # 保存状态
                        "registered_module_path": None
                    }
                # 尝试评估并添加因子
                added, added_factor_count, eval_result = self.evaluate_and_add_factor(
                    top_factor=top_factor,
                    logs=logs,
                    sota_pool_list=sota_pool_list,
                    origin_factor_pool_analysis_result=origin_factor_pool_analysis_result,
                    added_factor_count=added_factor_count,
                    correlation_threshold=0.99
                )
                
                # 如果初始评估未通过，尝试修订因子
                if not added:
                    revised_factor_expression = self.revise_factor(eval_result)
                    if revised_factor_expression:
                        top_factor["qlib_expression"] = revised_factor_expression["revised_factor_expression"]
                        # 再次尝试评估并添加修订后的因子
                        added, added_factor_count, eval_result = self.evaluate_and_add_factor(
                            top_factor=top_factor,
                            logs=logs,
                            sota_pool_list=sota_pool_list,
                            origin_factor_pool_analysis_result=origin_factor_pool_analysis_result,
                            added_factor_count=added_factor_count,
                            correlation_threshold=0.99
                        )

                # if len(sota_pool_list) % 5 == 0 and len(sota_pool_list) != 5:
                #     print(f"当前因子池大小: {len(sota_pool_list)}")
                #     print(f"当前因子: {top_factor['qlib_expression']}")
                    
                #     # 构建当前因子池的评估结果信息（用于LLM提示）
                #     should_continue_response =self.llm.call(
                #         prompt=f"""
                #         如果有一堆因子评估专家，如下是当前因子池，请根据这些信息，给出因子评估的结论:
                #         当前因子池:
                #         {sota_pool_list}
                #         原始因子池：
                #         {base_factor_pool_list}
                #         以下是原始因子池每次加入新的因子之后，在测试集上的ic和rank_ic评估结果，最后一次为当前因子池的评估结果:
                #         {origin_factor_pool_analysis_result}

                #         新的因子来源于机器学习或者RL算法，可能存在同质化问题，即使有新的因子进来，可能也无法为整个因子池的ic、rank_ic带来显著提升。
                #         这些因子专家们根据上述信息，分别判断新加入的因子是否能继续增加因子池在测试集的ic和rank_ic,还是停止迭代，进入下一阶段
                #         并给出理由。以json格式返回，键为should_continue和reason。
                #         should_continue: 是否继续迭代增加因子池在测试集的ic和rank_ic,还是停止迭代，进入下一阶段，模型优化，True或False
                #         reason: 判断理由
                #         """,
                #         stream=False
                #     )
                #     should_continue_response = self.llm.parse_json_response(should_continue_response)
                #     if should_continue_response["should_continue"]:
                #         continue
                #     else:
                #         break
        
        # 如果没有通过相关性检查的因子，返回空结果
        return {
            "status": "success",
            "logs": logs,
            "filtered_factors": filter_factors,
            "sota_pool_list": sota_pool_list,  # 保存状态
            "origin_factor_pool_analysis_result": origin_factor_pool_analysis_result,  # 保存状态
            "registered_module_path": None
        }

if __name__ == "__main__":
    """测试过滤高相关因子的功能"""
    print("=" * 80)
    print("测试因子相关性过滤功能")
    print("=" * 80)
    
    # 创建测试因子（注意：needs_cs_rank 应该是 Python 的 True，不是 true）
    filter_factors = [{
        "qlib_expression": "(Less($open, $close)-$low)/$open",#"WMA(Div(Mean(WMA($volume, 10), 20), Std(WMA($volume, 10), 20)), 20)",
        "ic": 0.029146308079361916,
        "needs_cs_rank":False,  # Python 布尔值
        "is_valid": True,
        "original_expression": "Rank(TsWMA(TsIr(TsWMA(volume,10),20),20))",
        "rank_ic_valid": 0.0042740413919091225
    }]
    
    # 创建 FactorEvalAgent 实例（需要传入 llm_service，这里用 None 占位）
    # 注意：filter_high_correlation_with_sota 不依赖 llm_service，所以可以传入 None
    agent = FactorEvalAgent(llm_service=None)
    
    print(f"\n测试因子:")
    print(f"  表达式: {filter_factors[0]['qlib_expression']}")
    print(f"  IC: {filter_factors[0]['ic']}")
    print(f"  需要 CSRank: {filter_factors[0]['needs_cs_rank']}")
    print(f"\n开始测试相关性过滤...")
    
    if filter_factors:
        top_factor = filter_factors[0]  # 只取排名第一的因子
        
        # 尝试自动检测 Qlib 数据路径
        import os
        possible_qlib_paths = [
            os.path.expanduser("~/.qlib/qlib_data/cn_data")
        ]
        
        qlib_path = None
        for path in possible_qlib_paths:
            if os.path.exists(path):
                qlib_path = path
                print(f"找到 Qlib 数据路径: {qlib_path}")
                break
        
        if qlib_path:
            # 手动初始化 Qlib（如果 filter_high_correlation_with_sota 内部初始化失败）
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "Qlib_MCP" / "modules"))
                from calculate_ic import init_qlib
                success, _ = init_qlib(provider_uri=qlib_path)
                if success:
                    print(f"Qlib 初始化成功: {qlib_path}")
                else:
                    print(f"警告: Qlib 初始化失败")
            except Exception as e:
                print(f"警告: 无法初始化 Qlib: {e}")
        
        should_keep = agent.filter_high_correlation_with_sota(
            top_factor,
            correlation_threshold=0.99,
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        
        if not should_keep:
            # 排名第一的因子与 SOTA 因子高相关，移除它
            filter_factors = filter_factors[1:]
            #logs.append(f"排名第一的因子与 SOTA 因子高相关，已移除")
        else:
            # 排名第一的因子通过相关性检查，保留它
            #logs.append(f"排名第一的因子通过相关性检查，已保留")
            
            # 合并新因子和 sota_pool，注册成新的因子池
            registered_module_path = agent._register_merged_factor_pool(
                new_factor=top_factor,
                sota_pool_path=None
            )
            workflow_config_path = agent.generate_workflow_config(
                module_path=registered_module_path,
                output_path=None,
                base_config_path=None,
                model_type="xgboost"
            )
            result = agent.call_mcp_tool(
                "qlib_benchmark_runner",
                {
                    "yaml_path": str(workflow_config_path),
                    "experiment_name": None
                }
            )
            if isinstance(result, str):
                result_dict = json.loads(result)
            else:
                result_dict = result
            #qrun_workflow(workflow_config_path)
            print(f"workflow_config_path: {workflow_config_path}")
            print(f"registered_module_path: {registered_module_path}")
            print(f"result: {result}")
            analysis_result = agent.get_pickle_statistics(result_dict["ic"])
            print(f"analysis_result: {analysis_result}")
    
