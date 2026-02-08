"""
因子评估 Agent（重构版）

这是一个精简的编排类，负责协调各个组件完成因子评估工作流。
原始代码约 2000 行，重构后约 400 行。

职责分离:
- MCPService: MCP 客户端通信
- FactorPoolManager: 因子池的加载/保存/注册
- CorrelationEvaluator: 因子相关性检查
- WorkflowConfigGenerator: Workflow 配置生成
- LLMFactorEvaluator: LLM 因子分析和评估
- FileUtils: 文件操作工具
"""
import copy
import random
from typing import Dict, Any, List, Optional

# 导入各组件
from .services.mcp_service import MCPService
from .services.factor_pool_manager import FactorPoolManager
from .evaluators.correlation_evaluator import CorrelationEvaluator
from .evaluators.llm_evaluator import LLMFactorEvaluator, convert_to_bool
from .generators.workflow_config_generator import WorkflowConfigGenerator
from .utils.file_utils import FileUtils


class FactorEvalAgent:
    """
    因子评估 Agent
    
    主要功能:
    1. 筛选和排序因子
    2. 检查因子与 SOTA 因子池的相关性
    3. 注册合并的因子池并运行基准测试
    4. 使用 LLM 进行因子评估
    5. 管理因子池的增删
    """
    
    def __init__(self, llm_service, mcp_server_path: Optional[str] = None):
        """
        初始化因子评估 Agent
        
        Args:
            llm_service: LLM 服务实例（需要有 call 和 parse_json_response 方法）
            mcp_server_path: MCP 服务器脚本路径
        """
        self.llm = llm_service
        
        # 初始化各组件
        self.mcp = MCPService(mcp_server_path)
        self.pool_manager = FactorPoolManager()
        self.correlation_evaluator = CorrelationEvaluator()
        self.workflow_generator = WorkflowConfigGenerator()
        self.llm_evaluator = LLMFactorEvaluator(llm_service) if llm_service else None
    
    # ==================== 公开 API ====================
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用的 MCP 工具列表"""
        return self.mcp.list_tools()
    
    def call_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """调用 MCP 工具"""
        return self.mcp.call_tool(tool_name, arguments)
    
    def filter_and_sort_factors(
        self, 
        factors: List[Dict[str, Any]], 
        ic_threshold: float = 0.03
    ) -> List[Dict[str, Any]]:
        """
        筛选和排序因子
        
        Args:
            factors: 因子列表，格式为 [{"expression": "...", "ic": IC值, "rank_ic_valid": Rank_IC值}, ...]
            ic_threshold: IC 阈值，只保留 IC 大于此值的因子
            
        Returns:
            筛选和排序后的因子列表，按 rank_ic_valid 降序排列
        """
        if not factors:
            return []
        
        filtered = []
        for factor in factors:
            if not isinstance(factor, dict):
                continue
            
            # 解析 IC 值
            ic = self._parse_numeric_value(factor.get("ic"))
            if ic is None or ic <= ic_threshold:
                continue
            
            # 解析 rank_ic_valid
            rank_ic = self._parse_numeric_value(factor.get("rank_ic_valid"))
            if rank_ic is None or rank_ic < 0:
                continue
            
            filtered.append(factor)
        
        # 按 rank_ic_valid 降序排序
        return sorted(
            filtered,
            key=lambda f: self._parse_numeric_value(f.get("rank_ic_valid")) or float('-inf'),
            reverse=True
        )
    
    def get_base_pool(self, path: Optional[str] = None) -> List[str]:
        """获取基础因子池"""
        return self.pool_manager.load_base_pool(path)
    
    def append_factor_to_sota_pool(self, factor: str, path: Optional[str] = None) -> bool:
        """将因子追加到 SOTA 因子池"""
        return self.pool_manager.append_to_sota_pool(factor, path)
    
    def get_origin_factor_pool_analysis_result(
        self, 
        base_pool_list: List[str]
    ) -> Optional[Dict[str, Any]]:
        """获取原始因子池的评估结果"""
        result = self._register_and_run_benchmark(
            new_factor=None,
            sota_factors=base_pool_list
        )
        
        if not result:
            return None
        
        return {
            "original_ic_analysis_result": result.get("ic_stats"),
            "original_rank_ic_analysis_result": result.get("rank_ic_stats"),
            "original_annualized_return_test_result": result.get("annualized_return"),
            "original_max_drawdown_test_result": result.get("max_drawdown")
        }
    
    def evaluate_and_analyze_factor(
        self,
        factor: Dict[str, Any],
        logs: List[str],
        sota_pool_list: List[str],
        origin_factor_pool_analysis_result: Optional[Dict[str, Any]] = None,
        correlation_threshold: float = 0.99
    ) -> Dict[str, Any]:
        """
        评估单个因子
        
        Args:
            factor: 要评估的因子字典
            logs: 日志列表
            sota_pool_list: SOTA 因子池列表
            origin_factor_pool_analysis_result: 原始因子池统计结果
            correlation_threshold: 相关性阈值
            
        Returns:
            评估结果字典
        """
        # 1. 检查与 SOTA 因子的相关性
        should_keep = self.correlation_evaluator.check_correlation(
            factor,
            sota_pool_list,
            threshold=correlation_threshold
        )
        
        if not should_keep:
            logs.append("因子与 SOTA 因子高相关，已移除")
            return {"should_keep": False}
        
        logs.append("因子通过相关性检查，已保留")
        
        # 2. 注册因子池并运行基准测试
        benchmark_result = self._register_and_run_benchmark(factor, sota_pool_list)
        
        if not benchmark_result:
            logs.append("警告: 因子池注册或基准测试失败")
            return {"should_keep": True, "registered_module_path": None}
        
        logs.append(f"已注册因子池: {benchmark_result.get('module_path')}")
        
        # 3. 如果没有 LLM 服务，直接返回基准测试结果
        if not self.llm_evaluator:
            return self._build_eval_result_without_llm(factor, benchmark_result)
        
        # 4. 使用 LLM 进行综合评估
        return self._evaluate_with_llm(
            factor, benchmark_result, sota_pool_list, origin_factor_pool_analysis_result
        )
    
    def revise_factor(
        self, 
        eval_result: Dict[str, Any], 
        sota_pool_list: List[str]
    ) -> Dict[str, Any]:
        """根据评估结果修订因子表达式"""
        if not self.llm_evaluator:
            return {}
        return self.llm_evaluator.revise_factor(eval_result, sota_pool_list)
    
    def process(
        self,
        factors: List[Dict[str, Any]],
        sota_pool_list: Optional[List[str]] = None,
        origin_factor_pool_analysis_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        主处理流程：评估因子列表并更新因子池
        
        Args:
            factors: 待评估的因子列表
            sota_pool_list: 当前 SOTA 因子池（可选）
            origin_factor_pool_analysis_result: 原始因子池分析结果（可选）
            
        Returns:
            处理结果字典
        """
        print("开始因子评估")
        logs = []
        
        # 1. 筛选因子
        filtered_factors = self.filter_and_sort_factors(factors)
        print(f"筛选后因子数量: {len(filtered_factors)}")
        
        # 2. 初始化因子池
        base_pool = self.pool_manager.load_base_pool()
        if sota_pool_list is None:
            sota_pool_list = copy.deepcopy(base_pool)
            logs.append(f"初始化 sota_pool_list，包含 {len(sota_pool_list)} 个因子")
        
        # 3. 获取原始因子池评估结果
        if origin_factor_pool_analysis_result is None:
            origin_factor_pool_analysis_result = {}
            origin_result = self.get_origin_factor_pool_analysis_result(base_pool)
            if origin_result:
                origin_factor_pool_analysis_result["原始因子池的ic和rank_ic"] = {
                    "ic": origin_result.get("original_ic_analysis_result", {}),
                    "rank_ic": origin_result.get("original_rank_ic_analysis_result", {}),
                    "annualized_return_test_result": origin_result.get("original_annualized_return_test_result"),
                    "max_drawdown_test_result": origin_result.get("original_max_drawdown_test_result")
                }
        
        # 4. 逐个评估因子
        added_count = 0
        max_pool_size = 40
        top_n = min(5, len(filtered_factors))
        
        for idx in range(top_n):
            if len(sota_pool_list) >= max_pool_size:
                break
            
            factor = filtered_factors[idx]
            
            try:
                # 评估并尝试添加
                added, added_count, eval_result = self._evaluate_and_try_add(
                    factor, logs, sota_pool_list, 
                    origin_factor_pool_analysis_result, added_count
                )
                
                # 如果未通过，尝试修订
                if not added and self.llm_evaluator:
                    revised = self.revise_factor(eval_result, sota_pool_list)
                    if revised and revised.get("revised_factor_expression"):
                        factor["qlib_expression"] = revised["revised_factor_expression"]
                        added, added_count, _ = self._evaluate_and_try_add(
                            factor, logs, sota_pool_list,
                            origin_factor_pool_analysis_result, added_count
                        )
            except Exception as e:
                import traceback
                factor_expr = factor.get("qlib_expression", factor.get("expression", "未知"))
                error_msg = f"因子 {idx+1}/{top_n} 处理失败: {str(e)}"
                logs.append(f"[FactorEval] {error_msg}")
                logs.append(f"[FactorEval] 因子表达式: {factor_expr[:100]}...")
                logs.append(f"[FactorEval] 错误详情: {traceback.format_exc()}")
                print(f"⚠️ {error_msg}")
                # 继续处理下一个因子，不中断流程
                continue
        
        # 5. 检查是否需要更多因子
        if len(sota_pool_list) <= max_pool_size and len(filtered_factors) <= top_n:
            return {
                "status": "need_more_factors",
                "logs": logs,
                "filtered_factors": filtered_factors,
                "current_node": "factor_mining",
                "sota_pool_list": sota_pool_list,
                "origin_factor_pool_analysis_result": origin_factor_pool_analysis_result
            }
        
        return {
            "status": "success",
            "logs": logs,
            "filtered_factors": filtered_factors,
            "sota_pool_list": sota_pool_list,
            "origin_factor_pool_analysis_result": origin_factor_pool_analysis_result
        }
    
    # ==================== 私有方法 ====================
    
    def _parse_numeric_value(self, value) -> Optional[float]:
        """解析可能是字符串的数值"""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                if value in ("未知", "") or not value.strip():
                    return None
                return float(value)
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _register_and_run_benchmark(
        self,
        new_factor: Optional[Dict[str, Any]],
        sota_factors: List[str]
    ) -> Optional[Dict[str, Any]]:
        """注册因子池并运行基准测试"""
        # 注册因子池
        module_path = self.pool_manager.register_merged_pool(new_factor, sota_factors)
        if not module_path:
            return None
        
        # 生成 workflow 配置
        try:
            workflow_path = self.workflow_generator.generate(
                module_path=module_path,
                model_type="xgboost"
            )
        except Exception as e:
            print(f"生成 workflow 配置失败: {e}")
            return None
        
        # 运行基准测试
        try:
            result = self.mcp.run_benchmark(workflow_path)
        except Exception as e:
            print(f"运行基准测试失败: {e}")
            return None
        
        # 提取统计信息
        return {
            "module_path": module_path,
            "workflow_path": workflow_path,
            "ic_stats": FileUtils.read_pickle_stats(result.get("ic", "")),
            "rank_ic_stats": FileUtils.read_pickle_stats(result.get("rank_ic", "")),
            "annualized_return": FileUtils.read_mlflow_metric(
                result.get("1day.excess_return_with_cost.annualized_return", "")
            ),
            "max_drawdown": FileUtils.read_mlflow_metric(
                result.get("1day.excess_return_with_cost.max_drawdown", "")
            )
        }
    
    def _build_eval_result_without_llm(
        self,
        factor: Dict[str, Any],
        benchmark_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建不使用 LLM 的评估结果"""
        return {
            "should_keep": True,
            "factor_expression": factor.get("qlib_expression"),
            "ic_analysis_result": benchmark_result.get("ic_stats"),
            "rank_ic_analysis_result": benchmark_result.get("rank_ic_stats"),
            "annualized_return_test_result": benchmark_result.get("annualized_return"),
            "max_drawdown_test_result": benchmark_result.get("max_drawdown")
        }
    
    def _evaluate_with_llm(
        self,
        factor: Dict[str, Any],
        benchmark_result: Dict[str, Any],
        sota_pool_list: List[str],
        origin_stats: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """使用 LLM 进行综合评估"""
        factor_expr = factor.get("qlib_expression", "")
        
        # 准备统计数据
        new_stats = {
            "ic": benchmark_result.get("ic_stats"),
            "rank_ic": benchmark_result.get("rank_ic_stats"),
            "annualized_return": benchmark_result.get("annualized_return"),
            "max_drawdown": benchmark_result.get("max_drawdown")
        }
        
        origin_pool_stats = origin_stats if origin_stats else {}
        
        # LLM 分析
        performance_analysis = self.llm_evaluator.analyze_factor_performance(
            factor_expr, origin_pool_stats, new_stats
        )
        
        economics_explanation = self.llm_evaluator.explain_factor_economics(factor_expr)
        
        comprehensive_eval = self.llm_evaluator.evaluate_factor_comprehensive(
            factor_expr, sota_pool_list, origin_pool_stats, new_stats,
            performance_analysis, economics_explanation
        )
        
        return {
            "should_keep": convert_to_bool(comprehensive_eval.get("if_keep", False)),
            "factor_expression": factor_expr,
            "ic_analysis_result": new_stats["ic"],
            "rank_ic_analysis_result": new_stats["rank_ic"],
            "annualized_return_test_result": new_stats["annualized_return"],
            "max_drawdown_test_result": new_stats["max_drawdown"],
            "factor_analysis_response": performance_analysis,
            "factor_economic_explanation_response": economics_explanation,
            "factor_evaluation_response": comprehensive_eval
        }
    
    def _evaluate_and_try_add(
        self,
        factor: Dict[str, Any],
        logs: List[str],
        sota_pool_list: List[str],
        origin_stats: Dict[str, Any],
        added_count: int
    ) -> tuple:
        """评估因子并尝试添加到池中"""
        # 获取最新的原始池统计
        latest_origin = list(origin_stats.values())[-1] if origin_stats else None
        
        eval_result = self.evaluate_and_analyze_factor(
            factor=factor,
            logs=logs,
            sota_pool_list=sota_pool_list,
            origin_factor_pool_analysis_result=latest_origin,
            correlation_threshold=0.99
        )
        
        if eval_result.get("should_keep"):
            factor_expr = factor.get("qlib_expression", "")
            print(f"保留因子: {factor_expr}")
            
            sota_pool_list.append(factor_expr)
            self.pool_manager.save_eval_result(factor_expr, eval_result, sota_pool_list)
            
            added_count += 1
            origin_stats[f"新加入第{added_count}个因子后因子池的ic和rank_ic"] = {
                "ic": eval_result.get("ic_analysis_result"),
                "rank_ic": eval_result.get("rank_ic_analysis_result"),
                "annualized_return_test_result": eval_result.get("annualized_return_test_result"),
                "max_drawdown_test_result": eval_result.get("max_drawdown_test_result")
            }
            
            return True, added_count, eval_result
        
        return False, added_count, eval_result

