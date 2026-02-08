"""
Factor Evaluation Agent (Refactored)

This is a streamlined orchestration class responsible for coordinating components to complete factor evaluation workflow.

Separation of responsibilities:
- MCPService: MCP client communication
- FactorPoolManager: Factor pool loading/saving/registration
- CorrelationEvaluator: Factor correlation checking
- WorkflowConfigGenerator: Workflow configuration generation
- LLMFactorEvaluator: LLM factor analysis and evaluation
- FileUtils: File operation utilities

Mining feedback generation (LLM-driven):
- _generate_mining_feedback: Main entry, generates guidance feedback for next round of GP mining
- _build_pool_report: Builds factor pool analysis report (current factors + historical attribution)
- _llm_analyze_and_suggest: LLM analyzes weaknesses and generates seed direction suggestions
"""
import copy
import random
import re
import json
from typing import Dict, Any, List, Optional
from collections import Counter

# Import components
from .services.mcp_service import MCPService
from .services.factor_pool_manager import FactorPoolManager
from .evaluators.correlation_evaluator import CorrelationEvaluator
from .evaluators.llm_evaluator import LLMFactorEvaluator, convert_to_bool
from .generators.workflow_config_generator import WorkflowConfigGenerator
from .utils.file_utils import FileUtils


class FactorEvalAgent:
    """
    Factor Evaluation Agent
    
    Main functions:
    1. Filter and sort factors
    2. Check correlation between factors and SOTA factor pool
    3. Register merged factor pool and run benchmark tests
    4. Use LLM for factor evaluation
    5. Manage factor pool additions and deletions
    """
    
    def __init__(self, llm_service, mcp_server_path: Optional[str] = None):
        """
        Initialize Factor Evaluation Agent
        
        Args:
            llm_service: LLM service instance (needs call and parse_json_response methods)
            mcp_server_path: MCP server script path
        """
        self.llm = llm_service
        
        # Initialize components
        self.mcp = MCPService(mcp_server_path)
        self.pool_manager = FactorPoolManager()
        self.correlation_evaluator = CorrelationEvaluator()
        self.workflow_generator = WorkflowConfigGenerator()
        self.llm_evaluator = LLMFactorEvaluator(llm_service) if llm_service else None
    
    # ==================== Public API ====================
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tool list"""
        return self.mcp.list_tools()
    
    def call_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Call MCP tool"""
        return self.mcp.call_tool(tool_name, arguments)
    
    def filter_and_sort_factors(
        self, 
        factors: List[Dict[str, Any]], 
        ic_threshold: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Filter and sort factors
        
        Args:
            factors: Factor list, format: [{"expression": "...", "ic": IC_value, "rank_ic_valid": Rank_IC_value}, ...]
            ic_threshold: IC threshold, only keep factors with IC greater than this value
            
        Returns:
            Filtered and sorted factor list, sorted by rank_ic_valid in descending order
        """
        if not factors:
            return []
        
        filtered = []
        for factor in factors:
            if not isinstance(factor, dict):
                continue
            
            # Parse IC value
            ic = self._parse_numeric_value(factor.get("ic"))
            if ic is None or ic <= ic_threshold:
                continue
            
            # Parse rank_ic_valid
            rank_ic = self._parse_numeric_value(factor.get("rank_ic_valid"))
            if rank_ic is None or rank_ic < 0:
                continue
            
            filtered.append(factor)
        
        # Sort by rank_ic_valid in descending order
        return sorted(
            filtered,
            key=lambda f: self._parse_numeric_value(f.get("rank_ic_valid")) or float('-inf'),
            reverse=True
        )
    
    def get_base_pool(self, path: Optional[str] = None) -> List[str]:
        """Get base factor pool"""
        return self.pool_manager.load_base_pool(path)
    
    def append_factor_to_sota_pool(self, factor: str, path: Optional[str] = None) -> bool:
        """Append factor to SOTA factor pool"""
        return self.pool_manager.append_to_sota_pool(factor, path)
    
    def get_origin_factor_pool_analysis_result(
        self, 
        base_pool_list: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get original factor pool evaluation results"""
        result = self._register_and_run_benchmark(
            new_factor=None,
            sota_factors=base_pool_list
        )
        #print(f"origin_factor_pool_analysis_result: {result}")
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
        Evaluate a single factor
        
        Args:
            factor: Factor dictionary to evaluate
            logs: Log list
            sota_pool_list: SOTA factor pool list
            origin_factor_pool_analysis_result: Original factor pool statistics
            correlation_threshold: Correlation threshold
            
        Returns:
            Evaluation result dictionary
        """
        # 1. Check correlation with SOTA factors
        should_keep = self.correlation_evaluator.check_correlation(
            factor,
            sota_pool_list,
            threshold=correlation_threshold
        )
        
        if not should_keep:
            logs.append("Factor highly correlated with SOTA factors, removed")
            return {"should_keep": False}
        
        logs.append("Factor passed correlation check, kept")
        
        # 2. Register factor pool and run benchmark test
        benchmark_result = self._register_and_run_benchmark(factor, sota_pool_list)
        #print(f"benchmark_result: {benchmark_result}")
        if not benchmark_result:
            logs.append("Warning: Factor pool registration or benchmark test failed")
            return {"should_keep": False, "registered_module_path": None}
        
        logs.append(f"Factor pool registered: {benchmark_result.get('module_path')}")
        
        # 3. If no LLM service, directly return benchmark test results
        if not self.llm_evaluator:
            return self._build_eval_result_without_llm(factor, benchmark_result)
        
        # 4. Use LLM for comprehensive evaluation
        return self._evaluate_with_llm(
            factor, benchmark_result, sota_pool_list, origin_factor_pool_analysis_result
        )
    
    def revise_factor(
        self, 
        eval_result: Dict[str, Any], 
        sota_pool_list: List[str]
    ) -> Dict[str, Any]:
        """Revise factor expression based on evaluation results"""
        if not self.llm_evaluator:
            return {}
        return self.llm_evaluator.revise_factor(eval_result, sota_pool_list)
    
    def process(
        self,
        factors: List[Dict[str, Any]],
        sota_pool_list: Optional[List[str]] = None,
        factor_pool_analysis_result_history: Optional[List[Dict[str, Any]]] = None,
        previous_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main processing flow: evaluate factor list and update factor pool
        
        Args:
            factors: List of factors to evaluate
            sota_pool_list: Current SOTA factor pool (optional)
            factor_pool_analysis_result_history: Factor pool analysis result history (optional)
            previous_feedback: Previous round's mining feedback (optional, for iterative optimization)
            
        Returns:
            Processing result dictionary
        """
        print("Starting factor evaluation")
        logs = []
        
        # 1. Filter factors
        filtered_factors = self.filter_and_sort_factors(factors)
        print(f"Number of factors after filtering: {len(filtered_factors)}")
        
        # 2. Initialize factor pool
        base_pool = self.pool_manager.load_base_pool()
        if sota_pool_list is None:
            sota_pool_list = copy.deepcopy(base_pool)
            logs.append(f"Initialized sota_pool_list, contains {len(sota_pool_list)} factors")
        
        # 3. Get original factor pool evaluation results
        if factor_pool_analysis_result_history is None:
            origin_factor_pool_analysis_result = {}
            factor_pool_analysis_result_history = []
            origin_result = self.get_origin_factor_pool_analysis_result(base_pool)
            #print(f"origin_result: {origin_result}")
            if origin_result:
                origin_factor_pool_analysis_result= {
                    "ic": origin_result.get("original_ic_analysis_result", {}),
                    "rank_ic": origin_result.get("original_rank_ic_analysis_result", {}),
                    "annualized_return_test_result": origin_result.get("original_annualized_return_test_result"),
                    "max_drawdown_test_result": origin_result.get("original_max_drawdown_test_result")
                }
                factor_pool_analysis_result_history.append({"stage_000__add": origin_factor_pool_analysis_result})
        print(f"factor_pool_analysis_result_history: {factor_pool_analysis_result_history}")
        # 4. Evaluate factors one by one
        added_count = 0
        max_pool_size = 40
        top_n = min(5, len(filtered_factors))
        
        for idx in range(top_n):
            if len(sota_pool_list) >= max_pool_size:
                break
            
            factor = filtered_factors[idx]
            
            try:
                # Evaluate and try to add
                added, added_count, eval_result = self._evaluate_and_try_add(
                    factor, logs, sota_pool_list, 
                    factor_pool_analysis_result_history, added_count
                )
                
                # If not passed, try revision
                if not added and self.llm_evaluator:
                    try:
                        revised = self.revise_factor(eval_result, sota_pool_list)
                        
                        # Handle case where revised might be a list
                        if isinstance(revised, list) and len(revised) > 0:
                            # If returns list, try each revision version
                            for revised_item in revised:
                                if isinstance(revised_item, dict) and revised_item.get("revised_factor_expression"):
                                    factor["qlib_expression"] = revised_item["revised_factor_expression"]
                                    added, added_count, _ = self._evaluate_and_try_add(
                                        factor, logs, sota_pool_list,
                                        factor_pool_analysis_result_history, added_count
                                    )
                                    if added:
                                        break  # If successfully added, stop trying other revision versions
                        elif isinstance(revised, dict) and revised.get("revised_factor_expression"):
                            # If returns dict, use directly
                            factor["qlib_expression"] = revised["revised_factor_expression"]
                            added, added_count, _ = self._evaluate_and_try_add(
                                factor, logs, sota_pool_list,
                                factor_pool_analysis_result_history, added_count
                            )
                    except Exception as e:
                        # Factor revision failed, log error but continue processing next factor
                        error_msg = f"[FactorEval] Factor {idx+1} revision failed: {str(e)}"
                        print(error_msg)
                        logs.append(error_msg)
                        import traceback
                        traceback.print_exc()
                        
            except Exception as e:
                # Factor evaluation failed, log error but continue processing next factor
                error_msg = f"[FactorEval] Factor {idx+1} evaluation failed: {str(e)}"
                print(error_msg)
                logs.append(error_msg)
                import traceback
                traceback.print_exc()
                continue  # Continue processing next factor
        
        # 5. Check if more factors are needed
        if len(sota_pool_list) <= max_pool_size:
            # Generate mining feedback to guide next round of GP mining
            mining_feedback = self._generate_mining_feedback(
                sota_pool_list=sota_pool_list,
                factor_pool_analysis_result_history=factor_pool_analysis_result_history,
                previous_feedback=previous_feedback
            )
            print(f"mining_feedback: {mining_feedback}")
            
            # Save mining feedback to file
            feedback_path = self.pool_manager.save_mining_feedback(
                mining_feedback=mining_feedback,
                sota_pool_list=sota_pool_list
            )
            if feedback_path:
                logs.append(f"[FactorEval] Mining feedback saved: {feedback_path}")
            
            return {
                "status": "need_more_factors",
                "logs": logs,
                "current_node": "factor_mining",
                "sota_pool_list": sota_pool_list,
                "factor_pool_analysis_result_history": factor_pool_analysis_result_history,
                "mining_feedback": mining_feedback,
                "mining_feedback_path": feedback_path
            }
        
        return {
            "status": "success",
            "logs": logs,
            "sota_pool_list": sota_pool_list,
            "factor_pool_analysis_result_history": factor_pool_analysis_result_history
        }
    
    # ==================== Private Methods ====================
    
    def _parse_numeric_value(self, value) -> Optional[float]:
        """Parse numeric value that might be a string"""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                if value in ("Unknown", "") or not value.strip():
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
        """Register factor pool and run benchmark test"""
        # Register factor pool
        module_path = self.pool_manager.register_merged_pool(new_factor, sota_factors)
        if not module_path:
            return None
        
        # Generate workflow configuration
        try:
            workflow_path = self.workflow_generator.generate(
                module_path=module_path,
                model_type="xgboost"
            )
        except Exception as e:
            print(f"Failed to generate workflow configuration: {e}")
            return None
        
        # Run benchmark test
        try:
            result = self.mcp.run_benchmark(workflow_path)
            #print(f"result: {result}")
        except Exception as e:
            print(f"Failed to run benchmark test: {e}")
            return None
        
        # Extract statistics
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
        """Build evaluation result without LLM"""
        return {
            "should_keep": False,
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
        """Use LLM for comprehensive evaluation"""
        factor_expr = factor.get("qlib_expression", "")
        
        # Prepare statistics
        new_stats = {
            "ic": benchmark_result.get("ic_stats"),
            "rank_ic": benchmark_result.get("rank_ic_stats"),
            "annualized_return": benchmark_result.get("annualized_return"),
            "max_drawdown": benchmark_result.get("max_drawdown")
        }
        
        #origin_pool_stats = origin_stats if origin_stats else {}
        origin_pool_stats = {}
        if origin_stats:
            # origin_stats format is {"stage_XXX__YYY": {...metrics...}}
            # Need to extract the actual metrics dict first
            if isinstance(origin_stats, dict) and len(origin_stats) == 1:
                # Extract the unique value (metrics dict)
                metrics_dict = next(iter(origin_stats.values()))
            else:
                # Compatible with old format: if origin_stats is already a metrics dict
                metrics_dict = origin_stats
            
            origin_pool_stats = {
                "ic": metrics_dict.get("ic"),
                "rank_ic": metrics_dict.get("rank_ic"),
                "annualized_return": metrics_dict.get("annualized_return_test_result"),
                "max_drawdown": metrics_dict.get("max_drawdown_test_result")
            }
        # LLM analysis
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
    
    # ==================== Knowledge Base Literature Search (MCP-driven) ====================
    
    def _call_knowledge_base_search(
        self,
        query: str,
        top_k: int = 5,
        year_start: Optional[int] = 2018,
        year_end: Optional[int] = 2025
    ) -> Dict[str, Any]:
        """
        Call knowledge base search tool to find academic literature related to query
        
        Reference FactorMiningAgent._call_gp_training style implementation
        
        Args:
            query: English search query (recommended to use English for best results)
            top_k: Number of results to return, default 5
            year_start: Start year, default 2018
            year_end: End year, default 2025
            
        Returns:
            dict: Dictionary containing retrieval results, format:
                - status: "success" or "error"
                - query: Original query
                - total_found: Number of results found
                - results: Result list, each item contains title, journal, text, score, etc.
        """
        # Build parameters
        search_params = {
            "query": query,
            "top_k": top_k,
            "year_start": year_start or 0,
            "year_end": year_end or 0
        }
        
        print(f"[FactorEvalAgent] 📚 Calling search_papers knowledge base search")
        print(f"[FactorEvalAgent] Query: {query[:100]}...")
        
        # Call MCP tool
        try:
            result_text = self.mcp.call_tool("search_papers", search_params)
            
            # Parse JSON result
            if isinstance(result_text, str):
                result = json.loads(result_text)
            else:
                result = result_text
            
            found_count = result.get("total_found", 0)
            print(f"[FactorEvalAgent] ✅ Literature search completed, found {found_count} related papers")
            
            return result
            
        except Exception as e:
            print(f"[FactorEvalAgent] ❌ Knowledge base search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "total_found": 0,
                "results": []
            }
    
    def _build_factor_search_query(self, factor_expr: str) -> str:
        """
        Build English search query from factor expression
        
        Map operators and features in Qlib factor expression to financial academic terminology
        
        Args:
            factor_expr: Qlib factor expression, e.g. "Corr($close, $volume, 5)"
            
        Returns:
            str: English search query
        """
        # Operator to academic terminology mapping (based on qlib_operators.json)
        operator_mapping = {
            # element_wise operators
            "Abs": "absolute value",
            "Sign": "sign indicator",
            "Log": "logarithm",
            "Not": "logical negation",
            "Mask": "masking filtering",
            "ChangeInstrument": "cross-asset",
            
            # binary operators
            "Add": "addition",
            "Sub": "subtraction difference",
            "Mul": "multiplication",
            "Div": "division ratio",
            "Power": "power exponentiation",
            "Greater": "maximum selection",
            "Less": "minimum selection",
            "Gt": "comparison",
            "Ge": "comparison",
            "Lt": "comparison",
            "Le": "comparison",
            "Eq": "comparison",
            "Ne": "comparison",
            "And": "logical and",
            "Or": "logical or",
            
            # ternary operators
            "If": "conditional selection",
            
            # rolling operators
            "Ref": "lagged value",
            "Mean": "moving average",
            "Sum": "rolling sum",
            "Std": "volatility standard deviation",
            "Var": "variance",
            "Skew": "skewness",
            "Kurt": "kurtosis",
            "Max": "rolling maximum",
            "Min": "rolling minimum",
            "Med": "median",
            "Mad": "mean absolute deviation",
            "Rank": "time series ranking percentile",
            "Quantile": "quantile percentile",
            "Count": "count non-null",
            "Delta": "momentum change",
            "Slope": "trend slope regression",
            "Rsquare": "r-squared coefficient of determination",
            "Resi": "regression residual",
            "WMA": "weighted moving average",
            "EMA": "exponential moving average",
            "IdxMax": "index of maximum",
            "IdxMin": "index of minimum",
            
            # pair_rolling operators
            "Corr": "correlation",
            "Cov": "covariance",
            
            # time_resample operators
            "TResample": "time resampling temporal aggregation",
            
            # pit_operators
            "P": "point-in-time aggregation",
            "PRef": "point-in-time reference"
        }
        
        # Feature to academic terminology mapping
        feature_mapping = {
            "$close": "stock price",
            "$open": "opening price",
            "$high": "high price",
            "$low": "low price",
            "$volume": "trading volume",
            "$vwap": "volume weighted average price",
            "$turn": "turnover rate",
            "$amount": "trading amount",
            "$factor": "adjustment factor"
        }
        
        # Extract operators from factor
        found_operators = []
        for op, term in operator_mapping.items():
            if op in factor_expr:
                found_operators.append(term)
        
        # Extract features from factor
        found_features = []
        for feat, term in feature_mapping.items():
            if feat in factor_expr:
                found_features.append(term)
        
        # Detect time window (numbers usually represent days)
        import re
        windows = re.findall(r'\b(\d+)\b', factor_expr)
        time_horizon = ""
        if windows:
            max_window = max(int(w) for w in windows)
            if max_window <= 5:
                time_horizon = "short-term"
            elif max_window <= 20:
                time_horizon = "medium-term"
            else:
                time_horizon = "long-term"
        
        # Build query
        query_parts = []
        
        # Add core concept
        query_parts.append("stock market alpha factor")
        
        # Add operator-related terms
        if found_operators:
            query_parts.extend(found_operators[:3])  # Take at most 3
        
        # Add feature-related terms
        if found_features:
            query_parts.extend(found_features[:2])  # Take at most 2
        
        # Add time range
        if time_horizon:
            query_parts.append(time_horizon)
        
        # Add general financial terms
        query_parts.append("quantitative trading")
        
        query = " ".join(query_parts)
        
        print(f"[FactorEvalAgent] 🔍 Factor expression: {factor_expr}")
        print(f"[FactorEvalAgent] 🔍 Generated search query: {query}")
        
        return query
    
    def _explain_factor_with_literature(
        self,
        factor_expr: str,
        eval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine literature query results, use LLM to explain factor's economic and financial rationality
        
        Args:
            factor_expr: Factor expression
            eval_result: Factor evaluation result
            
        Returns:
            dict: Factor explanation with literature support
                - search_query: Search query used
                - literature_results: Related literature found
                - literature_explanation: LLM explanation combined with literature
                - academic_support_level: Literature support level (high/medium/low)
        """
        # 1. Build search query
        search_query = self._build_factor_search_query(factor_expr)
        
        # 2. Call knowledge base search
        search_result = self._call_knowledge_base_search(
            query=search_query,
            top_k=5,
            year_start=2018,
            year_end=2025
        )
        
        literature_results = []
        if search_result.get("status") == "success":
            literature_results = search_result.get("results", [])
        
        # 3. If no LLM service, directly return literature results
        if not self.llm:
            return {
                "search_query": search_query,
                "literature_results": literature_results,
                "literature_explanation": None,
                "academic_support_level": "unknown"
            }
        
        # 4. Use LLM combined with literature to generate explanation
        literature_explanation = self._llm_explain_with_literature(
            factor_expr=factor_expr,
            literature_results=literature_results,
            eval_result=eval_result
        )
        
        # 5. Assess literature support level
        academic_support_level = self._assess_literature_support(
            literature_results=literature_results,
            explanation=literature_explanation
        )
        
        return {
            "search_query": search_query,
            "literature_results": literature_results,
            "literature_explanation": literature_explanation,
            "academic_support_level": academic_support_level
        }
    
    def _llm_explain_with_literature(
        self,
        factor_expr: str,
        literature_results: List[Dict[str, Any]],
        eval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM combined with literature results to explain factor's economic rationality
        
        Args:
            factor_expr: Factor expression
            literature_results: Literature search results
            eval_result: Factor evaluation result
            
        Returns:
            dict: LLM-generated explanation
        """
        # Format literature information
        literature_text = ""
        if literature_results:
            for i, paper in enumerate(literature_results[:5], 1):
                literature_text += f"\n### Paper {i}: {paper.get('title', 'Unknown')}\n"
                literature_text += f"- Journal: {paper.get('journal', 'Unknown')}\n"
                literature_text += f"- Publication time: {paper.get('publish_time', 'Unknown')}\n"
                literature_text += f"- Relevance score: {paper.get('score', 0):.3f}\n"
                literature_text += f"- Abstract: {paper.get('text', '')[:500]}...\n"
        else:
            literature_text = "No directly related academic literature found."
        
        # Format factor performance
        performance_text = ""
        if eval_result:
            ic_stats = eval_result.get("ic_analysis_result", {})
            performance_text = f"""
- IC mean: {ic_stats.get('mean', 'N/A')}
- IC std: {ic_stats.get('std', 'N/A')}
- Annualized return: {eval_result.get('annualized_return_test_result', 'N/A')}
- Max drawdown: {eval_result.get('max_drawdown_test_result', 'N/A')}
"""
        
        prompt = f"""You are a quantitative finance research expert. Please provide economic and financial theoretical explanations for this quantitative factor based on the following academic literature and factor performance data.

## Factor Expression
```
{factor_expr}
```

## Factor Performance in Backtest
{performance_text}

## Related Academic Literature
{literature_text}

## Task
Please complete the following analysis:

1. **Factor's Economic Principle**: Explain the economic/financial principle behind this factor, why might this pattern have predictive power?

2. **Literature Support**: 
   - Is there research in the above literature that directly supports the effectiveness of this factor?
   - How do related theories mentioned in the literature (such as behavioral finance, market microstructure, etc.) explain this factor?

3. **Potential Risks and Limitations**:
   - Based on academic research, under what conditions might this type of factor fail?
   - Are there risks of factor crowding or overtrading?

4. **Improvement Suggestions**:
   - Based on literature insights, are there ideas for improving this factor?

Please output in JSON format:
```json
{{
    "economic_principle": "Core economic/financial principle explanation of the factor",
    "literature_support": "How literature supports the effectiveness of this factor",
    "related_theories": ["Related financial theory 1", "Theory 2"],
    "potential_risks": ["Risk 1", "Risk 2"],
    "improvement_suggestions": ["Improvement suggestion 1", "Suggestion 2"],
    "confidence_level": "high/medium/low (based on literature support level)"
}}
```
"""
        
        try:
            response = self.llm.call(prompt=prompt, stream=False)
            result = self.llm.parse_json_response(response)
            return result or {}
        except Exception as e:
            print(f"[FactorEvalAgent] LLM literature explanation failed: {e}")
            return {}
    
    def _assess_literature_support(
        self,
        literature_results: List[Dict[str, Any]],
        explanation: Dict[str, Any]
    ) -> str:
        """
        Assess literature support level for factor
        
        Args:
            literature_results: Literature search results
            explanation: LLM-generated explanation
            
        Returns:
            str: Support level - "high", "medium", "low"
        """
        # Judge based on literature count and relevance scores
        if not literature_results:
            return "low"
        
        # Calculate average relevance score
        scores = [paper.get("score", 0) for paper in literature_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Combine with LLM confidence judgment
        llm_confidence = explanation.get("confidence_level", "medium")
        
        # Comprehensive judgment
        if avg_score > 0.7 and llm_confidence == "high":
            return "high"
        elif avg_score > 0.5 or llm_confidence in ["high", "medium"]:
            return "medium"
        else:
            return "low"
    
    def _evaluate_and_try_add(
        self,
        factor: Dict[str, Any],
        logs: List[str],
        sota_pool_list: List[str],
        factor_pool_analysis_result_history: List[Dict[str, Any]],
        added_count: int
    ) -> tuple:
        """Evaluate factor and try to add to pool"""
        # Get latest original pool statistics
        latest_origin = factor_pool_analysis_result_history[-1] if factor_pool_analysis_result_history else None
        
        eval_result = self.evaluate_and_analyze_factor(
            factor=factor,
            logs=logs,
            sota_pool_list=sota_pool_list,
            origin_factor_pool_analysis_result=latest_origin,
            correlation_threshold=0.99
        )
        
        if eval_result.get("should_keep"):
            factor_expr = factor.get("qlib_expression", "")
            print(f"Keeping factor: {factor_expr}")
            
            sota_pool_list.append(factor_expr)
            self.pool_manager.save_eval_result(factor_expr, eval_result, sota_pool_list)
            
            # === After factor successfully added, call literature search to explain its economic and financial rationality ===
            print(f"[FactorEvalAgent] 📖 Starting literature search to explain factor economic rationality...")
            literature_support = self._explain_factor_with_literature(
                factor_expr=factor_expr,
                eval_result=eval_result
            )
            logs.append(f"Literature support level: {literature_support.get('academic_support_level', 'unknown')}")
            
            if literature_support.get("literature_explanation"):
                explanation = literature_support["literature_explanation"]
                logs.append(f"Economic principle: {explanation.get('economic_principle', 'N/A')[:100]}...")
                print(f"[FactorEvalAgent] ✅ Literature explanation completed, support level: {literature_support.get('academic_support_level')}")
            
            # Save literature support information to file
            literature_path = self.pool_manager.save_literature_support(
                factor_expr=factor_expr,
                literature_support=literature_support,
                eval_result=eval_result,
                sota_pool_list=sota_pool_list
            )
            if literature_path:
                logs.append(f"[FactorEval] Literature support information saved: {literature_path}")
            
            added_count += 1
            factor_pool_analysis_result_history.append({f"stage_{added_count:03d}__add": {
                "ic": eval_result.get("ic_analysis_result"),
                "rank_ic": eval_result.get("rank_ic_analysis_result"),
                "annualized_return_test_result": eval_result.get("annualized_return_test_result"),
                "max_drawdown_test_result": eval_result.get("max_drawdown_test_result"),
                "factor_expression": factor_expr,
                "factor_economic_explanation_response": eval_result.get("factor_economic_explanation_response"),
                "factor_evaluation_response": eval_result.get("factor_evaluation_response"),
                "factor_analysis_response": eval_result.get("factor_analysis_response"),
                # === New: Literature support information ===
                "literature_support": literature_support
            }})
            
            return True, added_count, eval_result
        
        return False, added_count, eval_result
    
    # ==================== Mining Feedback Generation (LLM-driven) ====================
    
    def _generate_mining_feedback(
        self,
        sota_pool_list: List[str],
        factor_pool_analysis_result_history: List[Dict[str, Any]],
        previous_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate factor mining feedback to guide next round of GP mining
        
        Core: Based on historical data in factor_pool_analysis_result_history,
        analyze which factors improved returns, what weaknesses the current factor pool has, then let LLM generate seed directions
        """
        iteration = (previous_feedback.get("iteration", 0) + 1) if previous_feedback else 1
        
        # Build factor pool analysis report (for LLM to read)
        pool_report = self._build_pool_report(sota_pool_list, factor_pool_analysis_result_history)
        
        # LLM analyzes and generates suggestions
        llm_result = {}
        if self.llm:
            llm_result = self._llm_analyze_and_suggest(pool_report)
        
        # Calculate convergence information
        convergence_info = self._compute_convergence_info(
            pool_size=len(sota_pool_list),
            history=factor_pool_analysis_result_history,
            previous_feedback=previous_feedback
        )
        
        return {
            "iteration": iteration,
            "pool_report": pool_report,
            "pool_weaknesses": llm_result.get("pool_weaknesses", []),
            "suggested_directions": llm_result.get("suggested_directions", []),
            "suggested_seeds": llm_result.get("suggested_seeds", []),
            "gp_strategy_hints": llm_result.get("gp_strategy_hints", {}),
            "convergence_info": convergence_info
        }
    
    def _build_pool_report(
        self, 
        sota_pool_list: List[str], 
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Build factor pool analysis report (text format, for LLM analysis)
        
        Contains:
        1. All factors in current factor pool
        2. Performance changes and LLM analysis results for each newly added factor
        3. Overall return change trends
        """
        lines = []
        
        # === Part 1: Current factor pool ===
        lines.append("## I. Current SOTA Factor Pool")
        lines.append(f"Total {len(sota_pool_list)} factors:")
        for i, expr in enumerate(sota_pool_list, 1):
            lines.append(f"  {i}. {expr}")
        lines.append("")
        
        # === Part 2: Factor pool evolution history (attribution analysis) ===
        lines.append("## II. Factor Pool Evolution and Return Attribution")
        
        prev_ic, prev_return = None, None
        for item in history:
            if not isinstance(item, dict) or not item:
                continue

            stage, data = next(iter(item.items()))  # stage is "stage_001__add", data is the metrics dict

            ic_stats = data.get("ic", {})
            ic_mean = ic_stats.get("mean", 0) if isinstance(ic_stats, dict) else 0
            ann_return = data.get("annualized_return_test_result")
            max_dd = data.get("max_drawdown_test_result")
            factor_expr = data.get("factor_expression", "")
            economic_reason = data.get("factor_economic_explanation_response", "")

            lines.append(f"\n### {stage}")
            
            # If there's a factor expression, this is a newly added factor
            if factor_expr:
                lines.append(f"- New factor: `{factor_expr}`")
                if economic_reason:
                    # Extract summary of economic explanation (avoid too long)
                    reason_summary = str(economic_reason)[:200] + "..." if len(str(economic_reason)) > 200 else str(economic_reason)
                    lines.append(f"- Economic explanation: {reason_summary}")
            
            # Metrics
            lines.append(f"- Factor pool IC: {ic_mean:.4f}" + (f" (change: {ic_mean - prev_ic:+.4f})" if prev_ic else ""))
            if ann_return is not None:
                lines.append(f"- Annualized return: {ann_return:.2%}" + (f" (change: {ann_return - prev_return:+.2%})" if prev_return else ""))
            if max_dd is not None:
                lines.append(f"- Max drawdown: {max_dd:.2%}")
            
            prev_ic, prev_return = ic_mean, ann_return
        
        # === Part 3: Summary ===
        lines.append("\n## III. Summary")
        if history:
            # history is List[Dict[str, Any]], each element is {"stage_XXX__YYY": {...metrics...}}
            # Need to extract data (metrics dict) from each item
            stages_data = []
            for item in history:
                if isinstance(item, dict) and item:
                    # Extract metrics dict
                    data = next(iter(item.values()))
                    stages_data.append(data)
            
            if len(stages_data) >= 2:
                first_ic = stages_data[0].get("ic", {}).get("mean", 0) if isinstance(stages_data[0].get("ic"), dict) else 0
                last_ic = stages_data[-1].get("ic", {}).get("mean", 0) if isinstance(stages_data[-1].get("ic"), dict) else 0
                first_ret = stages_data[0].get("annualized_return_test_result")
                last_ret = stages_data[-1].get("annualized_return_test_result")
                
                lines.append(f"- IC total change: {first_ic:.4f} → {last_ic:.4f} ({last_ic - first_ic:+.4f})")
                if first_ret and last_ret:
                    lines.append(f"- Annualized return total change: {first_ret:.2%} → {last_ret:.2%} ({last_ret - first_ret:+.2%})")
                
                # Count actual number of newly added factors (entries with factor_expression)
                added_factors_count = sum(1 for data in stages_data if data.get("factor_expression"))
                lines.append(f"- Number of new factors: {added_factors_count}")
        
        return "\n".join(lines)
    
    def _llm_analyze_and_suggest(self, pool_report: str) -> Dict[str, Any]:
        """
        Let LLM analyze factor pool report, identify weaknesses and generate seed direction suggestions
        """
        from .prompts import FACTOR_POOL_ANALYSIS_PROMPT
        
        prompt = FACTOR_POOL_ANALYSIS_PROMPT.format(pool_report=pool_report)
        
        try:
            response = self.llm.call(prompt=prompt, stream=False)
            return self.llm.parse_json_response(response) or {}
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return {}
    
    def _compute_convergence_info(
        self,
        pool_size: int,
        history: List[Dict[str, Any]],
        previous_feedback: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate convergence information, determine if iteration should continue"""
        convergence_info = {
            "current_pool_size": pool_size,
            "is_converged": False,
            "convergence_reason": None
        }
        
        if not previous_feedback:
            return convergence_info
        
        prev_pool_size = previous_feedback.get("convergence_info", {}).get("current_pool_size", 0)
        iteration = previous_feedback.get("iteration", 0)
        
        # Condition 1: Factor pool size hasn't increased for two consecutive rounds
        if pool_size <= prev_pool_size and iteration >= 2:
            convergence_info["is_converged"] = True
            convergence_info["convergence_reason"] = "No new factors added in consecutive iterations"
        
        # Condition 2: Reached maximum iteration count
        if iteration >= 5:
            convergence_info["is_converged"] = True
            convergence_info["convergence_reason"] = "Reached maximum iteration count"
        
        return convergence_info

