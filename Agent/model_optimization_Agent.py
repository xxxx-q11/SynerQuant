"""Model Optimization Agent - Self-iterative tuning"""
import sys
import os
import json
import yaml
import copy
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from utils.file_process import (
    explore_repo_structure, find_training_scripts, select_training_script,
    read_file_for_llm, find_readme_files, get_top_factors_from_gp_json
)
from Agent.prompts import FACTOR_MINING_SYSTEM_PROMPT, FACTOR_MINING_ANALYSIS_PROMPT
from Agent.utils.file_utils import FileUtils

try:
    from utils.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP client unavailable")


# Tunable parameter range definition (model parameters only)
TUNABLE_PARAMS = {
    "model": {
        "TransformerModel": {
            # Model architecture parameters
            "d_model": {"type": "int", "range": [32, 64, 128, 256], "default": 64},
            "nhead": {"type": "int", "range": [2, 4, 8], "default": 2},  # Note: it's nhead not n_head
            "num_layers": {"type": "int", "range": [1, 2, 3, 4], "default": 2},
            "dropout": {"type": "float", "range": [0.0, 0.1, 0.2, 0.3], "default": 0.0},
            
            # Training hyperparameters
            "n_epochs": {"type": "int", "range": [50, 100, 150, 200], "default": 100},
            "lr": {"type": "float", "range": [0.0001, 0.0005, 0.001, 0.002], "default": 0.0001},
            "batch_size": {"type": "int", "range": [4096, 8192, 16384], "default": 8192},
            "early_stop": {"type": "int", "range": [3, 5, 10], "default": 5},
            "reg": {"type": "float", "range": [1e-4, 1e-3, 1e-2], "default": 1e-3},
        },
        "XGBModel": {
            "n_estimators": {"type": "int", "range": [200, 400, 647, 800, 1000], "default": 647},
            "max_depth": {"type": "int", "range": [4, 6, 8, 10], "default": 8},
            "eta": {"type": "float", "range": [0.01, 0.03, 0.05, 0.1], "default": 0.0421},
            "subsample": {"type": "float", "range": [0.7, 0.8, 0.9, 1.0], "default": 0.8789},
            "colsample_bytree": {"type": "float", "range": [0.7, 0.8, 0.9, 1.0], "default": 0.8879},
            "alpha": {"type": "float", "range": [0, 50, 100, 200], "default": 50.0},
            "lambda": {"type": "float", "range": [100, 200, 400, 600], "default": 150.0},
        },
        "LGBModel": {
            "num_leaves": {"type": "int", "range": [63, 127, 210, 255], "default": 210},
            "max_depth": {"type": "int", "range": [4, 6, 8, 10], "default": 8},
            "learning_rate": {"type": "float", "range": [0.05, 0.1, 0.15, 0.2], "default": 0.2},
            "subsample": {"type": "float", "range": [0.7, 0.8, 0.9], "default": 0.8789},
            "colsample_bytree": {"type": "float", "range": [0.7, 0.8, 0.9], "default": 0.8879},
            "lambda_l1": {"type": "float", "range": [0, 100, 200, 300], "default": 205.6999},
            "lambda_l2": {"type": "float", "range": [200, 400, 600, 800], "default": 580.9768},
        }
    }
}


class ModelOptimizationAgent:
    """Model Optimization Agent - Self-iterative model parameter tuning"""
    
    def __init__(self, llm_service, mcp_server_path: Optional[str] = None):
        """
        Initialize Model Optimization Agent
        
        Args:
            llm_service: LLM service instance (BaseAgent)
            mcp_server_path: MCP server script path
        """
        self.llm = llm_service
        self.mcp_client = None
        
        # Configuration paths
        self.template_path = Path("Qlib_MCP/workspace/qlib_benchmark/benchmarks/train_temp/template_model.yaml")
        self.output_dir = Path("Qlib_MCP/workspace/qlib_benchmark/benchmarks/train_temp")
        
        # Initialize MCP client
        self._init_mcp_client(mcp_server_path)
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
    
    def _init_mcp_client(self, mcp_server_path: Optional[str]) -> None:
        """Initialize MCP client"""
        if not MCP_AVAILABLE:
            return
        
        if mcp_server_path is None:
            current_dir = Path(__file__).parent.parent
            mcp_server_path = current_dir / "Qlib_MCP" / "mcp_server_inline.py"
        else:
            mcp_server_path = Path(mcp_server_path)
        
        if not mcp_server_path.exists():
            print(f"Warning: MCP server script does not exist: {mcp_server_path}")
            return
        
        try:
            self.mcp_client = SyncMCPClient(str(mcp_server_path))
            print(f"MCP client initialized successfully: {mcp_server_path}")
        except Exception as e:
            print(f"Warning: MCP client initialization failed: {e}")

    # ==================== Public API ====================
    
    def process(
        self,
        factors: list,
        max_iterations: int = 10,
        target_annualized_return: float = 0.30,
        target_max_drawdown: float = -0.10
    ) -> Dict[str, Any]:
        """
        Perform model optimization based on existing factor pool (self-iterative tuning)
        
        Args:
            factors: Factor list, format: [{"expression": "factor expression", "ic": IC_value}, ...]
            max_iterations: Maximum number of iterations
            target_annualized_return: Target annualized return
            target_max_drawdown: Target maximum drawdown
            
        Returns:
            Dictionary containing model configuration and optimization results
        """
        logs = []
        result = {
            "factors_count": len(factors),
            "status": "success",
            "logs": logs,
            "optimization_history": []
        }
        
        try:
            logs.append(f"[ModelOptimization] Received {len(factors)} factors")
            print(f"🚀 Starting model optimization, factor count: {len(factors)}")
            
            # Step 1: Save factors and register factor pool
            factors_file = self._save_factors_to_file(factors)
            logs.append(f"[ModelOptimization] Factor list saved to: {factors_file}")
            
            factor_pool_name = self._register_factor_pool(factors)
            logs.append(f"[ModelOptimization] Factor pool registered: {factor_pool_name}")
            result["factor_pool_name"] = factor_pool_name
            
            # Step 2: Generate initial yaml configuration
            module_name = self._to_snake_case(factor_pool_name)
            module_path = f"qlib_benchmark.factor_pools.{module_name}"
            
            yaml_config = self._generate_initial_yaml_config(
                factor_pool_name=factor_pool_name,
                module_path=module_path,
                factors_count=len(factors)
            )
            
            # Step 3: Iterative optimization
            best_result = None
            best_score = float('-inf')
            
            for iteration in range(1, max_iterations + 1):
                print(f"\n{'='*60}")
                print(f"🔄 Iteration {iteration}/{max_iterations}")
                print(f"{'='*60}")
                logs.append(f"[ModelOptimization] === Iteration {iteration} ===")
                
                # Save current configuration
                yaml_path = self._save_yaml_config(yaml_config, factor_pool_name, iteration)
                logs.append(f"[ModelOptimization] Configuration saved: {yaml_path}")
                
                # Run training
                print("📊 Running model training...")
                train_result = self._run_benchmark(yaml_path)
                
                if train_result is None:
                    logs.append("[ModelOptimization] Training failed, skipping this iteration")
                    continue
                
                # Read and parse results
                metrics = self._extract_metrics(train_result)
                print(f"📈 Training results:")
                print(f"   - IC mean: {metrics.get('ic_mean', 'N/A'):.4f}" if metrics.get('ic_mean') else "   - IC mean: N/A")
                print(f"   - Rank IC mean: {metrics.get('rank_ic_mean', 'N/A'):.4f}" if metrics.get('rank_ic_mean') else "   - Rank IC mean: N/A")
                print(f"   - Annualized return: {metrics.get('annualized_return', 'N/A'):.2%}" if metrics.get('annualized_return') else "   - Annualized return: N/A")
                print(f"   - Max drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}" if metrics.get('max_drawdown') else "   - Max drawdown: N/A")
                
                # Record this iteration's results
                # Extract model parameters actually used this round (for reviewing "parameter->metric" causal chain in prompt later)
                used_model_cfg = (yaml_config or {}).get("task", {}).get("model", {})
                used_model_class = used_model_cfg.get("class", "Unknown")
                used_model_kwargs = used_model_cfg.get("kwargs", {}) or {}

                # Calculate parameter differences compared to previous round (only compare model.kwargs, avoid stuffing unrelated config into history)
                prev_kwargs = {}
                if self.optimization_history:
                    prev_kwargs = (self.optimization_history[-1] or {}).get("model_kwargs", {}) or {}
                kwargs_diff: Dict[str, Dict[str, Any]] = {}
                all_keys = set(prev_kwargs.keys()) | set(used_model_kwargs.keys())
                for k in sorted(all_keys):
                    old_v = prev_kwargs.get(k, None)
                    new_v = used_model_kwargs.get(k, None)
                    if old_v != new_v:
                        kwargs_diff[k] = {"old": old_v, "new": new_v}

                iteration_record = {
                    "iteration": iteration,
                    "yaml_path": yaml_path,
                    "metrics": metrics,
                    "config_snapshot": copy.deepcopy(yaml_config),
                    "model_class": used_model_class,
                    "model_kwargs": copy.deepcopy(used_model_kwargs),
                    "model_kwargs_diff": kwargs_diff,
                }
                self.optimization_history.append(iteration_record)
                result["optimization_history"].append(iteration_record)
                
                # Calculate comprehensive score
                score = self._compute_optimization_score(metrics)
                print(f"   - Comprehensive score: {score:.4f}")
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_result = {
                        "iteration": iteration,
                        "yaml_path": yaml_path,
                        "metrics": metrics,
                        "config": copy.deepcopy(yaml_config)
                    }
                    logs.append(f"[ModelOptimization] 🎉 Found better configuration, score: {score:.4f}")
                
                # Check if target is reached
                # if self._check_target_reached(metrics, target_annualized_return, target_max_drawdown):
                #     logs.append("[ModelOptimization] ✅ Target metrics reached, ending optimization early")
                #     print("✅ Target metrics reached, ending optimization early")
                #     break
                
                # Use LLM to generate optimization suggestions and update configuration
                if iteration < max_iterations:
                    print("🤖 LLM analyzing and generating optimization suggestions...")
                    optimization_suggestion = self._llm_analyze_and_suggest(
                        current_metrics=metrics,
                        optimization_history=self.optimization_history,
                        current_config=yaml_config,
                        factors_count=len(factors)
                    )
                    
                    if optimization_suggestion:
                        logs.append(f"[ModelOptimization] LLM suggestion: {optimization_suggestion.get('summary', '')}")
                        yaml_config = self._apply_optimization_suggestion(
                            yaml_config, optimization_suggestion
                        )
            
            # Summarize results
            result["best_result"] = best_result
            result["best_score"] = best_score
            result["total_iterations"] = len(self.optimization_history)
            
            if best_result:
                logs.append(f"[ModelOptimization] Optimization completed, best score: {best_score:.4f}")
                logs.append(f"[ModelOptimization] Best configuration from iteration {best_result['iteration']}")
                
                # === Prepare complete model information for StrategyGenerationAgent ===
                result["yaml_config_path"] = best_result.get("yaml_path")
                result["factor_pool_name"] = factor_pool_name
                result["module_path"] = module_path
                result["best_metrics"] = best_result.get("metrics", {})
                
                # Extract model configuration information
                best_config = best_result.get("config", {})
                model_config = best_config.get("task", {}).get("model", {})
                result["model_class"] = model_config.get("class", "TransformerModel")
                result["model_kwargs"] = model_config.get("kwargs", {})
                
                # Extract strategy-related default configuration
                strategy_config = best_config.get("task", {}).get("backtest", {}).get("strategy", {})
                result["default_strategy_config"] = strategy_config
                
                logs.append(f"[ModelOptimization] YAML configuration path: {result['yaml_config_path']}")
                logs.append(f"[ModelOptimization] Factor pool: {factor_pool_name}, Model: {result['model_class']}")
            
        except Exception as e:
            import traceback
            logs.append(f"[ModelOptimization] Processing error: {str(e)}")
            logs.append(traceback.format_exc())
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    # ==================== Core Methods ====================
    
    def _generate_initial_yaml_config(
        self,
        factor_pool_name: str,
        module_path: str,
        factors_count: int
    ) -> Dict[str, Any]:
        """
        Generate initial yaml configuration based on template
        
        Args:
            factor_pool_name: Factor pool class name
            module_path: Factor pool module path
            factors_count: Factor count
            
        Returns:
            yaml configuration dictionary
        """
        # Read template
        with open(self.template_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Update factor pool configuration
        if 'task' in config and 'dataset' in config['task']:
            handler_config = config['task']['dataset']['kwargs']['handler']
            handler_config['class'] = factor_pool_name
            handler_config['module_path'] = module_path
        
        # Adjust model parameters based on factor count
        config = self._adjust_model_params_by_factor_count(config, factors_count)
        
        return config
    
    def _adjust_model_params_by_factor_count(
        self,
        config: Dict[str, Any],
        factors_count: int
    ) -> Dict[str, Any]:
        """Adjust model parameters based on factor count"""
        if 'task' not in config or 'model' not in config['task']:
            return config
        
        model_config = config['task']['model']
        model_class = model_config.get('class', '')
        
        # Select parameter tier based on factor count
        if factors_count < 20:
            size = "small"
        elif factors_count < 50:
            size = "medium"
        else:
            size = "large"
        
        # Adjust d_feat for Transformer model
        if model_class == "TransformerModel":
            if 'kwargs' not in model_config:
                model_config['kwargs'] = {}
            model_config['kwargs']['d_feat'] = factors_count
            print(f"[Config Generation] Transformer d_feat set to: {factors_count}")
        
        return config
    
    def _save_yaml_config(
        self,
        config: Dict[str, Any],
        factor_pool_name: str,
        iteration: int
    ) -> str:
        """Save yaml configuration to file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"workflow_config_{factor_pool_name}_iter{iteration}_{timestamp}.yaml"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return str(output_file)
    
    def _run_benchmark(self, yaml_path: str) -> Optional[Dict[str, Any]]:
        """Run benchmark"""
        if not self.mcp_client:
            print("Warning: MCP client not initialized, cannot run training")
            return None
        
        try:
            result = self.mcp_client.call_tool(
                "qlib_benchmark_runner",
                {"yaml_path": yaml_path}
            )
            
            if isinstance(result, str):
                return json.loads(result)
            return result
            
        except Exception as e:
            print(f"Benchmark run failed: {e}")
            return None
    
    def _extract_metrics(self, train_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from training results
        
        Args:
            train_result: Training results returned by MCP
            
        Returns:
            Dictionary containing IC, RankIC, annualized return, max drawdown and other metrics
        """
        metrics = {}
        
        # Read IC statistics
        ic_path = train_result.get("ic", "")
        if ic_path:
            ic_stats = FileUtils.read_pickle_stats(ic_path)
            if ic_stats:
                metrics["ic_mean"] = ic_stats.get("mean", 0)
                metrics["ic_std"] = ic_stats.get("std", 0)
                metrics["ic_stats"] = ic_stats
        
        # Read Rank IC statistics
        rank_ic_path = train_result.get("rank_ic", "")
        if rank_ic_path:
            rank_ic_stats = FileUtils.read_pickle_stats(rank_ic_path)
            if rank_ic_stats:
                metrics["rank_ic_mean"] = rank_ic_stats.get("mean", 0)
                metrics["rank_ic_std"] = rank_ic_stats.get("std", 0)
                metrics["rank_ic_stats"] = rank_ic_stats
        
        # Read annualized return
        annualized_return_path = train_result.get("1day.excess_return_with_cost.annualized_return", "")
        if annualized_return_path:
            metrics["annualized_return"] = FileUtils.read_mlflow_metric(annualized_return_path)
        
        # Read max drawdown
        max_drawdown_path = train_result.get("1day.excess_return_with_cost.max_drawdown", "")
        if max_drawdown_path:
            metrics["max_drawdown"] = FileUtils.read_mlflow_metric(max_drawdown_path)
        
        # Calculate IR (Information Ratio = IC mean / IC std)
        if metrics.get("ic_mean") and metrics.get("ic_std") and metrics["ic_std"] != 0:
            metrics["ir"] = metrics["ic_mean"] / metrics["ic_std"]
        
        # Calculate Rank IR
        if metrics.get("rank_ic_mean") and metrics.get("rank_ic_std") and metrics["rank_ic_std"] != 0:
            metrics["rank_ir"] = metrics["rank_ic_mean"] / metrics["rank_ic_std"]
        
        return metrics
    
    def _compute_optimization_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate comprehensive optimization score
        
        Score formula considers:
        - IC mean (higher is better)
        - IR (Information Ratio, higher is better)
        - Annualized return (higher is better)
        - Max drawdown (smaller absolute value is better)
        """
        score = 0.0
        
        # IC score (weight 20%)
        ic_mean = metrics.get("ic_mean", 0)
        if ic_mean:
            score += ic_mean * 100 * 0.2  # IC=0.05 → 1 point
        
        # IR score (weight 20%)
        ir = metrics.get("ir", 0)
        if ir:
            score += ir * 0.2  # IR=1 → 0.2 points
        
        # Annualized return score (weight 40%)
        annualized_return = metrics.get("annualized_return", 0)
        if annualized_return:
            score += annualized_return * 0.4  # 10% annualized → 0.04 points
        
        # Max drawdown penalty (weight 20%)
        max_drawdown = metrics.get("max_drawdown", 0)
        if max_drawdown:
            # Drawdown is negative, convert to positive for penalty
            score += max_drawdown * 0.2  # -10% drawdown → -0.02 points
        
        return score
    
    def _check_target_reached(
        self,
        metrics: Dict[str, Any],
        target_annualized_return: float,
        target_max_drawdown: float
    ) -> bool:
        """Check if target metrics are reached"""
        annualized_return = metrics.get("annualized_return")
        max_drawdown = metrics.get("max_drawdown")
        
        if annualized_return is None or max_drawdown is None:
            return False
        
        return (
            annualized_return >= target_annualized_return and
            max_drawdown >= target_max_drawdown  # Drawdown is negative, larger is better
        )
    
    def _llm_analyze_and_suggest(
        self,
        current_metrics: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        current_config: Dict[str, Any],
        factors_count: int
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze current results and generate optimization suggestions
        
        Args:
            current_metrics: Current iteration's metrics
            optimization_history: Historical optimization records
            current_config: Current configuration
            factors_count: Factor count
            
        Returns:
            Optimization suggestion dictionary
        """
        if not self.llm:
            return None
        
        # Build history summary
        history_summary = self._build_history_summary(optimization_history)
        
        # Get current model configuration
        model_config = current_config.get('task', {}).get('model', {})
        model_class = model_config.get('class', 'Unknown')
        model_params = model_config.get('kwargs', {})
        
        # Build tunable parameters description
        tunable_params_desc = self._build_tunable_params_description(model_class)
        
        prompt = f"""You are a quantitative model optimization expert. Please analyze the current model performance and provide parameter tuning suggestions.

## Current Model Information
- Model type: {model_class}
- Factor count: {factors_count}

## Current Model Parameters
{json.dumps(model_params, indent=2, ensure_ascii=False)}

## Current Performance Metrics
- IC mean: {current_metrics.get('ic_mean', 'N/A')}
- IC std: {current_metrics.get('ic_std', 'N/A')}
- IR (Information Ratio): {current_metrics.get('ir', 'N/A')}
- Rank IC mean: {current_metrics.get('rank_ic_mean', 'N/A')}
- Annualized return: {current_metrics.get('annualized_return', 'N/A')}
- Max drawdown: {current_metrics.get('max_drawdown', 'N/A')}

## Optimization History
{history_summary}

## Tunable Parameter Ranges
{tunable_params_desc}

Please analyze the problems with current performance and provide specific model parameter adjustment suggestions.

Notes:
1. If IC/IR is low, consider adjusting model complexity or regularization parameters
2. If annualized return is low but IC is high, may need to adjust model learning rate or iteration count
3. If drawdown is too large, consider increasing regularization strength (alpha/lambda)
4. Refer to history records, avoid repeating ineffective parameter combinations
5. Only suggest adjusting 1-3 key parameters each time
6. Only optimize model parameters, do not adjust strategy parameters

Please output JSON format:
{{
    "analysis": "Analysis of current performance",
    "issues": ["Issue 1", "Issue 2"],
    "summary": "Optimization direction summary",
    "model_params_update": {{
        "parameter_name": new_value
    }},
    "reasoning": "Reasoning for parameter adjustment"
}}

Only output JSON, no other content."""
        
        try:
            response = self.llm.call(prompt=prompt, stream=False)
            suggestion = self.llm.parse_json_response(response)
            
            if suggestion:
                print(f"💡 LLM analysis: {suggestion.get('analysis', '')}")
                print(f"   Optimization direction: {suggestion.get('summary', '')}")
            
            return suggestion
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return None
    
    def _build_history_summary(self, history: List[Dict[str, Any]]) -> str:
        """Build optimization history summary"""
        if not history:
            return "No history records"
        
        lines = []
        # Only keep last 10 records to avoid prompt being too long
        for record in history[-10:]:
            metrics = record.get("metrics", {}) or {}

            # Metrics summary (allow partial missing)
            ic_mean = metrics.get("ic_mean", None)
            annualized_return = metrics.get("annualized_return", None)
            max_drawdown = metrics.get("max_drawdown", None)

            metric_parts = []
            if ic_mean is not None:
                metric_parts.append(f"IC={ic_mean:.4f}")
            if annualized_return is not None:
                metric_parts.append(f"Annualized={annualized_return:.2%}")
            if max_drawdown is not None:
                metric_parts.append(f"Drawdown={max_drawdown:.2%}")
            metric_str = ", ".join(metric_parts) if metric_parts else "Incomplete metrics"

            # Parameter change summary: prefer diff (shorter, more causal chain), fallback to kwargs if no diff
            diff = record.get("model_kwargs_diff", {}) or {}
            if diff:
                # Only show up to 8 changes to prevent too many changes from bloating context
                diff_items = []
                for i, (k, v) in enumerate(diff.items()):
                    if i >= 8:
                        diff_items.append("... (more changes omitted)")
                        break
                    diff_items.append(f"{k}: {v.get('old')} -> {v.get('new')}")
                params_str = "; ".join(diff_items)
            else:
                kwargs = record.get("model_kwargs", {}) or {}
                # kwargs might be long, only take first 12 keys
                keys = sorted(kwargs.keys())[:12]
                shown = {k: kwargs.get(k) for k in keys}
                suffix = " ... (more parameters omitted)" if len(kwargs) > len(keys) else ""
                params_str = f"{json.dumps(shown, ensure_ascii=False)}{suffix}"

            model_class = record.get("model_class", "Unknown")
            lines.append(f"- Iteration {record.get('iteration', 'N/A')} [{model_class}]: {metric_str} | Parameters: {params_str}")
        
        return "\n".join(lines)
    
    def _build_tunable_params_description(self, model_class: str) -> str:
        """Build tunable parameters description (model parameters only)"""
        lines = []
        
        # Model parameters
        model_params = TUNABLE_PARAMS["model"].get(model_class, {})
        if model_params:
            lines.append(f"### {model_class} Model Parameters:")
            for param, info in model_params.items():
                lines.append(f"  - {param}: {info['range']} (default: {info['default']})")
        else:
            lines.append(f"### {model_class} Model Parameters:")
            lines.append("  (No tunable parameters defined)")
        
        return "\n".join(lines)
    
    def _apply_optimization_suggestion(
        self,
        config: Dict[str, Any],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply LLM's optimization suggestions to configuration (only update model parameters)
        
        Args:
            config: Current configuration
            suggestion: LLM's optimization suggestion
            
        Returns:
            Updated configuration
        """
        config = copy.deepcopy(config)
        
        # Update model parameters
        model_params_update = suggestion.get("model_params_update", {})
        if model_params_update and 'task' in config and 'model' in config['task']:
            if 'kwargs' not in config['task']['model']:
                config['task']['model']['kwargs'] = {}
            
            for param, value in model_params_update.items():
                config['task']['model']['kwargs'][param] = value
                print(f"   📝 Updated model parameter: {param} = {value}")
        
        return config
    
    # ==================== Helper Methods ====================
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tool list"""
        if not self.mcp_client:
            return []
        
        try:
            return self.mcp_client.list_tools()
        except Exception as e:
            print(f"Failed to get tool list: {e}")
            return []
    
    def call_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Call MCP tool"""
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized")
        
        try:
            return self.mcp_client.call_tool(tool_name, arguments)
        except Exception as e:
            raise RuntimeError(f"Failed to call tool {tool_name}: {str(e)}") from e
    
    def _register_factor_pool(self, factors: list) -> str:
        """Register factors as Handler class"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pool_name = f"CustomFactors_{timestamp}"
        
        factors_root = Path(__file__).parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
        factors_root_str = str(factors_root)
        if factors_root_str not in sys.path:
            sys.path.append(factors_root_str)
        
        from factor_pool_registry import register_factor_pool
        
        description = f"{len(factors)} factors mined by ModelOptimizationAgent"
        module_path = register_factor_pool(pool_name, factors, description)
        
        print(f"[ModelOptimization] Factor pool registered as: {pool_name}")
        print(f"[ModelOptimization] Module path: {module_path}")
        
        return pool_name
    
    def _save_factors_to_file(self, factors: list) -> str:
        """Save factor list to file"""
        current_dir = Path(__file__).parent.parent
        output_dir = current_dir / "Qlib_MCP" / "workspace" / "factors"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"optimized_factors_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "factors_count": len(factors),
                "factors": factors
            }, f, indent=2, ensure_ascii=False)
        
        return str(output_file)
    
    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# Maintain backward compatibility alias
model_optimization_Agent = ModelOptimizationAgent
