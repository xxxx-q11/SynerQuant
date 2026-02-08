"""LLM factor evaluator"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from ..utils.file_utils import ConfigLoader


class LLMFactorEvaluator:
    """Use LLM for factor analysis and evaluation"""
    
    def __init__(self, llm_service):
        """
        Initialize LLM evaluator
        
        Args:
            llm_service: LLM service instance (needs call and parse_json_response methods)
        """
        self.llm = llm_service
        self._qlib_operators = None
    
    @property
    def qlib_operators(self) -> Dict[str, Any]:
        """Lazy load Qlib operator configuration"""
        if self._qlib_operators is None:
            self._qlib_operators = ConfigLoader.load_qlib_operators()
        return self._qlib_operators
    
    def analyze_factor_performance(
        self,
        factor_expression: str,
        origin_pool_stats: Dict[str, Any],
        new_pool_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze factor performance after adding to factor pool
        
        Args:
            factor_expression: Factor expression
            origin_pool_stats: Original factor pool statistics
            new_pool_stats: Statistics after adding new factor
            
        Returns:
            Analysis result dictionary, containing conclusion and reason
        """
        prompt = f"""
If there are a group of factor evaluation experts, the following information is the IC evaluation results on the test set after a new factor is added to the factor pool and xgboost is used to optimize weights. Please give the factor evaluation conclusion based on the following factor evaluation results:

Original factor pool IC evaluation results:
{origin_pool_stats.get('ic', {})}

Original factor pool rank IC evaluation results:
{origin_pool_stats.get('rank_ic', {})}

Original factor pool annualized return and max drawdown on test set with transaction costs after using xgboost to optimize weights:
origin_factor_pool_annualized_return_test_result: {origin_pool_stats.get('annualized_return', 'N/A')}
origin_factor_pool_max_drawdown_test_result: {origin_pool_stats.get('max_drawdown', 'N/A')}

IC evaluation results and rank IC evaluation results on test set after new factor is added to factor pool and xgboost is used to optimize weights, as well as annualized return and max drawdown on test set with transaction costs:
ic_analysis_result: {new_pool_stats.get('ic', {})}
rank_ic_analysis_result: {new_pool_stats.get('rank_ic', {})}
annualized_return_test_result: {new_pool_stats.get('annualized_return', 'N/A')}
max_drawdown_test_result: {new_pool_stats.get('max_drawdown', 'N/A')}

These factor experts give the new factor evaluation conclusion and reason based on the above information.
Strictly follow JSON format, return in JSON format, do not use escape characters like newlines, keys are conclusion and reason.
"""
        time.sleep(1) # Wait 1 second to give LLM time to process the prompt
        response = self.llm.call(prompt=prompt, stream=False)
        return self.llm.parse_json_response(response)
    
    def explain_factor_economics(self, factor_expression: str) -> Dict[str, Any]:
        """
        Provide economic principle explanation for factor
        
        Args:
            factor_expression: Factor expression
            
        Returns:
            Explanation result dictionary, containing factor, conclusion and reason
        """
        prompt = f"""
If there are a group of factor evaluation experts, the following is a factor. These factor experts try to give an economic principle explanation based on the above factor. The rank in the factor formula is on the time series of the same stock:
{factor_expression}
IC is the mean IC on the training set, rank_ic is the mean rank_ic on the validation set.
Strictly follow JSON format, return in JSON format, do not use escape characters like newlines, keys are factor, conclusion and reason.
"""
        time.sleep(1)
        response = self.llm.call(prompt=prompt, stream=False)
        return self.llm.parse_json_response(response)
    
    def evaluate_factor_comprehensive(
        self,
        factor_expression: str,
        sota_pool: List[str],
        origin_pool_stats: Dict[str, Any],
        new_pool_stats: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        economics_explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensively evaluate whether factor should be kept
        
        Args:
            factor_expression: Factor expression
            sota_pool: Current SOTA factor pool
            origin_pool_stats: Original factor pool statistics
            new_pool_stats: Statistics after adding new factor
            performance_analysis: Performance analysis result
            economics_explanation: Economic explanation result
            
        Returns:
            Evaluation result dictionary, containing if_keep, conclusion and reason
        """
        prompt = f"""
If there are a group of factor evaluation experts, the following are the IC evaluation results on the test set after a new factor is added to the factor pool and xgboost is used to optimize weights, as well as the economic explanation based on the factor itself. Please give the factor evaluation conclusion based on this information:

Original factor pool. The IC values of factors in the original factor pool are defaulted to 0, but actually these factor IC values are not 0:
{sota_pool}

Original factor pool IC evaluation results after using xgboost to optimize weights:
{origin_pool_stats.get('ic', {})}

Original factor pool rank IC evaluation results:
{origin_pool_stats.get('rank_ic', {})}

Original factor pool annualized return and max drawdown on test set with transaction costs after using xgboost to optimize weights:
annualized_return_test_result: {origin_pool_stats.get('annualized_return', 'N/A')}
max_drawdown_test_result: {origin_pool_stats.get('max_drawdown', 'N/A')}

IC evaluation results and rank IC evaluation results on test set after new factor {factor_expression} is added to factor pool and xgboost is used to optimize weights, as well as annualized return and max drawdown on test set with transaction costs:
annualized_return_test_result: {new_pool_stats.get('annualized_return', 'N/A')}
max_drawdown_test_result: {new_pool_stats.get('max_drawdown', 'N/A')}
ic_analysis_result: {new_pool_stats.get('ic', {})}
rank_ic_analysis_result: {new_pool_stats.get('rank_ic', {})}

Evaluation after factor is added to factor pool:
{performance_analysis}

Economic explanation based on the factor itself:
{economics_explanation}

When evaluating results:
1. If the new factor (or factor combination) can improve annualized return, suggest using it as an alternative to replace the current best result. (Set "if_keep" option to "True").
2. Minor changes in other metrics are acceptable, as long as annualized return can be improved.
3. If the factor combination clearly does not conform to economic principles, suggest not keeping this factor (set "if_keep" option to "False").

Strictly follow JSON format, return in JSON format, do not use escape characters like newlines, keys are if_keep, conclusion and reason.
"""
        time.sleep(1)
        response = self.llm.call(prompt=prompt, stream=False)
        return self.llm.parse_json_response(response)
    
    def revise_factor(
        self,
        eval_result: Dict[str, Any],
        sota_pool: List[str]
    ) -> Dict[str, Any]:
        """
        Revise factor expression based on evaluation results
        
        Args:
            eval_result: Evaluation result dictionary
            sota_pool: Current SOTA factor pool
            
        Returns:
            Revision result dictionary, containing revised_factor_expression and reason
        """
        ops = self.qlib_operators
        
        prompt = f"""
If there are a group of factor evaluation experts, the following are the IC and rank IC evaluation results on the test set after a new factor is added to the current factor pool, as well as the explanation based on the factor formula itself:
{eval_result}

Original factor pool as follows: {sota_pool}

These factor experts, based on this information and combined with economic knowledge, fully considering the factors in the existing factor pool, try to modify the factor formula itself.
The modified factor should be different from the factors already in the factor pool, while improving its performance on IC and rank IC on the test set,
as well as performance on annualized return and max drawdown on the test set with transaction costs,
and provide the modified factor expression. Only return one modified factor expression, do not return multiple.

The modified factor expression needs to support qlib expression syntax. The operators supported by qlib are as follows: {ops}
Expressions like ".rolling(20).mean()" are incorrect, do not use qlib APIs, use qlib expression syntax directly.

Strictly follow JSON format, return in JSON format, do not use escape characters like newlines, keys are revised_factor_expression and reason.
revised_factor_expression: Modified factor expression
reason: Revision reason
"""
        response = self.llm.call(prompt=prompt, stream=False)
        return self.llm.parse_json_response(response)
    
    def evaluate_continue_iteration(
        self,
        sota_pool: List[str],
        base_pool: List[str],
        pool_history: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate whether to continue iterating and adding factors
        
        Args:
            sota_pool: Current SOTA factor pool
            base_pool: Base factor pool
            pool_history: Factor pool historical evaluation results
            
        Returns:
            Evaluation result dictionary, containing should_continue and reason
        """
        prompt = f"""
If there are a group of factor evaluation experts, the following is the current factor pool. Please give the factor evaluation conclusion based on this information:

Current factor pool:
{sota_pool}

Original factor pool:
{base_pool}

The following are the IC and rank IC evaluation results on the test set after each new factor is added to the original factor pool. The last one is the evaluation result of the current factor pool:
{pool_history}

New factors come from machine learning or RL algorithms, and there may be homogeneity issues. Even if new factors come in, they may not significantly improve the IC and rank_ic of the entire factor pool.

These factor experts, based on the above information, judge whether newly added factors can continue to increase the factor pool's IC and rank_ic on the test set, or stop iteration and enter the next stage.
And give reasons. Return in JSON format, keys are should_continue and reason.
should_continue: Whether to continue iterating to increase the factor pool's IC and rank_ic on the test set, or stop iteration and enter the next stage of model optimization, True or False
reason: Judgment reason
"""
        response = self.llm.call(prompt=prompt, stream=False)
        return self.llm.parse_json_response(response)


def convert_to_bool(value) -> bool:
    """
    Convert string or boolean value to boolean
    
    Args:
        value: May be string "True"/"False" or boolean True/False
        
    Returns:
        Converted boolean value
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ("true", "1", "yes", "t"):
            return True
        elif value_lower in ("false", "0", "no", "f", ""):
            return False
        else:
            return False
    return bool(value)

