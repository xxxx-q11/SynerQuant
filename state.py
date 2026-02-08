"""
Agent state definition
"""
from typing import TypedDict, List, Dict, Any, Annotated


def merge_lists(left: List, right: List) -> List:
    """Merge lists"""
    return left + right


class AgentState(TypedDict, total=False):
    """Quantitative trading Agent state"""
    # Input
    task: str
    # Data from each stage
    factors: List[Dict[str, Any]]
    model: Dict[str, Any]
    strategy: Dict[str, Any]
    risk_report: Dict[str, Any]
    
    # Logs
    logs: Annotated[List[str], merge_lists]
    
    # Current node
    current_node: str
    
    # Factor evaluation related state
    sota_pool_list: List[str]  # Current factor pool list
    factor_pool_analysis_result_history: List[Dict[str, Any]]  # Original factor pool analysis results
    selection_result: Dict[str, Any]  # Tool call records
    
    # Factor mining feedback information (for iterative optimization)
    mining_feedback: Dict[str, Any]  # Factor evaluation feedback, guides next round of mining
    # mining_feedback structure:
    # {
    #     "iteration": int,                    # Current iteration round
    #     "good_patterns": List[str],          # Descriptions of effective factor patterns
    #     "bad_patterns": List[str],           # Descriptions of ineffective factor patterns
    #     "missing_categories": List[str],     # Missing factor categories
    #     "elite_factors": List[str],          # Elite factors (expressions)
    #     "suggested_seeds": List[Dict],       # LLM-suggested seed factors for next round
    #     "evaluation_summary": Dict[str, Any],# Summary statistics of this round's evaluation
    #     "convergence_info": Dict[str, Any],  # Convergence information
    # }
    
    # Strategy generation related state
    current_holdings: Dict[str, float]  # Current holdings {stock_code: holding_weight}
    top_k_recommendations: List[Dict[str, Any]]  # Top_K stocks recommended by model
    trade_decisions: List[Dict[str, Any]]  # Trading decision records
    turnover_rate: float  # Current turnover rate
    backtest_config: Dict[str, Any]  # Backtest configuration
