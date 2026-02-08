"""
LangGraph main graph definition
"""
from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    factor_mining_node,
    factor_eval_node,
    model_optimization_node,
    strategy_generation_node,
    risk_control_node,
)

# Iteration control constants
MAX_MINING_ITERATIONS = 10  # Maximum number of iterations
MIN_POOL_SIZE_FOR_SUCCESS = 40  # Minimum factor pool size (can end early when reached)


def route_after_factor_eval(state: AgentState) -> str:
    """
    Route after factor_eval based on current_node and convergence status
    
    Returns:
        Name of the next node
    """
    current_node = state.get("current_node", "")
    
    # If not returning to factor mining, proceed directly to model optimization
    if current_node != "factor_mining":
        return "model_optimization"
    
    # Get mining feedback
    mining_feedback = state.get("mining_feedback", {})
    
    # Check iteration control conditions
    iteration = mining_feedback.get("iteration", 0)
    convergence_info = mining_feedback.get("convergence_info", {})
    sota_pool_list = state.get("sota_pool_list", [])
    
    # Condition 1: Reached maximum iteration count
    #return "model_optimization"
    if iteration >= MAX_MINING_ITERATIONS:
        print(f"[Graph] Reached maximum iteration count ({MAX_MINING_ITERATIONS}), proceeding to model optimization")
        return "model_optimization"
    
    # Condition 2: Factor pool has reached minimum required size
    if len(sota_pool_list) >= MIN_POOL_SIZE_FOR_SUCCESS:
        print(f"[Graph] Factor pool size ({len(sota_pool_list)}) has reached requirement, proceeding to model optimization")
        return "model_optimization"
    
    # Condition 3: Convergence detected
    # if convergence_info.get("is_converged", False):
    #     reason = convergence_info.get("convergence_reason", "Unknown")
    #     print(f"[Graph] Convergence detected ({reason}), proceeding to model optimization")
    #     return "model_optimization"
    
    # Continue iterative mining
    print(f"[Graph] Iteration {iteration + 1}, continuing factor mining")
    return "factor_mining"


def build_graph() -> StateGraph:
    """
    Build quantitative trading Agent workflow graph
    
    Flow: factor_mining -> factor_eval -> (if factors insufficient, return to factor_mining, otherwise continue) -> model_optimization -> strategy_generation -> risk_control
    """
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("factor_mining", factor_mining_node)
    graph.add_node("factor_eval", factor_eval_node)
    graph.add_node("model_optimization", model_optimization_node)
    graph.add_node("strategy_generation", strategy_generation_node)
    graph.add_node("risk_control", risk_control_node)
    
    # Set entry point
    graph.set_entry_point("factor_mining")
    
    # Add edges
    graph.add_edge("factor_mining", "factor_eval")
    
    # factor_eval uses conditional edge, determines next node based on current_node
    graph.add_conditional_edges(
        "factor_eval",
        route_after_factor_eval,
        {
            "factor_mining": "factor_mining",  # If factors insufficient, return to mining
            "model_optimization": "model_optimization"  # Otherwise continue evaluation flow
        }
    )
    
    # Other nodes execute sequentially
    graph.add_edge("model_optimization", "strategy_generation")
    graph.add_edge("strategy_generation", "risk_control")
    graph.add_edge("risk_control", END)
    
    return graph


def create_agent():
    """Create and compile Agent"""
    graph = build_graph()
    return graph.compile()

