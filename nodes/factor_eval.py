"""Factor evaluation node"""
from state import AgentState
from Agent.factor_eval_agent import FactorEvalAgent


def factor_eval_node(state: AgentState) -> dict:
    """Factor evaluation node - evaluate factors based on factor mining results"""
    logs = ["[FactorEval] Executing factor evaluation"]
    
    try:
        # Get factor list from state
        factors = state.get("factors", [])
        
        # Get existing sota_pool_list and factor_pool_analysis_result_history from state (if exists)
        sota_pool_list = state.get("sota_pool_list", None)
        factor_pool_analysis_result_history = state.get("factor_pool_analysis_result_history", None)
        
        # Get previous round's mining feedback (for iterative optimization)
        previous_feedback = state.get("mining_feedback", None)
        
        if previous_feedback:
            iteration = previous_feedback.get("iteration", 0)
            logs.append(f"[FactorEval] Detected iteration feedback, current is iteration {iteration + 1}")
        
        if not factors:
            logs.append("[FactorEval] Warning: No factor list received, skipping factor evaluation")
            return {
                "factors": factors,
                "logs": logs,
                "current_node": "factor_eval",
                "sota_pool_list": sota_pool_list,
                "factor_pool_analysis_result_history": factor_pool_analysis_result_history,
            }
        
        logs.append(f"[FactorEval] Received {len(factors)} factors")
        
        # Print factor information (for debugging)
        for i, factor in enumerate(factors[:5]):  # Only print first 5
            if isinstance(factor, dict):
                expression = factor.get("expression", "Unknown")
                ic = factor.get("ic", "Unknown")
                rank_ic_valid = factor.get("rank_ic_valid", "Unknown")
                logs.append(f"[FactorEval] Factor {i+1}: IC={ic}, Rank_IC_Valid={rank_ic_valid}, Expression={expression[:50]}...")
        
        # Call FactorEvalAgent for actual evaluation
        llm_service = get_llm_service()
        agent = FactorEvalAgent(llm_service)
        result = agent.process(
            factors, 
            sota_pool_list=sota_pool_list,
            factor_pool_analysis_result_history=factor_pool_analysis_result_history,
            previous_feedback=previous_feedback
        )
        print(f"[FactorEval] result: {result}")
        
        # Process return result
        if not isinstance(result, dict):
            logs.append("[FactorEval] Warning: Return result format incorrect")
            return {
                "logs": logs,
                "current_node": "factor_eval",
                "sota_pool_list": sota_pool_list,
                "factor_pool_analysis_result_history": factor_pool_analysis_result_history,
            }
        
        # Merge logs
        if "logs" in result:
            logs.extend(result["logs"])
        
        status = result.get("status")
        
        # Handle different cases based on status
        if status == "need_more_factors":
            # Need more factors, jump to factor_mining node
            mining_feedback = result.get("mining_feedback")
            
            if mining_feedback:
                iteration = mining_feedback.get("iteration", 1)
                suggested_directions = mining_feedback.get("suggested_directions", [])
                suggested_seeds = mining_feedback.get("suggested_seeds", [])
                pool_weaknesses = mining_feedback.get("pool_weaknesses", [])
                convergence_info = mining_feedback.get("convergence_info", {})
                
                logs.append(f"[FactorEval] Generated iteration {iteration} mining feedback")
                logs.append(f"[FactorEval] Factor pool weaknesses: {pool_weaknesses[:2]}...")
                logs.append(f"[FactorEval] Suggested directions: {len(suggested_directions)} items")
                logs.append(f"[FactorEval] Suggested seed factors: {len(suggested_seeds)} items")
                logs.append(f"[FactorEval] Convergence status: {convergence_info.get('is_converged', False)}")
            
            return {
                "logs": logs,
                "current_node": result.get("current_node", "factor_mining"),
                "sota_pool_list": result.get("sota_pool_list"),
                "factor_pool_analysis_result_history": result.get("factor_pool_analysis_result_history"),
                "mining_feedback": mining_feedback,
            }
        
        elif status == "success":
            # Evaluation completed successfully
            filtered_factors = result.get("filtered_factors", [])
            logs.append(f"[FactorEval] Factor evaluation completed, filtered factor count: {len(filtered_factors)}")
            logs.append(f"[FactorEval] Final SOTA factor pool size: {len(result.get('sota_pool_list', []))}")
            
            return {
                "logs": logs,
                "current_node": "factor_eval",  # Evaluation completed, stay at current node
                "sota_pool_list": result.get("sota_pool_list"),
                "factor_pool_analysis_result_history": result.get("factor_pool_analysis_result_history"),
                "filtered_factors": filtered_factors,
            }
        
        else:
            # Unknown status
            logs.append(f"[FactorEval] Unknown status: {status}")
            return {
                "logs": logs,
                "current_node": "factor_eval",
                "sota_pool_list": result.get("sota_pool_list", sota_pool_list),
                "factor_pool_analysis_result_history": result.get("factor_pool_analysis_result_history", factor_pool_analysis_result_history),
            }
        
    except Exception as e:
        error_msg = f"[FactorEval] Processing error: {str(e)}"
        logs.append(error_msg)
        print(f"Error: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # In case of exception, preserve existing state
        return {
            "logs": logs,
            "current_node": "factor_eval",
            "sota_pool_list": state.get("sota_pool_list"),
            "factor_pool_analysis_result_history": state.get("factor_pool_analysis_result_history"),
        }
def get_llm_service():
    """
    Get LLM service instance
    
    Returns:
        BaseAgent instance
    """
    from Agent.agent_factory import load_env_config, create_agent
    
    config = load_env_config()
    return create_agent(
        provider=config.get("provider", "qwen"),
        api_key=config.get("api_key"),
        model=config.get("model"),
        base_url=config.get("base_url"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens"),
        timeout=300 
    )