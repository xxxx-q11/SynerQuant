"""Model optimization node"""
from state import AgentState
from Agent.model_optimization_Agent import model_optimization_Agent


def model_optimization_node(state: AgentState) -> dict:
    """Model optimization node - optimize model based on factor mining results"""
    logs = ["[ModelOptimization] Executing model optimization"]
    model = {}
    
    try:
        # Get factor list from state
        # factors = state.get("factors", [])
        # Should get sota_pool_list from state, not factors
        sota_pool_list = state.get("sota_pool_list", [])
        if not sota_pool_list:
            # If no sota_pool_list, use factors as fallback
            sota_pool_list = state.get("factors", [])

        # Convert sota_pool_list (list of strings) to factor dictionary list
        factors = [{"expression": expr, "ic": 0.0} for expr in sota_pool_list]
        
        if not factors:
            logs.append("[ModelOptimization] Warning: No factor list received, skipping model optimization")
            return {
                "model": model,
                "logs": logs,
                "current_node": "model_optimization",
            }
        
        logs.append(f"[ModelOptimization] Received {len(factors)} factors")
        
        # Print factor information (for debugging)
        for i, factor in enumerate(factors[:5]):  # Only print first 5
            if isinstance(factor, dict):
                expression = factor.get("expression", "Unknown")
                ic = factor.get("ic", "Unknown")
                logs.append(f"[ModelOptimization] Factor {i+1}: IC={ic}, Expression={expression[:50]}...")
        
        # Call model_optimization_Agent for actual optimization
        llm_service = get_llm_service()
        agent = model_optimization_Agent(llm_service)
        result = agent.process(factors)
        
        # Process return result
        if isinstance(result, dict):
            model = result
            # Merge logs
            if "logs" in result:
                logs.extend(result["logs"])
            
            # Record optimization results
            if result.get("status") == "success":
                logs.append(f"[ModelOptimization] Model optimization completed, processed {result.get('factors_count', 0)} factors")
                
                # Record key information passed to strategy generation
                if result.get("yaml_config_path"):
                    logs.append(f"[ModelOptimization] -> Optimal config: {result['yaml_config_path']}")
                if result.get("factor_pool_name"):
                    logs.append(f"[ModelOptimization] -> Factor pool: {result['factor_pool_name']}")
                if result.get("model_class"):
                    logs.append(f"[ModelOptimization] -> Model type: {result['model_class']}")
                if result.get("best_metrics"):
                    metrics = result["best_metrics"]
                    logs.append(f"[ModelOptimization] -> Baseline performance: IC={metrics.get('ic_mean', 'N/A'):.4f}, "
                               f"Annualized={metrics.get('annualized_return', 'N/A'):.2%}" 
                               if metrics.get('ic_mean') and metrics.get('annualized_return') 
                               else f"[ModelOptimization] -> Baseline performance: {metrics}")
            else:
                logs.append(f"[ModelOptimization] Model optimization failed: {result.get('error', 'Unknown error')}")
        else:
            model = {"raw_result": result}
            logs.append("[ModelOptimization] Model optimization completed (non-dict result)")
        
    except Exception as e:
        logs.append(f"[ModelOptimization] Processing error: {str(e)}")
    
    return {
        "model": model,
        "logs": logs,
        "current_node": "model_optimization",
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
    )

