"""Strategy generation node - integrated with LangGraph"""
from typing import Dict, Any, Optional
from state import AgentState


def strategy_generation_node(state: AgentState) -> dict:
    """
    Strategy generation node
    
    In LangGraph workflow, responsible for:
    1. Get model optimization results from state (yaml_config_path, factor_pool_name, best_metrics, etc.)
    2. Call StrategyGenerationAgent.process() to generate strategy configuration
    3. Run backtest with Agent-enhanced strategy
    
    Args:
        state: AgentState current state
        
    Returns:
        Updated state dictionary
    """
    logs = []
    logs.append("[StrategyGeneration] Starting strategy generation")
    
    try:
        # Get model optimization results from state
        model_info = state.get("model", {})
        sota_pool_list = state.get("sota_pool_list", [])
        
        # Validate necessary information
        if not model_info.get("yaml_config_path"):
            logs.append("[StrategyGeneration] Warning: Model configuration path not found")
        
        if not sota_pool_list:
            logs.append("[StrategyGeneration] Warning: Factor pool is empty")
        
        # Get LLM service
        llm_service = _get_llm_service()
        
        # Create StrategyGenerationAgent and call process
        from Agent.strategy_generation_agent import StrategyGenerationAgent, StrategyConfig
        
        # Configure strategy parameters
        strategy_config = StrategyConfig(
            topk=50,
            n_drop=5,
            max_turnover=0.3,
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5,
            limit_threshold=0.095,
        )
        
        # Initialize Agent (enable LLM-enhanced decision making)
        agent = StrategyGenerationAgent(
            llm_service=llm_service,
            config=strategy_config,
            use_llm_decision=True,  # Enable LLM-enhanced decision making
        )
        
        # Call process method to execute strategy generation and backtest
        result = agent.process(
            model_info=model_info,
            sota_pool_list=sota_pool_list,
            run_backtest=True  # Run backtest
        )
        
        # Merge logs
        if result.get("logs"):
            logs.extend(result["logs"])
        
        # Build returned strategy information
        strategy = result.get("strategy", {})
        strategy["status"] = result.get("status", "success")
        
        # Add backtest results
        if result.get("backtest_metrics"):
            strategy["backtest_metrics"] = result["backtest_metrics"]
        
        if result.get("agent_yaml_path"):
            strategy["agent_yaml_path"] = result["agent_yaml_path"]
        
        return {
            "strategy": strategy,
            "logs": logs,
            "current_node": "strategy_generation",
        }
        
    except Exception as e:
        import traceback
        logs.append(f"[StrategyGeneration] Processing error: {str(e)}")
        logs.append(traceback.format_exc())
        return {
            "strategy": {"status": "error", "error": str(e)},
            "logs": logs,
            "current_node": "strategy_generation",
        }


def _get_llm_service():
    """
    Get LLM service instance
    
    Returns:
        BaseAgent instance or None
    """
    try:
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
    except Exception as e:
        print(f"[StrategyGeneration] Failed to get LLM service: {e}")
        return None


def _build_strategy_config(
    model_info: Dict[str, Any],
    factors: list
) -> Dict[str, Any]:
    """
    Build strategy configuration
    
    Generate AgentEnhancedStrategy configuration based on model optimization results
    
    Args:
        model_info: Model information passed from ModelOptimizationAgent, containing:
            - yaml_config_path: Optimal YAML configuration path
            - factor_pool_name: Factor pool name
            - module_path: Factor pool module path
            - model_class: Model type
            - model_kwargs: Model parameters
            - best_metrics: Baseline performance metrics
            - default_strategy_config: Default strategy configuration (optional)
        factors: Factor list (sota_pool_list)
        
    Returns:
        Strategy configuration dictionary
    """
    # Get default strategy configuration from model optimization results (if available)
    default_config = model_info.get("default_strategy_config", {})
    
    # Default strategy configuration
    config = {
        # Stock selection parameters (can be overridden from default config)
        "topk": default_config.get("topk", 50),
        "n_drop": default_config.get("n_drop", 5),
        
        # Trading constraints
        "max_turnover": 0.3,
        "min_trade_value": 10000,
        
        # Trading costs
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
        
        # Limit up/down restrictions
        "limit_threshold": 0.095,
        
        # Position control
        "risk_degree": 0.95,
        "hold_thresh": 1,
        
        # Agent decision configuration
        "use_agent_decision": True,
        "use_llm_decision": True,
        "decision_mode": "rule_based",
    }
    
    # Extract configuration paths from model_info
    if model_info.get("yaml_config_path"):
        config["yaml_config_path"] = model_info["yaml_config_path"]
    
    if model_info.get("factor_pool_name"):
        config["factor_pool_name"] = model_info["factor_pool_name"]
    
    if model_info.get("module_path"):
        config["module_path"] = model_info["module_path"]
    
    # Factor count (for adjusting strategy parameters)
    config["factors_count"] = len(factors)
    
    # Automatically adjust topk based on factor count
    if len(factors) < 20:
        # Small factor pool, reduce stock selection count
        config["topk"] = min(30, config["topk"])
    elif len(factors) > 50:
        # Large factor pool, can appropriately increase stock selection count
        config["topk"] = min(80, max(50, config["topk"]))
    
    return config


def create_strategy_for_backtest(
    state: AgentState,
    signal=None,
    llm_service=None
) -> Optional[Any]:
    """
    Create AgentEnhancedStrategy strategy instance for backtesting
    
    This function is called during actual backtesting, creating strategy using configuration passed from model optimization
    
    Args:
        state: AgentState current state, containing strategy information
        signal: Qlib prediction signal (if None, needs to be loaded from yaml_config_path)
        llm_service: LLM service (optional, for LLM-enhanced decision making)
        
    Returns:
        AgentEnhancedStrategy instance or None
    """
    from Agent.strategy_generation_agent import (
        create_strategy_generation_agent,
        create_agent_enhanced_strategy,
        AgentEnhancedStrategy,
        StrategyConfig
    )
    
    strategy_info = state.get("strategy", {})
    config = strategy_info.get("config", {})
    
    # Extract configuration from strategy information
    topk = config.get("topk", 50)
    n_drop = config.get("n_drop", 5)
    use_agent_decision = strategy_info.get("use_agent_decision", True)
    use_llm_decision = strategy_info.get("use_llm_decision", False)
    
    # Create strategy configuration
    strategy_config = StrategyConfig(
        topk=topk,
        n_drop=n_drop,
        max_turnover=config.get("max_turnover", 0.3),
        min_trade_value=config.get("min_trade_value", 10000),
        open_cost=config.get("open_cost", 0.0005),
        close_cost=config.get("close_cost", 0.0015),
        min_cost=config.get("min_cost", 5),
        limit_threshold=config.get("limit_threshold", 0.095),
    )
    
    # Create strategy generation Agent
    agent = create_strategy_generation_agent(
        llm_service=llm_service if use_llm_decision else None,
        config=strategy_config.__dict__,
        use_llm=use_llm_decision
    )
    
    # If signal is available, directly create AgentEnhancedStrategy
    if signal is not None:
        return create_agent_enhanced_strategy(
            signal=signal,
            topk=topk,
            n_drop=n_drop,
            agent=agent,
            llm_service=llm_service,
            use_llm=use_llm_decision,
            use_agent_decision=use_agent_decision,
            risk_degree=config.get("risk_degree", 0.95),
            hold_thresh=config.get("hold_thresh", 1),
        )
    
    # Return Agent (signal needs to be passed during backtesting)
    return agent


def get_strategy_yaml_config(state: AgentState) -> Optional[Dict[str, Any]]:
    """
    Get complete strategy YAML configuration for Qlib backtesting
    
    This function generates a configuration dictionary that can be directly used for Qlib workflow,
    replacing the original default strategy with AgentEnhancedStrategy
    
    Args:
        state: AgentState current state
        
    Returns:
        Complete YAML configuration dictionary, can be used for Qlib backtesting
    """
    import yaml
    from pathlib import Path
    from Agent.strategy_generation_agent import get_strategy_config_for_qlib
    
    strategy_info = state.get("strategy", {})
    config = strategy_info.get("config", {})
    
    # Get YAML configuration path generated by model optimization
    yaml_config_path = strategy_info.get("yaml_config_path")
    
    if not yaml_config_path or not Path(yaml_config_path).exists():
        print(f"[Warning] YAML configuration file does not exist: {yaml_config_path}")
        return None
    
    # Read original configuration
    with open(yaml_config_path, 'r', encoding='utf-8') as f:
        workflow_config = yaml.safe_load(f)
    
    # Generate AgentEnhancedStrategy configuration
    agent_strategy_config = get_strategy_config_for_qlib(
        topk=config.get("topk", 50),
        n_drop=config.get("n_drop", 5),
        use_agent=strategy_info.get("use_agent_decision", True),
        use_llm=strategy_info.get("use_llm_decision", True),
        use_env_llm_config=True,  # Load LLM configuration from environment variables
        risk_degree=config.get("risk_degree", 0.95),
        hold_thresh=config.get("hold_thresh", 1),
    )
    
    # Replace strategy configuration
    if 'task' in workflow_config and 'backtest' in workflow_config['task']:
        workflow_config['task']['backtest']['strategy'] = agent_strategy_config
    
    return workflow_config


def save_strategy_yaml_config(state: AgentState, output_path: Optional[str] = None) -> Optional[str]:
    """
    Save YAML configuration file with AgentEnhancedStrategy
    
    Args:
        state: AgentState current state
        output_path: Output file path (optional, auto-generated by default)
        
    Returns:
        Saved file path, returns None on failure
    """
    import yaml
    from pathlib import Path
    from datetime import datetime
    
    workflow_config = get_strategy_yaml_config(state)
    
    if workflow_config is None:
        return None
    
    # Generate output path
    if output_path is None:
        strategy_info = state.get("strategy", {})
        factor_pool_name = strategy_info.get("factor_pool_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = Path("Qlib_MCP/workspace/qlib_benchmark/benchmarks/train_temp")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"workflow_agent_strategy_{factor_pool_name}_{timestamp}.yaml")
    
    # Save configuration
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(workflow_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"[StrategyGeneration] Strategy configuration saved: {output_path}")
    return output_path
