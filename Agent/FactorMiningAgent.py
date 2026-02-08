"""因子挖掘 Agent"""
import sys
import os
import re
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.file_process import explore_repo_structure,find_training_scripts,select_training_script,read_file_for_llm,find_readme_files,get_top_factors_from_gp_json
from Agent.prompts import (
    FACTOR_MINING_SYSTEM_PROMPT, 
    FACTOR_MINING_ANALYSIS_PROMPT, 
    FACTOR_TEMPLATE_GENERATION_PROMPT,
    VALID_OPERATORS,
    VALID_FEATURES
)

try:
    from utils.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP client unavailable")


class FactorMiningAgent:
    """Factor Mining Agent"""
    
    def __init__(self, llm_service, mcp_server_path: Optional[str] = None):
        """
        Initialize Factor Mining Agent
        
        Args:
            llm_service: LLM service instance (BaseAgent)
            mcp_server_path: MCP server script path, defaults to Qlib_MCP/mcp_server_inline.py
        """
        self.llm = llm_service
        
        # Initialize MCP client
        if MCP_AVAILABLE:
            if mcp_server_path is None:
                # Default path: find Qlib_MCP/mcp_server_inline.py relative to current file
                current_dir = Path(__file__).parent.parent
                mcp_server_path = current_dir / "Qlib_MCP" / "mcp_server_inline.py"
                if not mcp_server_path.exists():
                    print(f"Warning: MCP server script does not exist: {mcp_server_path}")
                    self.mcp_client = None
                else:
                    try:
                        self.mcp_client = SyncMCPClient(str(mcp_server_path))
                        print(f"MCP client initialized successfully: {mcp_server_path}")
                    except Exception as e:
                        print(f"Warning: MCP client initialization failed: {e}")
                        self.mcp_client = None
            else:
                try:
                    self.mcp_client = SyncMCPClient(mcp_server_path)
                    print(f"MCP client initialized successfully: {mcp_server_path}")
                except Exception as e:
                    print(f"Warning: MCP client initialization failed: {e}")
                    self.mcp_client = None
        else:
            self.mcp_client = None
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get available MCP tool list
        
        Returns:
            Tool list, each tool contains name, description, inputSchema
        """
        if not self.mcp_client:
            return []
        
        try:
            tools = self.mcp_client.list_tools()
            return tools
        except Exception as e:
            print(f"Failed to get tool list: {e}")
            return []
    
    def call_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        Call MCP tool
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments dictionary
            
        Returns:
            Tool execution result (text format)
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized")
        
        try:
            result = self.mcp_client.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            error_msg = f"Failed to call tool {tool_name}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _call_gp_training(
        self,
        seed_factors: List[Dict[str, Any]] = None,
        previous_seed: Optional[int] = None,
        iteration: int = 0,
        instruments: str = "csi300",
        train_end_year: int = 2020,
        cuda: str = "0"
    ) -> tuple:
        """
        Concise GP training call method, supports seed factor guidance
        
        Args:
            seed_factors: Seed factor list, from mining_feedback["suggested_seeds"]
            previous_seed: Random seed used in previous round, for auto-increment to avoid duplicates
            iteration: Current iteration round
            instruments: Stock pool
            train_end_year: Training end year
            cuda: CUDA device
            
        Returns:
            tuple: (factor file path, selection result dictionary)
        """
        # Auto-increment seed to avoid duplicate experiments
        new_seed = (previous_seed + 1) if previous_seed is not None else iteration
        
        # Build parameters
        config_params = {
            "instruments": instruments,
            "seed": new_seed,
            "train_end_year": train_end_year,
            "freq": "day",
            "cuda": cuda,
            "seed_factors": seed_factors or []
        }
        
        selection_result = {
            "selected_tool": "train_GP_AlphaSAGE",
            "reason": f"Iteration {iteration}, using {len(seed_factors or [])} seed factors to guide GP mining",
            "suggested_parameters": {"seed": new_seed, "config_params": config_params},
            "seed_factors_count": len(seed_factors or [])
        }
        
        print(f"[FactorMiningAgent] 🚀 Calling train_GP_AlphaSAGE")
        print(f"[FactorMiningAgent] Parameters: seed={new_seed}, instruments={instruments}, seed_factors={len(seed_factors or [])}")
        
        # Call MCP tool
        try:
            factor_file = self.call_mcp_tool("train_GP_AlphaSAGE", {
                "config_params": config_params,
                "task_name": f"gp_iteration_{iteration}_seed_{new_seed}"
            })
            print(f"[FactorMiningAgent] ✅ Training completed, factor file: {factor_file}")
            return factor_file, selection_result
        except Exception as e:
            print(f"[FactorMiningAgent] ❌ GP training failed: {e}")
            raise
    
    def generate_factor_templates_with_llm(
        self, 
        mining_feedback: Dict[str, Any],
        existing_seeds: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call LLM to generate factor templates based on feedback information
        
        Args:
            mining_feedback: Factor evaluation feedback dictionary, containing:
                - iteration: Current iteration round
                - pool_weaknesses: Factor pool weakness list
                - suggested_directions: Suggested mining direction list
                - gp_strategy_hints: GP strategy hints
                - convergence_info: Convergence information
            existing_seeds: Existing seed factor list (to avoid duplicates)
            
        Returns:
            dict: Contains generated factor templates and generation strategy
                - factor_templates: Factor template list
                - generation_strategy: Generation strategy description
        """
        print("[FactorMiningAgent] Starting to call LLM to generate factor templates...")
        
        # Extract feedback information
        iteration = mining_feedback.get("iteration", 0)
        pool_weaknesses = mining_feedback.get("pool_weaknesses", [])
        suggested_directions = mining_feedback.get("suggested_directions", [])
        gp_strategy_hints = mining_feedback.get("gp_strategy_hints", {})
        convergence_info = mining_feedback.get("convergence_info", {})
        
        # Format feedback information
        pool_weaknesses_str = "\n".join([f"- {w}" for w in pool_weaknesses]) if pool_weaknesses else "None"
        suggested_directions_str = "\n".join([f"- {d}" for d in suggested_directions]) if suggested_directions else "None"
        gp_strategy_hints_str = json.dumps(gp_strategy_hints, indent=2, ensure_ascii=False) if gp_strategy_hints else "None"
        convergence_info_str = json.dumps(convergence_info, indent=2, ensure_ascii=False) if convergence_info else "None"
        
        # Format existing seed factors
        if existing_seeds:
            existing_seeds_str = "\n".join([
                f"- {s.get('expression', '')} (IC: {s.get('ic', 'N/A')})"
                for s in existing_seeds[:10]  # Only show first 10
            ])
        else:
            existing_seeds_str = "None"
        
        # Build prompt
        prompt = FACTOR_TEMPLATE_GENERATION_PROMPT.format(
            iteration=iteration,
            pool_weaknesses=pool_weaknesses_str,
            suggested_directions=suggested_directions_str,
            gp_strategy_hints=gp_strategy_hints_str,
            convergence_info=convergence_info_str,
            existing_seeds=existing_seeds_str
        )
        
        try:
            # Call LLM
            response = self.llm.call(
                prompt=prompt,
                stream=False
            )
            
            # Parse JSON response
            result = self.llm.parse_json_response(response)
            
            factor_templates = result.get("factor_templates", [])
            generation_strategy = result.get("generation_strategy", "")
            
            print(f"[FactorMiningAgent] LLM generated {len(factor_templates)} factor templates")
            print(f"[FactorMiningAgent] Generation strategy: {generation_strategy[:100]}...")
            
            # Print generated templates
            for i, template in enumerate(factor_templates[:5]):
                print(f"  {i+1}. [{template.get('category', 'unknown')}] {template.get('expression', '')}")
                print(f"     Description: {template.get('description', '')}")
            
            return {
                "factor_templates": factor_templates,
                "generation_strategy": generation_strategy
            }
            
        except Exception as e:
            print(f"[FactorMiningAgent] LLM call failed: {e}")
            # Return empty result, don't affect subsequent flow
            return {
                "factor_templates": [],
                "generation_strategy": f"LLM call failed: {e}"
            }
    
    def _convert_templates_to_seeds(
        self, 
        factor_templates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert factor templates to GP seed factor format
        
        Args:
            factor_templates: LLM-generated factor template list
            
        Returns:
            List[Dict]: GP seed factor list, format: [{"expression": "...", "ic": 0, "category": "..."}]
        """
        seeds = []
        for template in factor_templates:
            expression = template.get("expression", "")
            if expression:
                seeds.append({
                    "expression": expression,
                    "ic": 0.0,  # Initial IC is 0, GP will re-evaluate
                    "rank_ic": 0.0,
                    "category": template.get("category", "unknown"),
                    "description": template.get("description", ""),
                    "source": "llm_generated"  # Mark source
                })
        return seeds
    
    def _validate_factor_expression(self, expression: str) -> bool:
        """
        Validate if factor expression only uses valid operators and features
        
        Args:
            expression: Factor expression string
            
        Returns:
            bool: Whether expression is valid
        """
        if not expression or not isinstance(expression, str):
            return False
        
        # Extract all function names (operators): match "FuncName(" pattern
        operator_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*\('
        found_operators = set(re.findall(operator_pattern, expression))
        
        # Extract all feature names: match standalone lowercase words (not function calls)
        # Feature names are usually open_, close, high, low, volume, vwap
        feature_pattern = r'\b(open_|close|high|low|volume|vwap)\b'
        found_features = set(re.findall(feature_pattern, expression))
        
        # Check if all operators are valid
        invalid_operators = found_operators - VALID_OPERATORS
        if invalid_operators:
            print(f"[FactorMiningAgent] Found invalid operators: {invalid_operators} in '{expression[:50]}...'")
            return False
        
        # If no features or operators found, might be format error
        if not found_operators and not found_features:
            print(f"[FactorMiningAgent] Expression format abnormal: '{expression[:50]}...'")
            return False
        
        return True
    
    def _filter_valid_seeds(
        self, 
        seeds: List[Dict[str, Any]], 
        source_name: str = "seeds"
    ) -> List[Dict[str, Any]]:
        """
        Filter to keep only seed factors using valid operators
        
        Args:
            seeds: Seed factor list
            source_name: Source name (for logging)
            
        Returns:
            List[Dict]: Filtered valid seed factor list
        """
        valid_seeds = []
        invalid_count = 0
        
        for seed in seeds:
            expression = seed.get("expression", "")
            if self._validate_factor_expression(expression):
                valid_seeds.append(seed)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"[FactorMiningAgent] Filtered out {invalid_count} invalid factor expressions from {source_name}")
        
        return valid_seeds
    
    
    def process(
        self, 
        previous_selection_result: Optional[Dict[str, Any]] = None,
        mining_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process factor mining task
        
        Args:
            previous_selection_result: Previous selection result, containing parameters used last time (e.g., seed)
            mining_feedback: Factor evaluation feedback, used to guide this round's mining
                - iteration: Current iteration round
                - pool_report: Factor pool analysis report (text format)
                - pool_weaknesses: Factor pool weakness list
                - suggested_directions: Suggested mining direction list
                - suggested_seeds: LLM-suggested seed factor list
                - gp_strategy_hints: GP strategy hints (including recommended operators, features, windows, etc.)
                - convergence_info: Convergence information (including whether converged, convergence reason, etc.)
            
        Returns:
            Result dictionary containing factors and logs
        """

        logs = []
        factors = []
        # ============================================================
        # Call LLM to generate factor templates based on feedback
        # ============================================================
        llm_generated_seeds = []
        
        if mining_feedback:
            iteration = mining_feedback.get("iteration", 0)
            suggested_seeds = mining_feedback.get("suggested_seeds", [])
            pool_weaknesses = mining_feedback.get("pool_weaknesses", [])
            suggested_directions = mining_feedback.get("suggested_directions", [])
            gp_strategy_hints = mining_feedback.get("gp_strategy_hints", {})
            convergence_info = mining_feedback.get("convergence_info", {})
            
            print(f"[FactorMiningAgent] Iteration {iteration} mining")
            print(f"[FactorMiningAgent] Received {len(suggested_seeds)} suggested seed factors")
            print(f"[FactorMiningAgent] Factor pool weaknesses: {len(pool_weaknesses)} items")
            print(f"[FactorMiningAgent] Suggested directions: {len(suggested_directions)} items")
            
            # Call LLM to generate factor templates
            llm_result = self.generate_factor_templates_with_llm(
                mining_feedback=mining_feedback,
                existing_seeds=suggested_seeds  # Pass existing seeds to avoid duplicates
            )
            
            factor_templates = llm_result.get("factor_templates", [])
            
            # Convert templates to seed factor format
            llm_generated_seeds = self._convert_templates_to_seeds(factor_templates)
            
            print(f"[FactorMiningAgent] LLM generated {len(llm_generated_seeds)} factor templates as GP seeds")
            
            # Validate and filter invalid seed factors (ensure only AlphaSAGE-supported operators are used)
            valid_suggested_seeds = self._filter_valid_seeds(
                suggested_seeds, source_name="suggested_seeds (from factor_eval)"
            )
            valid_llm_seeds = self._filter_valid_seeds(
                llm_generated_seeds, source_name="llm_generated_seeds"
            )
            
            print(f"[FactorMiningAgent] After validation: suggested_seeds {len(suggested_seeds)} -> {len(valid_suggested_seeds)}")
            print(f"[FactorMiningAgent] After validation: llm_generated_seeds {len(llm_generated_seeds)} -> {len(valid_llm_seeds)}")
            
            # Merge validated seed factors
            all_seeds = valid_suggested_seeds + valid_llm_seeds
            
            # Update suggested_seeds in mining_feedback
            mining_feedback["suggested_seeds"] = all_seeds
            mining_feedback["llm_generated_count"] = len(valid_llm_seeds)
            mining_feedback["filtered_count"] = (len(suggested_seeds) - len(valid_suggested_seeds)) + (len(llm_generated_seeds) - len(valid_llm_seeds))
            print(f"[FactorMiningAgent] mining_feedback: {mining_feedback}")
            
            print(f"[FactorMiningAgent] Total seed factors after merging: {len(all_seeds)}")
        #mining_feedback = None
        # Use concise method to call GP training, pass seed factors
        factorcs_path, selection_result = self._call_gp_training(
            seed_factors=mining_feedback.get("suggested_seeds", []) if mining_feedback else [],
            previous_seed=previous_selection_result.get("suggested_parameters", {}).get("seed", 0) if previous_selection_result else None,
            iteration=mining_feedback.get("iteration", 0) if mining_feedback else 0
        )
        
        factors_list = get_top_factors_from_gp_json(factorcs_path)
        print(f"[FactorMiningAgent] Retrieved {len(factors_list)} factors")
        return factors_list, selection_result

    

    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                self.mcp_client.close()
            except:
                pass