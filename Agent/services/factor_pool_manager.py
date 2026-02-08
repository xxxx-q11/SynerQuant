"""Factor pool management service"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Import factor pool registry
try:
    qlib_benchmark_path = Path(__file__).parent.parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
    if str(qlib_benchmark_path) not in sys.path:
        sys.path.insert(0, str(qlib_benchmark_path))
    from factor_pool_registry import FactorPoolRegistry
    FACTOR_REGISTRY_AVAILABLE = True
except ImportError:
    FACTOR_REGISTRY_AVAILABLE = False
    FactorPoolRegistry = None


class FactorPoolManager:
    """Factor pool manager - responsible for factor pool registration, loading, saving, etc."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize factor pool manager
        
        Args:
            data_dir: Data directory, defaults to trading_agent/data/
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default factor pool file paths
        self.base_pool_path = self.data_dir / "base_pool.json"
        self.sota_pool_path = self.data_dir / "sota_pool.json"
        self.eval_results_dir = self.data_dir / "eval_results"
    
    def load_base_pool(self, path: Optional[Path] = None) -> List[str]:
        """
        Load base factor pool
        
        Args:
            path: Factor pool file path
            
        Returns:
            Factor expression list
        """
        path = path or self.base_pool_path
        return self._load_pool(path, "Base factor pool")
    
    def load_sota_pool(self, path: Optional[Path] = None) -> List[str]:
        """
        Load SOTA factor pool
        
        Args:
            path: Factor pool file path
            
        Returns:
            Factor expression list
        """
        path = path or self.sota_pool_path
        return self._load_pool(path, "SOTA factor pool")
    
    def _load_pool(self, path: Path, pool_name: str) -> List[str]:
        """Generic factor pool loading method"""
        path = Path(path)
        
        if not path.exists():
            print(f"Warning: {pool_name} file does not exist: {path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                factors = json.load(f)
            
            if isinstance(factors, list):
                print(f"Successfully loaded {pool_name}, contains {len(factors)} factors")
                return factors
            else:
                print(f"Warning: {pool_name} file format is incorrect")
                return []
                
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {pool_name} JSON file: {e}")
            return []
        except Exception as e:
            print(f"Error: Exception occurred while loading {pool_name}: {e}")
            return []
    
    def append_to_sota_pool(self, factor: str, path: Optional[Path] = None) -> bool:
        """
        Append factor to SOTA factor pool
        
        Args:
            factor: Factor expression
            path: SOTA factor pool path
            
        Returns:
            Whether append was successful
        """
        path = path or self.sota_pool_path
        
        if not isinstance(factor, str) or not factor.strip():
            print("Error: Factor must be a non-empty string")
            return False
        
        factor = factor.strip()
        
        try:
            # Read existing factors
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    factors = json.load(f)
            else:
                factors = []
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check for duplicates
            if factor in factors:
                print(f"Factor already exists in SOTA factor pool")
                return False
            
            # Append and save
            factors.append(factor)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(factors, f, ensure_ascii=False, indent=0)
            
            print(f"Successfully appended factor to SOTA factor pool, currently {len(factors)} factors")
            return True
            
        except Exception as e:
            print(f"Error: Failed to append factor: {e}")
            return False
    
    def register_merged_pool(
        self,
        new_factor: Optional[Dict[str, Any]],
        sota_factors: List[str]
    ) -> Optional[Tuple[str, str]]:
        """
        Merge new factor and SOTA factor pool and register as new factor pool
        
        Args:
            new_factor: New factor dictionary, containing 'qlib_expression' and 'ic'
            sota_factors: SOTA factor expression list
            
        Returns:
            (module_path, workflow_config_path) or None
        """
        if not FACTOR_REGISTRY_AVAILABLE:
            print("Warning: Factor pool registry unavailable")
            return None
        
        try:
            # Prepare factor list
            factors_for_registry = []
            
            # Add new factor
            if new_factor:
                expression = new_factor.get('qlib_expression')
                if expression:
                    factors_for_registry.append({
                        'expression': expression,
                        'ic': new_factor.get('ic', 0.0)
                    })
            
            # Add SOTA factors
            for sota_expr in sota_factors:
                if isinstance(sota_expr, str) and sota_expr.strip():
                    factors_for_registry.append({
                        'expression': sota_expr,
                        'ic': 0.0
                    })
            
            if not factors_for_registry:
                print("Warning: No factors to register")
                return None
            
            # Generate factor pool name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pool_name = f"CustomFactors_{timestamp}"
            
            # Generate description
            if new_factor:
                description = (
                    f"Merged factor pool\n"
                    f"New factor: {new_factor.get('qlib_expression', 'N/A')}\n"
                    f"SOTA factor count: {len(sota_factors)}\n"
                    f"Total factor count: {len(factors_for_registry)}"
                )
            else:
                description = (
                    f"Original factor pool\n"
                    f"SOTA factor count: {len(sota_factors)}\n"
                    f"Total factor count: {len(factors_for_registry)}"
                )
            
            # Register factor pool
            registry_dir = (
                Path(__file__).parent.parent.parent / 
                "Qlib_MCP" / "workspace" / "qlib_benchmark" / "factor_pools"
            )
            registry = FactorPoolRegistry(registry_dir=str(registry_dir))
            
            module_path = registry.register_factor_pool(
                pool_name=pool_name,
                factors=factors_for_registry,
                description=description
            )
            
            print(f"[Factor Pool Registration] Successfully registered: {pool_name}")
            print(f"[Factor Pool Registration] Module path: {module_path}")
            print(f"[Factor Pool Registration] Factor count: {len(factors_for_registry)}")
            
            return module_path
            
        except Exception as e:
            print(f"Warning: Failed to register factor pool: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_eval_result(
        self,
        factor: str,
        eval_result: Dict[str, Any],
        sota_pool: Optional[List[str]] = None
    ) -> str:
        """
        Save factor evaluation result
        
        Args:
            factor: Factor expression
            eval_result: Evaluation result dictionary
            sota_pool: SOTA factor pool at that time
            
        Returns:
            Saved file path
        """
        import hashlib
        
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        factor_hash = hashlib.md5(factor.encode('utf-8')).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_result_{factor_hash}_{timestamp}.json"
        file_path = self.eval_results_dir / filename
        
        # Build save data
        save_data = {
            "factor": factor,
            "timestamp": timestamp,
            "eval_result": eval_result,
            "SOTA_pool": sota_pool
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved evaluation result: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"Error: Failed to save evaluation result: {e}")
            return ""
    
    def save_mining_feedback(
        self,
        mining_feedback: Dict[str, Any],
        sota_pool_list: Optional[List[str]] = None
    ) -> str:
        """
        Save factor mining feedback
        
        Args:
            mining_feedback: Mining feedback dictionary, containing iteration, pool_report, suggested_directions, etc.
            sota_pool_list: SOTA factor pool list at that time
            
        Returns:
            Saved file path
        """
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        iteration = mining_feedback.get("iteration", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mining_feedback_iter{iteration}_{timestamp}.json"
        file_path = self.eval_results_dir / filename
        
        # Build save data
        save_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "mining_feedback": mining_feedback,
            "SOTA_pool": sota_pool_list
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved mining feedback: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"Error: Failed to save mining feedback: {e}")
            return ""
    
    def save_literature_support(
        self,
        factor_expr: str,
        literature_support: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
        sota_pool_list: Optional[List[str]] = None
    ) -> str:
        """
        Save factor literature support information
        
        Args:
            factor_expr: Factor expression
            literature_support: Literature support dictionary, containing search_query, literature_results, 
                             literature_explanation, academic_support_level, etc.
            eval_result: Factor evaluation result (optional)
            sota_pool_list: SOTA factor pool list at that time (optional)
            
        Returns:
            Saved file path
        """
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use hash value of factor expression or first few characters as part of filename
        factor_hash = hash(factor_expr) % 1000000
        filename = f"literature_support_{factor_hash}_{timestamp}.json"
        file_path = self.eval_results_dir / filename
        
        # Build save data
        save_data = {
            "timestamp": timestamp,
            "factor_expression": factor_expr,
            "literature_support": literature_support,
            "eval_result": eval_result,
            "SOTA_pool": sota_pool_list
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved literature support information: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"Error: Failed to save literature support information: {e}")
            return ""

