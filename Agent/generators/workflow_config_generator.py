"""Workflow configuration generator"""
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class WorkflowConfigGenerator:
    """Generate Qlib workflow configuration file based on factor pool"""
    
    # Model parameter configuration (based on factor count)
    MODEL_PARAMS = {
        "small": {  # Factor count < 20
            "lgbm": {
                'loss': 'mse',
                'colsample_bytree': 0.95,
                'learning_rate': 0.2,
                'subsample': 0.8789,
                'lambda_l1': 50.0,
                'lambda_l2': 150.0,
                'max_depth': 9,
                'num_leaves': 255,
                'num_threads': 20
            },
            "xgboost": {
                'colsample_bytree': 0.95,
                'eta': 0.0421,
                'eval_metric': 'rmse',
                'max_depth': 9,
                'n_estimators': 647,
                'nthread': 20,
                'subsample': 0.8789,
                'alpha': 50.0,
                'lambda': 150.0
            }
        },
        "medium": {  # Factor count 20-50
            "lgbm": {
                'loss': 'mse',
                'colsample_bytree': 0.8879,
                'learning_rate': 0.2,
                'subsample': 0.8789,
                'lambda_l1': 205.6999,
                'lambda_l2': 580.9768,
                'max_depth': 8,
                'num_leaves': 210,
                'num_threads': 20
            },
            "xgboost": {
                'colsample_bytree': 0.8879,
                'eta': 0.0421,
                'eval_metric': 'rmse',
                'max_depth': 8,
                'n_estimators': 647,
                'nthread': 20,
                'subsample': 0.8789
            }
        },
        "large": {  # Factor count >= 50
            "lgbm": {
                'loss': 'mse',
                'colsample_bytree': 0.7,
                'learning_rate': 0.15,
                'subsample': 0.8,
                'lambda_l1': 300.0,
                'lambda_l2': 800.0,
                'max_depth': 6,
                'num_leaves': 127,
                'num_threads': 20
            },
            "xgboost": {
                'colsample_bytree': 0.7,
                'eta': 0.03,
                'eval_metric': 'rmse',
                'max_depth': 6,
                'n_estimators': 800,
                'nthread': 20,
                'subsample': 0.8,
                'alpha': 300.0,
                'lambda': 800.0
            }
        }
    }
    
    def __init__(self, base_config_dir: Optional[Path] = None):
        """
        Initialize configuration generator
        
        Args:
            base_config_dir: Base configuration file directory
        """
        if base_config_dir is None:
            base_config_dir = (
                Path(__file__).parent.parent.parent / 
                "Qlib_MCP" / "workspace" / "qlib_benchmark" / "benchmarks" / "train_temp"
            )
        self.base_config_dir = Path(base_config_dir)
    
    def generate(
        self,
        module_path: str,
        output_path: Optional[str] = None,
        model_type: str = "xgboost",
        train_start_year: int = 2010,
        train_end_year: int = 2020,
        instruments: str = "csi300",
        segments: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Generate workflow configuration file
        
        Args:
            module_path: Factor pool module path (e.g., "qlib_benchmark.factor_pools.custom_factors_xxx")
            output_path: Output configuration file path
            model_type: Model type "xgboost" or "lgbm"
            train_start_year: Training set start year
            train_end_year: Training set end year
            instruments: Stock pool
            segments: Custom time period dictionary
            
        Returns:
            Generated configuration file path
        """
        # Load base configuration
        base_config_path = self.base_config_dir / "workflow_base_lightgbm_model.yaml"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base configuration file does not exist: {base_config_path}")
        
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Parse module path to get class name
        class_name, factors_count = self._parse_module_path(module_path)
        
        # Update handler configuration
        self._update_handler_config(
            config, module_path, class_name,
            train_start_year, train_end_year, instruments
        )
        
        # Update time period configuration
        self._update_segments(config, train_start_year, train_end_year, segments)
        
        # Update model configuration
        self._update_model_config(config, model_type, factors_count)
        
        # Generate output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_config_dir / f"workflow_config_{class_name}_{timestamp}.yaml"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        self._print_config_info(output_path, class_name, module_path, config, train_end_year, factors_count)
        
        return str(output_path)
    
    def _parse_module_path(self, module_path: str) -> tuple:
        """Parse module path, get class name and factor count"""
        module_parts = module_path.split('.')
        if len(module_parts) < 3 or module_parts[-2] != 'factor_pools':
            raise ValueError(f"Invalid module path format: {module_path}")
        
        module_name = module_parts[-1]
        factors_count = 20  # Default value
        
        # Try to read from metadata file
        qlib_benchmark_path = Path(__file__).parent.parent.parent / "Qlib_MCP" / "workspace" / "qlib_benchmark"
        metadata_file = qlib_benchmark_path / "factor_pools" / f"{module_name}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    class_name = metadata.get('class_name')
                    factors_count = metadata.get('factors_count', 20)
                    if class_name:
                        return class_name, factors_count
            except Exception as e:
                print(f"Warning: Failed to read metadata file: {e}")
        
        # Derive class name from module name (snake_case -> PascalCase)
        parts = module_name.split('_')
        class_name = ''.join(word.capitalize() for word in parts)
        return class_name, factors_count
    
    def _update_handler_config(
        self,
        config: Dict,
        module_path: str,
        class_name: str,
        train_start_year: int,
        train_end_year: int,
        instruments: str
    ) -> None:
        """Update handler configuration"""
        data_handler_config = config.get('data_handler_config', {})
        
        # Get from existing handler configuration
        if not data_handler_config:
            handler_config = config.get('task', {}).get('dataset', {}).get('kwargs', {}).get('handler', {})
            if isinstance(handler_config, dict) and 'kwargs' in handler_config:
                data_handler_config = handler_config['kwargs']
        
        data_handler_config.update({
            'start_time': f'{train_start_year}-01-01',
            'end_time': f'{train_end_year + 4}-12-31',
            'fit_start_time': f'{train_start_year}-01-01',
            'fit_end_time': f'{train_end_year}-12-31',
            'instruments': instruments
        })
        
        if 'task' in config and 'dataset' in config['task'] and 'kwargs' in config['task']['dataset']:
            config['task']['dataset']['kwargs']['handler'] = {
                'class': class_name,
                'module_path': module_path,
                'kwargs': data_handler_config
            }
    
    def _update_segments(
        self,
        config: Dict,
        train_start_year: int,
        train_end_year: int,
        segments: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """Update time period configuration"""
        if 'task' not in config or 'dataset' not in config['task']:
            return
        
        if segments is not None:
            segments_config = segments
        else:
            segments_config = {
                'train': [f'{train_start_year}-01-01', f'{train_end_year}-12-31'],
                'valid': [f'{train_end_year + 1}-01-01', f'{train_end_year + 1}-12-31'],
                'test': [f'{train_end_year + 2}-01-01', f'{train_end_year + 4}-12-31']
            }
        
        config['task']['dataset']['kwargs']['segments'] = segments_config
        
        # Synchronize backtest time
        if 'port_analysis_config' in config and 'backtest' in config['port_analysis_config']:
            config['port_analysis_config']['backtest']['start_time'] = segments_config['test'][0]
            config['port_analysis_config']['backtest']['end_time'] = segments_config['test'][1]
    
    def _update_model_config(
        self,
        config: Dict,
        model_type: str,
        factors_count: int
    ) -> None:
        """Update model configuration"""
        if 'task' not in config or 'model' not in config['task']:
            return
        
        # Select parameter tier based on factor count
        if factors_count < 20:
            params_key = "small"
        elif factors_count < 50:
            params_key = "medium"
        else:
            params_key = "large"
        
        if model_type == "xgboost":
            params = self.MODEL_PARAMS[params_key]["xgboost"]
            config['task']['model'] = {
                'class': 'XGBModel',
                'module_path': 'qlib.contrib.model.xgboost',
                'kwargs': params
            }
        else:
            params = self.MODEL_PARAMS[params_key]["lgbm"]
            if 'kwargs' not in config['task']['model']:
                config['task']['model']['kwargs'] = {}
            config['task']['model']['kwargs'].update(params)
        
        print(f"[Config Generation] Factor count: {factors_count}, using {params_key} configuration")
    
    def _print_config_info(
        self,
        output_path: Path,
        class_name: str,
        module_path: str,
        config: Dict,
        train_end_year: int,
        factors_count: int
    ) -> None:
        """Print configuration information"""
        print(f"[Config Generation] Configuration file generated: {output_path}")
        print(f"[Config Generation] Factor pool class name: {class_name}")
        print(f"[Config Generation] Module path: {module_path}")
        
        segments_info = config.get('task', {}).get('dataset', {}).get('kwargs', {}).get('segments', {})
        if segments_info:
            print(f"[Config Generation] Time period configuration (based on train_end_year={train_end_year}):")
            if 'train' in segments_info:
                print(f"  - Training set: {segments_info['train'][0]} to {segments_info['train'][1]}")
            if 'valid' in segments_info:
                print(f"  - Validation set: {segments_info['valid'][0]} to {segments_info['valid'][1]}")
            if 'test' in segments_info:
                print(f"  - Test set: {segments_info['test'][0]} to {segments_info['test'][1]}")

