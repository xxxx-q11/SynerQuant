# custom_factor_handler.py
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
from qlib.contrib.data.loader import QlibDataLoader

# Define default processors
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class CustomFactorHandler(DataHandlerLP):
    """Custom factor handler - supports custom factor list"""
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        custom_factors=None,  # New: custom factor list
        **kwargs,
    ):
        # Save custom factor list
        self.custom_factors = custom_factors
        
        # Process processor configuration
        from qlib.contrib.data.handler import check_transform_proc
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # Configure data loader
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),  # Use custom factors or default factors
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        """Define custom factors
        
        If custom_factors parameter is provided, use mined factors
        Otherwise use default example factors
        """
        # If custom factor list is provided, use it
        if self.custom_factors:
            return self._parse_custom_factors(self.custom_factors)
        
        # Otherwise use default example factors
        fields = []
        names = []
        
        # ========== Example factors (please replace with your 50 factors) ==========
        # Factors 1-10: Price-related factors
        fields += [
            "$close/$open",  # Closing price / opening price
            "($high-$low)/$close",  # Amplitude
            "($close-Ref($close, 1))/$close",  # Return rate
            "Mean($close, 5)/$close",  # 5-day moving average ratio
            "Mean($close, 10)/$close",  # 10-day moving average ratio
            "Mean($close, 20)/$close",  # 20-day moving average ratio
            "Std($close, 5)/$close",  # 5-day volatility
            "Std($close, 10)/$close",  # 10-day volatility
            "Max($high, 5)/$close",  # 5-day highest price ratio
            "Min($low, 5)/$close",  # 5-day lowest price ratio
        ]
        names += [f"PRICE_{i}" for i in range(1, 11)]
        
        # Factors 11-20: Volume-related factors
        fields += [
            "$volume/Mean($volume, 5)",  # Volume ratio
            "$volume/Mean($volume, 10)",  # Volume ratio
            "Corr($close, $volume, 5)",  # Price-volume correlation
            "Corr($close, $volume, 10)",  # Price-volume correlation
            "Mean($volume, 5)/Mean($volume, 20)",  # Volume moving average ratio
            "Std($volume, 5)/Mean($volume, 5)",  # Volume volatility
            "Log($volume+1)",  # Logarithmic volume
            "($volume-Ref($volume, 1))/$volume",  # Volume change rate
            "Sum($volume, 5)/Sum($volume, 20)",  # Volume cumulative ratio
            "Rank($volume, 20)",  # Volume ranking
        ]
        names += [f"VOLUME_{i}" for i in range(11, 21)]
        
        # Factors 21-30: Technical indicator factors
        fields += [
            "Ref($close, 5)/$close",  # 5-day return rate
            "Ref($close, 10)/$close",  # 10-day return rate
            "Ref($close, 20)/$close",  # 20-day return rate
            "Rsquare($close, 5)",  # 5-day R-squared
            "Rsquare($close, 10)",  # 10-day R-squared
            "Resi($close, 5)/$close",  # 5-day residual
            "Resi($close, 10)/$close",  # 10-day residual
            "Slope($close, 5)/$close",  # 5-day slope
            "Slope($close, 10)/$close",  # 10-day slope
            "Quantile($close, 20, 0.8)/$close",  # 80th percentile
        ]
        names += [f"TECH_{i}" for i in range(21, 31)]
        
        # Factors 31-40: Momentum factors
        fields += [
            "Mean($close>Ref($close, 1), 5)",  # Proportion of up days
            "Mean($close>Ref($close, 1), 10)",  # Proportion of up days
            "Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",  # RSI-like
            "Sum(Greater($close-Ref($close, 1), 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",  # RSI-like
            "($close-Min($low, 5))/(Max($high, 5)-Min($low, 5)+1e-12)",  # Stochastic indicator
            "($close-Min($low, 10))/(Max($high, 10)-Min($low, 10)+1e-12)",  # Stochastic indicator
            "IdxMax($high, 5)/5",  # Highest price position
            "IdxMin($low, 5)/5",  # Lowest price position
            "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",  # Price-volume change correlation
            "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",  # Price-volume change correlation
        ]
        names += [f"MOMENTUM_{i}" for i in range(31, 41)]
        
        # Factors 41-50: Other factors (you can add more custom factors)
        fields += [
            "$vwap/$close",  # VWAP ratio
            "($close-$vwap)/$close",  # Closing price and VWAP difference
            "Mean($vwap, 5)/$close",  # VWAP moving average
            "Rank($close, 20)",  # Price ranking
            "Rank($volume, 20)",  # Volume ranking
            "($high+$low+$close)/3/$close",  # Typical price
            "Abs($close-Ref($close, 1))/$close",  # Absolute change rate
            "Mean(Abs($close-Ref($close, 1)), 5)/$close",  # Mean absolute change
            "($close-Mean($close, 20))/Std($close, 20)",  # Z-score
            "Quantile($close, 20, 0.2)/$close",  # 20th percentile
        ]
        names += [f"OTHER_{i}" for i in range(41, 51)]
        
        # ========== Replace the above example factors with your 50 factors ==========
        # Your factors should use Qlib expression syntax, for example:
        # - "$close" represents closing price
        # - "Ref($close, 5)" represents closing price 5 days ago
        # - "Mean($close, 10)" represents 10-day moving average
        # - "Std($close, 5)" represents 5-day standard deviation
        # - "Corr($close, $volume, 10)" represents 10-day correlation
        # etc...
        
        return fields, names
    
    def _parse_custom_factors(self, factors_list):
        """Parse factor list passed from FactorMiningAgent
        
        Args:
            factors_list: Factor list, format [{"expression": "factor expression", "ic": IC value}, ...]
            
        Returns:
            (fields, names) tuple for Qlib feature configuration
        """
        fields = []
        names = []
        
        for i, factor in enumerate(factors_list):
            if isinstance(factor, dict):
                # Support two formats: {"expression": "...", "ic": ...} or {"factor expression": ic value}
                if "expression" in factor:
                    expression = factor["expression"]
                    ic = factor.get("ic", 0.0)
                else:
                    # Assume dictionary has only one key-value pair
                    expression = list(factor.keys())[0]
                    ic = factor[expression]
                
                fields.append(expression)
                names.append(f"FACTOR_{i+1}_IC{ic:.4f}" if isinstance(ic, (int, float)) else f"FACTOR_{i+1}")
        
        print(f"[CustomFactorHandler] Loaded {len(fields)} custom factors")
        return fields, names

    def get_label_config(self):
        """Define label (prediction target)"""
        # Default label: future 2-day return rate
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


def create_handler_from_factors(factors_list, **kwargs):
    """
    Create CustomFactorHandler instance from factor list
    
    Args:
        factors_list: Factor list, format [{"expression": "factor expression", "ic": IC value}, ...]
        **kwargs: Other parameters passed to CustomFactorHandler
        
    Returns:
        CustomFactorHandler instance
        
    Example:
        >>> factors = [{"expression": "$close/$open", "ic": 0.05}, ...]
        >>> handler = create_handler_from_factors(factors, instruments="csi300")
    """
    return CustomFactorHandler(custom_factors=factors_list, **kwargs)