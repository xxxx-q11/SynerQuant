"""
Strategy Generation Agent - Inherits Qlib Strategy class, makes intelligent trading decisions in backtesting

Reference Qlib TopkDropoutStrategy implementation
"""
import sys
import copy
import yaml
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas/numpy not available, some features limited")

# Try importing MCP client
try:
    from utils.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP client not available")

# Try importing file utilities
try:
    from Agent.utils.file_utils import FileUtils
except ImportError:
    FileUtils = None


# =============================================================================
# Data Structure Definitions
# =============================================================================

@dataclass
class TradeSignal:
    """Trading signal"""
    stock_code: str
    action: str  # "buy", "sell", "hold"
    target_weight: float  # Target weight (0-1)
    reason: str
    score: float = 0.0  # Model score
    current_weight: float = 0.0  # Current weight
    confidence: float = 1.0  # Confidence (0-1)
    priority: int = 0  # Priority, higher value means higher priority


@dataclass
class PortfolioState:
    """Portfolio state"""
    holdings: Dict[str, float]  # {stock_code: weight}
    cash_ratio: float  # Cash ratio
    total_value: float  # Total value
    date: str  # Current date


@dataclass 
class TradeDecision:
    """Trading decision result"""
    buy_list: List[TradeSignal]
    sell_list: List[TradeSignal]
    hold_list: List[TradeSignal]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """
    Holdings vs recommendations comparison result
    
    Reference TopkDropoutStrategy set comparison logic:
    - sell_candidates: Stocks in current holdings but not in Top K recommendations (to sell)
    - buy_candidates: Stocks in Top K recommendations but not in current holdings (to buy)
    - hold_candidates: Intersection of current holdings and Top K recommendations (continue holding)
    - ranked_holdings: Current holdings sorted by model score
    - ranked_recommendations: Recommended stocks sorted by score
    """
    sell_candidates: Set[str]  # A - B: In holdings but not in recommendations
    buy_candidates: Set[str]   # B - A: In recommendations but not in holdings
    hold_candidates: Set[str]  # A ∩ B: Intersection of both
    
    # Ranked lists with scores
    ranked_holdings: List[Tuple[str, float]]  # [(stock_code, score), ...] sorted by score descending
    ranked_recommendations: List[Tuple[str, float]]  # [(stock_code, score), ...] sorted by score descending
    
    # Suggested sell and buy lists (considering n_drop limit)
    suggested_sell: List[str] = field(default_factory=list)  # Stocks actually suggested to sell
    suggested_buy: List[str] = field(default_factory=list)   # Stocks actually suggested to buy
    
    # Score mapping
    score_map: Dict[str, float] = field(default_factory=dict)  # {stock_code: score}


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    topk: int = 50  # Top K number of stocks
    n_drop: int = 10  # Number of stocks to drop per rebalancing
    max_turnover: float = 0.3  # Maximum turnover rate
    min_trade_value: float = 10000  # Minimum trade value
    open_cost: float = 0.0005  # Buy cost
    close_cost: float = 0.0015  # Sell cost
    min_cost: float = 5  # Minimum trade cost
    limit_threshold: float = 0.095  # Limit up/down threshold


# =============================================================================
# Strategy Generation Agent Core Class
# =============================================================================

class StrategyGenerationAgent:
    """
    Strategy Generation Agent
    
    Responsible for reviewing held stocks and model-recommended Top_K stocks during backtesting,
    deciding which stocks to keep, buy, and sell
    
    Also supports running complete backtesting workflow in LangGraph
    """
    
    def __init__(
        self,
        llm_service=None,
        config: Optional[StrategyConfig] = None,
        use_llm_decision: bool = True,
        mcp_server_path: Optional[str] = None,
        news_data_path: Optional[str] = None,
        news_batch_size: int = 10
    ):
        """
        Initialize Strategy Generation Agent
        
        Args:
            llm_service: LLM service instance (optional, for intelligent decision making)
            config: Strategy configuration
            use_llm_decision: Whether to use LLM for decision making
            mcp_server_path: MCP server script path (for running backtests)
            news_data_path: News data path (can be directory path or single JSON file path)
                           - Directory path: Auto-load by month, filename format: eastmoney_news_YYYY_processed_YYYY_MM.json
                           - File path: Compatible with old version, loads single JSON file
            news_batch_size: Number of stocks to analyze per batch
        """
        self.llm = llm_service
        self.config = config or StrategyConfig()
        self.use_llm_decision = use_llm_decision
        
        # Trade history records
        self.trade_history: List[Dict[str, Any]] = []
        
        # Decision logs
        self.decision_logs: List[str] = []
        
        # MCP client (for running backtests)
        self.mcp_client = None
        self._init_mcp_client(mcp_server_path)
        
        # Configure paths
        # Detect if in container (by checking if /workspace exists)
        is_in_container = Path('/workspace').exists()
        default_news_path = news_data_path
        if is_in_container:
            # Container environment
            self.output_dir = Path("/workspace/qlib_benchmark/benchmarks/train_temp")
            if not default_news_path:
                default_news_path = '/workspace/news_data/csi_300'
            logger.info(f"[Path] Detected container environment, using path: {default_news_path}")
        else:
            # Host environment
            self.output_dir = Path("Qlib_MCP/workspace/qlib_benchmark/benchmarks/train_temp")
            if not default_news_path:
                default_news_path = 'Qlib_MCP/workspace/news_data/csi_300'
            logger.info(f"[Path] Detected host environment, using path: {default_news_path}")
        
        # News data configuration
        self.news_data_path = default_news_path 
        
        # News data configuration - supports directory path (load by month) or single file path
        self.news_batch_size = news_batch_size
        # Cache news data by month: {month: {date: {stock_code: [news_list]}}}
        # month format: "2025-01", "2025-02", etc.
        self._news_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Macro news path (in same directory as stock news)
        if self.news_data_path:
            news_path = Path(self.news_data_path)
            if news_path.exists():
                if news_path.is_dir():
                    # Directory mode: Macro_News.json in same directory
                    self.macro_news_path = news_path / "Macro_News.json"
                    # Midday summary path: Midday Market Summary.json in same directory
                    self.midday_summary_path = news_path / "Midday Market Summary.json"
                elif news_path.is_file():
                    # File mode: Macro_News.json in same directory
                    self.macro_news_path = news_path.parent / "Macro_News.json"
                    # Midday summary path: Midday Market Summary.json in same directory
                    self.midday_summary_path = news_path.parent / "Midday Market Summary.json"
                else:
                    self.macro_news_path = None
                    self.midday_summary_path = None
            else:
                self.macro_news_path = None
                self.midday_summary_path = None
        else:
            self.macro_news_path = None
            self.midday_summary_path = None
        
        # Short-term memory system: keep at most 3 trading days
        # Format: {date: {"macro_news_summary": str, "stock_reasons": {stock_code: reason}}}
        self._short_term_memory: Dict[str, Dict[str, Any]] = {}
        self._max_memory_days = 3
        
        # Compressed macro news cache: {date: compressed_text}
        self._compressed_macro_news_cache: Dict[str, str] = {}
        
        # Log news data path configuration
        logger.info(f"[News] News data path configuration: {self.news_data_path}")
        news_path = Path(self.news_data_path) if self.news_data_path else None
        if news_path and news_path.exists():
            if news_path.is_dir():
                logger.info(f"[News] Detected directory mode, path exists: {news_path.absolute()}")
            elif news_path.is_file():
                logger.info(f"[News] Detected file mode, path exists: {news_path.absolute()}")
        else:
            logger.warning(f"[News] News data path does not exist: {news_path.absolute() if news_path else 'None'}")
        
        if self.macro_news_path and self.macro_news_path.exists():
            logger.info(f"[News] Macro news path: {self.macro_news_path.absolute()}")
        else:
            logger.warning(f"[News] Macro news file does not exist: {self.macro_news_path}")
        
        if self.midday_summary_path and self.midday_summary_path.exists():
            logger.info(f"[News] Midday summary path: {self.midday_summary_path.absolute()}")
        else:
            logger.warning(f"[News] Midday summary file does not exist: {self.midday_summary_path}")
    
    def _init_mcp_client(self, mcp_server_path: Optional[str]) -> None:
        """Initialize MCP client"""
        if not MCP_AVAILABLE:
            return
        
        if mcp_server_path is None:
            current_dir = Path(__file__).parent.parent
            mcp_server_path = current_dir / "Qlib_MCP" / "mcp_server_inline.py"
        else:
            mcp_server_path = Path(mcp_server_path)
        
        if not mcp_server_path.exists():
            print(f"Warning: MCP server script does not exist: {mcp_server_path}")
            return
        
        try:
            self.mcp_client = SyncMCPClient(str(mcp_server_path))
            print(f"[StrategyGeneration] MCP client initialized successfully")
        except Exception as e:
            print(f"Warning: MCP client initialization failed: {e}")
    
    # =========================================================================
    # News Data Loading and Retrieval
    # =========================================================================
    
    def _get_month_from_date(self, date: str) -> str:
        """
        Extract month from date string
        
        Args:
            date: Date string, format like "2025-01-02" or "2025-01-02 10:00:00"
            
        Returns:
            Month string, format like "2025-01"
        """
        try:
            # Extract date part (remove time part)
            date_part = date[:10] if len(date) >= 10 else date
            # Extract year-month part
            year_month = date_part[:7]  # "2025-01"
            return year_month
        except Exception:
            logger.warning(f"[News] Cannot extract month from date {date}")
            return ""
    
    def _get_news_file_path(self, month: str) -> Optional[Path]:
        """
        Get news file path based on month
        
        Args:
            month: Month string, format like "2025-01"
            
        Returns:
            News file path, returns None if doesn't exist
        """
        if not self.news_data_path:
            logger.warning(f"[News] news_data_path not configured")
            return None
        
        logger.debug(f"[News] Starting to get file path, month={month}, news_data_path={self.news_data_path}")
        news_path = Path(self.news_data_path)
        
        # Check if path exists
        if not news_path.exists():
            logger.warning(f"[News] News data path does not exist: {news_path} (absolute path: {news_path.absolute()})")
            # Try to handle as directory (might be relative path issue)
            try:
                month_num = month.split('-')[1]
                year = month.split('-')[0]
                filename = f"eastmoney_news_{year}_processed_{year}_{month_num}.json"
                file_path = news_path / filename
                logger.debug(f"[News] Trying to build file path: {file_path} (absolute path: {file_path.absolute()})")
                if file_path.exists():
                    logger.info(f"[News] Found news file: {file_path}")
                    return file_path
                else:
                    logger.warning(f"[News] News file does not exist: {file_path}")
                    return None
            except (IndexError, AttributeError) as e:
                logger.error(f"[News] Month format error: {month}, error: {e}")
                return None
        
        # Determine if directory or file
        if news_path.is_dir():
            # Directory mode: load by month
            # Filename format: eastmoney_news_2025_processed_2025_MM.json
            logger.debug(f"[News] Detected directory mode: {news_path}")
            try:
                month_num = month.split('-')[1]  # Extract month number, e.g. "01"
                year = month.split('-')[0]  # Extract year, e.g. "2025"
                filename = f"eastmoney_news_{year}_processed_{year}_{month_num}.json"
                file_path = news_path / filename
                logger.debug(f"[News] Building file path: {file_path} (absolute path: {file_path.absolute()})")
                if file_path.exists():
                    logger.info(f"[News] Found news file: {file_path}")
                    return file_path
                else:
                    logger.warning(f"[News] News file does not exist: {file_path}")
                    # List files in directory to help debugging
                    try:
                        files = list(news_path.glob("*.json"))
                        logger.debug(f"[News] JSON files in directory: {[f.name for f in files[:10]]}")
                    except Exception as e:
                        logger.debug(f"[News] Cannot list directory files: {e}")
                    return None
            except (IndexError, AttributeError) as e:
                logger.error(f"[News] Month format error: {month}, error: {e}")
                return None
        elif news_path.is_file():
            # File mode: compatible with old version (single file)
            # In file mode, all months use the same file
            logger.debug(f"[News] Detected file mode: {news_path}")
            return news_path
        else:
            logger.warning(f"[News] Path is neither directory nor file: {news_path}")
            return None
    
    def _load_news_data_by_month(self, month: str) -> Dict[str, Any]:
        """
        Load news data by month (with caching)
        
        Args:
            month: Month string, format like "2025-01"
        
        News data format: {date: {stock_code: [news_list]}}
        
        Returns:
            News data dictionary for that month
        """
        # Check cache
        if month in self._news_data_cache:
            logger.debug(f"[News] Loading {month} data from cache")
            return self._news_data_cache[month]
        
        logger.debug(f"[News] Starting to load {month} news data")
        
        # Get file path
        file_path = self._get_news_file_path(month)
        if not file_path:
            logger.warning(f"[News] News data file does not exist: {month}, news_data_path={self.news_data_path}")
            self._news_data_cache[month] = {}
            return {}
        
        logger.info(f"[News] Loading news file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            logger.debug(f"[News] File loaded successfully, contains {len(all_data)} date keys")
            
            # Determine if directory mode or file mode
            news_path = Path(self.news_data_path)
            if news_path.exists() and news_path.is_dir():
                # Directory mode: directly use loaded data (already data for that month)
                month_data = all_data
                logger.debug(f"[News] Directory mode: using all data")
            else:
                # File mode: filter data for that month from all data
                month_data = {}
                for date, stocks_news in all_data.items():
                    if date.startswith(month):
                        month_data[date] = stocks_news
                logger.debug(f"[News] File mode: filtered {len(month_data)} dates from {len(all_data)} dates")
                # Cache entire file data to avoid reloading
                if "all_data" not in self._news_data_cache:
                    self._news_data_cache["all_data"] = all_data
            
            self._news_data_cache[month] = month_data
            logger.info(f"[News] Successfully loaded {month} news data, contains {len(month_data)} dates")
            # List first few dates for debugging
            if month_data:
                sample_dates = list(month_data.keys())[:5]
                logger.debug(f"[News] Sample dates: {sample_dates}")
            return month_data
        except Exception as e:
            logger.error(f"[News] Failed to load {month} news data: {e}", exc_info=True)
            self._news_data_cache[month] = {}
            return {}
    
    def _load_news_data(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Load news data (compatible with old interface, supports auto-selecting month by date)
        
        Args:
            date: Optional, date string. If provided, loads news data for corresponding month
        
        News data format: {date: {stock_code: [news_list]}}
        
        Returns:
            News data dictionary (if date provided, returns that month's; otherwise returns all cached data)
        """
        if date:
            # Load corresponding month by date
            month = self._get_month_from_date(date)
            if month:
                return self._load_news_data_by_month(month)
            return {}
        
        # Compatible with old code: return all cached data (merged)
        if not self._news_data_cache:
            return {}
        
        # Merge news data from all months
        merged_data = {}
        for month_data in self._news_data_cache.values():
            merged_data.update(month_data)
        return merged_data
    
    def _get_stock_news(
        self,
        stock_code: str,
        date: str
    ) -> List[Dict[str, Any]]:
        """
        Get news for specific stock on specific date
        
        Args:
            stock_code: Stock code (e.g. "SH600015")
            date: Date (e.g. "2025-01-02" or "2025-01-02 00:00:00")
            
        Returns:
            News list [{"publish_date": ..., "news_title": ..., "content": ...}, ...]
        """
        # Format date to YYYY-MM-DD format
        formatted_date = date[:10] if len(date) >= 10 else date
        logger.debug(f"[News] Getting news: stock_code={stock_code}, date={date} -> formatted_date={formatted_date}")
        
        # Load news data for corresponding month based on date
        news_data = self._load_news_data(date=formatted_date)
        
        if not news_data:
            logger.debug(f"[News] No news data found for {formatted_date}")
            return []
        
        logger.debug(f"[News] News data contains {len(news_data)} dates")
        
        # Try direct date match (using formatted date)
        date_news = news_data.get(formatted_date, {})
        
        # If no exact match, try prefix match
        if not date_news:
            logger.debug(f"[News] Exact match failed, trying prefix match")
            for d in news_data.keys():
                if d.startswith(formatted_date):
                    date_news = news_data[d]
                    logger.debug(f"[News] Using matched date key: {d} (query: {formatted_date})")
                    break
        
        if not date_news:
            logger.debug(f"[News] No date data found for {formatted_date}, available date examples: {list(news_data.keys())[:5]}")
            return []
        
        # Get news for this stock
        stock_news = date_news.get(stock_code, [])
        
        if stock_news:
            logger.debug(f"[News] Found {len(stock_news)} news items for {stock_code} on {formatted_date}")
        else:
            logger.debug(f"[News] No news for {stock_code} on {formatted_date}, this date has data for {len(date_news)} stocks")
        
        return stock_news
    
    def _get_stocks_news_batch(
        self,
        stock_codes: List[str],
        date: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch get news for multiple stocks
        
        Args:
            stock_codes: Stock code list
            date: Date
            
        Returns:
            {stock_code: [news_list], ...}
        """
        result = {}
        for code in stock_codes:
            news = self._get_stock_news(code, date)
            if news:
                result[code] = news
        return result
    
    # =========================================================================
    # Macro News Loading and Compression
    # =========================================================================
    
    def _load_macro_news(self, date: str) -> List[Dict[str, Any]]:
        """
        Load macro news from the most recent day before current trading day (excluding trading day itself)
        
        Args:
            date: Trading date, format like "2025-01-02" or "2025-01-02 10:00:00"
            
        Returns:
            Macro news list [{"publish_date": ..., "news_title": ..., "content": ...}, ...]
        """
        if not self.macro_news_path or not self.macro_news_path.exists():
            logger.debug(f"[MacroNews] Macro news file does not exist: {self.macro_news_path}")
            return []
        
        try:
            # Format date to YYYY-MM-DD
            formatted_date = date[:10] if len(date) >= 10 else date
            
            # Read macro news file
            with open(self.macro_news_path, 'r', encoding='utf-8') as f:
                macro_data = json.load(f)
            
            # Find most recent day's macro news (search backwards, up to 7 days, skip trading day itself)
            from datetime import datetime, timedelta
            current_date = datetime.strptime(formatted_date, "%Y-%m-%d")
            
            # Start from i=1, skip trading day itself (i=0)
            for i in range(1, 8):  # From 1 to 7, search for news from 1-7 days ago
                check_date = current_date - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                
                if date_str in macro_data:
                    news_list = macro_data[date_str]
                    logger.info(f"[MacroNews] Found macro news for {date_str} ({i} days before trading day {formatted_date}), total {len(news_list)} items")
                    return news_list
            
            logger.debug(f"[MacroNews] No macro news found within 1-7 days before {formatted_date} (trading day itself skipped)")
            return []
            
        except Exception as e:
            logger.error(f"[MacroNews] Failed to load macro news: {e}", exc_info=True)
            return []
    
    def _load_midday_summary(self, date: str) -> Optional[Dict[str, Any]]:
        """
        Load current trading day's midday summary
        
        Args:
            date: Trading date, format like "2025-01-02" or "2025-01-02 10:00:00"
            
        Returns:
            Midday summary dictionary {"datetime": ..., "content": ..., "title": ...} or None
        """
        if not self.midday_summary_path or not self.midday_summary_path.exists():
            logger.debug(f"[MiddaySummary] Midday summary file does not exist: {self.midday_summary_path}")
            return None
        
        try:
            # Format date to YYYY-MM-DD
            formatted_date = date[:10] if len(date) >= 10 else date
            
            # Read midday summary file
            with open(self.midday_summary_path, 'r', encoding='utf-8') as f:
                midday_data = json.load(f)
            
            # Find current day's midday summary
            # Midday summary datetime format is "2025-01-02 11:30:58", need to match date part
            for item in midday_data:
                if isinstance(item, dict):
                    item_datetime = item.get("datetime", "")
                    if item_datetime.startswith(formatted_date):
                        logger.info(f"[MiddaySummary] Found midday summary for {formatted_date}")
                        return item
            
            logger.debug(f"[MiddaySummary] No midday summary found for {formatted_date}")
            return None
            
        except Exception as e:
            logger.error(f"[MiddaySummary] Failed to load midday summary: {e}", exc_info=True)
            return None
    
    def _compress_macro_news(self, macro_news: List[Dict[str, Any]], date: str) -> str:
        """
        Compress macro news using LLM
        
        Args:
            macro_news: Macro news list
            date: Trading date
            
        Returns:
            Compressed macro news summary text
        """
        if not macro_news:
            return ""
        
        # Check cache
        if date in self._compressed_macro_news_cache:
            logger.debug(f"[MacroNews] Using cached compression result: {date}")
            return self._compressed_macro_news_cache[date]
        
        if not self.llm:
            # Without LLM, simply concatenate titles
            summary = "\n".join([f"- {news.get('news_title', '')}" for news in macro_news[:5]])
            logger.debug(f"[MacroNews] No LLM, using simple summary")
            return summary
        
        try:
            # Build compression prompt
            news_text = ""
            for i, news in enumerate(macro_news[:10]):  # Process at most 10 items
                title = news.get("news_title", "")
                content = news.get("content", "")[:500]  # At most 500 characters per news
                news_text += f"\n[{i+1}] {title}\n{content}...\n"
            
            prompt = f"""You are a professional financial analyst. Please compress the following macro news into a concise summary, focusing on information that affects the stock market and investment decisions.

## Trading Date
{date}

## Macro News
{news_text}

## Task
Please compress the above macro news into a summary within 200 characters, highlighting:
1. Monetary policy and fiscal policy changes
2. Economic data and industry data
3. Major policy and regulatory changes
4. Key information affecting the overall stock market

Only output the compressed summary, no other content.
"""
            
            logger.info(f"[MacroNews] Starting to compress {len(macro_news)} macro news items")
            response = self.llm.call(prompt=prompt, stream=False)
            
            # Extract text (if not JSON format)
            if isinstance(response, str):
                compressed = response.strip()
            elif isinstance(response, dict):
                compressed = response.get("summary", response.get("content", ""))
            else:
                compressed = str(response).strip()
            
            # Cache result
            self._compressed_macro_news_cache[date] = compressed
            logger.info(f"[MacroNews] Compression completed, summary length: {len(compressed)} characters")
            
            return compressed
            
        except Exception as e:
            logger.error(f"[MacroNews] Failed to compress macro news: {e}")
            # On failure, return simple summary
            summary = "\n".join([f"- {news.get('news_title', '')}" for news in macro_news[:5]])
            return summary
    
    # =========================================================================
    # Short-term Memory System
    # =========================================================================
    
    def _get_compressed_macro_news(self, date: str) -> str:
        """
        Get compressed macro news (with caching)
        
        Args:
            date: Trading date
            
        Returns:
            Compressed macro news summary
        """
        # Check cache first
        if date in self._compressed_macro_news_cache:
            return self._compressed_macro_news_cache[date]
        
        # Load and compress
        macro_news = self._load_macro_news(date)
        compressed = self._compress_macro_news(macro_news, date)
        
        return compressed
    
    def _save_short_term_memory(
        self,
        date: str,
        macro_news_summary: str,
        stock_reasons: Dict[str, str]
    ) -> None:
        """
        Save short-term memory
        
        Args:
            date: Trading date
            macro_news_summary: Compressed macro news summary
            stock_reasons: Stock decision reasons {stock_code: reason}
        """
        # Format date
        formatted_date = date[:10] if len(date) >= 10 else date
        
        # Compress stock reasons (if too many)
        if len(stock_reasons) > 50:
            # Only keep reasons for first 50 stocks
            stock_reasons = dict(list(stock_reasons.items())[:50])
        
        # Save memory
        self._short_term_memory[formatted_date] = {
            "macro_news_summary": macro_news_summary,
            "stock_reasons": stock_reasons.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Clean up old memory (only keep last 3 trading days)
        self._cleanup_short_term_memory()
        
        logger.info(f"[Memory] Saved short-term memory: {formatted_date}, macro news: {len(macro_news_summary)} chars, stock reasons: {len(stock_reasons)} items")
    
    def _cleanup_short_term_memory(self) -> None:
        """Clean up short-term memory, only keep last 3 trading days"""
        if len(self._short_term_memory) <= self._max_memory_days:
            return
        
        # Sort by date, keep latest
        sorted_dates = sorted(self._short_term_memory.keys(), reverse=True)
        dates_to_keep = sorted_dates[:self._max_memory_days]
        
        # Delete old memory
        dates_to_remove = [d for d in sorted_dates if d not in dates_to_keep]
        for date in dates_to_remove:
            del self._short_term_memory[date]
            logger.debug(f"[Memory] Deleted old memory: {date}")
    
    def _load_short_term_memory(self) -> str:
        """
        Load short-term memory, return formatted memory text
        
        Returns:
            Formatted short-term memory text
        """
        if not self._short_term_memory:
            return ""
        
        memory_parts = []
        # Sort by date descending (latest first)
        sorted_dates = sorted(self._short_term_memory.keys(), reverse=True)
        
        for date in sorted_dates:
            memory = self._short_term_memory[date]
            macro_summary = memory.get("macro_news_summary", "")
            stock_reasons = memory.get("stock_reasons", {})
            
            memory_text = f"## {date}\n"
            if macro_summary:
                memory_text += f"Macro news summary: {macro_summary}\n"
            if stock_reasons:
                # Only show reasons for first 10 stocks
                stock_items = list(stock_reasons.items())[:10]
                memory_text += "Important stock decision reasons:\n"
                for stock_code, reason in stock_items:
                    memory_text += f"  - {stock_code}: {reason[:100]}...\n"  # At most 100 chars per reason
            
            memory_parts.append(memory_text)
        
        if memory_parts:
            return "\n---\n".join(memory_parts)
        return ""
    
    # =========================================================================
    # News Analysis Methods (Batch Processing)
    # =========================================================================
    
    def _analyze_news_batch(
        self,
        batch_stocks: List[TradeSignal],
        stocks_news: Dict[str, List[Dict[str, Any]]],
        action_type: str,
        date: str,
        macro_news_summary: str = "",
        short_term_memory: str = "",
        midday_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch analyze stock news, return analysis results for each stock (including macro news, midday summary and short-term memory)
        
        Args:
            batch_stocks: List of stock trading signals in this batch
            stocks_news: Stock news dictionary {stock_code: [news_list]}
            action_type: Trading action type ("sell", "buy", "hold")
            date: Trading date
            macro_news_summary: Compressed macro news summary
            short_term_memory: Short-term memory text
            midday_summary: Current day's midday summary dictionary
            
        Returns:
            Analysis result dictionary {stock_code: {"recommendation": ..., "confidence": ..., "reason": ...}}
        """
        if not self.llm or not batch_stocks:
            return {}
        
        # Build batch analysis prompt (including macro news, midday summary and short-term memory)
        prompt = self._build_news_analysis_prompt(
            batch_stocks, stocks_news, action_type, date,
            macro_news_summary=macro_news_summary,
            short_term_memory=short_term_memory,
            midday_summary=midday_summary
        )
        
        # Retry logic: try at most 2 times (initial 1 + retry 1)
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[NewsAnalysis] Analyzing batch: {action_type}, stock count: {len(batch_stocks)}, attempt {attempt}/{max_attempts}")
                response = self.llm.call(prompt=prompt, stream=False)
                
                # Parse LLM response
                result = self.llm.parse_json_response(response)
                
                if result and "stocks_analysis" in result:
                    return result["stocks_analysis"]
                elif result:
                    return result
                
                return {}
                
            except Exception as e:
                logger.error(f"[NewsAnalysis] Batch analysis failed (attempt {attempt}/{max_attempts}): {e}")
                
                # If not last attempt, wait 2 seconds then retry
                if attempt < max_attempts:
                    logger.info(f"[NewsAnalysis] Waiting 2 seconds before retry...")
                    time.sleep(2)
                else:
                    # Last attempt also failed, return empty dict
                    logger.error(f"[NewsAnalysis] Batch analysis finally failed, retried {max_attempts} times")
                    return {}
        
        return {}
    
    def _build_news_analysis_prompt(
        self,
        batch_stocks: List[TradeSignal],
        stocks_news: Dict[str, List[Dict[str, Any]]],
        action_type: str,
        date: str,
        macro_news_summary: str = "",
        short_term_memory: str = "",
        midday_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build news analysis prompt (including macro news, midday summary and short-term memory)
        
        Args:
            batch_stocks: List of stock trading signals
            stocks_news: Stock news dictionary
            action_type: Trading action type
            date: Trading date
            macro_news_summary: Compressed macro news summary
            short_term_memory: Short-term memory text
            midday_summary: Current day's midday summary dictionary
            
        Returns:
            Prompt string
        """
        action_desc = {
            "sell": "Sell candidates",
            "buy": "Buy candidates",
            "hold": "Hold stocks"
        }.get(action_type, "Candidate stocks")
        
        # Build stock information
        stocks_info = []
        for signal in batch_stocks:
            stock_code = signal.stock_code
            news_list = stocks_news.get(stock_code, [])
            
            # Format news (limit to at most 5 news items per stock, at most 200 chars per news)
            news_text = ""
            if news_list:
                news_items = []
                for i, news in enumerate(news_list[:5]):
                    title = news.get("news_title", "")
                    content = news.get("content", "")[:200]
                    source = news.get("news_source", "")
                    publish_date = news.get("publish_date", "")
                    news_items.append(f"    [{i+1}] {title} ({source})({publish_date})\n        {content}...")
                news_text = "\n".join(news_items)
            else:
                news_text = "    No related news"
            
            stock_info = f"""
  Stock code: {stock_code}
  Model score: {signal.score:.4f}
  Current weight: {signal.current_weight:.2%}
  Target weight: {signal.target_weight:.2%}
  Reason: {signal.reason}
  Related news:
{news_text}
"""
            stocks_info.append(stock_info)
        
        stocks_info_text = "\n---".join(stocks_info)
        
        # Build macro news section
        macro_news_section = ""
        if macro_news_summary:
            macro_news_section = f"""
## Macro News Summary (Most Recent Day Before Current Trading Day)
{macro_news_summary}
"""
        
        # Build midday summary section
        midday_section = ""
        if midday_summary:
            midday_content = midday_summary.get("content", "")
            midday_title = midday_summary.get("title", "")
            midday_datetime = midday_summary.get("datetime", "")
            midday_section = f"""
## Current Day Midday Summary ({midday_datetime})
Title: {midday_title}
Content: {midday_content}

Note: Midday summary reflects overall market performance and sector hotspots for the day
"""
        
        # Build short-term memory section
        memory_section = ""
        if short_term_memory:
            memory_section = f"""
## Short-term Memory (Decision History from Last 3 Trading Days)
{short_term_memory}


"""
        
        prompt = f"""You are a professional quantitative investment analyst. Please analyze investment recommendations for each stock based on the following stock information, news, macro news, midday summary and historical memory.

## Trading Date
{date}
Current time is near market close on trading day.
## Macro News, Midday Summary and Short-term Memory
{macro_news_section}{midday_section}{memory_section}
## Stock Type
{action_desc}

## Stock Information and News
{stocks_info_text}

## Analysis Task
Please comprehensively analyze each stock combining the following information:
1. Impact of individual stock news on the stock (positive, negative or neutral)
2. Impact of macro news on overall market and industry
3. Market sentiment and sector performance reflected in midday summary, especially rise/fall of related sectors
4. Whether it supports current trading recommendation ({action_type})
5. Provide confidence assessment (between 0-1)
6. Macro news is from yesterday, midday summary is from today, buy/sell time is near market close today.
7. Pay attention to individual stock news publication time.
8. Combine macro news, midday summary, historical memory and individual stock news to give analysis results for stocks.

**Important: recommendation field return value rules**
- For **buy candidates**: Mainly refer to model score, assist with individual stock news. If there is clear negative news, stock price may continue to fall tomorrow, recommend not to buy, return "hold" (temporarily not buy). If individual stock news has no obvious negative, recommend to buy, return "buy"
- For **sell candidates**: Mainly refer to model score, assist with individual stock news. If there is clear positive news, stock price may continue to rise tomorrow, recommend not to sell, return "hold" (continue holding). If individual stock news has no obvious positive, recommend to sell, return "sell"

Please output analysis results in JSON format:
```json
{{
    "stocks_analysis": {{
        "Stock Code 1": {{
            "news_sentiment": "positive/negative/neutral",
            "recommendation": "buy/sell/hold/",
            "confidence": 0.8,
            "reason": "Comprehensive analysis result based on above information"
        }},
        "Stock Code 2": {{
            ...
        }}
    }},
    "batch_summary": "Overall analysis summary of this batch of stocks"
}}
```

Note:
1. Macro policy changes (such as monetary policy, fiscal policy) may affect overall market
2. Only output JSON, no other content
"""
        return prompt
    
    def _analyze_all_candidates_with_news(
        self,
        candidates: Dict[str, List[TradeSignal]],
        date: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform news analysis on buy and sell candidate stocks (batch processing, including macro news and short-term memory)
        Note: Do not analyze hold stocks
        
        Args:
            candidates: Candidate trading signals {"sell": [...], "buy": [...], "hold": [...]}
            date: Trading date
            
        Returns:
            Analysis results for all stocks {stock_code: {"recommendation": ..., "confidence": ..., "reason": ...}}
        """
        all_analysis = {}
        
        # Collect stocks to analyze (only analyze sell and buy, not hold)
        all_stocks = []
        stock_action_map = {}  # Record original action type for each stock
        
        for action_type in ["sell", "buy"]:
            signals = candidates.get(action_type, [])
            for signal in signals:
                all_stocks.append(signal)
                stock_action_map[signal.stock_code] = action_type
        
        if not all_stocks:
            return all_analysis
        
        # Get news for all stocks
        stock_codes = [s.stock_code for s in all_stocks]
        all_news = self._get_stocks_news_batch(stock_codes, date)
        
        logger.info(f"[NewsAnalysis] Total {len(all_stocks)} stocks need analysis (only buy and sell candidates), {len(all_news)} stocks have news")
        
        # =====================================================================
        # Load macro news, midday summary and short-term memory (shared by all batches)
        # =====================================================================
        macro_news_summary = self._get_compressed_macro_news(date)
        midday_summary = self._load_midday_summary(date)
        short_term_memory = self._load_short_term_memory()
        
        if macro_news_summary:
            logger.info(f"[NewsAnalysis] Loaded macro news summary, length: {len(macro_news_summary)} characters")
        if midday_summary:
            logger.info(f"[NewsAnalysis] Loaded current day midday summary: {midday_summary.get('title', '')}")
        if short_term_memory:
            logger.info(f"[NewsAnalysis] Loaded short-term memory, contains {len(self._short_term_memory)} trading days")
        
        # Batch processing - group by action, ensure same action within same batch
        batch_size = self.news_batch_size

        # Group stocks by action (only sell and buy)
        stocks_by_action = {"sell": [], "buy": []}
        for signal in all_stocks:
            action = stock_action_map[signal.stock_code]
            stocks_by_action[action].append(signal)

        # Process each action separately in batches
        for action_type in ["sell", "buy"]:
            action_stocks = stocks_by_action[action_type]
            if not action_stocks:
                continue
            
            # Batch stocks for this action
            for i in range(0, len(action_stocks), batch_size):
                batch = action_stocks[i:i + batch_size]
                batch_codes = [s.stock_code for s in batch]
                
                # Get news for this batch of stocks
                batch_news = {code: all_news.get(code, []) for code in batch_codes}
                
                # Main action type for this batch is current action_type
                main_action = action_type
                
                # Analyze this batch (pass macro news, midday summary and short-term memory)
                logger.info(f"[NewsAnalysis] Processing batch {action_type} - {i//batch_size + 1}, stocks: {batch_codes}")
                batch_analysis = self._analyze_news_batch(
                    batch, batch_news, main_action, date,
                    macro_news_summary=macro_news_summary,
                    short_term_memory=short_term_memory,
                    midday_summary=midday_summary
                )
                
                # Merge results
                all_analysis.update(batch_analysis)
                
        return all_analysis
    
    def _merge_news_analysis_to_decision(
        self,
        candidates: Dict[str, List[TradeSignal]],
        news_analysis: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge news analysis results, generate final adjustment suggestions
        Note: Only process buy and sell candidates, not hold stocks
        
        Args:
            candidates: Original candidate trading signals
            news_analysis: News analysis results
            
        Returns:
            Adjustment suggestion dictionary, compatible with _apply_llm_suggestions format
        """
        remove_from_sell = []
        remove_from_buy = []
        priority_adjustments = {}
        reasoning_parts = []
        
        # Process sell candidates
        for signal in candidates.get("sell", []):
            stock_code = signal.stock_code
            analysis = news_analysis.get(stock_code, {})
            
            if analysis:
                recommendation = analysis.get("recommendation", "").lower()
                confidence = analysis.get("confidence", 0.5)
                reason = analysis.get("reason", "")
                sentiment = analysis.get("news_sentiment", "neutral")
                
                # If news is positive and suggests not to sell, remove from sell candidates
                if recommendation in ["hold", "buy"]:
                    remove_from_sell.append(stock_code)
                    reasoning_parts.append(f"{stock_code}: Positive news, suggest keeping ({reason})")
                elif confidence < 0.3:
                    # Low confidence, no adjustment
                    pass
                else:
                    # Adjust priority: negative news increases sell priority
                    if sentiment == "negative":
                        priority_adjustments[stock_code] = signal.priority + 20
        
        # Process buy candidates
        for signal in candidates.get("buy", []):
            stock_code = signal.stock_code
            analysis = news_analysis.get(stock_code, {})
            
            if analysis:
                recommendation = analysis.get("recommendation", "").lower()
                confidence = analysis.get("confidence", 0.5)
                reason = analysis.get("reason", "")
                sentiment = analysis.get("news_sentiment", "neutral")
                
                # If news is negative and suggests not to buy, remove from buy candidates
                if recommendation in ["hold", "sell"]:
                    remove_from_buy.append(stock_code)
                    reasoning_parts.append(f"{stock_code}: Negative news, suggest not buying ({reason})")
                elif confidence < 0.3:
                    # Low confidence, no adjustment
                    pass
                else:
                    # Adjust priority: positive news increases buy priority
                    if sentiment == "positive":
                        priority_adjustments[stock_code] = signal.priority + 20
        
        # Note: Do not process hold candidates, hold stocks do not undergo news review
        
        # Build final suggestions
        final_reasoning = "Adjustments based on news analysis: " + "; ".join(reasoning_parts) if reasoning_parts else "News analysis found no significant adjustment suggestions"
        
        return {
            "remove_from_sell": remove_from_sell,
            "remove_from_buy": remove_from_buy,
            "move_to_sell": [],  # No longer move from hold to sell
            "priority_adjustments": priority_adjustments,
            "reasoning": final_reasoning,
            "news_analysis_summary": news_analysis
        }
    
    # =========================================================================
    # Core Decision Methods
    # =========================================================================
    
    def generate_trade_decision(
        self,
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]],
        market_data: Optional[Dict[str, Any]] = None,
        date: Optional[str] = None
    ) -> TradeDecision:
        """
        Generate trading decision - core entry method
        
        Args:
            current_holdings: Current holdings {stock_code: weight}
            top_k_recommendations: Model-recommended Top K stocks [(stock_code, score), ...]
            market_data: Market data (optional)
            date: Current date (optional)
            
        Returns:
            TradeDecision trading decision object
        """
        self.decision_logs.append(f"[{date}] Starting to generate trading decision")
        
        # Step 1: Set comparison analysis
        comparison_result = self._compare_holdings_and_recommendations(
            current_holdings, top_k_recommendations
        )
        
        # Step 2: Generate candidate trading list
        candidates = self._generate_trade_candidates(
            comparison_result,
            current_holdings,
            top_k_recommendations,
            market_data
        )
        
        # Step 3: Intelligent decision (optionally use LLM)
        # Note: Consistent with qlib TopkDropoutStrategy, do not apply constraints at decision stage
        # Constraint checking will be done at order generation time (in _convert_decision_to_orders)
        logger.info(f"[Decision] date={date}, use_llm_decision={self.use_llm_decision}, llm={self.llm is not None}")
        logger.info(f"[Decision] candidates: sell={len(candidates.get('sell', []))}, buy={len(candidates.get('buy', []))}, hold={len(candidates.get('hold', []))}")
        
        if self.use_llm_decision and self.llm:
            logger.info("[Decision] Using LLM for decision")
            final_decision = self._llm_enhanced_decision(
                candidates,
                current_holdings,
                top_k_recommendations,
                market_data,
                date
            )
        else:
            logger.info(f"[Decision] Using rule-based decision (use_llm_decision={self.use_llm_decision}, llm_exists={self.llm is not None})")
            final_decision = self._rule_based_decision(
                candidates,
                current_holdings,
                top_k_recommendations
            )
        
        # Step 5: Record trade history
        self._record_trade_history(final_decision, date)
        
        return final_decision
    
    # =========================================================================
    # Step 1: Set Comparison Analysis
    # =========================================================================
    
    def _compare_holdings_and_recommendations(
        self,
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]]
    ) -> ComparisonResult:
        """
        Compare current holdings and recommendation list, perform set analysis
        
        Reference Qlib TopkDropoutStrategy implementation logic:
        1. Get current holdings stock set A
        2. Get Top K recommended stock set B
        3. Calculate: sell_candidates = A - B, buy_candidates = B - A, hold = A ∩ B
        4. Sort holdings and recommendations by model score
        5. Determine actual stocks to sell and buy based on n_drop strategy
        
        Args:
            current_holdings: Current holdings {stock_code: weight}
            top_k_recommendations: Model-recommended Top K stocks [(stock_code, score), ...]
            
        Returns:
            ComparisonResult containing detailed comparison analysis results
        """
        # Step 1: Build sets
        holding_set = set(current_holdings.keys())  # Set A: Current holdings
        
        # Build recommendation set and score mapping
        score_map: Dict[str, float] = {}
        recommendation_set: Set[str] = set()
        
        for stock_code, score in top_k_recommendations:
            recommendation_set.add(stock_code)
            score_map[stock_code] = score
        
        # Set B: Top K recommended stocks
        topk_set = recommendation_set
        
        self.decision_logs.append(
            f"[Compare] Current holdings: {len(holding_set)} stocks, Top K recommendations: {len(topk_set)} stocks"
        )
        
        # Step 2: Set operations
        # sell_candidates: A - B (stocks in holdings but not in recommendations)
        sell_candidates = holding_set - topk_set
        
        # buy_candidates: B - A (stocks in recommendations but not in holdings)
        buy_candidates = topk_set - holding_set
        
        # hold_candidates: A ∩ B (intersection of both, continue holding)
        hold_candidates = holding_set & topk_set
        
        self.decision_logs.append(
            f"[Compare] Sell candidates: {len(sell_candidates)}, "
            f"Buy candidates: {len(buy_candidates)}, "
            f"Keep: {len(hold_candidates)}"
        )
        
        # Step 3: Sort by score
        # Sort current holdings by model score (holdings may not be in recommendation list, give default low score)
        ranked_holdings = self._rank_stocks_by_score(
            list(holding_set), 
            score_map, 
            default_score=float('-inf')
        )
        
        # Sort recommended stocks by score
        ranked_recommendations = self._rank_stocks_by_score(
            list(topk_set),
            score_map,
            default_score=0.0
        )
        
        # Step 4: Determine actual sell and buy lists (reference TopkDropoutStrategy n_drop logic)
        suggested_sell, suggested_buy = self._determine_trades_with_dropout(
            holding_set=holding_set,
            topk_set=topk_set,
            sell_candidates=sell_candidates,
            buy_candidates=buy_candidates,
            ranked_holdings=ranked_holdings,
            ranked_recommendations=ranked_recommendations,
            score_map=score_map
        )
        
        return ComparisonResult(
            sell_candidates=sell_candidates,
            buy_candidates=buy_candidates,
            hold_candidates=hold_candidates,
            ranked_holdings=ranked_holdings,
            ranked_recommendations=ranked_recommendations,
            suggested_sell=suggested_sell,
            suggested_buy=suggested_buy,
            score_map=score_map
        )
    
    def _rank_stocks_by_score(
        self,
        stock_list: List[str],
        score_map: Dict[str, float],
        default_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Sort stock list by model score
        
        Args:
            stock_list: Stock list
            score_map: Score mapping {stock_code: score}
            default_score: Default score (used when stock not in score_map)
            
        Returns:
            List sorted by score in descending order [(stock_code, score), ...]
        """
        scored_list = [
            (stock, score_map.get(stock, default_score))
            for stock in stock_list
        ]
        # Sort by score in descending order
        scored_list.sort(key=lambda x: (-x[1], x[0]))
        #scored_list.sort(key=lambda x: x[1], reverse=True)
        return scored_list
    
    def _determine_trades_with_dropout(
        self,
        holding_set: Set[str],
        topk_set: Set[str],
        sell_candidates: Set[str],
        buy_candidates: Set[str],
        ranked_holdings: List[Tuple[str, float]],
        ranked_recommendations: List[Tuple[str, float]],
        score_map: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Determine actual trading list based on TopkDropoutStrategy n_drop strategy
        
        Core logic (reference TopkDropoutStrategy):
        1. Merge current holdings and new recommended stocks, sort by score
        2. Select n_drop stocks from tail of merged list as sell candidates
        3. Only sell stocks in current holdings that belong to these low-score stocks
        4. Buy count = Sell count + (topk - current holdings count)
        
        This avoids "sell high buy low" situations
        
        Args:
            holding_set: Current holdings set
            topk_set: Top K recommendation set
            sell_candidates: Sell candidates from simple set difference
            buy_candidates: Buy candidates from simple set difference
            ranked_holdings: Holdings sorted by score
            ranked_recommendations: Recommendations sorted by score
            score_map: Score mapping
            
        Returns:
            (suggested_sell, suggested_buy) Actual suggested sell and buy lists
        """
        topk = self.config.topk
        n_drop = self.config.n_drop
        
        # Get new recommended stocks not in current holdings (sorted by score)
        # These are today's buy candidates
        new_recommendations = [
            (stock, score) for stock, score in ranked_recommendations
            if stock not in holding_set
        ]
        
        # Calculate how many stocks to buy to reach topk
        # Buy at most n_drop + (topk - current holdings count) stocks
        max_buy_count = n_drop + topk - len(holding_set)
        today_buy_candidates = [stock for stock, _ in new_recommendations[:max_buy_count]]
        
        # Merge: current holdings + new recommended stocks, sort by score
        combined_set = holding_set | set(today_buy_candidates)
        combined_ranked = self._rank_stocks_by_score(list(combined_set), score_map, float('-inf'))
        
        # Select n_drop stocks from tail of merged list as sell candidates
        # This ensures we sell the lowest-scored stocks, not simply stocks not in recommendations
        drop_candidates = [stock for stock, _ in combined_ranked[-n_drop:]] if len(combined_ranked) > n_drop else []
        
        # Actual sell: stocks in current holdings that belong to drop_candidates
        suggested_sell = [
            stock for stock in drop_candidates
            if stock in holding_set
        ]
        
        # Actual buy: determined based on sell count and holdings gap
        # Buy count = Sell count + max(0, topk - current holdings count)
        buy_count = len(suggested_sell) + max(0, topk - len(holding_set))
        suggested_buy = today_buy_candidates[:buy_count]
        
        self.decision_logs.append(
            f"[Dropout] After merge and sort, suggested sell: {suggested_sell}, suggested buy: {suggested_buy[:5]}..."
        )
        
        return suggested_sell, suggested_buy
    
    # =========================================================================
    # Step 2: Generate Candidate Trading List
    # =========================================================================
    
    def _generate_trade_candidates(
        self,
        comparison_result: ComparisonResult,
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, List[TradeSignal]]:
        """
        Generate candidate trading signals based on comparison results
        
        Args:
            comparison_result: ComparisonResult set comparison result
            current_holdings: Current holdings
            top_k_recommendations: Top K recommendations
            market_data: Market data
            
        Returns:
            Candidate trading signal dictionary {"sell": [...], "buy": [...], "hold": [...]}
        """
        candidates: Dict[str, List[TradeSignal]] = {
            "sell": [],
            "buy": [],
            "hold": []
        }
        
        score_map = comparison_result.score_map
        
        # Generate sell candidate signals
        for stock_code in comparison_result.suggested_sell:
            current_weight = current_holdings.get(stock_code, 0.0)
            score = score_map.get(stock_code, 0.0)
            
            signal = self._evaluate_sell_candidate(
                stock_code=stock_code,
                current_weight=current_weight,
                score=score,
                market_data=market_data
            )
            candidates["sell"].append(signal)
        
        # Generate buy candidate signals
        for stock_code in comparison_result.suggested_buy:
            score = score_map.get(stock_code, 0.0)
            
            signal = self._evaluate_buy_candidate(
                stock_code=stock_code,
                model_score=score,
                market_data=market_data
            )
            candidates["buy"].append(signal)
        
        # Generate hold candidate signals
        # Evaluate all holdings stocks not in sell candidates
        # This ensures all holdings stocks can undergo news risk screening
        sell_stock_set = set(comparison_result.suggested_sell)
        all_hold_stocks = set(current_holdings.keys()) - sell_stock_set
        
        for stock_code in all_hold_stocks:
            current_weight = current_holdings.get(stock_code, 0.0)
            score = score_map.get(stock_code, 0.0)
            
            signal = self._evaluate_hold_candidate(
                stock_code=stock_code,
                current_weight=current_weight,
                model_score=score,
                market_data=market_data
            )
            candidates["hold"].append(signal)
        
        self.decision_logs.append(
            f"[Candidates] Generated candidates: sell {len(candidates['sell'])}, "
            f"buy {len(candidates['buy'])}, "
            f"keep {len(candidates['hold'])}"
        )
        
        return candidates
    
    def _evaluate_sell_candidate(
        self,
        stock_code: str,
        current_weight: float,
        score: float,
        market_data: Optional[Dict[str, Any]]
    ) -> TradeSignal:
        """
        Evaluate single sell candidate stock
        
        Args:
            stock_code: Stock code
            current_weight: Current weight
            score: Model score
            market_data: Market data
            
        Returns:
            TradeSignal trading signal
        """
        # Build sell reasons
        reasons = []
        reasons.append(f"Low model score ({score:.4f})")
        reasons.append("Not in Top K recommendation list")
        
        # Check limit up/down restrictions
        is_tradable = self._check_limit_price(stock_code, "sell", market_data)
        if not is_tradable:
            reasons.append("Note: May be limit down, cannot sell")
        
        # Calculate priority (lower score, higher sell priority)
        priority = int(-score * 100) if score != float('-inf') else 100
        
        return TradeSignal(
            stock_code=stock_code,
            action="sell",
            target_weight=0.0,  # Target weight is 0, sell all
            reason="; ".join(reasons),
            score=score,
            current_weight=current_weight,
            confidence=1.0 if is_tradable else 0.5,
            priority=priority
        )
    
    def _evaluate_buy_candidate(
        self,
        stock_code: str,
        model_score: float,
        market_data: Optional[Dict[str, Any]]
    ) -> TradeSignal:
        """
        Evaluate single buy candidate stock
        
        Args:
            stock_code: Stock code
            model_score: Model score
            market_data: Market data
            
        Returns:
            TradeSignal trading signal
        """
        # Build buy reasons
        reasons = []
        reasons.append(f"High model score ({model_score:.4f})")
        reasons.append("In Top K recommendation list")
        
        # Check limit up/down restrictions
        is_tradable = self._check_limit_price(stock_code, "buy", market_data)
        if not is_tradable:
            reasons.append("Note: May be limit up, cannot buy")
        
        # Calculate target weight (equal weight allocation)
        target_weight = 1.0 / self.config.topk
        
        # Calculate priority (higher score, higher buy priority)
        priority = int(model_score * 100)
        
        return TradeSignal(
            stock_code=stock_code,
            action="buy",
            target_weight=target_weight,
            reason="; ".join(reasons),
            score=model_score,
            current_weight=0.0,
            confidence=1.0 if is_tradable else 0.5,
            priority=priority
        )
    
    def _evaluate_hold_candidate(
        self,
        stock_code: str,
        current_weight: float,
        model_score: float,
        market_data: Optional[Dict[str, Any]]
    ) -> TradeSignal:
        """
        Evaluate hold stock
        
        Args:
            stock_code: Stock code
            current_weight: Current weight
            model_score: Model score
            market_data: Market data
            
        Returns:
            TradeSignal trading signal (action is "hold")
        """
        # Calculate target weight (equal weight allocation)
        target_weight = 1.0 / self.config.topk
        
        # Keep unchanged
        reason = f"Continue holding, score {model_score:.4f}"
        
        return TradeSignal(
            stock_code=stock_code,
            action="hold",
            target_weight=target_weight,
            reason=reason,
            score=model_score,
            current_weight=current_weight,
            confidence=1.0,
            priority=int(model_score * 50)  # Hold stocks have moderate priority
        )
    
    # =========================================================================
    # Step 3: Apply Trading Constraints
    # =========================================================================
    
    def _apply_trade_constraints(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, List[TradeSignal]]:
        """
        Apply trading constraint conditions to filter candidate trades
        
        Constraints include:
        1. Turnover rate limit
        2. Limit up/down restrictions
        3. Liquidity constraints
        
        Args:
            candidates: Candidate trading signals
            current_holdings: Current holdings
            market_data: Market data
            
        Returns:
            Filtered candidate trading signals
        """
        filtered = {
            "sell": [],
            "buy": [],
            "hold": candidates.get("hold", [])  # Hold signals not filtered
        }
        
        # 1. Filter sell candidates
        for signal in candidates.get("sell", []):
            # Check limit up/down
            if signal.confidence < 0.5:  # May be limit up
                self.decision_logs.append(
                    f"[Constraint] {signal.stock_code} may be limit up, skip sell"
                )
                continue
            filtered["sell"].append(signal)
        
        # 2. Filter buy candidates
        for signal in candidates.get("buy", []):
            # Check limit up/down
            if signal.confidence < 0.5:  # May be limit down
                self.decision_logs.append(
                    f"[Constraint] {signal.stock_code} may be limit down, skip buy"
                )
                continue
            filtered["buy"].append(signal)
        
        # 3. Apply turnover rate limit
        filtered = self._check_turnover_limit(filtered, current_holdings)
        
        return filtered
    
    def _check_turnover_limit(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float]
    ) -> Dict[str, List[TradeSignal]]:
        """
        Check and limit turnover rate
        
        Turnover rate = (Sell amount + Buy amount) / (2 * Total assets)
        
        Args:
            candidates: Candidate trades
            current_holdings: Current holdings
            
        Returns:
            Trades after turnover rate constraint
        """
        max_turnover = self.config.max_turnover
        
        # Calculate current candidate turnover rate
        sell_weight = sum(s.current_weight for s in candidates.get("sell", []))
        buy_weight = sum(s.target_weight for s in candidates.get("buy", []))
        
        # Estimate turnover rate (simplified calculation)
        estimated_turnover = (sell_weight + buy_weight) / 2
        
        if estimated_turnover <= max_turnover:
            return candidates
        
        self.decision_logs.append(
            f"[Turnover] Estimated turnover rate {estimated_turnover:.2%} exceeds limit {max_turnover:.2%}, need to reduce trades"
        )
        
        # Sort by priority, keep high priority trades
        sell_list = sorted(candidates.get("sell", []), key=lambda x: x.priority, reverse=True)
        buy_list = sorted(candidates.get("buy", []), key=lambda x: x.priority, reverse=True)
        
        # Gradually reduce until turnover rate limit is met
        filtered_sell = []
        filtered_buy = []
        current_turnover = 0.0
        
        # Process sells first (sells release funds for buys)
        for signal in sell_list:
            if current_turnover + signal.current_weight / 2 <= max_turnover:
                filtered_sell.append(signal)
                current_turnover += signal.current_weight / 2
        
        # Then process buys
        for signal in buy_list:
            if current_turnover + signal.target_weight / 2 <= max_turnover:
                filtered_buy.append(signal)
                current_turnover += signal.target_weight / 2
        
        return {
            "sell": filtered_sell,
            "buy": filtered_buy,
            "hold": candidates.get("hold", [])
        }
    
    def _check_trade_cost_efficiency(
        self,
        signal: TradeSignal,
        current_weight: float,
        total_value: float = 1.0
    ) -> bool:
        """
        Check trade cost efficiency
        
        Determine if rebalancing returns can cover trading costs
        
        Args:
            signal: Trading signal
            current_weight: Current weight
            total_value: Total value (for calculating absolute cost)
            
        Returns:
            Whether it's worth trading
        """
        weight_change = abs(signal.target_weight - current_weight)
        trade_value = weight_change * total_value
        
        # Calculate trading cost
        # Rebalancing involves selling and buying, requires bilateral cost
        if signal.target_weight > current_weight:
            # Buy
            cost = trade_value * self.config.open_cost
        else:
            # Sell
            cost = trade_value * self.config.close_cost
        
        # Minimum trading cost
        cost = max(cost, self.config.min_cost / total_value if total_value > 0 else 0)
        
        # If rebalancing amount is too small, cost ratio too high, not worth it
        min_trade_ratio = self.config.min_trade_value / total_value if total_value > 0 else 0.01
        
        return weight_change > min_trade_ratio and weight_change > cost * 2
    
    def _check_liquidity_constraint(
        self,
        stock_code: str,
        trade_value: float,
        market_data: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Check liquidity constraint
        
        Args:
            stock_code: Stock code
            trade_value: Trade value
            market_data: Market data
            
        Returns:
            Whether liquidity requirements are met
        """
        if market_data is None:
            return True  # Default tradable when no market data
        
        stock_data = market_data.get(stock_code, {})
        volume = stock_data.get("volume", float('inf'))
        avg_price = stock_data.get("price", 1.0)
        
        # Check if trade value exceeds certain proportion of daily turnover
        daily_turnover = volume * avg_price
        max_participation = 0.1  # At most 10% of daily turnover
        
        return trade_value <= daily_turnover * max_participation
    
    def _check_limit_price(
        self,
        stock_code: str,
        action: str,
        market_data: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Check limit up/down restrictions
        
        Args:
            stock_code: Stock code
            action: Trading action (buy/sell)
            market_data: Market data
            
        Returns:
            Whether can trade (not limit up/down)
        """
        if market_data is None:
            return True  # Default tradable when no market data
        
        stock_data = market_data.get(stock_code, {})
        price_change = stock_data.get("price_change_pct", 0.0)
        
        limit = self.config.limit_threshold
        
        if action == "buy":
            # When buying, check if limit down (can buy) or limit up (cannot buy)
            return price_change < limit
        elif action == "sell":
            # When selling, check if limit up (can sell) or limit down (cannot sell)
            return price_change > -limit
        
        return True
    
    # =========================================================================
    # Step 4: Decision Methods
    # =========================================================================
    
    def _rule_based_decision(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]]
    ) -> TradeDecision:
        """
        Rule-based decision method
        
        Decision rules:
        1. Sort all trading signals by priority
        2. Execute sells first (release funds)
        3. Then execute buys
        
        Args:
            candidates: Candidate trades
            current_holdings: Current holdings
            top_k_recommendations: Top K recommendations
            
        Returns:
            TradeDecision trading decision
        """
        # Sort by priority
        sell_list = sorted(
            candidates.get("sell", []),
            key=lambda x: x.priority,
            reverse=True
        )
        
        buy_list = sorted(
            candidates.get("buy", []),
            key=lambda x: x.priority,
            reverse=True
        )
        
        hold_list = candidates.get("hold", [])
        
        # Build metadata
        metadata = {
            "decision_mode": "rule_based",
            "total_sell": len(sell_list),
            "total_buy": len(buy_list),
            "total_hold": len(hold_list),
            "sell_stocks": [s.stock_code for s in sell_list],
            "buy_stocks": [s.stock_code for s in buy_list],
        }
        
        self.decision_logs.append(
            f"[Decision] Rule-based decision completed: sell {len(sell_list)}, "
            f"buy {len(buy_list)}, keep {len(hold_list)}"
        )
        
        return TradeDecision(
            buy_list=buy_list,
            sell_list=sell_list,
            hold_list=hold_list,
            metadata=metadata
        )
    
    def _llm_enhanced_decision(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]],
        market_data: Optional[Dict[str, Any]],
        date: Optional[str]
    ) -> TradeDecision:
        """
        LLM-enhanced decision method (integrated with news analysis)
        
        Process:
        1. Batch get news data for candidate stocks
        2. Call LLM in batches to analyze news impact on stocks
        3. Merge all batch analysis results
        4. Adjust trading decisions based on news analysis results
        
        Args:
            candidates: Candidate trades
            current_holdings: Current holdings
            top_k_recommendations: Top K recommendations
            market_data: Market data
            date: Date
            
        Returns:
            TradeDecision trading decision
        """
        if not self.llm:
            logger.warning("[LLM] LLM service unavailable, fallback to rule-based decision")
            self.decision_logs.append("[LLM] LLM service unavailable, fallback to rule-based decision")
            return self._rule_based_decision(candidates, current_holdings, top_k_recommendations)
        
        try:
            # =====================================================================
            # Step 1: Batch analyze news (for all candidate stocks)
            # =====================================================================
            logger.info(f"[LLM] Starting news analysis, date: {date}")
            self.decision_logs.append(f"[LLM] Starting news analysis, date: {date}")
            
            # Format date (ensure matches news data date format)
            formatted_date = date[:10] if date and len(date) >= 10 else date
            
            # Batch analyze news for all candidate stocks
            news_analysis = self._analyze_all_candidates_with_news(candidates, formatted_date)
            
            logger.info(f"[LLM] News analysis completed, analyzed {len(news_analysis)} stocks")
            self.decision_logs.append(f"[LLM] News analysis completed, analyzed {len(news_analysis)} stocks")
            
            # =====================================================================
            # Step 2: Generate adjustment suggestions based on news analysis results
            # =====================================================================
            if news_analysis:
                # Convert news analysis results to decision adjustment suggestions
                news_based_suggestions = self._merge_news_analysis_to_decision(
                    candidates=candidates,
                    news_analysis=news_analysis
                )
                # Extract adjustment counts
                remove_sell_count = len(news_based_suggestions.get('remove_from_sell', []))
                remove_buy_count = len(news_based_suggestions.get('remove_from_buy', []))
                move_to_sell_count = len(news_based_suggestions.get('move_to_sell', []))
                
                logger.info(f"[LLM] News analysis suggestions: "
                           f"Remove from sell {remove_sell_count} (convert to hold), "
                           f"Remove from buy {remove_buy_count}, "
                           f"Remove from hold (convert to sell) {move_to_sell_count}, ")
                self.decision_logs.append(f"[LLM] News analysis suggestions: {news_based_suggestions.get('reasoning', '')}")
            else:
                # No news analysis results, use empty suggestions
                news_based_suggestions = {
                    "remove_from_sell": [],
                    "remove_from_buy": [],
                    'move_to_sell': [],
                    "priority_adjustments": {},
                    "reasoning": "No news data available for analysis"
                }
            
            # =====================================================================
            # Step 3: Optional - Perform comprehensive decision (combine market data and news analysis)
            # =====================================================================
            # If further LLM comprehensive analysis is needed, can build comprehensive prompt
            # Here we directly use news analysis results as decision basis
            
            # Build final LLM decision (including news analysis summary)
            llm_decision = news_based_suggestions
            
            # =====================================================================
            # Step 4: Apply adjustment suggestions to generate final decision
            # =====================================================================
            adjusted_decision = self._apply_llm_suggestions(
                candidates=candidates,
                llm_decision=llm_decision
            )
            
            # =====================================================================
            # Step 5: Save short-term memory (macro news + stock decision reasons)
            # =====================================================================
            # Collect decision reasons for all stocks
            stock_reasons = {}
            for signal in adjusted_decision.sell_list + adjusted_decision.buy_list + adjusted_decision.hold_list:
                stock_code = signal.stock_code
                # Get reason from news analysis, if not available use original reason
                analysis = news_analysis.get(stock_code, {})
                reason = analysis.get("reason", signal.reason)
                stock_reasons[stock_code] = reason
            
            # Get macro news summary
            macro_news_summary = self._get_compressed_macro_news(date)
            
            # Save short-term memory
            self._save_short_term_memory(
                date=date,
                macro_news_summary=macro_news_summary,
                stock_reasons=stock_reasons
            )
            
            # Add news analysis information to metadata
            adjusted_decision.metadata["news_analysis"] = {
                "analyzed_stocks": len(news_analysis),
                "summary": news_based_suggestions.get("reasoning", ""),
                "removed_from_sell": news_based_suggestions.get("remove_from_sell", []),
                "removed_from_buy": news_based_suggestions.get("remove_from_buy", []),
                "macro_news_loaded": bool(macro_news_summary),
                "short_term_memory_days": len(self._short_term_memory),
            }
            
            logger.info(f"[LLM] LLM-enhanced decision completed (including news analysis, macro news, short-term memory)")
            self.decision_logs.append(f"[LLM] LLM-enhanced decision completed (including news analysis, macro news, short-term memory)")
            
            return adjusted_decision
            
        except Exception as e:
            logger.error(f"[LLM] LLM decision failed: {str(e)}, fallback to rule-based decision")
            self.decision_logs.append(f"[LLM] LLM decision failed: {str(e)}, fallback to rule-based decision")
            import traceback
            logger.error(traceback.format_exc())
            return self._rule_based_decision(candidates, current_holdings, top_k_recommendations)
    
#     def _build_llm_decision_prompt(
#         self,
#         candidates: Dict[str, List[TradeSignal]],
#         current_holdings: Dict[str, float],
#         market_context: str
#     ) -> str:
#         """
#         Build LLM decision prompt
        
#         Args:
#             candidates: Candidate trades
#             current_holdings: Current holdings
#             market_context: Market context
            
#         Returns:
#             Prompt string
#         """
#         # Format candidate trading information
#         sell_info = "\n".join([
#             f"  - {s.stock_code}: Current weight {s.current_weight:.2%}, Score {s.score:.4f}, Reason: {s.reason}"
#             for s in candidates.get("sell", [])
#         ]) or "  None"
        
#         buy_info = "\n".join([
#             f"  - {s.stock_code}: Target weight {s.target_weight:.2%}, Score {s.score:.4f}, Reason: {s.reason}"
#             for s in candidates.get("buy", [])
#         ]) or "  None"
        
#         hold_info = "\n".join([
#             f"  - {s.stock_code}: Current weight {s.current_weight:.2%}, Score {s.score:.4f}"
#             for s in candidates.get("hold", [])[:10]  # Only show first 10
#         ]) or "  None"
        
#         prompt = f"""You are a quantitative trading strategy expert. Please review the following trading decision candidates and provide final recommendations.

# ## Market Context
# {market_context}

# ## Current Holdings Count
# {len(current_holdings)} stocks

# ## Candidate Sell Stocks
# {sell_info}

# ## Candidate Buy Stocks
# {buy_info}

# ## Continue Holding Stocks (First 10)
# {hold_info}

# ## Task
# Please analyze the above candidate trades, considering the following factors:
# 1. Are there stocks misjudged as sell candidates (e.g., short-term volatility but long-term positive)
# 2. Are there buy candidates with risks (e.g., excessive gains, chasing high risk)
# 3. Whether trading priority needs adjustment

# Please output your recommendations in JSON format:
# ```json
# {{
#     "remove_from_sell": ["Stock Code 1", "Stock Code 2"],  // Stocks recommended not to sell
#     "remove_from_buy": ["Stock Code 1"],  // Stocks recommended not to buy
#     "priority_adjustments": {{"Stock Code": new_priority}},  // Priority adjustments
#     "reasoning": "Overall analysis and recommendations"
# }}
# ```
# """
#         return prompt
    
    def _apply_llm_suggestions(
        self,
        candidates: Dict[str, List[TradeSignal]],
        llm_decision: Dict[str, Any]
    ) -> TradeDecision:
        """
        Apply LLM suggestions to adjust decision
        Note: Only process adjustments for buy and sell candidates, not hold stocks
        
        Args:
            candidates: Original candidate trades
            llm_decision: LLM decision suggestions
            
        Returns:
            Adjusted TradeDecision
        """
        remove_from_sell = set(llm_decision.get("remove_from_sell", []))
        remove_from_buy = set(llm_decision.get("remove_from_buy", []))
        priority_adjustments = llm_decision.get("priority_adjustments", {})
        
        # Filter sell list
        sell_list = [
            s for s in candidates.get("sell", [])
            if s.stock_code not in remove_from_sell
        ]
        
        # Filter buy list
        buy_list = [
            s for s in candidates.get("buy", [])
            if s.stock_code not in remove_from_buy
        ]
        
        # Hold list remains unchanged (do not process hold stocks)
        hold_list = candidates.get("hold", []).copy()
        
        # Adjust priority (only adjust sell and buy)
        for signal in sell_list + buy_list:
            if signal.stock_code in priority_adjustments:
                signal.priority = priority_adjustments[signal.stock_code]
        
        # Removed sell candidates converted to hold
        removed_sells = [
            TradeSignal(
                stock_code=s.stock_code,
                action="hold",
                target_weight=s.current_weight,
                reason=f"LLM suggests keeping: {llm_decision.get('reasoning', '')}",
                score=s.score,
                current_weight=s.current_weight,
                confidence=s.confidence,
                priority=s.priority
            )
            for s in candidates.get("sell", [])
            if s.stock_code in remove_from_sell
        ]
        
        hold_list.extend(removed_sells)
        
        return TradeDecision(
            buy_list=sorted(buy_list, key=lambda x: x.priority, reverse=True),
            sell_list=sorted(sell_list, key=lambda x: x.priority, reverse=True),
            hold_list=hold_list,
            metadata={
                "decision_mode": "llm_enhanced",
                "llm_reasoning": llm_decision.get("reasoning", ""),
                "removed_sells": list(remove_from_sell),
                "removed_buys": list(remove_from_buy),
            }
        )
    
    # =========================================================================
    # Step 5: History Records
    # =========================================================================
    
    def _record_trade_history(
        self,
        decision: TradeDecision,
        date: Optional[str]
    ) -> None:
        """
        Record trade history
        
        Args:
            decision: Trading decision
            date: Date
        """
        # Extract LLM suggested adjustment information from metadata
        removed_sells = decision.metadata.get("removed_sells", [])
        removed_buys = decision.metadata.get("removed_buys", [])
        llm_reasoning = decision.metadata.get("llm_reasoning", "")
        
        # Extract information from news_analysis (if exists)
        news_analysis = decision.metadata.get("news_analysis", {})
        removed_from_sell = news_analysis.get("removed_from_sell", [])
        removed_from_buy = news_analysis.get("removed_from_buy", [])
        news_summary = news_analysis.get("summary", "")
        
        # Merge information, prioritize direct information in metadata
        final_removed_sells = removed_sells if removed_sells else removed_from_sell
        final_removed_buys = removed_buys if removed_buys else removed_from_buy
        final_reasoning = llm_reasoning if llm_reasoning else news_summary
        
        record = {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "sell_count": len(decision.sell_list),
            "buy_count": len(decision.buy_list),
            "hold_count": len(decision.hold_list),
            "sell_stocks": [s.stock_code for s in decision.sell_list],
            "buy_stocks": [s.stock_code for s in decision.buy_list],
            "metadata": decision.metadata,
            "timestamp": datetime.now().isoformat(),
            # LLM suggested adjustment information
            "removed_sells": final_removed_sells,
            "removed_buys": final_removed_buys,
            "llm_reasoning": final_reasoning
        }
        
        # Format stock lists as strings
        actual_buy_stocks = [s.stock_code for s in decision.buy_list]
        actual_sell_stocks = [s.stock_code for s in decision.sell_list]
        actual_buy_str = ", ".join(actual_buy_stocks) if actual_buy_stocks else "None"
        actual_sell_str = ", ".join(actual_sell_stocks) if actual_sell_stocks else "None"
        removed_buy_str = ", ".join(final_removed_buys) if final_removed_buys else "None"
        removed_sell_str = ", ".join(final_removed_sells) if final_removed_sells else "None"
        
        # Print log
        current_date = date or datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"{current_date}: Actual buy={actual_buy_str}, "
            f"Actual sell={actual_sell_str}, "
            f"Removed buy={removed_buy_str}, "
            f"Removed sell={removed_sell_str}"
        )
        
        self.trade_history.append(record)
        
        self.decision_logs.append(
            f"[History] Trade record saved: {date}, "
            f"sell {len(decision.sell_list)} stocks, buy {len(decision.buy_list)} stocks"
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _calculate_target_weights(
        self,
        top_k_recommendations: List[Tuple[str, float]],
        method: str = "equal"
    ) -> Dict[str, float]:
        """
        Calculate target weights based on model scores
        
        Args:
            top_k_recommendations: Top K recommendations [(stock_code, score), ...]
            method: Weight calculation method
                - "equal": Equal weight allocation
                - "score_weighted": Weighted by score
                - "rank_weighted": Weighted by rank
            
        Returns:
            Target weight dictionary {stock_code: target_weight}
        """
        if not top_k_recommendations:
            return {}
        
        n = len(top_k_recommendations)
        
        if method == "equal":
            # Equal weight allocation
            weight = 1.0 / n
            return {stock: weight for stock, _ in top_k_recommendations}
        
        elif method == "score_weighted":
            # Weighted by score (needs normalization)
            scores = [score for _, score in top_k_recommendations]
            min_score = min(scores)
            # Shift to ensure all scores are positive
            shifted_scores = [s - min_score + 1 for s in scores]
            total = sum(shifted_scores)
            
            return {
                stock: shifted_scores[i] / total
                for i, (stock, _) in enumerate(top_k_recommendations)
            }
        
        elif method == "rank_weighted":
            # Weighted by rank (higher rank has higher weight)
            # Use 1/rank as weight
            weights = [1.0 / (i + 1) for i in range(n)]
            total = sum(weights)
            
            return {
                stock: weights[i] / total
                for i, (stock, _) in enumerate(top_k_recommendations)
            }
        
        else:
            # Default equal weight
            weight = 1.0 / n
            return {stock: weight for stock, _ in top_k_recommendations}
    
    def _calculate_trade_cost(
        self,
        trade_value: float,
        action: str
    ) -> float:
        """
        Calculate trading cost
        
        Args:
            trade_value: Trade value
            action: Trading action (buy/sell)
            
        Returns:
            Trading cost
        """
        if action == "buy":
            cost = trade_value * self.config.open_cost
        elif action == "sell":
            cost = trade_value * self.config.close_cost
        else:
            cost = 0.0
        
        # Minimum trading cost
        return max(cost, self.config.min_cost)
    
    def get_decision_logs(self) -> List[str]:
        """Get decision logs"""
        return self.decision_logs.copy()
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history.copy()
    
    def reset(self) -> None:
        """Reset Agent state"""
        self.trade_history.clear()
        self.decision_logs.clear()
    
    # =========================================================================
    # LangGraph Workflow Integration - process method
    # =========================================================================
    
    def process(
        self,
        model_info: Dict[str, Any],
        sota_pool_list: List[str],
        run_backtest: bool = True,
        run_both_versions: bool = True
    ) -> Dict[str, Any]:
        """
        Strategy generation main entry method - for LangGraph workflow
        
        Generate strategy configuration based on model optimization results, optionally run backtest
        If run_both_versions=True, will first run baseline (without LLM), then run LLM-enhanced version
        
        Args:
            model_info: Model optimization results, containing:
                - yaml_config_path: Optimal YAML config path
                - factor_pool_name: Factor pool name
                - module_path: Factor pool module path
                - model_class: Model type
                - model_kwargs: Model parameters
                - best_metrics: Baseline performance metrics
            sota_pool_list: Factor list
            run_backtest: Whether to run backtest
            run_both_versions: Whether to run both versions (baseline and LLM-enhanced) for comparison
            
        Returns:
            Dictionary containing strategy config and backtest results
        """
        logs = []
        logs.append("[StrategyGeneration] Starting strategy generation process")
        
        result = {
            "status": "success",
            "logs": logs,
            "strategy_config": {},
            "backtest_result": None,
            "baseline_result": None,
            "llm_enhanced_result": None,
        }
        
        try:
            # Step 1: Extract model optimization information
            yaml_config_path = model_info.get("yaml_config_path")
            factor_pool_name = model_info.get("factor_pool_name")
            module_path = model_info.get("module_path")
            model_class = model_info.get("model_class", "TransformerModel")
            best_metrics = model_info.get("best_metrics", {})
            
            logs.append(f"[StrategyGeneration] Received model optimization results:")
            logs.append(f"  - Config path: {yaml_config_path}")
            logs.append(f"  - Factor pool: {factor_pool_name}")
            logs.append(f"  - Model type: {model_class}")
            logs.append(f"  - Factor count: {len(sota_pool_list)}")
            
            if best_metrics:
                ic_mean = best_metrics.get('ic_mean')
                ann_ret = best_metrics.get('annualized_return')
                if ic_mean is not None:
                    logs.append(f"  - Baseline IC: {ic_mean:.4f}")
                if ann_ret is not None:
                    logs.append(f"  - Baseline annualized: {ann_ret:.2%}")
            
            # Step 2: Build strategy configuration
            strategy_config = self._build_strategy_config(model_info, sota_pool_list)
            result["strategy_config"] = strategy_config
            logs.append("[StrategyGeneration] Strategy configuration generated")
            logs.append(f"[StrategyGeneration] TopK: {strategy_config.get('topk', 50)}, N_Drop: {strategy_config.get('n_drop', 10)}")
            
            # Step 3: Run backtest (optional)
            if run_backtest and yaml_config_path and Path(yaml_config_path).exists():
                if run_both_versions and self.use_llm_decision:
                    # =============================================================
                    # Run both versions: baseline and LLM-enhanced
                    # =============================================================
                    logs.append("[StrategyGeneration] ========== Starting dual-version backtest comparison ==========")
                    
                    # 3.1: Run Baseline version (without LLM)
                    logs.append("[StrategyGeneration] ---------- Step 1/2: Running Baseline version (without LLM) ----------")
                    baseline_yaml_path = self._generate_agent_yaml_config(
                        original_yaml_path=yaml_config_path,
                        strategy_config=strategy_config,
                        factor_pool_name=factor_pool_name,
                        use_llm_decision=False,
                        version_suffix="baseline"
                    )
                    logs.append(f"[StrategyGeneration] Baseline strategy config saved: {baseline_yaml_path}")
                    
                    baseline_result = self._run_backtest(baseline_yaml_path)
                    baseline_metrics = None
                    if baseline_result:
                        baseline_metrics = self._extract_backtest_metrics(baseline_result)
                        result["baseline_result"] = baseline_result
                        result["baseline_metrics"] = baseline_metrics
                        
                        logs.append("[StrategyGeneration] Baseline backtest completed")
                        if baseline_metrics.get("ic_mean"):
                            logs.append(f"[StrategyGeneration] Baseline IC mean: {baseline_metrics['ic_mean']:.4f}")
                        if baseline_metrics.get("annualized_return"):
                            logs.append(f"[StrategyGeneration] Baseline annualized return: {baseline_metrics['annualized_return']:.2%}")
                        if baseline_metrics.get("max_drawdown"):
                            logs.append(f"[StrategyGeneration] Baseline max drawdown: {baseline_metrics['max_drawdown']:.2%}")
                    else:
                        logs.append("[StrategyGeneration] Warning: Baseline backtest returned no result")
                    
                    # 3.2: Run LLM-enhanced version
                    logs.append("[StrategyGeneration] ---------- Step 2/2: Running LLM-enhanced version ----------")
                    llm_yaml_path = self._generate_agent_yaml_config(
                        original_yaml_path=yaml_config_path,
                        strategy_config=strategy_config,
                        factor_pool_name=factor_pool_name,
                        use_llm_decision=True,
                        version_suffix="llm_enhanced"
                    )
                    logs.append(f"[StrategyGeneration] LLM-enhanced strategy config saved: {llm_yaml_path}")
                    
                    llm_result = self._run_backtest(llm_yaml_path)
                    llm_metrics = None
                    if llm_result:
                        llm_metrics = self._extract_backtest_metrics(llm_result)
                        result["llm_enhanced_result"] = llm_result
                        result["llm_enhanced_metrics"] = llm_metrics
                        result["backtest_result"] = llm_result  # Keep compatibility
                        result["backtest_metrics"] = llm_metrics
                        
                        logs.append("[StrategyGeneration] LLM-enhanced backtest completed")
                        if llm_metrics.get("ic_mean"):
                            logs.append(f"[StrategyGeneration] LLM-enhanced IC mean: {llm_metrics['ic_mean']:.4f}")
                        if llm_metrics.get("annualized_return"):
                            logs.append(f"[StrategyGeneration] LLM-enhanced annualized return: {llm_metrics['annualized_return']:.2%}")
                        if llm_metrics.get("max_drawdown"):
                            logs.append(f"[StrategyGeneration] LLM-enhanced max drawdown: {llm_metrics['max_drawdown']:.2%}")
                    else:
                        logs.append("[StrategyGeneration] Warning: LLM-enhanced backtest returned no result")
                    
                    # 3.3: Compare two versions
                    if baseline_metrics and llm_metrics:
                        logs.append("[StrategyGeneration] ========== Version comparison results ==========")
                        self._compare_two_versions(baseline_metrics, llm_metrics, logs)
                        result["comparison"] = self._generate_comparison_dict(baseline_metrics, llm_metrics)
                    
                    # Set final config path to use (prioritize LLM-enhanced version)
                    agent_yaml_path = llm_yaml_path if llm_yaml_path else baseline_yaml_path
                    
                else:
                    # =============================================================
                    # Run only one version (determined by use_llm_decision)
                    # =============================================================
                    use_llm = self.use_llm_decision if not run_both_versions else False
                    version_name = "LLM-enhanced" if use_llm else "Baseline"
                    logs.append(f"[StrategyGeneration] Starting {version_name} strategy backtest...")
                    
                    agent_yaml_path = self._generate_agent_yaml_config(
                        original_yaml_path=yaml_config_path,
                        strategy_config=strategy_config,
                        factor_pool_name=factor_pool_name,
                        use_llm_decision=use_llm,
                        version_suffix="baseline" if not use_llm else "llm_enhanced"
                    )
                    logs.append(f"[StrategyGeneration] {version_name} strategy config saved: {agent_yaml_path}")
                    
                    backtest_result = self._run_backtest(agent_yaml_path)
                    
                    if backtest_result:
                        result["backtest_result"] = backtest_result
                        
                        # Extract backtest metrics
                        backtest_metrics = self._extract_backtest_metrics(backtest_result)
                        result["backtest_metrics"] = backtest_metrics
                        
                        logs.append(f"[StrategyGeneration] {version_name} backtest completed")
                        if backtest_metrics.get("ic_mean"):
                            logs.append(f"[StrategyGeneration] {version_name} IC mean: {backtest_metrics['ic_mean']:.4f}")
                        if backtest_metrics.get("annualized_return"):
                            logs.append(f"[StrategyGeneration] {version_name} annualized return: {backtest_metrics['annualized_return']:.2%}")
                        if backtest_metrics.get("max_drawdown"):
                            logs.append(f"[StrategyGeneration] {version_name} max drawdown: {backtest_metrics['max_drawdown']:.2%}")
                        
                        # Compare with baseline
                        if best_metrics and backtest_metrics:
                            self._compare_with_baseline(best_metrics, backtest_metrics, logs)
                    else:
                        logs.append(f"[StrategyGeneration] Warning: {version_name} backtest returned no result")
            elif not run_backtest:
                logs.append("[StrategyGeneration] Skipping backtest (run_backtest=False)")
                agent_yaml_path = None
            else:
                logs.append(f"[StrategyGeneration] Warning: Original YAML config does not exist: {yaml_config_path}")
                agent_yaml_path = None
            
            # Build complete strategy information
            result["strategy"] = {
                "type": "AgentEnhancedStrategy",
                "config": strategy_config,
                "yaml_config_path": agent_yaml_path or yaml_config_path,
                "factor_pool_name": factor_pool_name,
                "module_path": module_path,
                "model_class": model_class,
                "factors_count": len(sota_pool_list),
                "use_agent_decision": True,
                "use_llm_decision": self.use_llm_decision,
            }
            
            logs.append("[StrategyGeneration] Strategy generation process completed")
            
        except Exception as e:
            import traceback
            logs.append(f"[StrategyGeneration] Processing error: {str(e)}")
            logs.append(traceback.format_exc())
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _build_strategy_config(
        self,
        model_info: Dict[str, Any],
        factors: List[str]
    ) -> Dict[str, Any]:
        """
        Build strategy configuration
        
        Args:
            model_info: Model optimization results
            factors: Factor list
            
        Returns:
            Strategy configuration dictionary
        """
        # Get default strategy configuration from model optimization results
        default_config = model_info.get("default_strategy_config", {})
        
        config = {
            # Stock selection parameters
            "topk": default_config.get("topk", self.config.topk),
            "n_drop": default_config.get("n_drop", self.config.n_drop),
            
            # Trading constraints
            "max_turnover": self.config.max_turnover,
            "min_trade_value": self.config.min_trade_value,
            
            # Trading costs
            "open_cost": self.config.open_cost,
            "close_cost": self.config.close_cost,
            "min_cost": self.config.min_cost,
            
            # Limit up/down restrictions
            "limit_threshold": self.config.limit_threshold,
            
            # Position control
            "risk_degree": 0.95,
            "hold_thresh": 1,
            
            # Agent decision configuration
            "use_agent_decision": True,
            "use_llm_decision": self.use_llm_decision,
        }
        
        # Factor count
        config["factors_count"] = len(factors)
        
        # Automatically adjust topk based on factor count
        if len(factors) < 20:
            config["topk"] = min(30, config["topk"])
        elif len(factors) > 50:
            config["topk"] = min(80, max(50, config["topk"]))
        
        return config
    
    def _generate_agent_yaml_config(
        self,
        original_yaml_path: str,
        strategy_config: Dict[str, Any],
        factor_pool_name: str,
        use_llm_decision: bool = False,
        version_suffix: str = ""
    ) -> str:
        """
        Generate YAML configuration file with AgentEnhancedStrategy
        
        Args:
            original_yaml_path: Original YAML config path
            strategy_config: Strategy configuration
            factor_pool_name: Factor pool name
            use_llm_decision: Whether to use LLM-enhanced decision
            version_suffix: Version suffix (for distinguishing baseline and llm_enhanced)
            
        Returns:
            New YAML configuration file path
        """
        # Read original configuration
        with open(original_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Generate AgentEnhancedStrategy configuration
        agent_strategy_config = {
            "class": "AgentEnhancedStrategy",
            "module_path": "Agent.strategy_generation_agent",
            "kwargs": {
                "signal": "<PRED>",
                "topk": strategy_config.get("topk", 50),
                "n_drop": strategy_config.get("n_drop", 10),
                "use_agent_decision": strategy_config.get("use_agent_decision", True),
                "use_llm_decision": use_llm_decision,
                "use_env_llm_config": True,
                "risk_degree": strategy_config.get("risk_degree", 0.95),
                "hold_thresh": strategy_config.get("hold_thresh", 1),
                "only_tradable": False,
                "forbid_all_trade_at_limit": True,
            }
        }
        
        # If using LLM decision, add news data configuration
        if use_llm_decision:
            if self.news_data_path:
                agent_strategy_config["kwargs"]["news_data_path"] = self.news_data_path
            if self.news_batch_size != 10:
                agent_strategy_config["kwargs"]["news_batch_size"] = self.news_batch_size
        
        back_test_start_time = '2025-01-01'
        back_test_end_time = '2025-06-30'
        # Replace strategy configuration - fix path issues
        # Option 1: Replace port_analysis_config.strategy
        if 'port_analysis_config' in config:
            config['port_analysis_config']['strategy'] = agent_strategy_config
        # Option 2: Compatible with task.backtest.strategy structure
        elif 'task' in config and 'backtest' in config['task']:
            config['task']['backtest']['strategy'] = agent_strategy_config
        # 1. Expand data_handler_config.end_time
        if 'data_handler_config' in config:
            if 'end_time' in config['data_handler_config']:
                config['data_handler_config']['end_time'] = back_test_end_time
                logger.info(f"[Strategy] Expanded data_handler_config.end_time to: {back_test_end_time}")

        # Modify backtest time
        if 'port_analysis_config' in config and 'backtest' in config['port_analysis_config']:
            config['port_analysis_config']['backtest']['start_time'] = back_test_start_time
            config['port_analysis_config']['backtest']['end_time'] = back_test_end_time
        # 2. Expand test segment to cover backtest time (critical fix!)
        if 'task' in config and 'dataset' in config['task']:
            dataset_config = config['task']['dataset']
            if 'kwargs' in dataset_config and 'segments' in dataset_config['kwargs']:
                segments = dataset_config['kwargs']['segments']
                if 'test' in segments and len(segments['test']) >= 2:
                    segments['test'][0] = "2024-12-31"
                    segments['test'][1] = back_test_end_time
                    segments['valid'][1] = "2024-12-31"
                    logger.info(f"[Strategy] Expanded test segment end_time to: {back_test_end_time}")
        # Save new configuration
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{version_suffix}" if version_suffix else ""
        output_path = self.output_dir / f"workflow_agent_strategy_{factor_pool_name}{suffix_str}_{timestamp}.yaml"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return str(output_path)
    
    def _run_backtest(self, yaml_path: str) -> Optional[Dict[str, Any]]:
        """
        Run backtest
        
        Args:
            yaml_path: YAML configuration path
            
        Returns:
            Backtest result dictionary
        """
        if not self.mcp_client:
            print("[StrategyGeneration] Warning: MCP client not initialized, cannot run backtest")
            return None
        
        try:
            result = self.mcp_client.call_tool(
                "qlib_benchmark_runner",
                {"yaml_path": yaml_path}
            )
            
            if isinstance(result, str):
                return json.loads(result)
            return result
            
        except Exception as e:
            print(f"[StrategyGeneration] Backtest execution failed: {e}")
            return None
    
    def _extract_backtest_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from backtest results
        
        Args:
            backtest_result: Backtest result
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        if FileUtils is None:
            return metrics
        
        # Read IC statistics
        ic_path = backtest_result.get("ic", "")
        if ic_path:
            ic_stats = FileUtils.read_pickle_stats(ic_path)
            if ic_stats:
                metrics["ic_mean"] = ic_stats.get("mean", 0)
                metrics["ic_std"] = ic_stats.get("std", 0)
        
        # Read Rank IC statistics
        rank_ic_path = backtest_result.get("rank_ic", "")
        if rank_ic_path:
            rank_ic_stats = FileUtils.read_pickle_stats(rank_ic_path)
            if rank_ic_stats:
                metrics["rank_ic_mean"] = rank_ic_stats.get("mean", 0)
                metrics["rank_ic_std"] = rank_ic_stats.get("std", 0)
        
        # Read annualized return
        ann_ret_path = backtest_result.get("1day.excess_return_with_cost.annualized_return", "")
        if ann_ret_path:
            metrics["annualized_return"] = FileUtils.read_mlflow_metric(ann_ret_path)
        
        # Read max drawdown
        max_dd_path = backtest_result.get("1day.excess_return_with_cost.max_drawdown", "")
        if max_dd_path:
            metrics["max_drawdown"] = FileUtils.read_mlflow_metric(max_dd_path)
        
        # Calculate IR
        if metrics.get("ic_mean") and metrics.get("ic_std") and metrics["ic_std"] != 0:
            metrics["ir"] = metrics["ic_mean"] / metrics["ic_std"]
        
        return metrics
    
    def _compare_with_baseline(
        self,
        baseline_metrics: Dict[str, Any],
        backtest_metrics: Dict[str, Any],
        logs: List[str]
    ) -> None:
        """
        Compare backtest results with baseline
        
        Args:
            baseline_metrics: Baseline metrics
            backtest_metrics: Backtest metrics
            logs: Log list
        """
        logs.append("[StrategyGeneration] === Comparison with Baseline ===")
        
        # IC comparison
        baseline_ic = baseline_metrics.get("ic_mean")
        backtest_ic = backtest_metrics.get("ic_mean")
        if baseline_ic and backtest_ic:
            diff = backtest_ic - baseline_ic
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  IC: {baseline_ic:.4f} -> {backtest_ic:.4f} ({sign} {abs(diff):.4f})")
        
        # Annualized return comparison
        baseline_ret = baseline_metrics.get("annualized_return")
        backtest_ret = backtest_metrics.get("annualized_return")
        if baseline_ret and backtest_ret:
            diff = backtest_ret - baseline_ret
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  Annualized return: {baseline_ret:.2%} -> {backtest_ret:.2%} ({sign} {abs(diff):.2%})")
        
        # Max drawdown comparison
        baseline_dd = baseline_metrics.get("max_drawdown")
        backtest_dd = backtest_metrics.get("max_drawdown")
        if baseline_dd and backtest_dd:
            diff = backtest_dd - baseline_dd
            # Drawdown is negative, larger (closer to 0) is better
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  Max drawdown: {baseline_dd:.2%} -> {backtest_dd:.2%} ({sign} {abs(diff):.2%})")
    
    def _compare_two_versions(
        self,
        baseline_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any],
        logs: List[str]
    ) -> None:
        """
        Compare backtest results of Baseline version and LLM-enhanced version
        
        Args:
            baseline_metrics: Baseline version metrics
            llm_metrics: LLM-enhanced version metrics
            logs: Log list
        """
        logs.append("[StrategyGeneration] === Baseline vs LLM-enhanced Version Comparison ===")
        
        # IC comparison
        baseline_ic = baseline_metrics.get("ic_mean")
        llm_ic = llm_metrics.get("ic_mean")
        if baseline_ic is not None and llm_ic is not None:
            diff = llm_ic - baseline_ic
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            improvement_pct = (diff / abs(baseline_ic) * 100) if baseline_ic != 0 else 0
            logs.append(f"  IC mean: Baseline {baseline_ic:.4f} -> LLM-enhanced {llm_ic:.4f} ({sign} {abs(diff):.4f}, {improvement_pct:+.2f}%)")
        
        # Rank IC comparison
        baseline_rank_ic = baseline_metrics.get("rank_ic_mean")
        llm_rank_ic = llm_metrics.get("rank_ic_mean")
        if baseline_rank_ic is not None and llm_rank_ic is not None:
            diff = llm_rank_ic - baseline_rank_ic
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  Rank IC: Baseline {baseline_rank_ic:.4f} -> LLM-enhanced {llm_rank_ic:.4f} ({sign} {abs(diff):.4f})")
        
        # Annualized return comparison
        baseline_ret = baseline_metrics.get("annualized_return")
        llm_ret = llm_metrics.get("annualized_return")
        if baseline_ret is not None and llm_ret is not None:
            diff = llm_ret - baseline_ret
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  Annualized return: Baseline {baseline_ret:.2%} -> LLM-enhanced {llm_ret:.2%} ({sign} {abs(diff):.2%})")
        
        # Max drawdown comparison
        baseline_dd = baseline_metrics.get("max_drawdown")
        llm_dd = llm_metrics.get("max_drawdown")
        if baseline_dd is not None and llm_dd is not None:
            diff = llm_dd - baseline_dd
            # Drawdown is negative, larger (closer to 0) is better
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  Max drawdown: Baseline {baseline_dd:.2%} -> LLM-enhanced {llm_dd:.2%} ({sign} {abs(diff):.2%})")
        
        # IR comparison
        baseline_ir = baseline_metrics.get("ir")
        llm_ir = llm_metrics.get("ir")
        if baseline_ir is not None and llm_ir is not None:
            diff = llm_ir - baseline_ir
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  IR (IC/Std): Baseline {baseline_ir:.4f} -> LLM-enhanced {llm_ir:.4f} ({sign} {abs(diff):.4f})")
        
        # Summary
        improvements = []
        if baseline_ic is not None and llm_ic is not None and llm_ic > baseline_ic:
            improvements.append("IC")
        if baseline_ret is not None and llm_ret is not None and llm_ret > baseline_ret:
            improvements.append("Annualized return")
        if baseline_dd is not None and llm_dd is not None and llm_dd > baseline_dd:
            improvements.append("Max drawdown")
        
        if improvements:
            logs.append(f"[StrategyGeneration] LLM-enhanced version improved on the following metrics: {', '.join(improvements)}")
        else:
            logs.append(f"[StrategyGeneration] LLM-enhanced version did not outperform Baseline on all metrics")
    
    def _generate_comparison_dict(
        self,
        baseline_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comparison dictionary for two versions
        
        Args:
            baseline_metrics: Baseline version metrics
            llm_metrics: LLM-enhanced version metrics
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "baseline": baseline_metrics.copy(),
            "llm_enhanced": llm_metrics.copy(),
            "differences": {}
        }
        
        # Calculate differences for each metric
        for key in ["ic_mean", "rank_ic_mean", "annualized_return", "max_drawdown", "ir"]:
            baseline_val = baseline_metrics.get(key)
            llm_val = llm_metrics.get(key)
            if baseline_val is not None and llm_val is not None:
                diff = llm_val - baseline_val
                diff_pct = (diff / abs(baseline_val) * 100) if baseline_val != 0 else 0
                comparison["differences"][key] = {
                    "absolute": diff,
                    "percentage": diff_pct,
                    "improved": diff > 0 if key != "max_drawdown" else diff > 0  # Drawdown larger is better (closer to 0)
                }
        
        return comparison


# =============================================================================
# Qlib Strategy Integration Class
# =============================================================================

# Try importing Qlib related modules
try:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.position import Position
    from qlib.backtest.signal import Signal, create_signal_from
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("Warning: Qlib module unavailable, AgentEnhancedStrategy will use mock mode")
    
    # Create mock classes
    class BaseStrategy:
        pass
    
    class Position:
        pass
    
    class Order:
        SELL = 0
        BUY = 1
    
    class OrderDir:
        SELL = 0
        BUY = 1
    
    class TradeDecisionWO:
        def __init__(self, order_list, strategy):
            self.order_list = order_list
            self.strategy = strategy


class AgentEnhancedStrategy(BaseStrategy if QLIB_AVAILABLE else object):
    """
    Agent-enhanced Qlib Strategy class
    
    Inherits from Qlib's BaseStrategy (similar to TopkDropoutStrategy), calls StrategyGenerationAgent
    for intelligent decision-making in generate_trade_decision
    
    Main features:
    1. Fully compatible with Qlib backtest framework
    2. Uses Agent for intelligent trading decisions
    3. Supports LLM-enhanced decisions (configurable via YAML)
    4. Retains original TopkDropoutStrategy core logic as fallback
    """
    
    def __init__(
        self,
        *,
        agent: Optional[StrategyGenerationAgent] = None,
        signal: Union[Any, Tuple, List, Dict, str] = None,
        topk: int = 50,
        n_drop: int = 10,
        method_sell: str = "bottom",
        method_buy: str = "top",
        hold_thresh: int = 1,
        only_tradable: bool = False,
        forbid_all_trade_at_limit: bool = True,
        risk_degree: float = 0.95,
        use_agent_decision: bool = True,
        use_llm_decision: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
        use_env_llm_config: bool = False,
        news_data_path: Optional[str] = None,
        news_batch_size: int = 10,
        **kwargs
    ):
        """
        Initialize Agent-enhanced strategy
        
        Args:
            agent: StrategyGenerationAgent instance, if None will be created automatically
            signal: Qlib prediction signal (supports multiple formats)
            topk: Top K stock count
            n_drop: Number of stocks to drop per rebalancing
            method_sell: Sell method ("bottom" or "random")
            method_buy: Buy method ("top" or "random")
            hold_thresh: Minimum holding days
            only_tradable: Whether to only consider tradable stocks
            forbid_all_trade_at_limit: Whether to forbid all trades at limit up/down
            risk_degree: Position ratio (0-1)
            use_agent_decision: Whether to use Agent decision (False falls back to rule-based strategy)
            use_llm_decision: Whether to use LLM-enhanced decision (including news analysis)
            llm_config: LLM configuration dictionary, containing provider, api_key, model, etc.
                Example: {"provider": "qwen", "api_key": "xxx", "model": "qwen-plus"}
            use_env_llm_config: Whether to load LLM config from environment variables/config file (lower priority than llm_config)
            news_data_path: News data path (can be directory path or single JSON file path)
                           - Directory path: Auto-load by month, filename format: eastmoney_news_YYYY_processed_YYYY_MM.json
                           - File path: Compatible with old version, loads single JSON file
                           Data format: {date: {stock_code: [news_list]}}
            news_batch_size: Number of stocks analyzed per batch (for batch processing in news analysis)
            **kwargs: Other parameters passed to base class
        """
        if QLIB_AVAILABLE:
            super().__init__(**kwargs)
        
        # Strategy parameters
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.risk_degree = risk_degree
        self.use_agent_decision = use_agent_decision
        self.use_llm_decision = use_llm_decision
        
        # Initialize Signal
        if QLIB_AVAILABLE and signal is not None:
            self.signal = create_signal_from(signal)
        else:
            self.signal = signal
        
        # Initialize Agent
        if agent is not None:
            self.agent = agent
        else:
            # Automatically create Agent
            config = StrategyConfig(
                topk=topk,
                n_drop=n_drop,
                max_turnover=0.3,
                open_cost=kwargs.get("open_cost", 0.0005),
                close_cost=kwargs.get("close_cost", 0.0015),
                min_cost=kwargs.get("min_cost", 5),
                limit_threshold=kwargs.get("limit_threshold", 0.095),
            )
            
            # Create LLM service (if enabled)
            llm_service = None
            if use_llm_decision:
                llm_service = self._create_llm_service(
                    llm_config=llm_config,
                    use_env_config=use_env_llm_config
                )
                if llm_service is None:
                    print("[Warning] LLM service creation failed, will use rule-based decision")
            
            self.agent = StrategyGenerationAgent(
                config=config,
                llm_service=llm_service,
                use_llm_decision=use_llm_decision and llm_service is not None,
                news_data_path=news_data_path,
                news_batch_size=news_batch_size
            )
    
    def _create_llm_service(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        use_env_config: bool = False
    ):
        """
        Create LLM service instance
        
        Supports two methods:
        1. Configure via llm_config dictionary
        2. Load from environment variables/config file (use_env_config=True)
        
        Args:
            llm_config: LLM configuration dictionary
            use_env_config: Whether to load config from environment variables
            
        Returns:
            LLM service instance or None
        """
        try:
            from Agent.agent_factory import load_env_config, create_agent
            
            # Prioritize provided configuration
            if llm_config:
                config = llm_config
            elif use_env_config:
                # Load from environment variables/config file
                config = load_env_config()
            else:
                print("[Warning] No LLM config provided, environment variable config not enabled")
                return None
            
            if not config:
                print("[Warning] LLM config is empty")
                return None
            
            # Create LLM Agent
            llm_service = create_agent(
                provider=config.get("provider", "qwen"),
                api_key=config.get("api_key"),
                model=config.get("model"),
                base_url=config.get("base_url"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens"),
            )
            
            print(f"[Info] LLM service created successfully: provider={config.get('provider')}, model={config.get('model')}")
            return llm_service
            
        except ImportError as e:
            print(f"[Warning] Cannot import agent_factory: {e}")
            return None
        except Exception as e:
            print(f"[Warning] Failed to create LLM service: {e}")
            return None
    
    def get_risk_degree(self, trade_step=None) -> float:
        """Get risk degree (position ratio)"""
        return self.risk_degree
    
    def generate_trade_decision(self, execute_result=None):
        """
        Generate trading decision - core interface method for Qlib backtest framework
        
        This method is called by Qlib backtest framework on each trading day
        
        Reference TopkDropoutStrategy.generate_trade_decision implementation
        
        Args:
            execute_result: Previous execution result (usually None or execution report)
            
        Returns:
            TradeDecisionWO: Qlib format trading decision object
        """
        if not QLIB_AVAILABLE:
            # Mock mode
            return self._generate_trade_decision_mock(execute_result)
        
        # =====================================================================
        # Step 1: Get trading time and prediction signal
        # =====================================================================
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        logger.info(f"[Strategy] ========== trade_step={trade_step}, trade_time={trade_start_time} ==========")
        
        # Get prediction scores
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        # Handle DataFrame format (only take first column)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        
        if pred_score is None:
            logger.warning(f"[Strategy] pred_score is None for trade_step={trade_step}, returning empty decision")
            return TradeDecisionWO([], self)
        
        logger.info(f"[Strategy] pred_score length={len(pred_score)}")
        
        # =====================================================================
        # Step 2: Get current holdings information
        # =====================================================================
        current_position: Position = copy.deepcopy(self.trade_position)
        cash = current_position.get_cash()
        current_stock_list = current_position.get_stock_list()
        
        # Convert to format needed by Agent
        current_holdings = self._position_to_holdings(current_position)
        
        # =====================================================================
        # Step 3: Get Top K recommendations
        # =====================================================================
        top_k_recommendations = self._get_top_k_from_signal(pred_score)
        
        # =====================================================================
        # Step 4: Get market data
        # =====================================================================
        all_stocks = list(set(current_stock_list) | set([s for s, _ in top_k_recommendations]))
        market_data = self._get_market_data(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            stock_list=all_stocks
        )
        
        # =====================================================================
        # Step 5: Call Agent to generate trading decision
        # =====================================================================
        logger.info(f"[Strategy] trade_step={trade_step}, use_agent_decision={self.use_agent_decision}, holdings={len(current_holdings)}, recommendations={len(top_k_recommendations)}")
        
        if self.use_agent_decision:
            logger.info(f"[Strategy] Calling agent.generate_trade_decision for {trade_start_time}")
            agent_decision = self.agent.generate_trade_decision(
                current_holdings=current_holdings,
                top_k_recommendations=top_k_recommendations,
                market_data=market_data,
                date=str(trade_start_time)
            )
            logger.info(f"[Strategy] Agent decision: sell={len(agent_decision.sell_list) if agent_decision else 0}, buy={len(agent_decision.buy_list) if agent_decision else 0}")
        else:
            logger.info("[Strategy] Skipping agent decision (use_agent_decision=False)")
            agent_decision = None
        
        # =====================================================================
        # Step 6: Convert Agent decision to Qlib Order list
        # =====================================================================
        order_list = self._convert_decision_to_orders(
            agent_decision=agent_decision,
            current_position=current_position,
            pred_score=pred_score,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            cash=cash
        )
        
        return TradeDecisionWO(order_list, self)
    
    def _position_to_holdings(self, position: Position) -> Dict[str, float]:
        """
        Convert Qlib Position object to holdings weight dictionary
        
        Args:
            position: Qlib Position object
            
        Returns:
            Holdings weight dictionary {stock_code: weight}
        """
        if not QLIB_AVAILABLE:
            return {}
        
        holdings = {}
        stock_list = position.get_stock_list()
        
        # Calculate total value
        total_value = position.calculate_value()
        
        if total_value <= 0:
            return holdings
        
        for stock in stock_list:
            stock_value = position.get_stock_amount(stock) * position.get_stock_price(stock)
            holdings[stock] = stock_value / total_value
        
        return holdings
    
    def _get_top_k_from_signal(
        self,
        pred_score: pd.Series
    ) -> List[Tuple[str, float]]:
        """
        Get Top K recommended stocks from prediction signal
        
        Args:
            pred_score: Prediction score Series, index is stock code
            
        Returns:
            Top K recommendation list [(stock_code, score), ...]
        """
        if pred_score is None or len(pred_score) == 0:
            return []
        
        # Sort by score in descending order, take Top K
        sorted_scores = pred_score.sort_values(ascending=False)
        top_k = sorted_scores.head(self.topk)
        
        return [(stock, score) for stock, score in top_k.items()]
    
    def _get_market_data(
        self,
        trade_start_time,
        trade_end_time,
        stock_list: List[str]
    ) -> Dict[str, Any]:
        """
        Get market data (price, volume, limit up/down information, etc.)
        
        Args:
            trade_start_time: Trading start time
            trade_end_time: Trading end time
            stock_list: Stock list
            
        Returns:
            Market data dictionary {stock_code: {price, volume, price_change_pct, is_tradable}}
        """
        if not QLIB_AVAILABLE:
            return {}
        
        market_data = {}
        
        for stock in stock_list:
            try:
                # Check if tradable
                is_tradable = self.trade_exchange.is_stock_tradable(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time
                )
                
                # Get price
                price = self.trade_exchange.get_deal_price(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY
                )

                volume = self.trade_exchange.get_volume(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    method="sum"  # Or "ts_data_last" to get last value
                )
                # $change field is loaded when Exchange is initialized, represents price change
                change = self.trade_exchange.get_quote_info(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    field="$change",
                    method="ts_data_last"
                )
                # $change is already in percentage form (e.g., 0.05 means 5%)
                price_change_pct = change if change is not None else 0.0
                
                market_data[stock] = {
                    "price": price,
                    "is_tradable": is_tradable,
                    "price_change_pct": price_change_pct,  # Can calculate if needed
                    "volume": volume,  # Simplified processing
                }
            except Exception:
                market_data[stock] = {
                    "price": 1.0,
                    "is_tradable": True,
                    "price_change_pct": 0.0,
                    "volume": float('inf'),
                }
        
        return market_data
    
    def _convert_decision_to_orders(
        self,
        agent_decision: Optional[TradeDecision],
        current_position: Position,
        pred_score: pd.Series,
        trade_start_time,
        trade_end_time,
        cash: float
    ) -> List[Order]:
        """
        Convert Agent's TradeDecision to Qlib Order list
        
        If agent_decision is None, fall back to rule-based logic similar to TopkDropoutStrategy
        
        Args:
            agent_decision: Agent's trading decision
            current_position: Current position
            pred_score: Prediction score
            trade_start_time: Trading start time
            trade_end_time: Trading end time
            cash: Current cash
            
        Returns:
            Qlib Order list
        """
        sell_order_list = []
        buy_order_list = []
        
        if agent_decision is None:
            # Fall back to rule-based strategy
            return self._generate_orders_rule_based(
                current_position, pred_score, trade_start_time, trade_end_time, cash
            )
        
        # =====================================================================
        # Process sell orders
        # =====================================================================
        for signal in agent_decision.sell_list:
            stock_code = signal.stock_code
            
            # Check if tradable
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock_code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            # Check holding days (first check if stock is in holdings)
            if current_position.get_stock_amount(code=stock_code) == 0:
                continue  # If holdings is 0, skip
            time_per_step = self.trade_calendar.get_freq()
            if current_position.get_stock_count(stock_code, bar=time_per_step) < self.hold_thresh:
                continue
            
            # Get sell amount
            sell_amount = current_position.get_stock_amount(code=stock_code)
            
            if sell_amount > 0:
                sell_order = Order(
                    stock_id=stock_code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                
                # Check if order is executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    # Simulate execution to update cash
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_position
                    )
                    cash += trade_val - trade_cost
        
        # =====================================================================
        # Process buy orders
        # =====================================================================
        buy_stocks = [signal.stock_code for signal in agent_decision.buy_list]
        
        if len(buy_stocks) > 0:
            # Calculate buy amount per stock
            value = cash * self.risk_degree / len(buy_stocks)
            
            for signal in agent_decision.buy_list:
                stock_code = signal.stock_code
                
                # Check if tradable
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=stock_code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
                ):
                    continue
                
                # Get buy price
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=stock_code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY
                )
                
                if buy_price <= 0:
                    continue
                
                # Calculate buy amount
                buy_amount = value / buy_price
                factor = self.trade_exchange.get_factor(
                    stock_id=stock_code,
                    start_time=trade_start_time,
                    end_time=trade_end_time
                )
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
                
                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=stock_code,
                        amount=buy_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.BUY,
                    )
                    buy_order_list.append(buy_order)
        
        return sell_order_list + buy_order_list
    
    def _generate_orders_rule_based(
        self,
        current_position: Position,
        pred_score: pd.Series,
        trade_start_time,
        trade_end_time,
        cash: float
    ) -> List[Order]:
        """
        Rule-based order generation (fallback solution, similar to TopkDropoutStrategy)
        
        Used when Agent decision is unavailable
        
        Args:
            current_position: Current position
            pred_score: Prediction score
            trade_start_time: Trading start time
            trade_end_time: Trading end time
            cash: Current cash
            
        Returns:
            Order list
        """
        sell_order_list = []
        buy_order_list = []
        
        current_stock_list = current_position.get_stock_list()
        
        # Sort current holdings by score
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False, kind='mergesort').index
        
        # Get new recommended stocks (high-score stocks not in current holdings)
        today = list(
            pred_score[~pred_score.index.isin(last)]
            .sort_values(ascending=False)
            .index[:self.n_drop + self.topk - len(last)]
        )
        
        # Merge and sort
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        
        # Determine sell stocks (lowest-scored n_drop stocks)
        sell = last[last.isin(list(comb)[-self.n_drop:])] if len(comb) > self.n_drop else pd.Index([])
        
        # Determine buy stocks
        buy = today[:len(sell) + self.topk - len(last)]
        
        # Generate sell orders
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                if current_position.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                
                sell_amount = current_position.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_position
                    )
                    cash += trade_val - trade_cost
        
        # Generate buy orders
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
        
        return sell_order_list + buy_order_list
    
    def _generate_trade_decision_mock(self, execute_result=None):
        """
        Mock mode trading decision generation (when Qlib is unavailable)
        
        Args:
            execute_result: Execution result
            
        Returns:
            Mock TradeDecisionWO object
        """
        return TradeDecisionWO([], self)


# =============================================================================
# Factory Functions
# =============================================================================

def create_strategy_generation_agent(
    llm_service=None,
    config: Optional[Dict[str, Any]] = None,
    use_llm: bool = False
) -> StrategyGenerationAgent:
    """
    Factory function to create strategy generation Agent
    
    Args:
        llm_service: LLM service instance
        config: Strategy configuration dictionary
        use_llm: Whether to use LLM decision
        
    Returns:
        StrategyGenerationAgent instance
    """
    strategy_config = StrategyConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(strategy_config, key):
                setattr(strategy_config, key, value)
    
    return StrategyGenerationAgent(
        llm_service=llm_service,
        config=strategy_config,
        use_llm_decision=use_llm
    )


def create_agent_enhanced_strategy(
    signal: Any,
    topk: int = 50,
    n_drop: int = 10,
    agent: Optional[StrategyGenerationAgent] = None,
    llm_service=None,
    use_llm: bool = False,
    use_agent_decision: bool = True,
    **kwargs
) -> AgentEnhancedStrategy:
    """
    Factory function to create Agent-enhanced strategy
    
    Args:
        signal: Qlib prediction signal (required)
        topk: Top K stock count
        n_drop: Number to drop per rebalancing
        agent: StrategyGenerationAgent instance (optional, if None will be created automatically)
        llm_service: LLM service (optional, for creating Agent)
        use_llm: Whether to use LLM decision
        use_agent_decision: Whether to use Agent decision (False uses pure rule-based strategy)
        **kwargs: Other parameters (such as risk_degree, hold_thresh, etc.)
        
    Returns:
        AgentEnhancedStrategy instance
        
    Example:
        ```python
        # Basic usage
        strategy = create_agent_enhanced_strategy(
            signal=pred_signal,
            topk=50,
            n_drop=10
        )
        
        # Use LLM enhancement
        strategy = create_agent_enhanced_strategy(
            signal=pred_signal,
            topk=50,
            n_drop=10,
            llm_service=my_llm,
            use_llm=True
        )
        
        # Custom Agent
        my_agent = create_strategy_generation_agent(config={"topk": 30})
        strategy = create_agent_enhanced_strategy(
            signal=pred_signal,
            agent=my_agent
        )
        ```
    """
    # If no Agent provided, create automatically
    if agent is None:
        agent_config = {
            "topk": topk,
            "n_drop": n_drop,
            "open_cost": kwargs.get("open_cost", 0.0005),
            "close_cost": kwargs.get("close_cost", 0.0015),
            "min_cost": kwargs.get("min_cost", 5),
            "limit_threshold": kwargs.get("limit_threshold", 0.095),
            "max_turnover": kwargs.get("max_turnover", 0.3),
        }
        agent = create_strategy_generation_agent(
            llm_service=llm_service,
            config=agent_config,
            use_llm=use_llm
        )
    
    return AgentEnhancedStrategy(
        agent=agent,
        signal=signal,
        topk=topk,
        n_drop=n_drop,
        use_agent_decision=use_agent_decision,
        **kwargs
    )


def get_strategy_config_for_qlib(
    topk: int = 50,
    n_drop: int = 10,
    use_agent: bool = True,
    use_llm: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
    use_env_llm_config: bool = False,
    news_data_path: Optional[str] = None,
    news_batch_size: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate strategy configuration dictionary for Qlib yaml configuration file
    
    Can be directly used in strategy configuration in workflow yaml file
    
    Args:
        topk: Top K stock count
        n_drop: Number to drop per rebalancing
        use_agent: Whether to use Agent decision
        use_llm: Whether to use LLM-enhanced decision (including news analysis)
        llm_config: LLM configuration dictionary (optional)
        use_env_llm_config: Whether to load LLM config from environment variables
        news_data_path: News data path (can be directory path or single JSON file path)
                       - Directory path: Auto-load by month, filename format: eastmoney_news_YYYY_processed_YYYY_MM.json
                       - File path: Compatible with old version, loads single JSON file
        news_batch_size: Number of stocks analyzed per batch
        **kwargs: Other strategy parameters
        
    Returns:
        Strategy configuration dictionary, can be directly used in Qlib yaml config
        
    Example:
        ```python
        # Basic configuration (without LLM)
        config = get_strategy_config_for_qlib(topk=50, n_drop=10)
        
        # Use LLM config from environment variables (including news analysis)
        config = get_strategy_config_for_qlib(
            topk=50, 
            n_drop=10, 
            use_llm=True,
            use_env_llm_config=True,
            news_data_path="/path/to/news_data.json",
            news_batch_size=10
        )
        
        # Use custom LLM config
        config = get_strategy_config_for_qlib(
            topk=50,
            n_drop=10,
            use_llm=True,
            llm_config={
                "provider": "qwen",
                "api_key": "your-api-key",
                "model": "qwen-plus"
            },
            news_data_path="/path/to/news_data.json"
        )
        ```
    """
    strategy_kwargs = {
        "signal": "<PRED>",
        "topk": topk,
        "n_drop": n_drop,
        "use_agent_decision": use_agent,
        "use_llm_decision": use_llm,
        "use_env_llm_config": use_env_llm_config,
        "risk_degree": kwargs.get("risk_degree", 0.95),
        "hold_thresh": kwargs.get("hold_thresh", 1),
        "only_tradable": kwargs.get("only_tradable", False),
        "forbid_all_trade_at_limit": kwargs.get("forbid_all_trade_at_limit", True),
    }
    
    # Add LLM configuration (if provided)
    if llm_config:
        strategy_kwargs["llm_config"] = llm_config
    
    # Add news data configuration (if provided)
    if news_data_path:
        strategy_kwargs["news_data_path"] = news_data_path
    if news_batch_size != 10:
        strategy_kwargs["news_batch_size"] = news_batch_size
    
    return {
        "class": "AgentEnhancedStrategy",
        "module_path": "Agent.strategy_generation_agent",
        "kwargs": strategy_kwargs
    }

