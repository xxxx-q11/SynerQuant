"""
策略生成 Agent - 继承 Qlib Strategy 类，在回测中进行智能交易决策

参考 Qlib TopkDropoutStrategy 实现
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

# 配置日志
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
    print("警告: pandas/numpy 不可用，部分功能受限")

# 尝试导入 MCP 客户端
try:
    from utils.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("警告: MCP 客户端不可用")

# 尝试导入文件工具
try:
    from Agent.utils.file_utils import FileUtils
except ImportError:
    FileUtils = None


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class TradeSignal:
    """交易信号"""
    stock_code: str
    action: str  # "buy", "sell", "hold", "adjust"
    target_weight: float  # 目标权重 (0-1)
    reason: str
    score: float = 0.0  # 模型评分
    current_weight: float = 0.0  # 当前权重
    confidence: float = 1.0  # 置信度 (0-1)
    priority: int = 0  # 优先级，数值越大越优先


@dataclass
class PortfolioState:
    """持仓状态"""
    holdings: Dict[str, float]  # {stock_code: weight}
    cash_ratio: float  # 现金比例
    total_value: float  # 总价值
    date: str  # 当前日期


@dataclass 
class TradeDecision:
    """交易决策结果"""
    buy_list: List[TradeSignal]
    sell_list: List[TradeSignal]
    hold_list: List[TradeSignal]
    adjust_list: List[TradeSignal]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """
    持仓与推荐对比结果
    
    参考 TopkDropoutStrategy 的集合对比逻辑：
    - sell_candidates: 当前持仓中不在 Top K 推荐中的股票（待卖出）
    - buy_candidates: Top K 推荐中不在当前持仓中的股票（待买入）
    - hold_candidates: 当前持仓与 Top K 推荐的交集（继续持有）
    - ranked_holdings: 当前持仓按模型评分排序
    - ranked_recommendations: 推荐股票按评分排序
    """
    sell_candidates: Set[str]  # A - B: 持仓有但推荐无
    buy_candidates: Set[str]   # B - A: 推荐有但持仓无  
    hold_candidates: Set[str]  # A ∩ B: 两者交集
    
    # 带评分的排序列表
    ranked_holdings: List[Tuple[str, float]]  # [(stock_code, score), ...] 按分数降序
    ranked_recommendations: List[Tuple[str, float]]  # [(stock_code, score), ...] 按分数降序
    
    # 建议的卖出和买入列表（考虑 n_drop 限制）
    suggested_sell: List[str] = field(default_factory=list)  # 实际建议卖出的股票
    suggested_buy: List[str] = field(default_factory=list)   # 实际建议买入的股票
    
    # 评分映射
    score_map: Dict[str, float] = field(default_factory=dict)  # {stock_code: score}


@dataclass
class StrategyConfig:
    """策略配置"""
    topk: int = 50  # Top K 股票数量
    n_drop: int = 5  # 每次调仓丢弃数量
    max_turnover: float = 0.3  # 最大换手率
    min_trade_value: float = 10000  # 最小交易金额
    open_cost: float = 0.0005  # 买入成本
    close_cost: float = 0.0015  # 卖出成本
    min_cost: float = 5  # 最小交易成本
    limit_threshold: float = 0.095  # 涨跌停阈值


# =============================================================================
# 策略生成 Agent 核心类
# =============================================================================

class StrategyGenerationAgent:
    """
    策略生成 Agent
    
    负责在回测过程中，审阅持仓股票与模型推荐的 Top_K 股票，
    决定保留、买入和卖出哪些股票
    
    同时支持在 LangGraph 工作流中运行完整的回测流程
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
        初始化策略生成 Agent
        
        Args:
            llm_service: LLM 服务实例 (可选，用于智能决策)
            config: 策略配置
            use_llm_decision: 是否使用 LLM 进行决策
            mcp_server_path: MCP 服务器脚本路径（用于运行回测）
            news_data_path: 新闻数据路径（可以是目录路径或单个JSON文件路径）
                           - 目录路径：按月份自动加载，文件名格式为 eastmoney_news_YYYY_processed_YYYY_MM.json
                           - 文件路径：兼容旧版本，加载单个JSON文件
            news_batch_size: 每批次分析的股票数量
        """
        self.llm = llm_service
        self.config = config or StrategyConfig()
        self.use_llm_decision = use_llm_decision
        
        # 交易历史记录
        self.trade_history: List[Dict[str, Any]] = []
        
        # 决策日志
        self.decision_logs: List[str] = []
        
        # MCP 客户端（用于运行回测）
        self.mcp_client = None
        self._init_mcp_client(mcp_server_path)
        
        # 配置路径
        # 检测是否在容器内（通过检查 /workspace 是否存在）
        is_in_container = Path('/workspace').exists()
        default_news_path = None
        if is_in_container:
            # 容器环境
            self.output_dir = Path("/workspace/qlib_benchmark/benchmarks/train_temp")
            default_news_path = '/workspace/news_data/csi_300'
            logger.info(f"[Path] 检测到容器环境，使用路径: {default_news_path}")
        else:
            # 主机环境
            self.output_dir = Path("Qlib_MCP/workspace/qlib_benchmark/benchmarks/train_temp")
            default_news_path = 'Qlib_MCP/workspace/news_data/csi_300'
            logger.info(f"[Path] 检测到主机环境，使用路径: {default_news_path}")
        
        # 新闻数据配置
        self.news_data_path = default_news_path 
        
        # 新闻数据配置
        # 新闻数据配置 - 支持目录路径（按月份加载）或单个文件路径
        self.news_batch_size = news_batch_size
        # 按月份缓存新闻数据: {month: {date: {stock_code: [news_list]}}}
        # month格式: "2025-01", "2025-02" 等
        self._news_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # 宏观新闻路径（与个股新闻在同一目录）
        if self.news_data_path:
            news_path = Path(self.news_data_path)
            if news_path.exists():
                if news_path.is_dir():
                    # 目录模式：Macro_News.json 在同一目录
                    self.macro_news_path = news_path / "Macro_News.json"
                    # 午盘播报路径：Midday Market Summary.json 在同一目录
                    self.midday_summary_path = news_path / "Midday Market Summary.json"
                elif news_path.is_file():
                    # 文件模式：Macro_News.json 在同一目录
                    self.macro_news_path = news_path.parent / "Macro_News.json"
                    # 午盘播报路径：Midday Market Summary.json 在同一目录
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
        
        # 短期记忆系统：最多保留3个交易日
        # 格式: {date: {"macro_news_summary": str, "stock_reasons": {stock_code: reason}}}
        self._short_term_memory: Dict[str, Dict[str, Any]] = {}
        self._max_memory_days = 3
        
        # 压缩后的宏观新闻缓存: {date: compressed_text}
        self._compressed_macro_news_cache: Dict[str, str] = {}
        
        # 记录新闻数据路径配置
        logger.info(f"[News] 新闻数据路径配置: {self.news_data_path}")
        news_path = Path(self.news_data_path) if self.news_data_path else None
        if news_path and news_path.exists():
            if news_path.is_dir():
                logger.info(f"[News] 检测到目录模式，路径存在: {news_path.absolute()}")
            elif news_path.is_file():
                logger.info(f"[News] 检测到文件模式，路径存在: {news_path.absolute()}")
        else:
            logger.warning(f"[News] 新闻数据路径不存在: {news_path.absolute() if news_path else 'None'}")
        
        if self.macro_news_path and self.macro_news_path.exists():
            logger.info(f"[News] 宏观新闻路径: {self.macro_news_path.absolute()}")
        else:
            logger.warning(f"[News] 宏观新闻文件不存在: {self.macro_news_path}")
        
        if self.midday_summary_path and self.midday_summary_path.exists():
            logger.info(f"[News] 午盘播报路径: {self.midday_summary_path.absolute()}")
        else:
            logger.warning(f"[News] 午盘播报文件不存在: {self.midday_summary_path}")
    
    def _init_mcp_client(self, mcp_server_path: Optional[str]) -> None:
        """初始化 MCP 客户端"""
        if not MCP_AVAILABLE:
            return
        
        if mcp_server_path is None:
            current_dir = Path(__file__).parent.parent
            mcp_server_path = current_dir / "Qlib_MCP" / "mcp_server_inline.py"
        else:
            mcp_server_path = Path(mcp_server_path)
        
        if not mcp_server_path.exists():
            print(f"警告: MCP 服务器脚本不存在: {mcp_server_path}")
            return
        
        try:
            self.mcp_client = SyncMCPClient(str(mcp_server_path))
            print(f"[StrategyGeneration] MCP 客户端初始化成功")
        except Exception as e:
            print(f"警告: MCP 客户端初始化失败: {e}")
    
    # =========================================================================
    # 新闻数据加载与获取
    # =========================================================================
    
    def _get_month_from_date(self, date: str) -> str:
        """
        从日期字符串中提取月份
        
        Args:
            date: 日期字符串，格式如 "2025-01-02" 或 "2025-01-02 10:00:00"
            
        Returns:
            月份字符串，格式如 "2025-01"
        """
        try:
            # 提取日期部分（去掉时间部分）
            date_part = date[:10] if len(date) >= 10 else date
            # 提取年月部分
            year_month = date_part[:7]  # "2025-01"
            return year_month
        except Exception:
            logger.warning(f"[News] 无法从日期 {date} 提取月份")
            return ""
    
    def _get_news_file_path(self, month: str) -> Optional[Path]:
        """
        根据月份获取新闻文件路径
        
        Args:
            month: 月份字符串，格式如 "2025-01"
            
        Returns:
            新闻文件路径，如果不存在则返回None
        """
        if not self.news_data_path:
            logger.warning(f"[News] news_data_path 未配置")
            return None
        
        logger.debug(f"[News] 开始获取文件路径，month={month}, news_data_path={self.news_data_path}")
        news_path = Path(self.news_data_path)
        
        # 检查路径是否存在
        if not news_path.exists():
            logger.warning(f"[News] 新闻数据路径不存在: {news_path} (绝对路径: {news_path.absolute()})")
            # 尝试作为目录处理（可能是相对路径问题）
            try:
                month_num = month.split('-')[1]
                year = month.split('-')[0]
                filename = f"eastmoney_news_{year}_processed_{year}_{month_num}.json"
                file_path = news_path / filename
                logger.debug(f"[News] 尝试构建文件路径: {file_path} (绝对路径: {file_path.absolute()})")
                if file_path.exists():
                    logger.info(f"[News] 找到新闻文件: {file_path}")
                    return file_path
                else:
                    logger.warning(f"[News] 新闻文件不存在: {file_path}")
                    return None
            except (IndexError, AttributeError) as e:
                logger.error(f"[News] 月份格式错误: {month}, 错误: {e}")
                return None
        
        # 判断是目录还是文件
        if news_path.is_dir():
            # 目录模式：按月份加载
            # 文件名格式: eastmoney_news_2025_processed_2025_MM.json
            logger.debug(f"[News] 检测到目录模式: {news_path}")
            try:
                month_num = month.split('-')[1]  # 提取月份数字，如 "01"
                year = month.split('-')[0]  # 提取年份，如 "2025"
                filename = f"eastmoney_news_{year}_processed_{year}_{month_num}.json"
                file_path = news_path / filename
                logger.debug(f"[News] 构建文件路径: {file_path} (绝对路径: {file_path.absolute()})")
                if file_path.exists():
                    logger.info(f"[News] 找到新闻文件: {file_path}")
                    return file_path
                else:
                    logger.warning(f"[News] 新闻文件不存在: {file_path}")
                    # 列出目录中的文件，帮助调试
                    try:
                        files = list(news_path.glob("*.json"))
                        logger.debug(f"[News] 目录中的JSON文件: {[f.name for f in files[:10]]}")
                    except Exception as e:
                        logger.debug(f"[News] 无法列出目录文件: {e}")
                    return None
            except (IndexError, AttributeError) as e:
                logger.error(f"[News] 月份格式错误: {month}, 错误: {e}")
                return None
        elif news_path.is_file():
            # 文件模式：兼容旧版本（单个文件）
            # 在文件模式下，所有月份都使用同一个文件
            logger.debug(f"[News] 检测到文件模式: {news_path}")
            return news_path
        else:
            logger.warning(f"[News] 路径既不是目录也不是文件: {news_path}")
            return None
    
    def _load_news_data_by_month(self, month: str) -> Dict[str, Any]:
        """
        按月份加载新闻数据（带缓存）
        
        Args:
            month: 月份字符串，格式如 "2025-01"
        
        新闻数据格式: {date: {stock_code: [news_list]}}
        
        Returns:
            该月份的新闻数据字典
        """
        # 检查缓存
        if month in self._news_data_cache:
            logger.debug(f"[News] 从缓存加载 {month} 月份数据")
            return self._news_data_cache[month]
        
        logger.debug(f"[News] 开始加载 {month} 月份新闻数据")
        
        # 获取文件路径
        file_path = self._get_news_file_path(month)
        if not file_path:
            logger.warning(f"[News] 新闻数据文件不存在: {month}, news_data_path={self.news_data_path}")
            self._news_data_cache[month] = {}
            return {}
        
        logger.info(f"[News] 正在加载新闻文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            logger.debug(f"[News] 文件加载成功，包含 {len(all_data)} 个日期键")
            
            # 判断是目录模式还是文件模式
            news_path = Path(self.news_data_path)
            if news_path.exists() and news_path.is_dir():
                # 目录模式：直接使用加载的数据（已经是该月份的数据）
                month_data = all_data
                logger.debug(f"[News] 目录模式：使用全部数据")
            else:
                # 文件模式：从所有数据中筛选出该月份的数据
                month_data = {}
                for date, stocks_news in all_data.items():
                    if date.startswith(month):
                        month_data[date] = stocks_news
                logger.debug(f"[News] 文件模式：从 {len(all_data)} 个日期中筛选出 {len(month_data)} 个日期")
                # 缓存整个文件的数据，避免重复加载
                if "all_data" not in self._news_data_cache:
                    self._news_data_cache["all_data"] = all_data
            
            self._news_data_cache[month] = month_data
            logger.info(f"[News] 成功加载 {month} 月份新闻数据，包含 {len(month_data)} 个日期")
            # 列出前几个日期，帮助调试
            if month_data:
                sample_dates = list(month_data.keys())[:5]
                logger.debug(f"[News] 示例日期: {sample_dates}")
            return month_data
        except Exception as e:
            logger.error(f"[News] 加载 {month} 月份新闻数据失败: {e}", exc_info=True)
            self._news_data_cache[month] = {}
            return {}
    
    def _load_news_data(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        加载新闻数据（兼容旧接口，支持按日期自动选择月份）
        
        Args:
            date: 可选，日期字符串。如果提供，则加载对应月份的新闻数据
        
        新闻数据格式: {date: {stock_code: [news_list]}}
        
        Returns:
            新闻数据字典（如果提供了date，返回该月份的；否则返回所有已缓存的数据）
        """
        if date:
            # 按日期加载对应月份
            month = self._get_month_from_date(date)
            if month:
                return self._load_news_data_by_month(month)
            return {}
        
        # 兼容旧代码：返回所有已缓存的数据（合并）
        if not self._news_data_cache:
            return {}
        
        # 合并所有月份的新闻数据
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
        获取特定日期特定股票的新闻
        
        Args:
            stock_code: 股票代码（如 "SH600015"）
            date: 日期（如 "2025-01-02" 或 "2025-01-02 00:00:00"）
            
        Returns:
            新闻列表 [{"publish_date": ..., "news_title": ..., "content": ...}, ...]
        """
        # 格式化日期为 YYYY-MM-DD 格式
        formatted_date = date[:10] if len(date) >= 10 else date
        logger.debug(f"[News] 获取新闻: stock_code={stock_code}, date={date} -> formatted_date={formatted_date}")
        
        # 根据日期加载对应月份的新闻数据
        news_data = self._load_news_data(date=formatted_date)
        
        if not news_data:
            logger.debug(f"[News] 未找到 {formatted_date} 的新闻数据")
            return []
        
        logger.debug(f"[News] 新闻数据包含 {len(news_data)} 个日期")
        
        # 尝试直接匹配日期（使用格式化后的日期）
        date_news = news_data.get(formatted_date, {})
        
        # 如果没有精确匹配，尝试匹配日期前缀
        if not date_news:
            logger.debug(f"[News] 精确匹配失败，尝试前缀匹配")
            for d in news_data.keys():
                if d.startswith(formatted_date):
                    date_news = news_data[d]
                    logger.debug(f"[News] 使用匹配的日期键: {d} (查询: {formatted_date})")
                    break
        
        if not date_news:
            logger.debug(f"[News] 未找到 {formatted_date} 的日期数据，可用日期示例: {list(news_data.keys())[:5]}")
            return []
        
        # 获取该股票的新闻
        stock_news = date_news.get(stock_code, [])
        
        if stock_news:
            logger.debug(f"[News] 找到 {stock_code} 在 {formatted_date} 的 {len(stock_news)} 条新闻")
        else:
            logger.debug(f"[News] {stock_code} 在 {formatted_date} 没有新闻，该日期有 {len(date_news)} 只股票的数据")
        
        return stock_news
    
    def _get_stocks_news_batch(
        self,
        stock_codes: List[str],
        date: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取多只股票的新闻
        
        Args:
            stock_codes: 股票代码列表
            date: 日期
            
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
    # 宏观新闻加载与压缩
    # =========================================================================
    
    def _load_macro_news(self, date: str) -> List[Dict[str, Any]]:
        """
        加载当前交易日前最近一天的宏观新闻（不包含交易日当天）
        
        Args:
            date: 交易日期，格式如 "2025-01-02" 或 "2025-01-02 10:00:00"
            
        Returns:
            宏观新闻列表 [{"publish_date": ..., "news_title": ..., "content": ...}, ...]
        """
        if not self.macro_news_path or not self.macro_news_path.exists():
            logger.debug(f"[MacroNews] 宏观新闻文件不存在: {self.macro_news_path}")
            return []
        
        try:
            # 格式化日期为 YYYY-MM-DD
            formatted_date = date[:10] if len(date) >= 10 else date
            
            # 读取宏观新闻文件
            with open(self.macro_news_path, 'r', encoding='utf-8') as f:
                macro_data = json.load(f)
            
            # 查找最近一天的宏观新闻（向前查找，最多查找7天，跳过交易日当天）
            from datetime import datetime, timedelta
            current_date = datetime.strptime(formatted_date, "%Y-%m-%d")
            
            # 从 i=1 开始，跳过交易日当天（i=0）
            for i in range(1, 8):  # 从1到7，查找前1-7天的新闻
                check_date = current_date - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                
                if date_str in macro_data:
                    news_list = macro_data[date_str]
                    logger.info(f"[MacroNews] 找到 {date_str} 的宏观新闻（交易日 {formatted_date} 的前 {i} 天），共 {len(news_list)} 条")
                    return news_list
            
            logger.debug(f"[MacroNews] 未找到 {formatted_date} 前1-7天内的宏观新闻（已跳过交易日当天）")
            return []
            
        except Exception as e:
            logger.error(f"[MacroNews] 加载宏观新闻失败: {e}", exc_info=True)
            return []
    
    def _load_midday_summary(self, date: str) -> Optional[Dict[str, Any]]:
        """
        加载当前交易日的午盘播报
        
        Args:
            date: 交易日期，格式如 "2025-01-02" 或 "2025-01-02 10:00:00"
            
        Returns:
            午盘播报字典 {"datetime": ..., "content": ..., "title": ...} 或 None
        """
        if not self.midday_summary_path or not self.midday_summary_path.exists():
            logger.debug(f"[MiddaySummary] 午盘播报文件不存在: {self.midday_summary_path}")
            return None
        
        try:
            # 格式化日期为 YYYY-MM-DD
            formatted_date = date[:10] if len(date) >= 10 else date
            
            # 读取午盘播报文件
            with open(self.midday_summary_path, 'r', encoding='utf-8') as f:
                midday_data = json.load(f)
            
            # 查找当日的午盘播报
            # 午盘播报的datetime格式为 "2025-01-02 11:30:58"，需要匹配日期部分
            for item in midday_data:
                if isinstance(item, dict):
                    item_datetime = item.get("datetime", "")
                    if item_datetime.startswith(formatted_date):
                        logger.info(f"[MiddaySummary] 找到 {formatted_date} 的午盘播报")
                        return item
            
            logger.debug(f"[MiddaySummary] 未找到 {formatted_date} 的午盘播报")
            return None
            
        except Exception as e:
            logger.error(f"[MiddaySummary] 加载午盘播报失败: {e}", exc_info=True)
            return None
    
    def _compress_macro_news(self, macro_news: List[Dict[str, Any]], date: str) -> str:
        """
        使用 LLM 压缩宏观新闻
        
        Args:
            macro_news: 宏观新闻列表
            date: 交易日期
            
        Returns:
            压缩后的宏观新闻摘要文本
        """
        if not macro_news:
            return ""
        
        # 检查缓存
        if date in self._compressed_macro_news_cache:
            logger.debug(f"[MacroNews] 使用缓存的压缩结果: {date}")
            return self._compressed_macro_news_cache[date]
        
        if not self.llm:
            # 无 LLM 时，简单拼接标题
            summary = "\n".join([f"- {news.get('news_title', '')}" for news in macro_news[:5]])
            logger.debug(f"[MacroNews] 无LLM，使用简单摘要")
            return summary
        
        try:
            # 构建压缩提示词
            news_text = ""
            for i, news in enumerate(macro_news[:10]):  # 最多处理10条
                title = news.get("news_title", "")
                content = news.get("content", "")[:500]  # 每条新闻最多500字
                news_text += f"\n[{i+1}] {title}\n{content}...\n"
            
            prompt = f"""你是一个专业的金融分析师。请将以下宏观新闻压缩成简洁的摘要，重点关注对股市和投资决策有影响的信息。

## 交易日期
{date}

## 宏观新闻
{news_text}

## 任务
请将以上宏观新闻压缩成200字以内的摘要，突出：
1. 货币政策、财政政策变化
2. 经济数据、行业数据
3. 重大政策、法规变化
4. 对股市整体影响的关键信息

只输出压缩后的摘要，不要其他内容。
"""
            
            logger.info(f"[MacroNews] 开始压缩 {len(macro_news)} 条宏观新闻")
            response = self.llm.call(prompt=prompt, stream=False)
            
            # 提取文本（如果不是JSON格式）
            if isinstance(response, str):
                compressed = response.strip()
            elif isinstance(response, dict):
                compressed = response.get("summary", response.get("content", ""))
            else:
                compressed = str(response).strip()
            
            # 缓存结果
            self._compressed_macro_news_cache[date] = compressed
            logger.info(f"[MacroNews] 压缩完成，摘要长度: {len(compressed)} 字")
            
            return compressed
            
        except Exception as e:
            logger.error(f"[MacroNews] 压缩宏观新闻失败: {e}")
            # 失败时返回简单摘要
            summary = "\n".join([f"- {news.get('news_title', '')}" for news in macro_news[:5]])
            return summary
    
    # =========================================================================
    # 短期记忆系统
    # =========================================================================
    
    def _get_compressed_macro_news(self, date: str) -> str:
        """
        获取压缩后的宏观新闻（带缓存）
        
        Args:
            date: 交易日期
            
        Returns:
            压缩后的宏观新闻摘要
        """
        # 先检查缓存
        if date in self._compressed_macro_news_cache:
            return self._compressed_macro_news_cache[date]
        
        # 加载并压缩
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
        保存短期记忆
        
        Args:
            date: 交易日期
            macro_news_summary: 压缩后的宏观新闻摘要
            stock_reasons: 股票决策理由 {stock_code: reason}
        """
        # 格式化日期
        formatted_date = date[:10] if len(date) >= 10 else date
        
        # 压缩股票理由（如果太多）
        if len(stock_reasons) > 50:
            # 只保留前50只股票的理由
            stock_reasons = dict(list(stock_reasons.items())[:50])
        
        # 保存记忆
        self._short_term_memory[formatted_date] = {
            "macro_news_summary": macro_news_summary,
            "stock_reasons": stock_reasons.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 清理旧记忆（只保留最近3个交易日）
        self._cleanup_short_term_memory()
        
        logger.info(f"[Memory] 保存短期记忆: {formatted_date}, 宏观新闻: {len(macro_news_summary)}字, 股票理由: {len(stock_reasons)}条")
    
    def _cleanup_short_term_memory(self) -> None:
        """清理短期记忆，只保留最近3个交易日"""
        if len(self._short_term_memory) <= self._max_memory_days:
            return
        
        # 按日期排序，保留最新的
        sorted_dates = sorted(self._short_term_memory.keys(), reverse=True)
        dates_to_keep = sorted_dates[:self._max_memory_days]
        
        # 删除旧记忆
        dates_to_remove = [d for d in sorted_dates if d not in dates_to_keep]
        for date in dates_to_remove:
            del self._short_term_memory[date]
            logger.debug(f"[Memory] 删除旧记忆: {date}")
    
    def _load_short_term_memory(self) -> str:
        """
        加载短期记忆，返回格式化的记忆文本
        
        Returns:
            格式化的短期记忆文本
        """
        if not self._short_term_memory:
            return ""
        
        memory_parts = []
        # 按日期倒序（最新的在前）
        sorted_dates = sorted(self._short_term_memory.keys(), reverse=True)
        
        for date in sorted_dates:
            memory = self._short_term_memory[date]
            macro_summary = memory.get("macro_news_summary", "")
            stock_reasons = memory.get("stock_reasons", {})
            
            memory_text = f"## {date}\n"
            if macro_summary:
                memory_text += f"宏观新闻摘要: {macro_summary}\n"
            if stock_reasons:
                # 只显示前10只股票的理由
                stock_items = list(stock_reasons.items())[:10]
                memory_text += "重要股票决策理由:\n"
                for stock_code, reason in stock_items:
                    memory_text += f"  - {stock_code}: {reason[:100]}...\n"  # 每条理由最多100字
            
            memory_parts.append(memory_text)
        
        if memory_parts:
            return "\n---\n".join(memory_parts)
        return ""
    
    # =========================================================================
    # 新闻分析方法（批次处理）
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
        批次分析股票新闻，返回每只股票的分析结果（包含宏观新闻、午盘播报和短期记忆）
        
        Args:
            batch_stocks: 本批次的股票交易信号列表
            stocks_news: 股票新闻字典 {stock_code: [news_list]}
            action_type: 交易动作类型 ("sell", "buy", "hold")
            date: 交易日期
            macro_news_summary: 压缩后的宏观新闻摘要
            short_term_memory: 短期记忆文本
            midday_summary: 当日午盘播报字典
            
        Returns:
            分析结果字典 {stock_code: {"recommendation": ..., "confidence": ..., "reason": ...}}
        """
        if not self.llm or not batch_stocks:
            return {}
        
        # 构建批次分析提示词（包含宏观新闻、午盘播报和短期记忆）
        prompt = self._build_news_analysis_prompt(
            batch_stocks, stocks_news, action_type, date,
            macro_news_summary=macro_news_summary,
            short_term_memory=short_term_memory,
            midday_summary=midday_summary
        )
        
        # 重试逻辑：最多尝试2次（初始1次 + 重试1次）
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[NewsAnalysis] 分析批次: {action_type}, 股票数: {len(batch_stocks)}, 尝试 {attempt}/{max_attempts}")
                response = self.llm.call(prompt=prompt, stream=False)
                
                # 解析 LLM 响应
                result = self.llm.parse_json_response(response)
                
                if result and "stocks_analysis" in result:
                    return result["stocks_analysis"]
                elif result:
                    return result
                
                return {}
                
            except Exception as e:
                logger.error(f"[NewsAnalysis] 批次分析失败 (尝试 {attempt}/{max_attempts}): {e}")
                
                # 如果不是最后一次尝试，等待2秒后重试
                if attempt < max_attempts:
                    logger.info(f"[NewsAnalysis] 等待2秒后重试...")
                    time.sleep(2)
                else:
                    # 最后一次尝试也失败，返回空字典
                    logger.error(f"[NewsAnalysis] 批次分析最终失败，已重试 {max_attempts} 次")
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
        构建新闻分析提示词（包含宏观新闻、午盘播报和短期记忆）
        
        Args:
            batch_stocks: 股票交易信号列表
            stocks_news: 股票新闻字典
            action_type: 交易动作类型
            date: 交易日期
            macro_news_summary: 压缩后的宏观新闻摘要
            short_term_memory: 短期记忆文本
            midday_summary: 当日午盘播报字典
            
        Returns:
            提示词字符串
        """
        action_desc = {
            "sell": "卖出候选",
            "buy": "买入候选",
            "hold": "持有股票"
        }.get(action_type, "候选股票")
        
        # 构建股票信息
        stocks_info = []
        for signal in batch_stocks:
            stock_code = signal.stock_code
            news_list = stocks_news.get(stock_code, [])
            
            # 格式化新闻（限制每只股票最多5条新闻，每条新闻最多200字）
            news_text = ""
            if news_list:
                news_items = []
                for i, news in enumerate(news_list[:5]):
                    title = news.get("news_title", "")
                    content = news.get("content", "")[:200]
                    source = news.get("news_source", "")
                    news_items.append(f"    [{i+1}] {title} ({source})\n        {content}...")
                news_text = "\n".join(news_items)
            else:
                news_text = "    无相关新闻"
            
            stock_info = f"""
  股票代码: {stock_code}
  模型评分: {signal.score:.4f}
  当前权重: {signal.current_weight:.2%}
  目标权重: {signal.target_weight:.2%}
  原因: {signal.reason}
  相关新闻:
{news_text}
"""
            stocks_info.append(stock_info)
        
        stocks_info_text = "\n---".join(stocks_info)
        
        # 构建宏观新闻部分
        macro_news_section = ""
        if macro_news_summary:
            macro_news_section = f"""
## 宏观新闻摘要（当前交易日前最近一天）
{macro_news_summary}

注意：请结合宏观新闻背景分析个股，宏观政策变化可能影响整体市场情绪和行业走势。如果宏观产生较大风险，应慎重买入。
"""
        
        # 构建午盘播报部分
        midday_section = ""
        if midday_summary:
            midday_content = midday_summary.get("content", "")
            midday_title = midday_summary.get("title", "")
            midday_datetime = midday_summary.get("datetime", "")
            midday_section = f"""
## 当日午盘播报（{midday_datetime}）
标题: {midday_title}
内容: {midday_content}

注意：午盘播报反映了当日市场的整体表现和板块热点，请结合午盘播报中的市场情绪、板块涨跌情况来分析个股。如果午盘播报显示市场整体下跌或相关板块表现不佳，应慎重买入。
"""
        
        # 构建短期记忆部分
        memory_section = ""
        if short_term_memory:
            memory_section = f"""
## 短期记忆（最近3个交易日的决策历史）
{short_term_memory}


"""
        
        prompt = f"""你是一个专业的量化投资分析师。请根据以下股票信息、新闻、宏观新闻、午盘播报和历史记忆，分析每只股票的投资建议。

## 交易日期
{date}
当前时间为交易日临近收盘时间。
## 宏观新闻、午盘播报和短期记忆
{macro_news_section}{midday_section}{memory_section}
## 股票类型
{action_desc}

## 股票信息与新闻
{stocks_info_text}

## 分析任务
请针对每只股票，结合以下信息综合分析：
1. 个股新闻对该股票的影响（正面、负面还是中性）
2. 宏观新闻对整体市场和行业的影响
3. 午盘播报反映的当日市场情绪和板块表现，特别是相关板块的涨跌情况
4. 是否支持当前的交易建议（{action_type}）
5. 给出置信度评估（0-1之间）
6.宏观新闻为昨日的新闻，午盘播报为当日的新闻，买入卖出股票的时间为当日临近收盘时间。
7.结合宏观新闻、午盘播报和历史记忆以及个股新闻，给出对股票的分析结果。

**重要：recommendation 字段的返回值规则**
- 对于**买入候选**：主要参考模型评分，辅助个股新闻参考，如有利空，建议不买入，返回 "hold"（暂不买入）
- 对于**卖出候选**：主要参考模型评分，辅助个股新闻参考，如有利好，建议不卖出，返回 "hold"（继续持有）
- 对于**持有股票**：权重不影响买入、卖出的判断，如果个股新闻有明显利空，建议卖出，返回 "sell"；如果建议继续持有，返回 "hold"

请以 JSON 格式输出分析结果：
```json
{{
    "stocks_analysis": {{
        "股票代码1": {{
            "news_sentiment": "positive/negative/neutral",
            "recommendation": "buy/sell/hold/",
            "confidence": 0.8,
            "reason": "综合上述信息，给出分析结果"
        }},
        "股票代码2": {{
            ...
        }}
    }},
    "batch_summary": "本批次股票的整体分析摘要"
}}
```

注意：
1. 宏观政策变化（如货币政策、财政政策）可能影响整体市场
2. 午盘播报中的板块涨跌情况对相关股票有重要参考价值，如果相关板块领跌，应慎重买入
3. 只输出 JSON，不要其他内容
"""
        return prompt
    
    def _analyze_all_candidates_with_news(
        self,
        candidates: Dict[str, List[TradeSignal]],
        date: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        对所有候选股票进行新闻分析（分批处理，包含宏观新闻和短期记忆）
        
        Args:
            candidates: 候选交易信号 {"sell": [...], "buy": [...], "hold": [...]}
            date: 交易日期
            
        Returns:
            所有股票的分析结果 {stock_code: {"recommendation": ..., "confidence": ..., "reason": ...}}
        """
        all_analysis = {}
        
        # 收集所有需要分析的股票
        all_stocks = []
        stock_action_map = {}  # 记录每只股票的原始动作类型
        
        # 修改：增加 adjust 类型，确保调仓候选也能进行新闻风险筛查
        for action_type in ["sell", "buy", "hold", "adjust"]:
            signals = candidates.get(action_type, [])
            for signal in signals:
                all_stocks.append(signal)
                # adjust 类型在新闻分析时归类为 hold 处理
                stock_action_map[signal.stock_code] = "hold" if action_type == "adjust" else action_type
        
        if not all_stocks:
            return all_analysis
        
        # 获取所有股票的新闻
        stock_codes = [s.stock_code for s in all_stocks]
        all_news = self._get_stocks_news_batch(stock_codes, date)
        
        logger.info(f"[NewsAnalysis] 总共 {len(all_stocks)} 只股票，{len(all_news)} 只有新闻")
        
        # =====================================================================
        # 加载宏观新闻、午盘播报和短期记忆（所有批次共享）
        # =====================================================================
        macro_news_summary = self._get_compressed_macro_news(date)
        midday_summary = self._load_midday_summary(date)
        short_term_memory = self._load_short_term_memory()
        
        if macro_news_summary:
            logger.info(f"[NewsAnalysis] 已加载宏观新闻摘要，长度: {len(macro_news_summary)} 字")
        if midday_summary:
            logger.info(f"[NewsAnalysis] 已加载当日午盘播报: {midday_summary.get('title', '')}")
        if short_term_memory:
            logger.info(f"[NewsAnalysis] 已加载短期记忆，包含 {len(self._short_term_memory)} 个交易日")
        
        # 分批处理
        # batch_size = self.news_batch_size
        # for i in range(0, len(all_stocks), batch_size):
        #     batch = all_stocks[i:i + batch_size]
        #     batch_codes = [s.stock_code for s in batch]
            
        #     # 获取本批次股票的新闻
        #     batch_news = {code: all_news.get(code, []) for code in batch_codes}
            
        #     # 确定本批次的主要动作类型（用于提示词）
        #     action_counts = {}
        #     for signal in batch:
        #         action = stock_action_map[signal.stock_code]
        #         action_counts[action] = action_counts.get(action, 0) + 1
        #     main_action = max(action_counts.keys(), key=lambda k: action_counts[k])
            
        #     # 分析本批次（传入宏观新闻、午盘播报和短期记忆）
        #     logger.info(f"[NewsAnalysis] 处理批次 {i//batch_size + 1}, 股票: {batch_codes}")
        #     batch_analysis = self._analyze_news_batch(
        #         batch, batch_news, main_action, date,
        #         macro_news_summary=macro_news_summary,
        #         short_term_memory=short_term_memory,
        #         midday_summary=midday_summary
        #     )
            
        #     # 合并结果
        #     all_analysis.update(batch_analysis)
        # 分批处理 - 按action分组，确保同一batch内action一致
        batch_size = self.news_batch_size

        # 按action分组股票
        stocks_by_action = {"sell": [], "buy": [], "hold": []}
        for signal in all_stocks:
            action = stock_action_map[signal.stock_code]
            stocks_by_action[action].append(signal)

        # 对每个action分别分批处理
        for action_type in ["sell", "buy", "hold"]:
            action_stocks = stocks_by_action[action_type]
            if not action_stocks:
                continue
            
            # 对该action的股票进行分批
            for i in range(0, len(action_stocks), batch_size):
                batch = action_stocks[i:i + batch_size]
                batch_codes = [s.stock_code for s in batch]
                
                # 获取本批次股票的新闻
                batch_news = {code: all_news.get(code, []) for code in batch_codes}
                
                # 本批次的主要动作类型就是当前的action_type
                main_action = action_type
                
                # 分析本批次（传入宏观新闻、午盘播报和短期记忆）
                logger.info(f"[NewsAnalysis] 处理批次 {action_type} - {i//batch_size + 1}, 股票: {batch_codes}")
                batch_analysis = self._analyze_news_batch(
                    batch, batch_news, main_action, date,
                    macro_news_summary=macro_news_summary,
                    short_term_memory=short_term_memory,
                    midday_summary=midday_summary
                )
                
                # 合并结果
                all_analysis.update(batch_analysis)
                
        return all_analysis
    
    def _merge_news_analysis_to_decision(
        self,
        candidates: Dict[str, List[TradeSignal]],
        news_analysis: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        将新闻分析结果合并，生成最终的调整建议
        
        Args:
            candidates: 原始候选交易信号
            news_analysis: 新闻分析结果
            
        Returns:
            调整建议字典，格式与 _apply_llm_suggestions 兼容
        """
        remove_from_sell = []
        remove_from_buy = []
        move_to_sell = []  # 从 hold 移到 sell 的股票
        priority_adjustments = {}
        reasoning_parts = []
        
        # 处理卖出候选
        for signal in candidates.get("sell", []):
            stock_code = signal.stock_code
            analysis = news_analysis.get(stock_code, {})
            
            if analysis:
                recommendation = analysis.get("recommendation", "").lower()
                confidence = analysis.get("confidence", 0.5)
                reason = analysis.get("reason", "")
                sentiment = analysis.get("news_sentiment", "neutral")
                
                # 如果新闻正面且建议不卖出，移除卖出候选
                if recommendation in ["hold", "buy"]:
                    remove_from_sell.append(stock_code)
                    reasoning_parts.append(f"{stock_code}: 正面新闻，建议保留 ({reason})")
                elif confidence < 0.3:
                    # 低置信度，不调整
                    pass
                else:
                    # 调整优先级：负面新闻增加卖出优先级
                    if sentiment == "negative":
                        priority_adjustments[stock_code] = signal.priority + 20
        
        # 处理买入候选
        for signal in candidates.get("buy", []):
            stock_code = signal.stock_code
            analysis = news_analysis.get(stock_code, {})
            
            if analysis:
                recommendation = analysis.get("recommendation", "").lower()
                confidence = analysis.get("confidence", 0.5)
                reason = analysis.get("reason", "")
                sentiment = analysis.get("news_sentiment", "neutral")
                
                # 如果新闻负面且建议不买入，移除买入候选
                #if sentiment == "negative" and recommendation in ["hold", "sell"]:
                if recommendation in ["hold", "sell"]:
                    remove_from_buy.append(stock_code)
                    reasoning_parts.append(f"{stock_code}: 负面新闻，建议不买入 ({reason})")
                elif confidence < 0.3:
                    # 低置信度，不调整
                    pass
                else:
                    # 调整优先级：正面新闻增加买入优先级
                    if sentiment == "positive":
                        priority_adjustments[stock_code] = signal.priority + 20
        
        # 处理持有候选和调仓候选（已持有的股票）
        # 修改：同时处理 hold 和 adjust 候选，确保所有持仓股票都能进行新闻风险筛查
        for action_type in ["hold", "adjust"]:
            for signal in candidates.get(action_type, []):
                stock_code = signal.stock_code
                analysis = news_analysis.get(stock_code, {})
                
                if analysis:
                    recommendation = analysis.get("recommendation", "").lower()
                    confidence = analysis.get("confidence", 0.5)
                    reason = analysis.get("reason", "")
                    sentiment = analysis.get("news_sentiment", "neutral")
                    
                    # 如果新闻强烈负面且建议卖出，考虑将其移到卖出候选
                    if recommendation == "sell" :
                        move_to_sell.append(stock_code)
                        reasoning_parts.append(f"{stock_code}: 持有股票出现负面新闻，建议卖出 ({reason})")
                    elif recommendation in ["buy", "hold"] :
                        # 正面新闻，可以增加优先级（虽然已经是 hold，但可以影响后续决策）
                        priority_adjustments[stock_code] = signal.priority + 10
                        reasoning_parts.append(f"{stock_code}: 持有股票出现正面新闻，增强持有信心 ({reason})")
        
        # 构建最终建议
        final_reasoning = "基于新闻分析的调整: " + "; ".join(reasoning_parts) if reasoning_parts else "新闻分析未发现显著调整建议"
        
        return {
            "remove_from_sell": remove_from_sell,
            "remove_from_buy": remove_from_buy,
            "move_to_sell": move_to_sell,
            "priority_adjustments": priority_adjustments,
            "reasoning": final_reasoning,
            "news_analysis_summary": news_analysis
        }
    
    # =========================================================================
    # 核心决策方法
    # =========================================================================
    
    def generate_trade_decision(
        self,
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]],
        market_data: Optional[Dict[str, Any]] = None,
        date: Optional[str] = None
    ) -> TradeDecision:
        """
        生成交易决策 - 核心入口方法
        
        Args:
            current_holdings: 当前持仓 {stock_code: weight}
            top_k_recommendations: 模型推荐的 Top K 股票 [(stock_code, score), ...]
            market_data: 市场数据 (可选)
            date: 当前日期 (可选)
            
        Returns:
            TradeDecision 交易决策对象
        """
        self.decision_logs.append(f"[{date}] 开始生成交易决策")
        
        # Step 1: 集合对比分析
        comparison_result = self._compare_holdings_and_recommendations(
            current_holdings, top_k_recommendations
        )
        
        # Step 2: 生成候选交易列表
        candidates = self._generate_trade_candidates(
            comparison_result,
            current_holdings,
            top_k_recommendations,
            market_data
        )
        
        # Step 3: 智能决策（可选使用 LLM）
        # 注意：与 qlib TopkDropoutStrategy 保持一致，不在决策阶段应用约束
        # 约束检查将在订单生成时进行（在 _convert_decision_to_orders 中）
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
        
        # Step 5: 记录交易历史
        self._record_trade_history(final_decision, date)
        
        return final_decision
    
    # =========================================================================
    # Step 1: 集合对比分析
    # =========================================================================
    
    def _compare_holdings_and_recommendations(
        self,
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]]
    ) -> ComparisonResult:
        """
        对比当前持仓和推荐列表，进行集合分析
        
        参考 Qlib TopkDropoutStrategy 的实现逻辑：
        1. 获取当前持仓股票集合 A
        2. 获取 Top K 推荐股票集合 B  
        3. 计算: sell_candidates = A - B, buy_candidates = B - A, hold = A ∩ B
        4. 按模型评分对持仓和推荐进行排序
        5. 根据 n_drop 策略确定实际要卖出和买入的股票
        
        Args:
            current_holdings: 当前持仓 {stock_code: weight}
            top_k_recommendations: 模型推荐的 Top K 股票 [(stock_code, score), ...]
            
        Returns:
            ComparisonResult 包含详细的对比分析结果
        """
        # Step 1: 构建集合
        holding_set = set(current_holdings.keys())  # 集合 A: 当前持仓
        
        # 构建推荐集合和评分映射
        score_map: Dict[str, float] = {}
        recommendation_set: Set[str] = set()
        
        for stock_code, score in top_k_recommendations:
            recommendation_set.add(stock_code)
            score_map[stock_code] = score
        
        # 集合 B: Top K 推荐股票
        topk_set = recommendation_set
        
        self.decision_logs.append(
            f"[Compare] 当前持仓: {len(holding_set)} 只, Top K 推荐: {len(topk_set)} 只"
        )
        
        # Step 2: 集合运算
        # sell_candidates: A - B (持仓中有但推荐中没有的股票)
        sell_candidates = holding_set - topk_set
        
        # buy_candidates: B - A (推荐中有但持仓中没有的股票)
        buy_candidates = topk_set - holding_set
        
        # hold_candidates: A ∩ B (两者交集，继续持有)
        hold_candidates = holding_set & topk_set
        
        self.decision_logs.append(
            f"[Compare] 卖出候选: {len(sell_candidates)}, "
            f"买入候选: {len(buy_candidates)}, "
            f"保留: {len(hold_candidates)}"
        )
        
        # Step 3: 按评分排序
        # 对当前持仓按模型评分排序（持仓可能不在推荐列表中，给默认低分）
        ranked_holdings = self._rank_stocks_by_score(
            list(holding_set), 
            score_map, 
            default_score=float('-inf')
        )
        
        # 对推荐股票按评分排序
        ranked_recommendations = self._rank_stocks_by_score(
            list(topk_set),
            score_map,
            default_score=0.0
        )
        
        # Step 4: 确定实际卖出和买入列表（参考 TopkDropoutStrategy 的 n_drop 逻辑）
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
        按模型评分对股票列表进行排序
        
        Args:
            stock_list: 股票列表
            score_map: 评分映射 {stock_code: score}
            default_score: 默认评分（当股票不在 score_map 中时使用）
            
        Returns:
            按评分降序排列的列表 [(stock_code, score), ...]
        """
        scored_list = [
            (stock, score_map.get(stock, default_score))
            for stock in stock_list
        ]
        # 按评分降序排序
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
        根据 TopkDropoutStrategy 的 n_drop 策略确定实际交易列表
        
        核心逻辑（参考 TopkDropoutStrategy）：
        1. 将当前持仓和新推荐股票合并，按评分排序
        2. 从合并列表的尾部选择 n_drop 只股票作为卖出候选
        3. 只卖出当前持仓中属于这些低分股票的部分
        4. 买入数量 = 卖出数量 + (topk - 当前持仓数)
        
        这样可以避免 "卖高买低" 的情况
        
        Args:
            holding_set: 当前持仓集合
            topk_set: Top K 推荐集合
            sell_candidates: 简单集合差得到的卖出候选
            buy_candidates: 简单集合差得到的买入候选
            ranked_holdings: 按评分排序的持仓
            ranked_recommendations: 按评分排序的推荐
            score_map: 评分映射
            
        Returns:
            (suggested_sell, suggested_buy) 实际建议的卖出和买入列表
        """
        topk = self.config.topk
        n_drop = self.config.n_drop
        
        # 获取不在当前持仓中的新推荐股票（按评分排序）
        # 这些是今天想要买入的候选
        new_recommendations = [
            (stock, score) for stock, score in ranked_recommendations
            if stock not in holding_set
        ]
        
        # 计算需要买入多少只股票以达到 topk
        # 最多买入 n_drop + (topk - 当前持仓数) 只
        max_buy_count = n_drop + topk - len(holding_set)
        today_buy_candidates = [stock for stock, _ in new_recommendations[:max_buy_count]]
        
        # 合并：当前持仓 + 新推荐股票，按评分排序
        combined_set = holding_set | set(today_buy_candidates)
        combined_ranked = self._rank_stocks_by_score(list(combined_set), score_map, float('-inf'))
        
        # 从合并列表的尾部选择 n_drop 只股票作为卖出候选
        # 这样可以确保我们卖出的是评分最低的股票，而不是简单地卖出不在推荐中的股票
        drop_candidates = [stock for stock, _ in combined_ranked[-n_drop:]] if len(combined_ranked) > n_drop else []
        
        # 实际卖出：当前持仓中属于 drop_candidates 的股票
        suggested_sell = [
            stock for stock in drop_candidates
            if stock in holding_set
        ]
        
        # 实际买入：根据卖出数量和持仓缺口确定
        # 买入数量 = 卖出数量 + max(0, topk - 当前持仓数)
        buy_count = len(suggested_sell) + max(0, topk - len(holding_set))
        suggested_buy = today_buy_candidates[:buy_count]
        
        self.decision_logs.append(
            f"[Dropout] 合并排序后, 建议卖出: {suggested_sell}, 建议买入: {suggested_buy[:5]}..."
        )
        
        return suggested_sell, suggested_buy
    
    # =========================================================================
    # Step 2: 生成候选交易列表
    # =========================================================================
    
    def _generate_trade_candidates(
        self,
        comparison_result: ComparisonResult,
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, List[TradeSignal]]:
        """
        根据对比结果生成候选交易信号
        
        Args:
            comparison_result: ComparisonResult 集合对比结果
            current_holdings: 当前持仓
            top_k_recommendations: Top K 推荐
            market_data: 市场数据
            
        Returns:
            候选交易信号字典 {"sell": [...], "buy": [...], "hold": [...], "adjust": [...]}
        """
        candidates: Dict[str, List[TradeSignal]] = {
            "sell": [],
            "buy": [],
            "hold": [],
            "adjust": []
        }
        
        score_map = comparison_result.score_map
        
        # 生成卖出候选信号
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
        
        # 生成买入候选信号
        for stock_code in comparison_result.suggested_buy:
            score = score_map.get(stock_code, 0.0)
            
            signal = self._evaluate_buy_candidate(
                stock_code=stock_code,
                model_score=score,
                market_data=market_data
            )
            candidates["buy"].append(signal)
        
        # 生成保留/调仓候选信号
        # 修改：对所有不在卖出候选中的持仓股票进行评估（而不仅仅是 hold_candidates 交集）
        # 这样才能确保所有持仓股票都能进行新闻风险筛查
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
            
            if signal.action == "adjust":
                candidates["adjust"].append(signal)
            else:
                candidates["hold"].append(signal)
        
        self.decision_logs.append(
            f"[Candidates] 生成候选: 卖出 {len(candidates['sell'])}, "
            f"买入 {len(candidates['buy'])}, "
            f"保留 {len(candidates['hold'])}, "
            f"调仓 {len(candidates['adjust'])}"
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
        评估单个卖出候选股票
        
        Args:
            stock_code: 股票代码
            current_weight: 当前权重
            score: 模型评分
            market_data: 市场数据
            
        Returns:
            TradeSignal 交易信号
        """
        # 构建卖出理由
        reasons = []
        reasons.append(f"模型评分较低({score:.4f})")
        reasons.append("不在 Top K 推荐列表中")
        
        # 检查涨跌停限制
        is_tradable = self._check_limit_price(stock_code, "sell", market_data)
        if not is_tradable:
            reasons.append("注意：可能跌停无法卖出")
        
        # 计算优先级（评分越低，卖出优先级越高）
        priority = int(-score * 100) if score != float('-inf') else 100
        
        return TradeSignal(
            stock_code=stock_code,
            action="sell",
            target_weight=0.0,  # 目标权重为 0，全部卖出
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
        评估单个买入候选股票
        
        Args:
            stock_code: 股票代码
            model_score: 模型评分
            market_data: 市场数据
            
        Returns:
            TradeSignal 交易信号
        """
        # 构建买入理由
        reasons = []
        reasons.append(f"模型评分较高({model_score:.4f})")
        reasons.append("在 Top K 推荐列表中")
        
        # 检查涨跌停限制
        is_tradable = self._check_limit_price(stock_code, "buy", market_data)
        if not is_tradable:
            reasons.append("注意：可能涨停无法买入")
        
        # 计算目标权重（等权分配）
        target_weight = 1.0 / self.config.topk
        
        # 计算优先级（评分越高，买入优先级越高）
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
        评估保留股票是否需要调仓
        
        Args:
            stock_code: 股票代码
            current_weight: 当前权重
            model_score: 模型评分
            market_data: 市场数据
            
        Returns:
            TradeSignal 交易信号（action 为 "hold" 或 "adjust"）
        """
        # 计算目标权重（等权分配）
        target_weight = 1.0 / self.config.topk
        
        # 计算权重偏差
        weight_diff = abs(current_weight - target_weight)
        weight_diff_ratio = weight_diff / target_weight if target_weight > 0 else 0
        
        # 判断是否需要调仓（权重偏差超过阈值）
        # 考虑交易成本，偏差较小时不调仓
        adjust_threshold = 0.2  # 权重偏差超过 20% 才调仓
        
        if weight_diff_ratio > adjust_threshold:
            # 需要调仓
            action = "adjust"
            reason = f"权重偏差 {weight_diff_ratio:.1%}，当前 {current_weight:.2%} -> 目标 {target_weight:.2%}"
        else:
            # 保持不变
            action = "hold"
            reason = f"继续持有，评分 {model_score:.4f}，权重偏差 {weight_diff_ratio:.1%} 在可接受范围内"
        
        return TradeSignal(
            stock_code=stock_code,
            action=action,
            target_weight=target_weight,
            reason=reason,
            score=model_score,
            current_weight=current_weight,
            confidence=1.0,
            priority=int(model_score * 50)  # 保留股票优先级适中
        )
    
    # =========================================================================
    # Step 3: 应用交易约束
    # =========================================================================
    
    def _apply_trade_constraints(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, List[TradeSignal]]:
        """
        应用交易约束条件过滤候选交易
        
        约束包括：
        1. 换手率限制
        2. 交易成本效率检查
        3. 涨跌停限制
        4. 流动性约束
        
        Args:
            candidates: 候选交易信号
            current_holdings: 当前持仓
            market_data: 市场数据
            
        Returns:
            过滤后的候选交易信号
        """
        filtered = {
            "sell": [],
            "buy": [],
            "hold": candidates.get("hold", []),  # 保留信号不过滤
            "adjust": []
        }
        
        # 1. 过滤卖出候选
        for signal in candidates.get("sell", []):
            # 检查涨跌停
            if signal.confidence < 0.5:  # 可能涨停
                self.decision_logs.append(
                    f"[Constraint] {signal.stock_code} 可能涨停，跳过卖出"
                )
                continue
            filtered["sell"].append(signal)
        
        # 2. 过滤买入候选
        for signal in candidates.get("buy", []):
            # 检查涨跌停
            if signal.confidence < 0.5:  # 可能跌停
                self.decision_logs.append(
                    f"[Constraint] {signal.stock_code} 可能跌停，跳过买入"
                )
                continue
            filtered["buy"].append(signal)
        
        # 3. 过滤调仓候选
        for signal in candidates.get("adjust", []):
            # 检查调仓成本是否划算
            if self._check_trade_cost_efficiency(signal, signal.current_weight, 1.0):
                filtered["adjust"].append(signal)
            else:
                # 调仓成本不划算，转为保留
                hold_signal = TradeSignal(
                    stock_code=signal.stock_code,
                    action="hold",
                    target_weight=signal.current_weight,
                    reason=f"调仓成本不划算，保持当前权重",
                    score=signal.score,
                    current_weight=signal.current_weight,
                    confidence=signal.confidence,
                    priority=signal.priority
                )
                filtered["hold"].append(hold_signal)
        
        # 4. 应用换手率限制
        filtered = self._check_turnover_limit(filtered, current_holdings)
        
        return filtered
    
    def _check_turnover_limit(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float]
    ) -> Dict[str, List[TradeSignal]]:
        """
        检查并限制换手率
        
        换手率 = (卖出金额 + 买入金额) / (2 * 总资产)
        
        Args:
            candidates: 候选交易
            current_holdings: 当前持仓
            
        Returns:
            换手率约束后的交易
        """
        max_turnover = self.config.max_turnover
        
        # 计算当前候选的换手率
        sell_weight = sum(s.current_weight for s in candidates.get("sell", []))
        buy_weight = sum(s.target_weight for s in candidates.get("buy", []))
        
        # 换手率估算（简化计算）
        estimated_turnover = (sell_weight + buy_weight) / 2
        
        if estimated_turnover <= max_turnover:
            return candidates
        
        self.decision_logs.append(
            f"[Turnover] 估算换手率 {estimated_turnover:.2%} 超过限制 {max_turnover:.2%}，需要削减交易"
        )
        
        # 按优先级排序，保留高优先级的交易
        sell_list = sorted(candidates.get("sell", []), key=lambda x: x.priority, reverse=True)
        buy_list = sorted(candidates.get("buy", []), key=lambda x: x.priority, reverse=True)
        
        # 逐步削减，直到满足换手率限制
        filtered_sell = []
        filtered_buy = []
        current_turnover = 0.0
        
        # 先处理卖出（卖出释放资金用于买入）
        for signal in sell_list:
            if current_turnover + signal.current_weight / 2 <= max_turnover:
                filtered_sell.append(signal)
                current_turnover += signal.current_weight / 2
        
        # 再处理买入
        for signal in buy_list:
            if current_turnover + signal.target_weight / 2 <= max_turnover:
                filtered_buy.append(signal)
                current_turnover += signal.target_weight / 2
        
        return {
            "sell": filtered_sell,
            "buy": filtered_buy,
            "hold": candidates.get("hold", []),
            "adjust": candidates.get("adjust", [])
        }
    
    def _check_trade_cost_efficiency(
        self,
        signal: TradeSignal,
        current_weight: float,
        total_value: float = 1.0
    ) -> bool:
        """
        检查交易成本效率
        
        判断调仓的收益是否能覆盖交易成本
        
        Args:
            signal: 交易信号
            current_weight: 当前权重
            total_value: 总价值（用于计算绝对成本）
            
        Returns:
            是否值得交易
        """
        weight_change = abs(signal.target_weight - current_weight)
        trade_value = weight_change * total_value
        
        # 计算交易成本
        # 调仓涉及卖出和买入，需要双边成本
        if signal.target_weight > current_weight:
            # 买入
            cost = trade_value * self.config.open_cost
        else:
            # 卖出
            cost = trade_value * self.config.close_cost
        
        # 最小交易成本
        cost = max(cost, self.config.min_cost / total_value if total_value > 0 else 0)
        
        # 如果调仓金额太小，成本占比过高，则不划算
        min_trade_ratio = self.config.min_trade_value / total_value if total_value > 0 else 0.01
        
        return weight_change > min_trade_ratio and weight_change > cost * 2
    
    def _check_liquidity_constraint(
        self,
        stock_code: str,
        trade_value: float,
        market_data: Optional[Dict[str, Any]]
    ) -> bool:
        """
        检查流动性约束
        
        Args:
            stock_code: 股票代码
            trade_value: 交易金额
            market_data: 市场数据
            
        Returns:
            是否满足流动性要求
        """
        if market_data is None:
            return True  # 无市场数据时默认可交易
        
        stock_data = market_data.get(stock_code, {})
        volume = stock_data.get("volume", float('inf'))
        avg_price = stock_data.get("price", 1.0)
        
        # 检查交易金额是否超过日成交量的一定比例
        daily_turnover = volume * avg_price
        max_participation = 0.1  # 最多占日成交量 10%
        
        return trade_value <= daily_turnover * max_participation
    
    def _check_limit_price(
        self,
        stock_code: str,
        action: str,
        market_data: Optional[Dict[str, Any]]
    ) -> bool:
        """
        检查涨跌停限制
        
        Args:
            stock_code: 股票代码
            action: 交易动作 (buy/sell)
            market_data: 市场数据
            
        Returns:
            是否可以交易（未涨跌停）
        """
        if market_data is None:
            return True  # 无市场数据时默认可交易
        
        stock_data = market_data.get(stock_code, {})
        price_change = stock_data.get("price_change_pct", 0.0)
        
        limit = self.config.limit_threshold
        
        if action == "buy":
            # 买入时检查是否跌停（跌停可以买入）或涨停（涨停无法买入）
            return price_change < limit
        elif action == "sell":
            # 卖出时检查是否涨停（涨停可以卖出）或跌停（跌停无法卖出）
            return price_change > -limit
        
        return True
    
    # =========================================================================
    # Step 4: 决策方法
    # =========================================================================
    
    def _rule_based_decision(
        self,
        candidates: Dict[str, List[TradeSignal]],
        current_holdings: Dict[str, float],
        top_k_recommendations: List[Tuple[str, float]]
    ) -> TradeDecision:
        """
        基于规则的决策方法
        
        决策规则：
        1. 按优先级排序所有交易信号
        2. 先执行卖出（释放资金）
        3. 再执行买入
        4. 最后处理调仓
        
        Args:
            candidates: 候选交易
            current_holdings: 当前持仓
            top_k_recommendations: Top K 推荐
            
        Returns:
            TradeDecision 交易决策
        """
        # 按优先级排序
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
        adjust_list = candidates.get("adjust", [])
        
        # 构建元数据
        metadata = {
            "decision_mode": "rule_based",
            "total_sell": len(sell_list),
            "total_buy": len(buy_list),
            "total_hold": len(hold_list),
            "total_adjust": len(adjust_list),
            "sell_stocks": [s.stock_code for s in sell_list],
            "buy_stocks": [s.stock_code for s in buy_list],
        }
        
        self.decision_logs.append(
            f"[Decision] 规则决策完成: 卖出 {len(sell_list)}, "
            f"买入 {len(buy_list)}, 保留 {len(hold_list)}, 调仓 {len(adjust_list)}"
        )
        
        return TradeDecision(
            buy_list=buy_list,
            sell_list=sell_list,
            hold_list=hold_list,
            adjust_list=adjust_list,
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
        LLM 增强的决策方法（集成新闻分析）
        
        流程：
        1. 批次获取候选股票的新闻数据
        2. 分批次调用 LLM 分析新闻对股票的影响
        3. 合并所有批次的分析结果
        4. 基于新闻分析结果调整交易决策
        
        Args:
            candidates: 候选交易
            current_holdings: 当前持仓
            top_k_recommendations: Top K 推荐
            market_data: 市场数据
            date: 日期
            
        Returns:
            TradeDecision 交易决策
        """
        if not self.llm:
            logger.warning("[LLM] LLM 服务不可用，回退到规则决策")
            self.decision_logs.append("[LLM] LLM 服务不可用，回退到规则决策")
            return self._rule_based_decision(candidates, current_holdings, top_k_recommendations)
        
        try:
            # =====================================================================
            # Step 1: 批次分析新闻（针对所有候选股票）
            # =====================================================================
            logger.info(f"[LLM] 开始新闻分析，日期: {date}")
            self.decision_logs.append(f"[LLM] 开始新闻分析，日期: {date}")
            
            # 格式化日期（确保与新闻数据的日期格式匹配）
            formatted_date = date[:10] if date and len(date) >= 10 else date
            
            # 批次分析所有候选股票的新闻
            news_analysis = self._analyze_all_candidates_with_news(candidates, formatted_date)
            
            logger.info(f"[LLM] 新闻分析完成，共分析 {len(news_analysis)} 只股票")
            self.decision_logs.append(f"[LLM] 新闻分析完成，共分析 {len(news_analysis)} 只股票")
            
            # =====================================================================
            # Step 2: 基于新闻分析结果生成调整建议
            # =====================================================================
            if news_analysis:
                # 将新闻分析结果转换为决策调整建议
                news_based_suggestions = self._merge_news_analysis_to_decision(
                    candidates=candidates,
                    news_analysis=news_analysis
                )
                # 提取各项调整数量
                remove_sell_count = len(news_based_suggestions.get('remove_from_sell', []))
                remove_buy_count = len(news_based_suggestions.get('remove_from_buy', []))
                move_to_sell_count = len(news_based_suggestions.get('move_to_sell', []))
                
                logger.info(f"[LLM] 新闻分析建议: "
                           f"移除卖出 {remove_sell_count} (转为持有), "
                           f"移除买入 {remove_buy_count}, "
                           f"移除持有(转为卖出) {move_to_sell_count}, ")
                self.decision_logs.append(f"[LLM] 新闻分析建议: {news_based_suggestions.get('reasoning', '')}")
                # logger.info(f"[LLM] 新闻分析建议: 移除卖出 {len(news_based_suggestions.get('remove_from_sell', []))}, "
                #            f"移除买入 {len(news_based_suggestions.get('remove_from_buy', []))}")
                # self.decision_logs.append(f"[LLM] 新闻分析建议: {news_based_suggestions.get('reasoning', '')}")
            else:
                # 无新闻分析结果，使用空建议
                news_based_suggestions = {
                    "remove_from_sell": [],
                    "remove_from_buy": [],
                    'move_to_sell': [],
                    "priority_adjustments": {},
                    "reasoning": "无新闻数据可供分析"
                }
            
            # =====================================================================
            # Step 3: 可选 - 进行综合决策（结合市场数据和新闻分析）
            # =====================================================================
            # 如果需要进一步的 LLM 综合分析，可以构建综合提示词
            # 这里我们直接使用新闻分析结果作为决策依据
            
            # 构建最终的 LLM 决策（包含新闻分析摘要）
            llm_decision = news_based_suggestions
            
            # =====================================================================
            # Step 4: 应用调整建议生成最终决策
            # =====================================================================
            adjusted_decision = self._apply_llm_suggestions(
                candidates=candidates,
                llm_decision=llm_decision
            )
            
            # =====================================================================
            # Step 5: 保存短期记忆（宏观新闻 + 股票决策理由）
            # =====================================================================
            # 收集所有股票的决策理由
            stock_reasons = {}
            for signal in adjusted_decision.sell_list + adjusted_decision.buy_list + adjusted_decision.hold_list:
                stock_code = signal.stock_code
                # 从新闻分析中获取理由，如果没有则使用原始理由
                analysis = news_analysis.get(stock_code, {})
                reason = analysis.get("reason", signal.reason)
                stock_reasons[stock_code] = reason
            
            # 获取宏观新闻摘要
            macro_news_summary = self._get_compressed_macro_news(date)
            
            # 保存短期记忆
            self._save_short_term_memory(
                date=date,
                macro_news_summary=macro_news_summary,
                stock_reasons=stock_reasons
            )
            
            # 在元数据中添加新闻分析信息
            adjusted_decision.metadata["news_analysis"] = {
                "analyzed_stocks": len(news_analysis),
                "summary": news_based_suggestions.get("reasoning", ""),
                "removed_from_sell": news_based_suggestions.get("remove_from_sell", []),
                "removed_from_buy": news_based_suggestions.get("remove_from_buy", []),
                "macro_news_loaded": bool(macro_news_summary),
                "short_term_memory_days": len(self._short_term_memory),
            }
            
            logger.info(f"[LLM] LLM 增强决策完成（含新闻分析、宏观新闻、短期记忆）")
            self.decision_logs.append(f"[LLM] LLM 增强决策完成（含新闻分析、宏观新闻、短期记忆）")
            
            return adjusted_decision
            
        except Exception as e:
            logger.error(f"[LLM] LLM 决策失败: {str(e)}，回退到规则决策")
            self.decision_logs.append(f"[LLM] LLM 决策失败: {str(e)}，回退到规则决策")
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
#         构建 LLM 决策提示词
        
#         Args:
#             candidates: 候选交易
#             current_holdings: 当前持仓
#             market_context: 市场上下文
            
#         Returns:
#             提示词字符串
#         """
#         # 格式化候选交易信息
#         sell_info = "\n".join([
#             f"  - {s.stock_code}: 当前权重 {s.current_weight:.2%}, 评分 {s.score:.4f}, 原因: {s.reason}"
#             for s in candidates.get("sell", [])
#         ]) or "  无"
        
#         buy_info = "\n".join([
#             f"  - {s.stock_code}: 目标权重 {s.target_weight:.2%}, 评分 {s.score:.4f}, 原因: {s.reason}"
#             for s in candidates.get("buy", [])
#         ]) or "  无"
        
#         hold_info = "\n".join([
#             f"  - {s.stock_code}: 当前权重 {s.current_weight:.2%}, 评分 {s.score:.4f}"
#             for s in candidates.get("hold", [])[:10]  # 只显示前10只
#         ]) or "  无"
        
#         prompt = f"""你是一个量化交易策略专家。请审阅以下交易决策候选，并给出最终建议。

# ## 市场上下文
# {market_context}

# ## 当前持仓数量
# {len(current_holdings)} 只股票

# ## 候选卖出股票
# {sell_info}

# ## 候选买入股票
# {buy_info}

# ## 继续持有股票（前10只）
# {hold_info}

# ## 任务
# 请分析以上候选交易，考虑以下因素：
# 1. 是否有股票被误判为卖出候选（例如短期波动但长期向好）
# 2. 是否有买入候选存在风险（例如涨幅过大追高风险）
# 3. 是否需要调整交易优先级

# 请以 JSON 格式输出你的建议：
# ```json
# {{
#     "remove_from_sell": ["股票代码1", "股票代码2"],  // 建议不卖出的股票
#     "remove_from_buy": ["股票代码1"],  // 建议不买入的股票
#     "priority_adjustments": {{"股票代码": 新优先级}},  // 优先级调整
#     "reasoning": "整体分析和建议"
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
        应用 LLM 的建议调整决策
        
        Args:
            candidates: 原始候选交易
            llm_decision: LLM 的决策建议
            
        Returns:
            调整后的 TradeDecision
        """
        remove_from_sell = set(llm_decision.get("remove_from_sell", []))
        remove_from_buy = set(llm_decision.get("remove_from_buy", []))
        move_to_sell = set(llm_decision.get("move_to_sell", []))
        priority_adjustments = llm_decision.get("priority_adjustments", {})
        
        # 过滤卖出列表
        sell_list = [
            s for s in candidates.get("sell", [])
            if s.stock_code not in remove_from_sell
        ]
        
        # 过滤买入列表
        buy_list = [
            s for s in candidates.get("buy", [])
            if s.stock_code not in remove_from_buy
        ]
        
        # 从 hold 和 adjust 中移除要卖出的股票
        hold_list = [
            s for s in candidates.get("hold", [])
            if s.stock_code not in move_to_sell
        ]
        
        # 修改：同时处理 adjust 候选，过滤掉要卖出的股票
        adjust_list = [
            s for s in candidates.get("adjust", [])
            if s.stock_code not in move_to_sell
        ]
        
        # 将 move_to_sell 的股票转为卖出信号
        # 修改：同时从 hold 和 adjust 中查找要转换为卖出的股票
        new_sell_signals = []
        for action_type in ["hold", "adjust"]:
            for signal in candidates.get(action_type, []):
                if signal.stock_code in move_to_sell:
                    sell_signal = TradeSignal(
                        stock_code=signal.stock_code,
                        action="sell",
                        target_weight=0.0,
                        reason=f"新闻分析建议卖出: {llm_decision.get('reasoning', '')}",
                        score=signal.score,
                        current_weight=signal.current_weight,
                        confidence=0.7,  # 基于新闻分析的置信度
                        priority=signal.priority + 30  # 提高优先级
                    )
                    new_sell_signals.append(sell_signal)
        
        sell_list.extend(new_sell_signals)
        
        # 调整优先级（包含 adjust_list）
        for signal in sell_list + buy_list + hold_list + adjust_list:
            if signal.stock_code in priority_adjustments:
                signal.priority = priority_adjustments[signal.stock_code]
        
        # 被移除的卖出候选转为保留
        removed_sells = [
            TradeSignal(
                stock_code=s.stock_code,
                action="hold",
                target_weight=s.current_weight,
                reason=f"LLM 建议保留: {llm_decision.get('reasoning', '')}",
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
            adjust_list=adjust_list,  # 修改：使用过滤后的 adjust_list
            metadata={
                "decision_mode": "llm_enhanced",
                "llm_reasoning": llm_decision.get("reasoning", ""),
                "removed_sells": list(remove_from_sell),
                "removed_buys": list(remove_from_buy),
                "moved_to_sell": list(move_to_sell),
            }
        )
    
    # =========================================================================
    # Step 5: 历史记录
    # =========================================================================
    
    def _record_trade_history(
        self,
        decision: TradeDecision,
        date: Optional[str]
    ) -> None:
        """
        记录交易历史
        
        Args:
            decision: 交易决策
            date: 日期
        """
        # 从metadata中提取LLM建议的调整信息
        removed_sells = decision.metadata.get("removed_sells", [])
        removed_buys = decision.metadata.get("removed_buys", [])
        moved_to_sell = decision.metadata.get("moved_to_sell", [])
        llm_reasoning = decision.metadata.get("llm_reasoning", "")
        
        # 从news_analysis中提取信息（如果存在）
        news_analysis = decision.metadata.get("news_analysis", {})
        removed_from_sell = news_analysis.get("removed_from_sell", [])
        removed_from_buy = news_analysis.get("removed_from_buy", [])
        news_summary = news_analysis.get("summary", "")
        
        # 合并信息，优先使用metadata中的直接信息
        final_removed_sells = removed_sells if removed_sells else removed_from_sell
        final_removed_buys = removed_buys if removed_buys else removed_from_buy
        final_reasoning = llm_reasoning if llm_reasoning else news_summary
        
        record = {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "sell_count": len(decision.sell_list),
            "buy_count": len(decision.buy_list),
            "hold_count": len(decision.hold_list),
            "adjust_count": len(decision.adjust_list),
            "sell_stocks": [s.stock_code for s in decision.sell_list],
            "buy_stocks": [s.stock_code for s in decision.buy_list],
            "metadata": decision.metadata,
            "timestamp": datetime.now().isoformat(),
            # LLM建议的调整信息
            "removed_sells": final_removed_sells,
            "removed_buys": final_removed_buys,
            "moved_to_sell": moved_to_sell,
            "llm_reasoning": final_reasoning
        }
        
        # 格式化股票列表为字符串
        actual_buy_stocks = [s.stock_code for s in decision.buy_list]
        actual_sell_stocks = [s.stock_code for s in decision.sell_list]
        actual_buy_str = "、".join(actual_buy_stocks) if actual_buy_stocks else "无"
        actual_sell_str = "、".join(actual_sell_stocks) if actual_sell_stocks else "无"
        removed_buy_str = "、".join(final_removed_buys) if final_removed_buys else "无"
        removed_sell_str = "、".join(final_removed_sells) if final_removed_sells else "无"
        moved_to_sell_str = "、".join(moved_to_sell) if moved_to_sell else "无"
        
        # 打印日志
        current_date = date or datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"{current_date}: 实际买入={actual_buy_str}、"
            f"实际卖出={actual_sell_str}、"
            f"移除买入={removed_buy_str}、"
            f"移除卖出={removed_sell_str}、"
            f"持有转为卖出={moved_to_sell_str}"
        )
        
        self.trade_history.append(record)
        
        self.decision_logs.append(
            f"[History] 交易记录已保存: {date}, "
            f"卖出 {len(decision.sell_list)} 只, 买入 {len(decision.buy_list)} 只"
        )
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _calculate_target_weights(
        self,
        top_k_recommendations: List[Tuple[str, float]],
        method: str = "equal"
    ) -> Dict[str, float]:
        """
        根据模型评分计算目标权重
        
        Args:
            top_k_recommendations: Top K 推荐 [(stock_code, score), ...]
            method: 权重计算方法
                - "equal": 等权分配
                - "score_weighted": 按评分加权
                - "rank_weighted": 按排名加权
            
        Returns:
            目标权重字典 {stock_code: target_weight}
        """
        if not top_k_recommendations:
            return {}
        
        n = len(top_k_recommendations)
        
        if method == "equal":
            # 等权分配
            weight = 1.0 / n
            return {stock: weight for stock, _ in top_k_recommendations}
        
        elif method == "score_weighted":
            # 按评分加权（需要归一化）
            scores = [score for _, score in top_k_recommendations]
            min_score = min(scores)
            # 平移确保所有分数为正
            shifted_scores = [s - min_score + 1 for s in scores]
            total = sum(shifted_scores)
            
            return {
                stock: shifted_scores[i] / total
                for i, (stock, _) in enumerate(top_k_recommendations)
            }
        
        elif method == "rank_weighted":
            # 按排名加权（排名靠前权重更高）
            # 使用 1/rank 作为权重
            weights = [1.0 / (i + 1) for i in range(n)]
            total = sum(weights)
            
            return {
                stock: weights[i] / total
                for i, (stock, _) in enumerate(top_k_recommendations)
            }
        
        else:
            # 默认等权
            weight = 1.0 / n
            return {stock: weight for stock, _ in top_k_recommendations}
    
    def _calculate_trade_cost(
        self,
        trade_value: float,
        action: str
    ) -> float:
        """
        计算交易成本
        
        Args:
            trade_value: 交易金额
            action: 交易动作 (buy/sell)
            
        Returns:
            交易成本
        """
        if action == "buy":
            cost = trade_value * self.config.open_cost
        elif action == "sell":
            cost = trade_value * self.config.close_cost
        else:
            cost = 0.0
        
        # 最小交易成本
        return max(cost, self.config.min_cost)
    
    def get_decision_logs(self) -> List[str]:
        """获取决策日志"""
        return self.decision_logs.copy()
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """获取交易历史"""
        return self.trade_history.copy()
    
    def reset(self) -> None:
        """重置 Agent 状态"""
        self.trade_history.clear()
        self.decision_logs.clear()
    
    # =========================================================================
    # LangGraph 工作流集成 - process 方法
    # =========================================================================
    
    def process(
        self,
        model_info: Dict[str, Any],
        sota_pool_list: List[str],
        run_backtest: bool = True,
        run_both_versions: bool = True
    ) -> Dict[str, Any]:
        """
        策略生成主入口方法 - 用于 LangGraph 工作流
        
        根据模型优化结果生成策略配置，可选择运行回测
        如果 run_both_versions=True，会先运行 baseline（不使用LLM），再运行 LLM 增强版本
        
        Args:
            model_info: 模型优化结果，包含:
                - yaml_config_path: 最优 YAML 配置路径
                - factor_pool_name: 因子池名称
                - module_path: 因子池模块路径
                - model_class: 模型类型
                - model_kwargs: 模型参数
                - best_metrics: 基线性能指标
            sota_pool_list: 因子列表
            run_backtest: 是否运行回测
            run_both_versions: 是否运行两个版本（baseline 和 LLM 增强）进行对比
            
        Returns:
            包含策略配置和回测结果的字典
        """
        logs = []
        logs.append("[StrategyGeneration] 开始策略生成流程")
        
        result = {
            "status": "success",
            "logs": logs,
            "strategy_config": {},
            "backtest_result": None,
            "baseline_result": None,
            "llm_enhanced_result": None,
        }
        
        try:
            # Step 1: 提取模型优化信息
            yaml_config_path = model_info.get("yaml_config_path")
            factor_pool_name = model_info.get("factor_pool_name")
            module_path = model_info.get("module_path")
            model_class = model_info.get("model_class", "TransformerModel")
            best_metrics = model_info.get("best_metrics", {})
            
            logs.append(f"[StrategyGeneration] 接收到模型优化结果:")
            logs.append(f"  - 配置路径: {yaml_config_path}")
            logs.append(f"  - 因子池: {factor_pool_name}")
            logs.append(f"  - 模型类型: {model_class}")
            logs.append(f"  - 因子数量: {len(sota_pool_list)}")
            
            if best_metrics:
                ic_mean = best_metrics.get('ic_mean')
                ann_ret = best_metrics.get('annualized_return')
                if ic_mean is not None:
                    logs.append(f"  - 基线 IC: {ic_mean:.4f}")
                if ann_ret is not None:
                    logs.append(f"  - 基线年化: {ann_ret:.2%}")
            
            # Step 2: 构建策略配置
            strategy_config = self._build_strategy_config(model_info, sota_pool_list)
            result["strategy_config"] = strategy_config
            logs.append("[StrategyGeneration] 策略配置已生成")
            logs.append(f"[StrategyGeneration] TopK: {strategy_config.get('topk', 50)}, N_Drop: {strategy_config.get('n_drop', 5)}")
            
            # Step 3: 运行回测（可选）
            if run_backtest and yaml_config_path and Path(yaml_config_path).exists():
                if run_both_versions and self.use_llm_decision:
                    # =============================================================
                    # 运行两个版本：baseline 和 LLM 增强
                    # =============================================================
                    logs.append("[StrategyGeneration] ========== 开始运行双版本回测对比 ==========")
                    
                    # 3.1: 运行 Baseline 版本（不使用 LLM）
                    logs.append("[StrategyGeneration] ---------- 步骤 1/2: 运行 Baseline 版本（不使用 LLM） ----------")
                    baseline_yaml_path = self._generate_agent_yaml_config(
                        original_yaml_path=yaml_config_path,
                        strategy_config=strategy_config,
                        factor_pool_name=factor_pool_name,
                        use_llm_decision=False,
                        version_suffix="baseline"
                    )
                    logs.append(f"[StrategyGeneration] Baseline 策略配置已保存: {baseline_yaml_path}")
                    
                    baseline_result = self._run_backtest(baseline_yaml_path)
                    baseline_metrics = None
                    if baseline_result:
                        baseline_metrics = self._extract_backtest_metrics(baseline_result)
                        result["baseline_result"] = baseline_result
                        result["baseline_metrics"] = baseline_metrics
                        
                        logs.append("[StrategyGeneration] Baseline 回测完成")
                        if baseline_metrics.get("ic_mean"):
                            logs.append(f"[StrategyGeneration] Baseline IC 均值: {baseline_metrics['ic_mean']:.4f}")
                        if baseline_metrics.get("annualized_return"):
                            logs.append(f"[StrategyGeneration] Baseline 年化收益: {baseline_metrics['annualized_return']:.2%}")
                        if baseline_metrics.get("max_drawdown"):
                            logs.append(f"[StrategyGeneration] Baseline 最大回撤: {baseline_metrics['max_drawdown']:.2%}")
                    else:
                        logs.append("[StrategyGeneration] 警告: Baseline 回测未返回结果")
                    
                    # 3.2: 运行 LLM 增强版本
                    logs.append("[StrategyGeneration] ---------- 步骤 2/2: 运行 LLM 增强版本 ----------")
                    llm_yaml_path = self._generate_agent_yaml_config(
                        original_yaml_path=yaml_config_path,
                        strategy_config=strategy_config,
                        factor_pool_name=factor_pool_name,
                        use_llm_decision=True,
                        version_suffix="llm_enhanced"
                    )
                    logs.append(f"[StrategyGeneration] LLM 增强策略配置已保存: {llm_yaml_path}")
                    
                    llm_result = self._run_backtest(llm_yaml_path)
                    llm_metrics = None
                    if llm_result:
                        llm_metrics = self._extract_backtest_metrics(llm_result)
                        result["llm_enhanced_result"] = llm_result
                        result["llm_enhanced_metrics"] = llm_metrics
                        result["backtest_result"] = llm_result  # 保留兼容性
                        result["backtest_metrics"] = llm_metrics
                        
                        logs.append("[StrategyGeneration] LLM 增强回测完成")
                        if llm_metrics.get("ic_mean"):
                            logs.append(f"[StrategyGeneration] LLM 增强 IC 均值: {llm_metrics['ic_mean']:.4f}")
                        if llm_metrics.get("annualized_return"):
                            logs.append(f"[StrategyGeneration] LLM 增强年化收益: {llm_metrics['annualized_return']:.2%}")
                        if llm_metrics.get("max_drawdown"):
                            logs.append(f"[StrategyGeneration] LLM 增强最大回撤: {llm_metrics['max_drawdown']:.2%}")
                    else:
                        logs.append("[StrategyGeneration] 警告: LLM 增强回测未返回结果")
                    
                    # 3.3: 对比两个版本
                    if baseline_metrics and llm_metrics:
                        logs.append("[StrategyGeneration] ========== 版本对比结果 ==========")
                        self._compare_two_versions(baseline_metrics, llm_metrics, logs)
                        result["comparison"] = self._generate_comparison_dict(baseline_metrics, llm_metrics)
                    
                    # 设置最终使用的配置路径（优先使用 LLM 增强版本）
                    agent_yaml_path = llm_yaml_path if llm_yaml_path else baseline_yaml_path
                    
                else:
                    # =============================================================
                    # 只运行一个版本（根据 use_llm_decision 决定）
                    # =============================================================
                    use_llm = self.use_llm_decision if not run_both_versions else False
                    version_name = "LLM 增强" if use_llm else "Baseline"
                    logs.append(f"[StrategyGeneration] 开始运行 {version_name} 策略回测...")
                    
                    agent_yaml_path = self._generate_agent_yaml_config(
                        original_yaml_path=yaml_config_path,
                        strategy_config=strategy_config,
                        factor_pool_name=factor_pool_name,
                        use_llm_decision=use_llm,
                        version_suffix="baseline" if not use_llm else "llm_enhanced"
                    )
                    logs.append(f"[StrategyGeneration] {version_name} 策略配置已保存: {agent_yaml_path}")
                    
                    backtest_result = self._run_backtest(agent_yaml_path)
                    
                    if backtest_result:
                        result["backtest_result"] = backtest_result
                        
                        # 提取回测指标
                        backtest_metrics = self._extract_backtest_metrics(backtest_result)
                        result["backtest_metrics"] = backtest_metrics
                        
                        logs.append(f"[StrategyGeneration] {version_name} 回测完成")
                        if backtest_metrics.get("ic_mean"):
                            logs.append(f"[StrategyGeneration] {version_name} IC 均值: {backtest_metrics['ic_mean']:.4f}")
                        if backtest_metrics.get("annualized_return"):
                            logs.append(f"[StrategyGeneration] {version_name} 年化收益: {backtest_metrics['annualized_return']:.2%}")
                        if backtest_metrics.get("max_drawdown"):
                            logs.append(f"[StrategyGeneration] {version_name} 最大回撤: {backtest_metrics['max_drawdown']:.2%}")
                        
                        # 与基线对比
                        if best_metrics and backtest_metrics:
                            self._compare_with_baseline(best_metrics, backtest_metrics, logs)
                    else:
                        logs.append(f"[StrategyGeneration] 警告: {version_name} 回测未返回结果")
            elif not run_backtest:
                logs.append("[StrategyGeneration] 跳过回测（run_backtest=False）")
                agent_yaml_path = None
            else:
                logs.append(f"[StrategyGeneration] 警告: 原始 YAML 配置不存在: {yaml_config_path}")
                agent_yaml_path = None
            
            # 构建完整策略信息
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
            
            logs.append("[StrategyGeneration] 策略生成流程完成")
            
        except Exception as e:
            import traceback
            logs.append(f"[StrategyGeneration] 处理出错: {str(e)}")
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
        构建策略配置
        
        Args:
            model_info: 模型优化结果
            factors: 因子列表
            
        Returns:
            策略配置字典
        """
        # 从模型优化结果中获取默认策略配置
        default_config = model_info.get("default_strategy_config", {})
        
        config = {
            # 选股参数
            "topk": default_config.get("topk", self.config.topk),
            "n_drop": default_config.get("n_drop", self.config.n_drop),
            
            # 交易约束
            "max_turnover": self.config.max_turnover,
            "min_trade_value": self.config.min_trade_value,
            
            # 交易成本
            "open_cost": self.config.open_cost,
            "close_cost": self.config.close_cost,
            "min_cost": self.config.min_cost,
            
            # 涨跌停限制
            "limit_threshold": self.config.limit_threshold,
            
            # 仓位控制
            "risk_degree": 0.95,
            "hold_thresh": 1,
            
            # Agent 决策配置
            "use_agent_decision": True,
            "use_llm_decision": self.use_llm_decision,
        }
        
        # 因子数量
        config["factors_count"] = len(factors)
        
        # 根据因子数量自动调整 topk
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
        生成带有 AgentEnhancedStrategy 的 YAML 配置文件
        
        Args:
            original_yaml_path: 原始 YAML 配置路径
            strategy_config: 策略配置
            factor_pool_name: 因子池名称
            use_llm_decision: 是否使用 LLM 增强决策
            version_suffix: 版本后缀（用于区分 baseline 和 llm_enhanced）
            
        Returns:
            新 YAML 配置文件路径
        """
        # 读取原始配置
        with open(original_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 生成 AgentEnhancedStrategy 配置
        agent_strategy_config = {
            "class": "AgentEnhancedStrategy",
            "module_path": "Agent.strategy_generation_agent",
            "kwargs": {
                "signal": "<PRED>",
                "topk": strategy_config.get("topk", 50),
                "n_drop": strategy_config.get("n_drop", 5),
                "use_agent_decision": strategy_config.get("use_agent_decision", True),
                "use_llm_decision": use_llm_decision,
                "use_env_llm_config": True,
                "risk_degree": strategy_config.get("risk_degree", 0.95),
                "hold_thresh": strategy_config.get("hold_thresh", 1),
                "only_tradable": False,
                "forbid_all_trade_at_limit": True,
            }
        }
        
        # 如果使用 LLM 决策，添加新闻数据配置
        if use_llm_decision:
            if self.news_data_path:
                agent_strategy_config["kwargs"]["news_data_path"] = self.news_data_path
            if self.news_batch_size != 10:
                agent_strategy_config["kwargs"]["news_batch_size"] = self.news_batch_size
        
        back_test_start_time = '2025-01-01'
        back_test_end_time = '2025-06-30'
        # 替换策略配置 - 修复路径问题
        # 方案1: 替换 port_analysis_config.strategy
        if 'port_analysis_config' in config:
            config['port_analysis_config']['strategy'] = agent_strategy_config
        # 方案2: 兼容 task.backtest.strategy 结构
        elif 'task' in config and 'backtest' in config['task']:
            config['task']['backtest']['strategy'] = agent_strategy_config
        # 1. 扩大 data_handler_config.end_time
        if 'data_handler_config' in config:
            if 'end_time' in config['data_handler_config']:
                config['data_handler_config']['end_time'] = back_test_end_time
                logger.info(f"[Strategy] 扩大 data_handler_config.end_time 到: {back_test_end_time}")

        # 修改回测时间
        if 'port_analysis_config' in config and 'backtest' in config['port_analysis_config']:
            config['port_analysis_config']['backtest']['start_time'] = back_test_start_time
            config['port_analysis_config']['backtest']['end_time'] = back_test_end_time
        # 2. 扩大 test segment 以覆盖回测时间（关键修复！）
        if 'task' in config and 'dataset' in config['task']:
            dataset_config = config['task']['dataset']
            if 'kwargs' in dataset_config and 'segments' in dataset_config['kwargs']:
                segments = dataset_config['kwargs']['segments']
                if 'test' in segments and len(segments['test']) >= 2:
                    segments['test'][0] = back_test_start_time
                    segments['test'][1] = back_test_end_time
                    segments['valid'][1] = "2024-12-31"
                    logger.info(f"[Strategy] 扩大 test segment end_time 到: {back_test_end_time}")
        # 保存新配置
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{version_suffix}" if version_suffix else ""
        output_path = self.output_dir / f"workflow_agent_strategy_{factor_pool_name}{suffix_str}_{timestamp}.yaml"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return str(output_path)
    
    def _run_backtest(self, yaml_path: str) -> Optional[Dict[str, Any]]:
        """
        运行回测
        
        Args:
            yaml_path: YAML 配置路径
            
        Returns:
            回测结果字典
        """
        if not self.mcp_client:
            print("[StrategyGeneration] 警告: MCP 客户端未初始化，无法运行回测")
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
            print(f"[StrategyGeneration] 运行回测失败: {e}")
            return None
    
    def _extract_backtest_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从回测结果中提取关键指标
        
        Args:
            backtest_result: 回测结果
            
        Returns:
            指标字典
        """
        metrics = {}
        
        if FileUtils is None:
            return metrics
        
        # 读取 IC 统计
        ic_path = backtest_result.get("ic", "")
        if ic_path:
            ic_stats = FileUtils.read_pickle_stats(ic_path)
            if ic_stats:
                metrics["ic_mean"] = ic_stats.get("mean", 0)
                metrics["ic_std"] = ic_stats.get("std", 0)
        
        # 读取 Rank IC 统计
        rank_ic_path = backtest_result.get("rank_ic", "")
        if rank_ic_path:
            rank_ic_stats = FileUtils.read_pickle_stats(rank_ic_path)
            if rank_ic_stats:
                metrics["rank_ic_mean"] = rank_ic_stats.get("mean", 0)
                metrics["rank_ic_std"] = rank_ic_stats.get("std", 0)
        
        # 读取年化收益
        ann_ret_path = backtest_result.get("1day.excess_return_with_cost.annualized_return", "")
        if ann_ret_path:
            metrics["annualized_return"] = FileUtils.read_mlflow_metric(ann_ret_path)
        
        # 读取最大回撤
        max_dd_path = backtest_result.get("1day.excess_return_with_cost.max_drawdown", "")
        if max_dd_path:
            metrics["max_drawdown"] = FileUtils.read_mlflow_metric(max_dd_path)
        
        # 计算 IR
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
        将回测结果与基线对比
        
        Args:
            baseline_metrics: 基线指标
            backtest_metrics: 回测指标
            logs: 日志列表
        """
        logs.append("[StrategyGeneration] === 与基线对比 ===")
        
        # IC 对比
        baseline_ic = baseline_metrics.get("ic_mean")
        backtest_ic = backtest_metrics.get("ic_mean")
        if baseline_ic and backtest_ic:
            diff = backtest_ic - baseline_ic
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  IC: {baseline_ic:.4f} -> {backtest_ic:.4f} ({sign} {abs(diff):.4f})")
        
        # 年化收益对比
        baseline_ret = baseline_metrics.get("annualized_return")
        backtest_ret = backtest_metrics.get("annualized_return")
        if baseline_ret and backtest_ret:
            diff = backtest_ret - baseline_ret
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  年化收益: {baseline_ret:.2%} -> {backtest_ret:.2%} ({sign} {abs(diff):.2%})")
        
        # 最大回撤对比
        baseline_dd = baseline_metrics.get("max_drawdown")
        backtest_dd = backtest_metrics.get("max_drawdown")
        if baseline_dd and backtest_dd:
            diff = backtest_dd - baseline_dd
            # 回撤是负数，变大（更接近0）是好的
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  最大回撤: {baseline_dd:.2%} -> {backtest_dd:.2%} ({sign} {abs(diff):.2%})")
    
    def _compare_two_versions(
        self,
        baseline_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any],
        logs: List[str]
    ) -> None:
        """
        对比 Baseline 版本和 LLM 增强版本的回测结果
        
        Args:
            baseline_metrics: Baseline 版本指标
            llm_metrics: LLM 增强版本指标
            logs: 日志列表
        """
        logs.append("[StrategyGeneration] === Baseline vs LLM 增强版本对比 ===")
        
        # IC 对比
        baseline_ic = baseline_metrics.get("ic_mean")
        llm_ic = llm_metrics.get("ic_mean")
        if baseline_ic is not None and llm_ic is not None:
            diff = llm_ic - baseline_ic
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            improvement_pct = (diff / abs(baseline_ic) * 100) if baseline_ic != 0 else 0
            logs.append(f"  IC 均值: Baseline {baseline_ic:.4f} -> LLM增强 {llm_ic:.4f} ({sign} {abs(diff):.4f}, {improvement_pct:+.2f}%)")
        
        # Rank IC 对比
        baseline_rank_ic = baseline_metrics.get("rank_ic_mean")
        llm_rank_ic = llm_metrics.get("rank_ic_mean")
        if baseline_rank_ic is not None and llm_rank_ic is not None:
            diff = llm_rank_ic - baseline_rank_ic
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  Rank IC: Baseline {baseline_rank_ic:.4f} -> LLM增强 {llm_rank_ic:.4f} ({sign} {abs(diff):.4f})")
        
        # 年化收益对比
        baseline_ret = baseline_metrics.get("annualized_return")
        llm_ret = llm_metrics.get("annualized_return")
        if baseline_ret is not None and llm_ret is not None:
            diff = llm_ret - baseline_ret
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  年化收益: Baseline {baseline_ret:.2%} -> LLM增强 {llm_ret:.2%} ({sign} {abs(diff):.2%})")
        
        # 最大回撤对比
        baseline_dd = baseline_metrics.get("max_drawdown")
        llm_dd = llm_metrics.get("max_drawdown")
        if baseline_dd is not None and llm_dd is not None:
            diff = llm_dd - baseline_dd
            # 回撤是负数，变大（更接近0）是好的
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  最大回撤: Baseline {baseline_dd:.2%} -> LLM增强 {llm_dd:.2%} ({sign} {abs(diff):.2%})")
        
        # IR 对比
        baseline_ir = baseline_metrics.get("ir")
        llm_ir = llm_metrics.get("ir")
        if baseline_ir is not None and llm_ir is not None:
            diff = llm_ir - baseline_ir
            sign = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            logs.append(f"  IR (IC/Std): Baseline {baseline_ir:.4f} -> LLM增强 {llm_ir:.4f} ({sign} {abs(diff):.4f})")
        
        # 总结
        improvements = []
        if baseline_ic is not None and llm_ic is not None and llm_ic > baseline_ic:
            improvements.append("IC")
        if baseline_ret is not None and llm_ret is not None and llm_ret > baseline_ret:
            improvements.append("年化收益")
        if baseline_dd is not None and llm_dd is not None and llm_dd > baseline_dd:
            improvements.append("最大回撤")
        
        if improvements:
            logs.append(f"[StrategyGeneration] LLM 增强版本在以下指标上有提升: {', '.join(improvements)}")
        else:
            logs.append(f"[StrategyGeneration] LLM 增强版本未在所有指标上超越 Baseline")
    
    def _generate_comparison_dict(
        self,
        baseline_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成两个版本的对比字典
        
        Args:
            baseline_metrics: Baseline 版本指标
            llm_metrics: LLM 增强版本指标
            
        Returns:
            对比字典
        """
        comparison = {
            "baseline": baseline_metrics.copy(),
            "llm_enhanced": llm_metrics.copy(),
            "differences": {}
        }
        
        # 计算各项指标的差异
        for key in ["ic_mean", "rank_ic_mean", "annualized_return", "max_drawdown", "ir"]:
            baseline_val = baseline_metrics.get(key)
            llm_val = llm_metrics.get(key)
            if baseline_val is not None and llm_val is not None:
                diff = llm_val - baseline_val
                diff_pct = (diff / abs(baseline_val) * 100) if baseline_val != 0 else 0
                comparison["differences"][key] = {
                    "absolute": diff,
                    "percentage": diff_pct,
                    "improved": diff > 0 if key != "max_drawdown" else diff > 0  # 回撤越大越好（更接近0）
                }
        
        return comparison


# =============================================================================
# Qlib Strategy 集成类
# =============================================================================

# 尝试导入 Qlib 相关模块
try:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.position import Position
    from qlib.backtest.signal import Signal, create_signal_from
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("警告: Qlib 模块不可用，AgentEnhancedStrategy 将使用模拟模式")
    
    # 创建模拟类
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
    Agent 增强的 Qlib 策略类
    
    继承 Qlib 的 BaseStrategy（类似 TopkDropoutStrategy），在 generate_trade_decision 中
    调用 StrategyGenerationAgent 进行智能决策
    
    主要特点：
    1. 完全兼容 Qlib 回测框架
    2. 使用 Agent 进行智能交易决策
    3. 支持 LLM 增强决策（可通过 YAML 配置）
    4. 保留原有 TopkDropoutStrategy 的核心逻辑作为备选
    """
    
    def __init__(
        self,
        *,
        agent: Optional[StrategyGenerationAgent] = None,
        signal: Union[Any, Tuple, List, Dict, str] = None,
        topk: int = 50,
        n_drop: int = 5,
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
        初始化 Agent 增强策略
        
        Args:
            agent: StrategyGenerationAgent 实例，如果为 None 则自动创建
            signal: Qlib 预测信号（兼容多种格式）
            topk: Top K 股票数量
            n_drop: 每次调仓丢弃数量
            method_sell: 卖出方法 ("bottom" 或 "random")
            method_buy: 买入方法 ("top" 或 "random")
            hold_thresh: 最小持仓天数
            only_tradable: 是否只考虑可交易股票
            forbid_all_trade_at_limit: 涨跌停时是否禁止所有交易
            risk_degree: 仓位比例 (0-1)
            use_agent_decision: 是否使用 Agent 决策（False 则回退到规则策略）
            use_llm_decision: 是否使用 LLM 增强决策（含新闻分析）
            llm_config: LLM 配置字典，包含 provider, api_key, model 等
                示例: {"provider": "qwen", "api_key": "xxx", "model": "qwen-plus"}
            use_env_llm_config: 是否从环境变量/配置文件加载 LLM 配置（优先级低于 llm_config）
            news_data_path: 新闻数据路径（可以是目录路径或单个JSON文件路径）
                           - 目录路径：按月份自动加载，文件名格式为 eastmoney_news_YYYY_processed_YYYY_MM.json
                           - 文件路径：兼容旧版本，加载单个JSON文件
                           数据格式: {date: {stock_code: [news_list]}}
            news_batch_size: 每批次分析的股票数量（用于新闻分析时的批处理）
            **kwargs: 其他参数传递给基类
        """
        if QLIB_AVAILABLE:
            super().__init__(**kwargs)
        
        # 策略参数
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
        
        # 初始化 Signal
        if QLIB_AVAILABLE and signal is not None:
            self.signal = create_signal_from(signal)
        else:
            self.signal = signal
        
        # 初始化 Agent
        if agent is not None:
            self.agent = agent
        else:
            # 自动创建 Agent
            config = StrategyConfig(
                topk=topk,
                n_drop=n_drop,
                max_turnover=0.3,
                open_cost=kwargs.get("open_cost", 0.0005),
                close_cost=kwargs.get("close_cost", 0.0015),
                min_cost=kwargs.get("min_cost", 5),
                limit_threshold=kwargs.get("limit_threshold", 0.095),
            )
            
            # 创建 LLM 服务（如果启用）
            llm_service = None
            if use_llm_decision:
                llm_service = self._create_llm_service(
                    llm_config=llm_config,
                    use_env_config=use_env_llm_config
                )
                if llm_service is None:
                    print("[Warning] LLM 服务创建失败，将使用规则决策")
            
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
        创建 LLM 服务实例
        
        支持两种方式：
        1. 通过 llm_config 字典配置
        2. 通过环境变量/配置文件加载（use_env_config=True）
        
        Args:
            llm_config: LLM 配置字典
            use_env_config: 是否从环境变量加载配置
            
        Returns:
            LLM 服务实例或 None
        """
        try:
            from Agent.agent_factory import load_env_config, create_agent
            
            # 优先使用传入的配置
            if llm_config:
                config = llm_config
            elif use_env_config:
                # 从环境变量/配置文件加载
                config = load_env_config()
            else:
                print("[Warning] 未提供 LLM 配置，也未启用环境变量配置")
                return None
            
            if not config:
                print("[Warning] LLM 配置为空")
                return None
            
            # 创建 LLM Agent
            llm_service = create_agent(
                provider=config.get("provider", "qwen"),
                api_key=config.get("api_key"),
                model=config.get("model"),
                base_url=config.get("base_url"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens"),
            )
            
            print(f"[Info] LLM 服务创建成功: provider={config.get('provider')}, model={config.get('model')}")
            return llm_service
            
        except ImportError as e:
            print(f"[Warning] 无法导入 agent_factory: {e}")
            return None
        except Exception as e:
            print(f"[Warning] 创建 LLM 服务失败: {e}")
            return None
    
    def get_risk_degree(self, trade_step=None) -> float:
        """获取风险度（仓位比例）"""
        return self.risk_degree
    
    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策 - Qlib 回测框架的核心接口方法
        
        此方法会被 Qlib 回测框架在每个交易日调用
        
        参考 TopkDropoutStrategy.generate_trade_decision 的实现
        
        Args:
            execute_result: 上一次执行结果（通常为 None 或执行报告）
            
        Returns:
            TradeDecisionWO: Qlib 格式的交易决策对象
        """
        if not QLIB_AVAILABLE:
            # 模拟模式
            return self._generate_trade_decision_mock(execute_result)
        
        # =====================================================================
        # Step 1: 获取交易时间和预测信号
        # =====================================================================
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        logger.info(f"[Strategy] ========== trade_step={trade_step}, trade_time={trade_start_time} ==========")
        
        # 获取预测分数
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        # 处理 DataFrame 格式（只取第一列）
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        
        if pred_score is None:
            logger.warning(f"[Strategy] pred_score is None for trade_step={trade_step}, returning empty decision")
            return TradeDecisionWO([], self)
        
        logger.info(f"[Strategy] pred_score length={len(pred_score)}")
        
        # =====================================================================
        # Step 2: 获取当前持仓信息
        # =====================================================================
        current_position: Position = copy.deepcopy(self.trade_position)
        cash = current_position.get_cash()
        current_stock_list = current_position.get_stock_list()
        
        # 转换为 Agent 需要的格式
        current_holdings = self._position_to_holdings(current_position)
        
        # =====================================================================
        # Step 3: 获取 Top K 推荐
        # =====================================================================
        top_k_recommendations = self._get_top_k_from_signal(pred_score)
        
        # =====================================================================
        # Step 4: 获取市场数据
        # =====================================================================
        all_stocks = list(set(current_stock_list) | set([s for s, _ in top_k_recommendations]))
        market_data = self._get_market_data(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            stock_list=all_stocks
        )
        
        # =====================================================================
        # Step 5: 调用 Agent 生成交易决策
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
        # Step 6: 将 Agent 决策转换为 Qlib Order 列表
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
        将 Qlib Position 对象转换为持仓权重字典
        
        Args:
            position: Qlib Position 对象
            
        Returns:
            持仓权重字典 {stock_code: weight}
        """
        if not QLIB_AVAILABLE:
            return {}
        
        holdings = {}
        stock_list = position.get_stock_list()
        
        # 计算总价值
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
        从预测信号中获取 Top K 推荐股票
        
        Args:
            pred_score: 预测分数 Series，index 为股票代码
            
        Returns:
            Top K 推荐列表 [(stock_code, score), ...]
        """
        if pred_score is None or len(pred_score) == 0:
            return []
        
        # 按分数降序排序，取 Top K
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
        获取市场数据（价格、成交量、涨跌停信息等）
        
        Args:
            trade_start_time: 交易开始时间
            trade_end_time: 交易结束时间
            stock_list: 股票列表
            
        Returns:
            市场数据字典 {stock_code: {price, volume, price_change_pct, is_tradable}}
        """
        if not QLIB_AVAILABLE:
            return {}
        
        market_data = {}
        
        for stock in stock_list:
            try:
                # 检查是否可交易
                is_tradable = self.trade_exchange.is_stock_tradable(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time
                )
                
                # 获取价格
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
                    method="sum"  # 或 "ts_data_last" 获取最后值
                )
                # $change 字段在 Exchange 初始化时已加载，表示价格变化
                change = self.trade_exchange.get_quote_info(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    field="$change",
                    method="ts_data_last"
                )
                # $change 已经是百分比形式（例如 0.05 表示 5%）
                price_change_pct = change if change is not None else 0.0
                
                market_data[stock] = {
                    "price": price,
                    "is_tradable": is_tradable,
                    "price_change_pct": price_change_pct,  # 如果需要可以计算
                    "volume": volume,  # 简化处理
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
        将 Agent 的 TradeDecision 转换为 Qlib Order 列表
        
        如果 agent_decision 为 None，则回退到类似 TopkDropoutStrategy 的规则逻辑
        
        Args:
            agent_decision: Agent 的交易决策
            current_position: 当前持仓
            pred_score: 预测分数
            trade_start_time: 交易开始时间
            trade_end_time: 交易结束时间
            cash: 当前现金
            
        Returns:
            Qlib Order 列表
        """
        sell_order_list = []
        buy_order_list = []
        
        if agent_decision is None:
            # 回退到规则策略
            return self._generate_orders_rule_based(
                current_position, pred_score, trade_start_time, trade_end_time, cash
            )
        
        # =====================================================================
        # 处理卖出订单
        # =====================================================================
        for signal in agent_decision.sell_list:
            stock_code = signal.stock_code
            
            # 检查是否可交易
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock_code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            # 检查持仓天数（先检查股票是否在持仓中）
            if current_position.get_stock_amount(code=stock_code) == 0:
                continue  # 如果持仓为0，跳过
            time_per_step = self.trade_calendar.get_freq()
            if current_position.get_stock_count(stock_code, bar=time_per_step) < self.hold_thresh:
                continue
            # 检查持仓天数
            # time_per_step = self.trade_calendar.get_freq()
            # if current_position.get_stock_count(stock_code, bar=time_per_step) < self.hold_thresh:
            #     continue
            
            # 获取卖出数量
            sell_amount = current_position.get_stock_amount(code=stock_code)
            
            if sell_amount > 0:
                sell_order = Order(
                    stock_id=stock_code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                
                # 检查订单是否可执行
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    # 模拟执行以更新现金
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_position
                    )
                    cash += trade_val - trade_cost
        
        # =====================================================================
        # 处理买入订单
        # =====================================================================
        buy_stocks = [signal.stock_code for signal in agent_decision.buy_list]
        
        if len(buy_stocks) > 0:
            # 计算每只股票的买入金额
            value = cash * self.risk_degree / len(buy_stocks)
            
            for signal in agent_decision.buy_list:
                stock_code = signal.stock_code
                
                # 检查是否可交易
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=stock_code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
                ):
                    continue
                
                # 获取买入价格
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=stock_code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY
                )
                
                if buy_price <= 0:
                    continue
                
                # 计算买入数量
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
        规则基础的订单生成（回退方案，类似 TopkDropoutStrategy）
        
        当 Agent 决策不可用时使用此方法
        
        Args:
            current_position: 当前持仓
            pred_score: 预测分数
            trade_start_time: 交易开始时间
            trade_end_time: 交易结束时间
            cash: 当前现金
            
        Returns:
            Order 列表
        """
        sell_order_list = []
        buy_order_list = []
        
        current_stock_list = current_position.get_stock_list()
        
        # 当前持仓按分数排序
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False, kind='mergesort').index
        #last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        
        # 获取新推荐股票（不在当前持仓中的高分股票）
        today = list(
            pred_score[~pred_score.index.isin(last)]
            .sort_values(ascending=False)
            .index[:self.n_drop + self.topk - len(last)]
        )
        
        # 合并并排序
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        
        # 确定卖出股票（评分最低的 n_drop 只）
        sell = last[last.isin(list(comb)[-self.n_drop:])] if len(comb) > self.n_drop else pd.Index([])
        
        # 确定买入股票
        buy = today[:len(sell) + self.topk - len(last)]
        
        # 生成卖出订单
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
        
        # 生成买入订单
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
        模拟模式的交易决策生成（当 Qlib 不可用时）
        
        Args:
            execute_result: 执行结果
            
        Returns:
            模拟的 TradeDecisionWO 对象
        """
        return TradeDecisionWO([], self)


# =============================================================================
# 工厂函数
# =============================================================================

def create_strategy_generation_agent(
    llm_service=None,
    config: Optional[Dict[str, Any]] = None,
    use_llm: bool = False
) -> StrategyGenerationAgent:
    """
    创建策略生成 Agent 的工厂函数
    
    Args:
        llm_service: LLM 服务实例
        config: 策略配置字典
        use_llm: 是否使用 LLM 决策
        
    Returns:
        StrategyGenerationAgent 实例
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
    n_drop: int = 5,
    agent: Optional[StrategyGenerationAgent] = None,
    llm_service=None,
    use_llm: bool = False,
    use_agent_decision: bool = True,
    **kwargs
) -> AgentEnhancedStrategy:
    """
    创建 Agent 增强策略的工厂函数
    
    Args:
        signal: Qlib 预测信号（必需）
        topk: Top K 股票数量
        n_drop: 每次丢弃数量
        agent: StrategyGenerationAgent 实例（可选，如果为 None 则自动创建）
        llm_service: LLM 服务（可选，用于创建 Agent）
        use_llm: 是否使用 LLM 决策
        use_agent_decision: 是否使用 Agent 决策（False 则使用纯规则策略）
        **kwargs: 其他参数（如 risk_degree, hold_thresh 等）
        
    Returns:
        AgentEnhancedStrategy 实例
        
    Example:
        ```python
        # 基本用法
        strategy = create_agent_enhanced_strategy(
            signal=pred_signal,
            topk=50,
            n_drop=5
        )
        
        # 使用 LLM 增强
        strategy = create_agent_enhanced_strategy(
            signal=pred_signal,
            topk=50,
            n_drop=5,
            llm_service=my_llm,
            use_llm=True
        )
        
        # 自定义 Agent
        my_agent = create_strategy_generation_agent(config={"topk": 30})
        strategy = create_agent_enhanced_strategy(
            signal=pred_signal,
            agent=my_agent
        )
        ```
    """
    # 如果没有提供 Agent，自动创建
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
    n_drop: int = 5,
    use_agent: bool = True,
    use_llm: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
    use_env_llm_config: bool = False,
    news_data_path: Optional[str] = None,
    news_batch_size: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    生成用于 Qlib yaml 配置文件的策略配置字典
    
    可以直接用于 workflow yaml 文件中的 strategy 配置
    
    Args:
        topk: Top K 股票数量
        n_drop: 每次丢弃数量
        use_agent: 是否使用 Agent 决策
        use_llm: 是否使用 LLM 增强决策（含新闻分析）
        llm_config: LLM 配置字典（可选）
        use_env_llm_config: 是否从环境变量加载 LLM 配置
        news_data_path: 新闻数据路径（可以是目录路径或单个JSON文件路径）
                       - 目录路径：按月份自动加载，文件名格式为 eastmoney_news_YYYY_processed_YYYY_MM.json
                       - 文件路径：兼容旧版本，加载单个JSON文件
        news_batch_size: 每批次分析的股票数量
        **kwargs: 其他策略参数
        
    Returns:
        策略配置字典，可直接用于 Qlib yaml 配置
        
    Example:
        ```python
        # 基本配置（不使用 LLM）
        config = get_strategy_config_for_qlib(topk=50, n_drop=5)
        
        # 使用环境变量中的 LLM 配置（含新闻分析）
        config = get_strategy_config_for_qlib(
            topk=50, 
            n_drop=5, 
            use_llm=True,
            use_env_llm_config=True,
            news_data_path="/path/to/news_data.json",
            news_batch_size=10
        )
        
        # 使用自定义 LLM 配置
        config = get_strategy_config_for_qlib(
            topk=50,
            n_drop=5,
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
    
    # 添加 LLM 配置（如果提供）
    if llm_config:
        strategy_kwargs["llm_config"] = llm_config
    
    # 添加新闻数据配置（如果提供）
    if news_data_path:
        strategy_kwargs["news_data_path"] = news_data_path
    if news_batch_size != 10:
        strategy_kwargs["news_batch_size"] = news_batch_size
    
    return {
        "class": "AgentEnhancedStrategy",
        "module_path": "Agent.strategy_generation_agent",
        "kwargs": strategy_kwargs
    }

