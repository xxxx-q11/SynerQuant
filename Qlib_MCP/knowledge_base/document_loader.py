"""
Paper document loader

Responsible for loading and parsing JSON format academic papers, and converting them to indexable document chunks.
"""

import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Generator
from datetime import datetime

from .config import ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class PaperDocument:
    """
    Paper document data structure
    
    Represents an indexable document chunk, containing text content and metadata.
    """
    
    # Unique identifier
    doc_id: str
    
    # Text content (for Embedding)
    text: str
    
    # Document type: "abstract" | "section" | "full"
    doc_type: str
    
    # Metadata
    paper_id: str
    title: str
    journal: str
    publish_time: str
    
    # Optional metadata
    section_title: Optional[str] = None
    section_index: Optional[int] = None
    abstract: Optional[str] = None
    
    # Additional fields
    key_topics: List[str] = field(default_factory=list)
    open_access: bool = True
    pdf_link: Optional[str] = None
    
    # CAMP extraction results (if available)
    camp_context: Optional[str] = None
    camp_independent_var: Optional[str] = None
    camp_dependent_var: Optional[str] = None
    camp_mechanism: Optional[str] = None
    camp_pattern: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for storage)"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "doc_type": self.doc_type,
            "paper_id": self.paper_id,
            "title": self.title,
            "journal": self.journal,
            "publish_time": self.publish_time,
            "section_title": self.section_title,
            "section_index": self.section_index,
            "abstract": self.abstract,
            "key_topics": self.key_topics,
            "open_access": self.open_access,
            "pdf_link": self.pdf_link,
            "camp_context": self.camp_context,
            "camp_independent_var": self.camp_independent_var,
            "camp_dependent_var": self.camp_dependent_var,
            "camp_mechanism": self.camp_mechanism,
            "camp_pattern": self.camp_pattern,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperDocument":
        """Create document from dictionary"""
        return cls(**data)
    
    def get_embedding_text(self) -> str:
        """Get text for Embedding (SPECTER2 format)"""
        if self.doc_type == "abstract":
            return f"{self.title} [SEP] {self.text}"
        elif self.doc_type == "section":
            return f"{self.title} - {self.section_title} [SEP] {self.text}"
        else:
            return f"{self.title} [SEP] {self.text}"


class PaperDocumentLoader:
    """
    Paper document loader
    
    Responsible for:
    1. Traversing paper directory structure
    2. Parsing JSON files
    3. Document chunking according to configuration
    4. Generating PaperDocument objects
    
    Usage example:
        loader = PaperDocumentLoader(config)
        for doc in loader.load_all():
            print(doc.title, doc.doc_type)
    """
    
    def __init__(
        self,
        data_dir: str,
        config: Optional[ChunkingConfig] = None,
    ):
        """
        Initialize loader
        
        Args:
            data_dir: Paper data root directory
            config: Chunking configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config or ChunkingConfig()
        
        # Compile regex patterns for skipping sections
        self._skip_patterns = [
            re.compile(pattern) for pattern in self.config.skip_sections
        ]
        
        logger.info(f"PaperDocumentLoader initialized: {self.data_dir}")
    
    def _should_skip_section(self, section_title: str) -> bool:
        """Check if this section should be skipped"""
        for pattern in self._skip_patterns:
            if pattern.search(section_title):
                return True
        return False

    def _is_intro_section(self, section_title: str) -> bool:
        """
        Determine if it's an introduction section
        
        Currently uses simple rule:
        - Title contains word "Introduction" (case-insensitive)
        """
        if not section_title:
            return False
        return re.search(r"(?i)\\bintroduction\\b", section_title) is not None
    
    def _parse_paper_json(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Parse paper JSON file"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON {json_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error reading {json_path}: {e}")
            return None
    
    def _extract_camp(self, paper_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Extract CAMP structured information"""
        camp = paper_data.get("Extract_CAMP", {})
        return {
            "camp_context": camp.get("Context"),
            "camp_independent_var": camp.get("A (Independent Variable)"),
            "camp_dependent_var": camp.get("B (Dependent Variable)"),
            "camp_mechanism": camp.get("Mechanism"),
            "camp_pattern": camp.get("Pattern"),
        }
    
    def _extract_key_topics(self, paper_data: Dict[str, Any]) -> List[str]:
        """Extract key topics"""
        judge = paper_data.get("Qwen3JudgeField", {})
        return judge.get("key_topics", [])

    def _get_intro_text(self, paper_data: Dict[str, Any]) -> str:
        """Get introduction (Introduction) section text, return full text of first introduction if exists, otherwise return empty string"""
        sections = paper_data.get("Sections", []) or []
        for section in sections:
            title = section.get("title", "") or ""
            if self._is_intro_section(title):
                return (section.get("text") or "").strip()
        return ""
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """Fixed size chunking"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
        
        return chunks
    
    def load_paper(self, json_path: Path) -> Generator[PaperDocument, None, None]:
        """
        加载单篇论文

        当前策略：**每篇论文只生成一个文档块**，内容为「摘要 + 引言」拼接。
        这样可以：
        - 保持索引结构简单（每篇 1 条记录）
        - 兼顾高召回（引言通常包含研究动机和背景）
        """
        paper_data = self._parse_paper_json(json_path)
        if paper_data is None:
            return

        paper_id = paper_data.get("id", "")
        title = paper_data.get("title", "") or ""
        journal = paper_data.get("journal", "") or ""
        publish_time = paper_data.get("publish_time", "") or ""

        abstract = (paper_data.get("Abstract") or "").strip()
        intro = self._get_intro_text(paper_data)

        # 如果摘要和引言都为空，则跳过
        if not abstract and not intro:
            return

        parts = []
        if abstract:
            parts.append("Abstract:\n" + abstract)
        if intro:
            parts.append("Introduction:\n" + intro)

        main_text = "\n\n".join(parts)
        camp = self._extract_camp(paper_data)

        yield PaperDocument(
            doc_id=f"{paper_id}_paper",
            text=main_text,
            doc_type="paper",
            paper_id=paper_id,
            title=title,
            journal=journal,
            publish_time=publish_time,
            abstract=abstract or None,
            key_topics=self._extract_key_topics(paper_data),
            open_access=paper_data.get("open_access", True),
            pdf_link=paper_data.get("pdf_link"),
            **camp,
        )
    
    def find_paper_files(self) -> Generator[Path, None, None]:
        """
        查找所有论文 JSON 文件
        
        Yields:
            JSON 文件路径
        """
        for json_path in self.data_dir.rglob("article.json"):
            yield json_path
    
    def load_all(
        self,
        limit: Optional[int] = None,
        journals: Optional[List[str]] = None,
        year_range: Optional[tuple] = None,
    ) -> Generator[PaperDocument, None, None]:
        """
        加载所有论文
        
        Args:
            limit: 最大加载数量（用于测试）
            journals: 期刊过滤列表
            year_range: 年份范围 (start_year, end_year)
        
        Yields:
            PaperDocument 对象
        """
        count = 0
        paper_count = 0
        
        for json_path in self.find_paper_files():
            # 期刊过滤
            if journals:
                journal_match = False
                for journal in journals:
                    if journal.lower() in str(json_path).lower():
                        journal_match = True
                        break
                if not journal_match:
                    continue
            
            # 年份过滤（从路径解析）
            if year_range:
                path_str = str(json_path)
                year_match = re.search(r"(\d{4})-(?:article|letter)", path_str)
                if year_match:
                    year = int(year_match.group(1))
                    if not (year_range[0] <= year <= year_range[1]):
                        continue
            
            for doc in self.load_paper(json_path):
                yield doc
                count += 1
                
                if limit and count >= limit:
                    logger.info(f"Reached limit of {limit} documents")
                    return
            
            paper_count += 1
            if paper_count % 100 == 0:
                logger.info(f"Processed {paper_count} papers, {count} documents")
        
        logger.info(f"Total: {paper_count} papers, {count} documents")
    
    def count_papers(self) -> int:
        """统计论文数量"""
        return sum(1 for _ in self.find_paper_files())
    
    def get_journals(self) -> List[str]:
        """获取所有期刊名称"""
        journals = set()
        for path in self.data_dir.iterdir():
            if path.is_dir():
                journals.add(path.name)
        return sorted(journals)


class MetadataLoader:
    """
    元数据加载器
    
    从 crawl_info_filter.json 加载论文元数据索引。
    """
    
    def __init__(self, metadata_file: str):
        """
        初始化元数据加载器
        
        Args:
            metadata_file: 元数据 JSON 文件路径
        """
        self.metadata_file = Path(metadata_file)
        self._metadata = None
    
    def load(self) -> Dict[str, Any]:
        """加载元数据"""
        if self._metadata is None:
            logger.info(f"Loading metadata from {self.metadata_file}")
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        return self._metadata
    
    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """获取指定论文的元信息"""
        metadata = self.load()
        
        for journal, years in metadata.items():
            for year_key, year_data in years.items():
                articles = year_data.get("articles", [])
                for article in articles:
                    if article.get("id") == paper_id:
                        return article
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取元数据统计信息"""
        metadata = self.load()
        
        stats = {
            "journals": {},
            "total_papers": 0,
            "years": set(),
        }
        
        for journal, years in metadata.items():
            journal_count = 0
            for year_key, year_data in years.items():
                count = year_data.get("filter_article_count", 0)
                journal_count += count
                
                # 提取年份
                year_match = re.search(r"(\d{4})", year_key)
                if year_match:
                    stats["years"].add(int(year_match.group(1)))
            
            stats["journals"][journal] = journal_count
            stats["total_papers"] += journal_count
        
        stats["years"] = sorted(stats["years"])
        return stats
