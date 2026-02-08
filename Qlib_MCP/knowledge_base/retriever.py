"""
Paper retriever

Provides high-level retrieval interface, encapsulating Embedding and vector retrieval logic.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .config import KnowledgeBaseConfig, RetrievalConfig
from .embeddings import SPECTER2Embeddings, get_embeddings
from .vector_store import QdrantVectorStore, SearchResult
from .document_loader import PaperDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Retrieval result (includes formatted output)"""
    
    results: List[SearchResult]
    query: str
    total_found: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "total_found": self.total_found,
            "results": [r.to_dict() for r in self.results],
        }
    
    def format_markdown(self) -> str:
        """Format as Markdown text"""
        lines = [f"## Retrieval Results: {self.query}\n"]
        lines.append(f"Found {self.total_found} relevant results\n")
        
        for i, result in enumerate(self.results, 1):
            doc = result.document
            lines.append(f"### {i}. {doc.title}")
            lines.append(f"- **Journal**: {doc.journal}")
            lines.append(f"- **Publication Time**: {doc.publish_time}")
            lines.append(f"- **Similarity**: {result.score:.3f}")
            lines.append(f"- **Document Type**: {doc.doc_type}")
            
            if doc.section_title:
                lines.append(f"- **Section**: {doc.section_title}")
            
            if doc.key_topics:
                lines.append(f"- **Key Topics**: {', '.join(doc.key_topics[:5])}")
            
            # Abstract preview
            text_preview = doc.text[:300] + "..." if len(doc.text) > 300 else doc.text
            lines.append(f"\n> {text_preview}\n")
            
            if doc.pdf_link:
                lines.append(f"[View Original]({doc.pdf_link})\n")
            
            lines.append("---\n")
        
        return "\n".join(lines)
    
    def format_brief(self) -> str:
        """Format as brief text"""
        lines = [f"Query: {self.query} | Found: {self.total_found}\n"]
        
        for i, result in enumerate(self.results, 1):
            doc = result.document
            lines.append(
                f"{i}. [{result.score:.2f}] {doc.title} "
                f"({doc.journal}, {doc.publish_time})"
            )
        
        return "\n".join(lines)


class PaperRetriever:
    """
    Paper retriever
    
    Provides semantic retrieval functionality, supporting multiple filter conditions.
    
    Usage example:
        retriever = PaperRetriever(config)
        results = retriever.search(
            query="urban air pollution health impact",
            top_k=5,
            journals=["Nature*"],
            year_range=(2020, 2024),
        )
        print(results.format_markdown())
    """
    
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        """
        Initialize retriever
        
        Args:
            config: Knowledge base configuration
        """
        self.config = config or KnowledgeBaseConfig()
        
        # Lazy initialization of components
        self._embeddings = None
        self._vector_store = None
        
        logger.info("PaperRetriever initialized")
    
    @property
    def embeddings(self):
        """Get Embedding model (lazy loading)"""
        if self._embeddings is None:
            self._embeddings = get_embeddings(self.config.embedding)
        return self._embeddings
    
    @property
    def vector_store(self):
        """Get vector storage (lazy loading)"""
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(self.config.qdrant)
        return self._vector_store
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        journals: Optional[List[str]] = None,
        year_range: Optional[tuple] = None,
        doc_type: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Semantic retrieval
        
        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity threshold
            journals: Journal filter list
            year_range: Year range (start_year, end_year)
            doc_type: Document type filter ("abstract" | "section")
        
        Returns:
            Retrieval result
        """
        # Parameter defaults
        top_k = top_k or self.config.retrieval.default_top_k
        top_k = min(top_k, self.config.retrieval.max_top_k)
        # 如果调用方未显式传入，使用配置；允许 0.0 作为合法阈值
        if score_threshold is None:
            score_threshold = self.config.retrieval.score_threshold
        
        # 生成查询向量
        query_embedding = self.embeddings.embed_query(query)
        
        # 构建过滤条件（不包括年份，年份在结果中后处理）
        filter_conditions = {}
        if journals:
            filter_conditions["journal"] = journals[0]  # TODO: 支持多期刊
        if doc_type:
            filter_conditions["doc_type"] = doc_type
        
        # 如果有年份过滤，多取一些结果以便后处理
        fetch_k = top_k * 3 if year_range else top_k
        
        # 执行检索
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions if filter_conditions else None,
        )
        
        # 年份过滤（在结果中后处理）
        if year_range:
            start_year, end_year = year_range
            filtered_results = []
            for r in results:
                try:
                    # 从 publish_time 提取年份 (格式: "YYYY-MM-DD")
                    pub_year = int(r.document.publish_time[:4])
                    if start_year <= pub_year <= end_year:
                        filtered_results.append(r)
                except (ValueError, TypeError, IndexError):
                    # 如果无法解析年份，保留该结果
                    filtered_results.append(r)
            results = filtered_results[:top_k]
        
        return RetrievalResult(
            results=results,
            query=query,
            total_found=len(results),
        )
    
    def search_similar_papers(
        self,
        paper_title: str,
        paper_abstract: str = "",
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> RetrievalResult:
        """
        查找相似论文
        
        Args:
            paper_title: 论文标题
            paper_abstract: 论文摘要
            top_k: 返回结果数量
            exclude_self: 是否排除自身
        
        Returns:
            检索结果
        """
        # 使用 SPECTER2 格式
        query_text = f"{paper_title} [SEP] {paper_abstract}" if paper_abstract else paper_title
        query_embedding = self.embeddings.embed_query(query_text)
        
        # 多取一些结果以便排除自身
        fetch_k = top_k + 5 if exclude_self else top_k
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
        )
        
        # 排除自身
        if exclude_self:
            results = [
                r for r in results
                if r.document.title.lower() != paper_title.lower()
            ][:top_k]
        
        return RetrievalResult(
            results=results,
            query=f"Similar to: {paper_title}",
            total_found=len(results),
        )
    
    def search_by_topics(
        self,
        topics: List[str],
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        按主题检索
        
        Args:
            topics: 主题列表
            top_k: 返回结果数量
        
        Returns:
            检索结果
        """
        # 将主题组合成查询
        query = " ".join(topics)
        return self.search(query=query, top_k=top_k)
    
    def get_paper_context(
        self,
        paper_id: str,
        include_similar: bool = True,
        similar_count: int = 3,
    ) -> Dict[str, Any]:
        """
        获取论文的完整上下文
        
        Args:
            paper_id: 论文 ID
            include_similar: 是否包含相似论文
            similar_count: 相似论文数量
        
        Returns:
            论文上下文信息
        """
        # 获取论文的所有文档块
        paper_docs = self.vector_store.search_by_paper_id(paper_id)
        
        if not paper_docs:
            return {"error": f"Paper not found: {paper_id}"}
        
        # 提取基本信息
        first_doc = paper_docs[0].document
        context = {
            "paper_id": paper_id,
            "title": first_doc.title,
            "journal": first_doc.journal,
            "publish_time": first_doc.publish_time,
            "abstract": first_doc.abstract,
            "sections": [],
            "key_topics": first_doc.key_topics,
        }
        
        # 收集章节信息
        for result in paper_docs:
            doc = result.document
            if doc.doc_type == "section":
                context["sections"].append({
                    "title": doc.section_title,
                    "text": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                })
        
        # 查找相似论文
        if include_similar and first_doc.abstract:
            similar_results = self.search_similar_papers(
                paper_title=first_doc.title,
                paper_abstract=first_doc.abstract,
                top_k=similar_count,
            )
            context["similar_papers"] = [
                {
                    "title": r.document.title,
                    "journal": r.document.journal,
                    "similarity": r.score,
                }
                for r in similar_results.results
            ]
        
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return self.vector_store.get_collection_info()
    
    def close(self) -> None:
        """关闭资源"""
        if self._vector_store is not None:
            self._vector_store.close()


class HybridRetriever(PaperRetriever):
    """
    混合检索器（语义 + 关键词）
    
    结合语义检索和关键词匹配，提高检索准确性。
    """
    
    def __init__(
        self,
        config: Optional[KnowledgeBaseConfig] = None,
        keyword_weight: float = 0.3,
    ):
        """
        初始化混合检索器
        
        Args:
            config: 知识库配置
            keyword_weight: 关键词匹配权重 (0-1)
        """
        super().__init__(config)
        self.keyword_weight = keyword_weight
    
    def _keyword_match_score(
        self,
        query: str,
        document: PaperDocument,
    ) -> float:
        """计算关键词匹配分数"""
        query_terms = set(query.lower().split())
        
        # 在标题和摘要中查找匹配
        doc_text = f"{document.title} {document.abstract or ''} {document.text}".lower()
        
        matches = sum(1 for term in query_terms if term in doc_text)
        return matches / len(query_terms) if query_terms else 0
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他过滤参数
        
        Returns:
            检索结果（按混合分数排序）
        """
        # 先进行语义检索，获取更多候选
        result = super().search(query, top_k=(top_k or 5) * 2, **kwargs)
        
        # 计算混合分数
        scored_results = []
        for r in result.results:
            keyword_score = self._keyword_match_score(query, r.document)
            hybrid_score = (
                (1 - self.keyword_weight) * r.score +
                self.keyword_weight * keyword_score
            )
            scored_results.append(SearchResult(
                document=r.document,
                score=hybrid_score,
            ))
        
        # 重新排序
        scored_results.sort(key=lambda x: x.score, reverse=True)
        top_results = scored_results[:top_k or 5]
        
        return RetrievalResult(
            results=top_results,
            query=query,
            total_found=len(top_results),
        )


