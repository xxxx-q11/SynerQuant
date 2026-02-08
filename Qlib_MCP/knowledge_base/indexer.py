"""
Paper index builder

Responsible for loading papers, generating Embeddings, and building vector index.
"""

import logging
import time
from typing import List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from .config import KnowledgeBaseConfig
from .document_loader import PaperDocumentLoader, PaperDocument
from .embeddings import get_embeddings
from .vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Indexing statistics"""
    
    total_papers: int = 0
    total_documents: int = 0
    indexed_documents: int = 0
    failed_documents: int = 0
    elapsed_seconds: float = 0
    
    def to_dict(self) -> dict:
        return {
            "total_papers": self.total_papers,
            "total_documents": self.total_documents,
            "indexed_documents": self.indexed_documents,
            "failed_documents": self.failed_documents,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "docs_per_second": round(
                self.indexed_documents / self.elapsed_seconds, 2
            ) if self.elapsed_seconds > 0 else 0,
        }
    
    def __str__(self) -> str:
        return (
            f"IndexingStats("
            f"papers={self.total_papers}, "
            f"docs={self.total_documents}, "
            f"indexed={self.indexed_documents}, "
            f"failed={self.failed_documents}, "
            f"time={self.elapsed_seconds:.1f}s)"
        )


class PaperIndexer:
    """
    Paper index builder
    
    Responsible for:
    1. Loading paper documents
    2. Generating SPECTER2 Embeddings
    3. Storing to Qdrant vector database
    
    Usage example:
        indexer = PaperIndexer(config)
        stats = indexer.build_index()
        print(f"Indexed {stats.indexed_documents} documents")
    """
    
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        """
        Initialize index builder
        
        Args:
            config: Knowledge base configuration
        """
        self.config = config or KnowledgeBaseConfig()
        
        # Initialize components
        self._loader = None
        self._embeddings = None
        self._vector_store = None
        
        logger.info("PaperIndexer initialized")
    
    @property
    def loader(self) -> PaperDocumentLoader:
        """Get document loader"""
        if self._loader is None:
            self._loader = PaperDocumentLoader(
                data_dir=self.config.data_dir,
                config=self.config.chunking,
            )
        return self._loader
    
    @property
    def embeddings(self):
        """Get Embedding model"""
        if self._embeddings is None:
            self._embeddings = get_embeddings(self.config.embedding)
        return self._embeddings
    
    @property
    def vector_store(self) -> QdrantVectorStore:
        """Get vector storage"""
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(self.config.qdrant)
        return self._vector_store
    
    def build_index(
        self,
        limit: Optional[int] = None,
        journals: Optional[List[str]] = None,
        year_range: Optional[tuple] = None,
        batch_size: int = 32,
        clear_existing: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> IndexingStats:
        """
        Build index
        
        Args:
            limit: Maximum number of documents (for testing)
            journals: Journal filter
            year_range: Year range
            batch_size: Embedding batch size
            clear_existing: Whether to clear existing index
            progress_callback: Progress callback function
        
        Returns:
            Indexing statistics
        """
        stats = IndexingStats()
        start_time = time.time()
        
        # Clear existing index
        if clear_existing:
            logger.info("Clearing existing index...")
            self.vector_store.clear()
        
        # Collect documents
        logger.info("Loading documents...")
        documents: List[PaperDocument] = []
        paper_ids = set()
        
        for doc in self.loader.load_all(
            limit=limit,
            journals=journals,
            year_range=year_range,
        ):
            documents.append(doc)
            paper_ids.add(doc.paper_id)
        
        stats.total_documents = len(documents)
        stats.total_papers = len(paper_ids)
        
        if not documents:
            logger.warning("No documents to index")
            return stats
        
        logger.info(f"Loaded {stats.total_documents} documents from {stats.total_papers} papers")
        
        # Process in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Indexing"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]
            
            try:
                # Generate Embeddings
                texts = [doc.get_embedding_text() for doc in batch_docs]
                embeddings = self.embeddings.embed_documents(
                    texts,
                    show_progress=False,
                )
                
                # Store to vector database
                self.vector_store.add_documents(batch_docs, embeddings)
                stats.indexed_documents += len(batch_docs)
                
            except Exception as e:
                logger.error(f"Failed to index batch {batch_idx}: {e}")
                stats.failed_documents += len(batch_docs)
            
            # Progress callback
            if progress_callback:
                progress_callback(stats.indexed_documents, stats.total_documents)
        
        stats.elapsed_seconds = time.time() - start_time
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def update_paper(self, paper_json_path: str) -> bool:
        """
        Update index for a single paper
        
        Args:
            paper_json_path: 论文 JSON 文件路径
        
        Returns:
            是否成功
        """
        path = Path(paper_json_path)
        if not path.exists():
            logger.error(f"Paper not found: {path}")
            return False
        
        try:
            # 加载论文文档
            documents = list(self.loader.load_paper(path))
            if not documents:
                logger.warning(f"No documents extracted from {path}")
                return False
            
            # 删除旧索引
            paper_id = documents[0].paper_id
            self.vector_store.delete_by_paper_id(paper_id)
            
            # 生成 Embedding
            texts = [doc.get_embedding_text() for doc in documents]
            embeddings = self.embeddings.embed_documents(texts, show_progress=False)
            
            # 添加新索引
            self.vector_store.add_documents(documents, embeddings)
            
            logger.info(f"Updated index for paper {paper_id}: {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update paper {path}: {e}")
            return False
    
    def get_index_stats(self) -> dict:
        """获取索引统计信息"""
        return self.vector_store.get_collection_info()
    
    def close(self) -> None:
        """关闭资源"""
        if self._vector_store is not None:
            self._vector_store.close()


class IncrementalIndexer(PaperIndexer):
    """
    增量索引构建器
    
    支持增量更新，只索引新增或修改的论文。
    """
    
    def __init__(
        self,
        config: Optional[KnowledgeBaseConfig] = None,
        checkpoint_file: str = ".indexer_checkpoint.json",
    ):
        """
        初始化增量索引器
        
        Args:
            config: 知识库配置
            checkpoint_file: 检查点文件路径
        """
        super().__init__(config)
        self.checkpoint_file = Path(checkpoint_file)
        self._indexed_papers = self._load_checkpoint()
    
    def _load_checkpoint(self) -> set:
        """加载检查点"""
        if self.checkpoint_file.exists():
            import json
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                return set(data.get("indexed_papers", []))
        return set()
    
    def _save_checkpoint(self) -> None:
        """保存检查点"""
        import json
        with open(self.checkpoint_file, "w") as f:
            json.dump({"indexed_papers": list(self._indexed_papers)}, f)
    
    def build_index(
        self,
        **kwargs,
    ) -> IndexingStats:
        """
        增量构建索引
        
        只索引尚未处理的论文。
        """
        stats = IndexingStats()
        start_time = time.time()
        
        batch_size = kwargs.get("batch_size", 32)
        
        # 收集新文档
        documents: List[PaperDocument] = []
        new_paper_ids = set()
        
        for doc in self.loader.load_all(
            limit=kwargs.get("limit"),
            journals=kwargs.get("journals"),
            year_range=kwargs.get("year_range"),
        ):
            if doc.paper_id not in self._indexed_papers:
                documents.append(doc)
                new_paper_ids.add(doc.paper_id)
        
        stats.total_documents = len(documents)
        stats.total_papers = len(new_paper_ids)
        
        if not documents:
            logger.info("No new documents to index")
            return stats
        
        logger.info(f"Found {stats.total_documents} new documents from {stats.total_papers} papers")
        
        # 分批处理
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Indexing"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]
            
            try:
                texts = [doc.get_embedding_text() for doc in batch_docs]
                embeddings = self.embeddings.embed_documents(texts, show_progress=False)
                self.vector_store.add_documents(batch_docs, embeddings)
                stats.indexed_documents += len(batch_docs)
                
                # 更新检查点
                for doc in batch_docs:
                    self._indexed_papers.add(doc.paper_id)
                
            except Exception as e:
                logger.error(f"Failed to index batch {batch_idx}: {e}")
                stats.failed_documents += len(batch_docs)
        
        # 保存检查点
        self._save_checkpoint()
        
        stats.elapsed_seconds = time.time() - start_time
        logger.info(f"Incremental indexing complete: {stats}")
        
        return stats

