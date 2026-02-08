"""
Qdrant vector storage wrapper

Provides vector storage, retrieval, update, and other functions.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .config import QdrantConfig
from .document_loader import PaperDocument

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result"""
    document: PaperDocument
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
        }


class QdrantVectorStore:
    """
    Qdrant vector storage
    
    Supports three storage modes:
    - memory: In-memory storage (lost after restart)
    - local: Local file storage
    - remote: Remote Qdrant server
    
    Usage example:
        store = QdrantVectorStore(config)
        store.add_documents(documents, embeddings)
        results = store.search(query_embedding, top_k=5)
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize vector storage
        
        Args:
            config: Qdrant configuration
        """
        self.config = config or QdrantConfig()
        self._client = None
        
        logger.info(f"QdrantVectorStore initialized (mode={self.config.mode})")
    
    def _get_client(self):
        """Get Qdrant client (lazy initialization)"""
        if self._client is not None:
            return self._client
        
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Create client based on mode
        if self.config.mode == "memory":
            self._client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant storage")
        
        elif self.config.mode == "local":
            path = Path(self.config.local_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
            logger.info(f"Using local Qdrant storage: {path}")
        
        elif self.config.mode == "remote":
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
            )
            logger.info(f"Connected to remote Qdrant: {self.config.host}:{self.config.port}")
        
        else:
            raise ValueError(f"Unknown Qdrant mode: {self.config.mode}")
        
        # Ensure collection exists
        self._ensure_collection()
        
        return self._client
    
    def _ensure_collection(self) -> None:
        """Ensure collection exists"""
        from qdrant_client.models import Distance, VectorParams
        
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.config.collection_name not in collection_names:
            # Distance metric mapping
            distance_map = {
                "cosine": Distance.COSINE,
                "euclid": Distance.EUCLID,
                "dot": Distance.DOT,
            }
            
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance_map.get(
                        self.config.distance_metric,
                        Distance.COSINE,
                    ),
                ),
            )
            logger.info(f"Created collection: {self.config.collection_name}")
        else:
            logger.info(f"Collection exists: {self.config.collection_name}")
    
    def add_documents(
        self,
        documents: List[PaperDocument],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to vector storage
        
        Args:
            documents: Document list
            embeddings: Embedding vector matrix
            batch_size: Batch size
        
        Returns:
            Number of successfully added documents
        """
        from qdrant_client.models import PointStruct
        
        client = self._get_client()
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents count ({len(documents)}) != embeddings count ({len(embeddings)})"
            )
        
        points = []
        for doc, embedding in zip(documents, embeddings):
            point = PointStruct(
                id=hash(doc.doc_id) % (2**63),  # Convert to integer ID
                vector=embedding.tolist(),
                payload=doc.to_dict(),
            )
            points.append(point)
        
        # Upload in batches
        total_added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
            total_added += len(batch)
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Uploaded {total_added}/{len(points)} documents")
        
        logger.info(f"Successfully added {total_added} documents")
        return total_added
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Vector retrieval
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity threshold
            filter_conditions: Filter conditions
        
        Returns:
            List of search results
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        client = self._get_client()
        
        # Build filter
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            
            # Journal filtering
            if "journal" in filter_conditions:
                journal = filter_conditions["journal"]
                if isinstance(journal, str):
                    if journal.endswith("*"):
                        # Prefix matching (Qdrant doesn't directly support, needs multiple conditions)
                        pass  # TODO: Implement prefix matching
                    else:
                        must_conditions.append(
                            FieldCondition(
                                key="journal",
                                match=MatchValue(value=journal),
                            )
                        )
            
            # Year range filtering (via publish_year integer field)
            # Note: Need to add publish_year field to documents, or manually filter in results
            if "year_range" in filter_conditions:
                start_year, end_year = filter_conditions["year_range"]
                # Qdrant Range requires numeric type, so use integer year
                # Here we skip Qdrant filtering, manually filter in results
                pass  # TODO: Add publish_year field support
            
            # Document type filtering
            if "doc_type" in filter_conditions:
                must_conditions.append(
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=filter_conditions["doc_type"]),
                    )
                )
            
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)
        
        # Execute search
        results = client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )
        
        # Convert results
        search_results = []
        for hit in results:
            doc = PaperDocument.from_dict(hit.payload)
            search_results.append(SearchResult(document=doc, score=hit.score))
        
        return search_results
    
    def search_by_paper_id(self, paper_id: str) -> List[SearchResult]:
        """
        Retrieve all related documents by paper ID
        
        Args:
            paper_id: Paper ID
        
        Returns:
            All document chunks for this paper
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = self._get_client()
        
        results = client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            ),
            limit=100,
        )[0]
        
        return [
            SearchResult(
                document=PaperDocument.from_dict(point.payload),
                score=1.0,
            )
            for point in results
        ]
    
    def delete_by_paper_id(self, paper_id: str) -> int:
        """
        Delete all documents for specified paper
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Number of deleted documents
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = self._get_client()
        
        # 先查询数量
        docs = self.search_by_paper_id(paper_id)
        count = len(docs)
        
        if count > 0:
            client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted {count} documents for paper {paper_id}")
        
        return count
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取 collection 信息"""
        client = self._get_client()
        
        info = client.get_collection(self.config.collection_name)
        
        return {
            "name": self.config.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.name,
            "vector_size": self.config.vector_size,
            "distance_metric": self.config.distance_metric,
        }
    
    def clear(self) -> None:
        """清空 collection"""
        client = self._get_client()
        
        # 删除并重建 collection
        client.delete_collection(self.config.collection_name)
        self._ensure_collection()
        
        logger.info(f"Cleared collection: {self.config.collection_name}")
    
    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")

