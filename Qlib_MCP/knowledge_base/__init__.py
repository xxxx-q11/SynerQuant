"""
Academic Paper Knowledge Base
=============================

Academic paper knowledge base system based on SPECTER2 + Qdrant.

Main components:
- config: Configuration management
- embeddings: SPECTER2 Embedding model wrapper
- document_loader: Paper JSON document loader
- vector_store: Qdrant vector storage
- retriever: Semantic retriever
- indexer: Index builder
- tool: Agent tool wrapper

Usage example:
    from knowledge_base import KnowledgeBaseConfig, PaperRetriever
    
    config = KnowledgeBaseConfig()
    retriever = PaperRetriever(config)
    results = retriever.search("urban air pollution health impact", top_k=5)
"""

from .config import KnowledgeBaseConfig
from .embeddings import SPECTER2Embeddings
from .document_loader import PaperDocument, PaperDocumentLoader
from .vector_store import QdrantVectorStore
from .retriever import PaperRetriever
from .indexer import PaperIndexer

__version__ = "1.0.0"
__all__ = [
    "KnowledgeBaseConfig",
    "SPECTER2Embeddings",
    "PaperDocument",
    "PaperDocumentLoader",
    "QdrantVectorStore",
    "PaperRetriever",
    "PaperIndexer",
]


