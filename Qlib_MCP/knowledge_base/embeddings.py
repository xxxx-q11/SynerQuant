"""
SPECTER2 Embedding model wrapper

SPECTER2 is an Embedding model designed by Allen AI specifically for academic papers,
supporting multiple task adapters (proximity, adhoc_query, classification).
"""

import os
import logging
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

"""
SPECTER2 Embedding model wrapper

SPECTER2 is an Embedding model designed by Allen AI specifically for academic papers,
supporting multiple task adapters (proximity, adhoc_query, classification).
"""

# Set HuggingFace mirror (for Mainland China access)
HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_MIRROR

# Local cache priority paths (ModelScope downloads)
LOCAL_MODEL_PATHS = {
    "sentence-transformers/allenai-specter": "/root/.cache/modelscope/sentence-transformers/allenai-specter",
    "allenai/specter2": "/root/.cache/modelscope/sentence-transformers/allenai-specter",
}


@dataclass
class EmbeddingResult:
    """Embedding result"""
    embeddings: np.ndarray
    texts: List[str]
    model_name: str


class SPECTER2Embeddings:
    """
    SPECTER2 academic paper Embedding model
    
    Supported adapters:
    - allenai/specter2: Base model
    - allenai/specter2_proximity: Paper similarity retrieval (recommended for RAG)
    - allenai/specter2_adhoc_query: Query-document matching
    - allenai/specter2_classification: Classification tasks
    
    Usage example:
        embeddings = SPECTER2Embeddings()
        vectors = embeddings.embed_documents(["paper title. paper abstract"])
        query_vector = embeddings.embed_query("research topic")
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize SPECTER2 Embedding model
        
        Args:
            config: Embedding configuration, if None then use default configuration
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._adapter_loaded = False
        
        # Lazy load model
        logger.info(f"SPECTER2Embeddings initialized (lazy loading)")
    
    def _load_model(self) -> None:
        """Load model (lazy loading) - using sentence-transformers"""
        if self._model is not None:
            return
        
        logger.info(f"Loading SPECTER2 model: {self.config.model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine device (sentence-transformers doesn't accept "auto")
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # Load model using sentence-transformers
            # Note: allenai/specter2 needs special handling, use SciBERT as alternative
            model_mapping = {
                "allenai/specter2": "sentence-transformers/allenai-specter",  # Original SPECTER
                "allenai/specter2_base": "sentence-transformers/allenai-specter",
                "allenai/specter2_proximity": "sentence-transformers/allenai-specter",
            }
            
            model_name = model_mapping.get(self.config.model_name, self.config.model_name)
            
            # If separate HF endpoint is configured, apply it first (otherwise use global HF_ENDPOINT)
            if getattr(self.config, "hf_endpoint", None):
                os.environ["HF_ENDPOINT"] = self.config.hf_endpoint  # type: ignore
            
            # Prefer local ModelScope cache
            local_path = LOCAL_MODEL_PATHS.get(model_name)
            load_path = local_path if local_path and os.path.exists(local_path) else model_name
            if local_path and os.path.exists(local_path):
                logger.info(f"Using local cached model: {local_path}")
            else:
                logger.info(f"Using remote/mirror model: {model_name}")
            
            logger.info(f"Using model: {load_path} on device: {device}")
            
            self._model = SentenceTransformer(load_path, device=device)
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load SPECTER2 model: {e}")
            raise
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed document list
        
        Args:
            texts: Document text list
            batch_size: Batch size, defaults to configuration value
            show_progress: Whether to show progress bar
        
        Returns:
            Embedding vector matrix, shape (n_docs, embedding_dim)
        """
        self._load_model()
        
        batch_size = batch_size or self.config.batch_size
        
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed single query
        
        Args:
            query: Query text
        
        Returns:
            Embedding vector, shape (embedding_dim,)
        """
        embeddings = self.embed_documents([query], show_progress=False)
        return embeddings[0]
    
    def embed_paper(
        self,
        title: str,
        abstract: str = "",
    ) -> np.ndarray:
        """
        Embed paper (using standard format)
        
        SPECTER2 recommended paper format: "title [SEP] abstract"
        
        Args:
            title: Paper title
            abstract: Paper abstract
        
        Returns:
            Embedding vector
        """
        # SPECTER2 recommended format
        text = f"{title} [SEP] {abstract}" if abstract else title
        return self.embed_query(text)
    
    @property
    def embedding_dimension(self) -> int:
        """Return Embedding dimension"""
        return 768  # SPECTER2 fixed output 768 dimensions
    
    def __repr__(self) -> str:
        return (
            f"SPECTER2Embeddings("
            f"model={self.config.model_name}, "
            f"adapter={self.config.adapter_name}, "
            f"device={self.config.device})"
        )


class FallbackEmbeddings:
    """
    Fallback Embedding model (used when SPECTER2 is unavailable)
    
    Uses sentence-transformers general model as alternative.
    """
    
    # ModelScope cache path mapping
    LOCAL_MODEL_PATHS = {
        "all-MiniLM-L6-v2": "/root/.cache/modelscope/sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2": "/root/.cache/modelscope/sentence-transformers/all-MiniLM-L6-v2",
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化回退模型
        
        Args:
            model_name: sentence-transformers 模型名称或本地路径
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"FallbackEmbeddings initialized with {model_name}")
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import os
            
            # 优先使用本地缓存路径
            local_path = self.LOCAL_MODEL_PATHS.get(self.model_name)
            if local_path and os.path.exists(local_path):
                logger.info(f"Loading from local cache: {local_path}")
                self._model = SentenceTransformer(local_path)
                return
            
            # 尝试从 HuggingFace 加载
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                raise
    
    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """对文档列表进行 Embedding"""
        self._load_model()
        return self._model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """对查询进行 Embedding"""
        self._load_model()
        return self._model.encode(
            query,
            normalize_embeddings=True,
        )
    
    @property
    def embedding_dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


def get_embeddings(config: Optional[EmbeddingConfig] = None) -> Union[SPECTER2Embeddings, FallbackEmbeddings]:
    """
    获取 Embedding 模型（带自动回退）
    
    Args:
        config: Embedding 配置
    
    Returns:
        Embedding 模型实例
    """
    config = config or EmbeddingConfig()
    
    # 如果配置了使用回退模型，直接使用 FallbackEmbeddings
    if config.use_fallback:
        model_name = config.model_name if "specter" not in config.model_name.lower() else "all-MiniLM-L6-v2"
        logger.info(f"Using fallback embeddings: {model_name}")
        return FallbackEmbeddings(model_name)
    
    try:
        return SPECTER2Embeddings(config)
    except Exception as e:
        logger.warning(f"Failed to load SPECTER2, falling back to MiniLM: {e}")
        return FallbackEmbeddings()
 