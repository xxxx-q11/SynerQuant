"""
Agent tool wrapper

Wraps knowledge base retrieval functionality as tools callable by AgentScope Agent.
"""

import json
import logging
from typing import Optional, List, Any, Dict

from .config import KnowledgeBaseConfig
from .retriever import PaperRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# Global retriever instance (singleton pattern)
_retriever: Optional[PaperRetriever] = None


def get_retriever(config: Optional[KnowledgeBaseConfig] = None) -> PaperRetriever:
    """
    Get retriever instance (singleton)
    
    Args:
        config: Knowledge base configuration
    
    Returns:
        PaperRetriever instance
    """
    global _retriever
    if _retriever is None:
        # Use configuration (supports environment variable override), defaults to global qdrant_data
        config = config or KnowledgeBaseConfig()
        # If fallback is explicitly enabled, automatically adjust vector dimension to match MiniLM
        if config.embedding.use_fallback:
            config.qdrant.vector_size = 384
        _retriever = PaperRetriever(config)
    return _retriever


def _is_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    import re
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def _translate_to_english(text: str) -> str:
    """
    Translate Chinese query to English (simple implementation)
    Note: This is a simple placeholder implementation, should use translation API in practice
    """
    # If contains Chinese, return prompt message
    if _is_chinese(text):
        # Should call translation service here, but to avoid adding dependencies, return original text and log warning
        logger.warning(f"Query contains Chinese characters: {text}. SPECTER2 embedding model works best with English queries.")
        # Return original text, but suggest Agent use English query
        return text
    return text


def query_knowledge_base(
    query: str,
    top_k: int = 5,
    journals: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    doc_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query academic paper knowledge base
    
    Performs semantic search in academic paper database, returns most relevant paper content.
    Supports filtering by journal, year, and document type.
    
    **Important**: SPECTER2 model works best with English queries. If query contains Chinese, please translate to English first.
    
    Args:
        query: Natural language query describing research topic or question (**recommended to use English**)
        top_k: Number of results to return, default 5, maximum 20
        journals: Journal filter (optional), supported journals include:
            - Nature, Nature Communications, Nature Human Behaviour
            - Nature Climate Change, Nature Sustainability
            - Scientific Reports, etc.
        year_start: Start year (optional), e.g. 2020
        year_end: End year (optional), e.g. 2024
        doc_type: Document type filter (optional):
            - "paper": Full paper level document (current indexing strategy: abstract + introduction merged into one chunk)
    
    Returns:
        Retrieval result dictionary containing:
        - status: Status ("success" or "error")
        - query: Original query
        - total_found: Number of results found
        - results: Result list, each item contains:
            - title: Paper title
            - journal: Journal name
            - publish_time: Publication time
            - score: Similarity score
            - text: Matched text content
            - section_title: Section title (if section)
            - key_topics: Key topics list
            - pdf_link: Original paper link
    
    Examples:
        # Simple query
        query_knowledge_base("urban air pollution health impact")
        
        # Query with filters
        query_knowledge_base(
            query="climate change economic policy",
            top_k=10,
            journals="Nature Climate Change",
            year_start=2020,
            year_end=2024
        )
        
        # Search abstracts only
        query_knowledge_base(
            query="machine learning in economics",
            doc_type="abstract"
        )
    """
    try:
        retriever = get_retriever()
        
        # Parameter validation
        top_k = min(max(1, top_k), 20)
        
        # Check query language: if contains Chinese, log warning
        if _is_chinese(query):
            logger.warning(
                f"Query contains Chinese characters: '{query}'. "
                "SPECTER2 embedding model is optimized for English. "
                "For best results, please translate the query to English before calling this function."
            )
            # Return error message requiring English query
            return {
                "status": "error",
                "error": (
                    "Query contains Chinese characters. SPECTER2 embedding model requires English queries. "
                    "Please translate your query to English before searching. "
                    f"Original query: {query}"
                ),
                "query": query,
                "total_found": 0,
                "results": [],
            }
        
        # Build year range
        year_range = None
        if year_start or year_end:
            year_range = (
                year_start or 2000,
                year_end or 2025,
            )
        
        # Journal list
        journal_list = [journals] if journals else None
        
        # Execute retrieval
        result = retriever.search(
            query=query,
            top_k=top_k,
            journals=journal_list,
            year_range=year_range,
            doc_type=doc_type,
        )
        
        # Format results
        formatted_results = []
        for r in result.results:
            doc = r.document
            formatted_results.append({
                "title": doc.title,
                "journal": doc.journal,
                "publish_time": doc.publish_time,
                "score": round(r.score, 3),
                "doc_type": doc.doc_type,
                "section_title": doc.section_title,
                "text": doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text,
                "key_topics": doc.key_topics[:5] if doc.key_topics else [],
                "pdf_link": doc.pdf_link,
                "paper_id": doc.paper_id,
            })
        
        return {
            "status": "success",
            "query": query,
            "total_found": result.total_found,
            "results": formatted_results,
        }
        
    except Exception as e:
        logger.error(f"Knowledge base query failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def get_paper_details(paper_id: str) -> Dict[str, Any]:
    """
    Get paper detailed information
    
    Get complete paper content and context information based on paper ID,
    including all section content and similar paper recommendations.
    
    Args:
        paper_id: Paper ID (obtained from retrieval results)
    
    Returns:
        Paper details dictionary containing:
        - paper_id: Paper ID
        - title: Title
        - journal: Journal
        - publish_time: Publication time
        - abstract: Abstract
        - sections: Section list
        - key_topics: Key topics
        - similar_papers: Similar papers list
    
    Example:
        get_paper_details("s41562-024-01817-8")
    """
    try:
        retriever = get_retriever()
        context = retriever.get_paper_context(
            paper_id=paper_id,
            include_similar=True,
            similar_count=3,
        )
        
        if "error" in context:
            return {
                "status": "error",
                "error": context["error"],
            }
        
        return {
            "status": "success",
            **context,
        }
        
    except Exception as e:
        logger.error(f"Failed to get paper details: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def find_similar_papers(
    title: str,
    abstract: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Find similar papers
    
    Given a paper's title and abstract, find most similar papers in knowledge base.
    Suitable for literature review, related work search, etc.
    
    Args:
        title: Paper title
        abstract: Paper abstract (optional, provides more precise retrieval when provided)
        top_k: Number of results to return
    
    Returns:
        Similar papers list
    
    Example:
        find_similar_papers(
            title="The impact of urban air pollution on public health",
            abstract="This study investigates...",
            top_k=5
        )
    """
    try:
        retriever = get_retriever()
        result = retriever.search_similar_papers(
            paper_title=title,
            paper_abstract=abstract,
            top_k=top_k,
        )
        
        return {
            "status": "success",
            "query_title": title,
            "total_found": result.total_found,
            "similar_papers": [
                {
                    "title": r.document.title,
                    "journal": r.document.journal,
                    "publish_time": r.document.publish_time,
                    "similarity": round(r.score, 3),
                    "abstract": r.document.abstract[:300] + "..." 
                        if r.document.abstract and len(r.document.abstract) > 300 
                        else r.document.abstract,
                    "pdf_link": r.document.pdf_link,
                }
                for r in result.results
            ],
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar papers: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def get_knowledge_base_stats() -> Dict[str, Any]:
    """
    Get knowledge base statistics
    
    Returns basic statistics of knowledge base, including number of indexed documents,
    list of supported journals, etc.
    
    Returns:
        Statistics dictionary
    """
    try:
        retriever = get_retriever()
        stats = retriever.get_stats()
        
        return {
            "status": "success",
            "collection_name": stats.get("name"),
            "total_documents": stats.get("points_count", 0),
            "vector_size": stats.get("vector_size"),
            "status": stats.get("status"),
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ============================================================================
# AgentScope tool wrapper (compatible with design_agent.py format)
# ============================================================================

def create_agentscope_tool():
    """
    Create AgentScope compatible tool function
    
    Returns a tool function that can be directly registered to AgentScope Toolkit.
    
    Usage:
        from knowledge_base.tool import create_agentscope_tool
        from agentscope.tool import Toolkit
        
        toolkit = Toolkit()
        toolkit.register_tool_function(create_agentscope_tool())
    """
    try:
        from agentscope.tool import ToolResponse
        from agentscope.message import TextBlock
        
        def query_academic_knowledge(
            query: str,
            top_k: int = 5,
            journals: str = "",
            year_start: int = 0,
            year_end: int = 0,
        ) -> ToolResponse:
            """
            Query academic paper knowledge base to get academic literature related to research question.
            
            This tool can help you:
            1. Understand background knowledge of a research field
            2. Find literature evidence supporting your research hypothesis
            3. Discover related research methods and theoretical frameworks
            
            Args:
                query: Natural language query describing research topic you want to understand
                top_k: Number of results to return (1-20)
                journals: Journal filter (optional)
                year_start: Start year (optional, 0 means no filter)
                year_end: End year (optional, 0 means no filter)
            
            Returns:
                Response containing related paper information
            """
            # Call core retrieval function
            result = query_knowledge_base(
                query=query,
                top_k=top_k,
                journals=journals if journals else None,
                year_start=year_start if year_start > 0 else None,
                year_end=year_end if year_end > 0 else None,
            )
            
            # Format as XML-style response
            if result["status"] == "success":
                results_json = json.dumps(result["results"], ensure_ascii=False, indent=2)
                response_text = (
                    f"<status>success</status>"
                    f"<query>{query}</query>"
                    f"<total_found>{result['total_found']}</total_found>"
                    f"<results>{results_json}</results>"
                )
            else:
                response_text = (
                    f"<status>error</status>"
                    f"<error>{result.get('error', 'Unknown error')}</error>"
                )
            
            return ToolResponse(
                content=[TextBlock(type="text", text=response_text)]
            )
        
        return query_academic_knowledge
        
    except ImportError:
        logger.warning("AgentScope not installed, returning plain function")
        return query_knowledge_base


# ============================================================================
# MCP tool wrapper (for MCP server integration)
# ============================================================================

def register_mcp_tools(mcp_server):
    """
    Register MCP tools
    
    Register knowledge base tools to MCP server.
    
    Args:
        mcp_server: FastMCP server instance
    
    Usage:
        from fastmcp import FastMCP
        from knowledge_base.tool import register_mcp_tools
        
        mcp = FastMCP("knowledge-base")
        register_mcp_tools(mcp)
    """
    
    @mcp_server.tool()
    def search_papers(
        query: str,
        top_k: int = 5,
        journal: str = "",
        year_start: int = 0,
        year_end: int = 0,
    ) -> str:
        """
        Search academic paper knowledge base
        
        Args:
            query: Search query
            top_k: Number of results to return
            journal: Journal filter
            year_start: Start year
            year_end: End year
        """
        result = query_knowledge_base(
            query=query,
            top_k=top_k,
            journals=journal if journal else None,
            year_start=year_start if year_start > 0 else None,
            year_end=year_end if year_end > 0 else None,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @mcp_server.tool()
    def get_paper_info(paper_id: str) -> str:
        """
        Get paper detailed information
        
        Args:
            paper_id: Paper ID
        """
        result = get_paper_details(paper_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @mcp_server.tool()
    def similar_papers(
        title: str,
        abstract: str = "",
        top_k: int = 5,
    ) -> str:
        """
        Find similar papers
        
        Args:
            title: Paper title
            abstract: Paper abstract
            top_k: Number of results to return
        """
        result = find_similar_papers(title, abstract, top_k)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @mcp_server.tool()
    def knowledge_base_info() -> str:
        """Get knowledge base statistics"""
        result = get_knowledge_base_stats()
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    logger.info("Registered MCP tools for knowledge base")


