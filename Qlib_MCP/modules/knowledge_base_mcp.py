"""Knowledge Base MCP Service Module

Provides MCP interface for academic paper knowledge base, supports paper search functionality only.
"""

import sys
import os
from types import SimpleNamespace

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.tool import query_knowledge_base


def run(args: SimpleNamespace):
    """Run knowledge base search tool
    
    Args:
        args: SimpleNamespace containing the following attributes:
            - query: Search query (required, English recommended)
            - top_k: Number of results to return, default 5
            - journal: Journal filter (optional)
            - year_start: Start year (optional)
            - year_end: End year (optional)
    
    Returns:
        Retrieval result dictionary
    """
    query = getattr(args, 'query', '')
    top_k = getattr(args, 'top_k', 5)
    journal = getattr(args, 'journal', None)
    year_start = getattr(args, 'year_start', None)
    year_end = getattr(args, 'year_end', None)
    
    result = query_knowledge_base(
        query=query,
        top_k=top_k,
        journals=journal,
        year_start=year_start,
        year_end=year_end,
    )
    
    return result

