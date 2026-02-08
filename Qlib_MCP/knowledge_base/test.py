"""
MCP literature search test script

Directly calls knowledge base literature search functionality (for testing)
"""

import json
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.tool import (
    query_knowledge_base,
    get_knowledge_base_stats,
    get_paper_details,
    find_similar_papers
)
from knowledge_base.config import KnowledgeBaseConfig


def setup_config(data_dir: str):
    """
    Configure knowledge base
    
    Args:
        data_dir: Paper library data directory path
    """
    # Create configuration, set data directory
    config = KnowledgeBaseConfig()
    config.data_dir = data_dir
    
    # Try to find metadata file (if exists)
    metadata_file = Path(data_dir) / "crawl_info_filter.json"
    if metadata_file.exists():
        config.metadata_file = str(metadata_file)
    else:
        # Try to find other possible metadata files
        json_files = list(Path(data_dir).glob("*.json"))
        if json_files:
            config.metadata_file = str(json_files[0])
            print(f"Using metadata file: {config.metadata_file}")
    
    return config


def test_search_papers(query: str, **kwargs):
    """
    Test paper search functionality
    
    Args:
        query: Search query (recommended to use English)
        **kwargs: Search parameters (top_k, journals, year_start, year_end, doc_type)
    """
    # Extract top_k from kwargs, default value is 5
    top_k = kwargs.pop('top_k', 5)
    
    print(f"\n{'='*60}")
    print(f"Testing paper search: {query}")
    print(f"{'='*60}\n")
    
    try:
        # Directly call query function
        result = query_knowledge_base(
            query=query,
            top_k=top_k,
            **kwargs
        )
        
        # Print results
        if result.get("status") == "success":
            print(f"✓ Search successful! Found {result.get('total_found', 0)} results\n")
            
            for i, paper in enumerate(result.get("results", []), 1):
                print(f"【Result {i}】")
                print(f"Title: {paper.get('title', 'N/A')}")
                print(f"Journal: {paper.get('journal', 'N/A')}")
                print(f"Publication Time: {paper.get('publish_time', 'N/A')}")
                print(f"Similarity Score: {paper.get('score', 0):.3f}")
                print(f"Document Type: {paper.get('doc_type', 'N/A')}")
                
                if paper.get('section_title'):
                    print(f"Section: {paper.get('section_title')}")
                
                text = paper.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"Content Preview: {preview}")
                
                if paper.get('pdf_link'):
                    print(f"PDF Link: {paper.get('pdf_link')}")
                
                if paper.get('paper_id'):
                    print(f"Paper ID: {paper.get('paper_id')}")
                
                print("-" * 60)
        else:
            print(f"✗ Search failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(f"\nDetailed error:\n{result['traceback']}")
    
    except Exception as e:
        print(f"✗ Error calling function: {e}")
        import traceback
        traceback.print_exc()


def test_knowledge_base_info():
    """Test getting knowledge base statistics"""
    print(f"\n{'='*60}")
    print("Testing knowledge base statistics retrieval")
    print(f"{'='*60}\n")
    
    try:
        result = get_knowledge_base_stats()
        
        if result.get("status") == "success":
            print("✓ Knowledge base information:")
            print(f"  Collection Name: {result.get('collection_name', 'N/A')}")
            print(f"  Total Documents: {result.get('total_documents', 0)}")
            print(f"  Vector Dimension: {result.get('vector_size', 'N/A')}")
            print(f"  Status: {result.get('status', 'N/A')}")
        else:
            print(f"✗ Failed to get information: {result.get('error', 'Unknown error')}")
            print("\nTip: If Qdrant version compatibility error occurs, try:")
            print("  1. Delete old Qdrant local storage directory (usually in qdrant_data directory)")
            print("  2. Re-run index building script")
    
    except Exception as e:
        print(f"✗ Error calling function: {e}")
        import traceback
        traceback.print_exc()


def test_get_paper_details(paper_id: str):
    """Test getting paper detailed information"""
    print(f"\n{'='*60}")
    print(f"Testing paper details retrieval: {paper_id}")
    print(f"{'='*60}\n")
    
    try:
        result = get_paper_details(paper_id)
        
        if result.get("status") == "success":
            print("✓ Paper details:")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  Journal: {result.get('journal', 'N/A')}")
            print(f"  Publication Time: {result.get('publish_time', 'N/A')}")
            
            abstract = result.get('abstract', '')
            if abstract:
                preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                print(f"  Abstract: {preview}")
            
            sections = result.get('sections', [])
            if sections:
                print(f"  Number of Sections: {len(sections)}")
                for i, section in enumerate(sections[:3], 1):  # Only show first 3 sections
                    print(f"    {i}. {section.get('title', 'N/A')}")
        else:
            print(f"✗ Failed to get details: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"✗ Error calling function: {e}")
        import traceback
        traceback.print_exc()


def fix_qdrant_storage(config):
    """
    Try to fix Qdrant storage compatibility issues
    
    Args:
        config: Knowledge base configuration
    """
    from pathlib import Path
    import shutil
    
    qdrant_path = Path(config.qdrant.local_path)
    
    if qdrant_path.exists():
        print(f"\nDetected Qdrant local storage: {qdrant_path}")
        print("If encountering version compatibility issues, you can try:")
        print(f"  1. Backup current storage: Rename {qdrant_path} to {qdrant_path}_backup")
        print(f"  2. Delete old storage: Delete {qdrant_path} directory")
        print("  3. Rebuild index")
        
        response = input("\nDo you want to delete old Qdrant storage and recreate? (y/N): ")
        if response.lower() == 'y':
            backup_path = qdrant_path.parent / f"{qdrant_path.name}_backup"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(str(qdrant_path), str(backup_path))
            print(f"✓ Backed up to: {backup_path}")
            print("Please re-run index building script")
            return True
    
    return False


def main():
    """Main function"""
    # Paper library address
    data_dir = r"C:\edge_download\python_project\Articles1219"
    
    print("=" * 60)
    print("Literature Search Test Script")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    
    # Check if directory exists
    if not Path(data_dir).exists():
        print(f"\n✗ Error: Data directory does not exist: {data_dir}")
        print("Please check if the path is correct")
        return
    
    try:
        # Configure knowledge base
        print("\nInitializing knowledge base...")
        config = setup_config(data_dir)
        
        # Check and try to fix Qdrant storage issues
        try:
            from knowledge_base.tool import get_retriever
            retriever = get_retriever(config)
            print("✓ Knowledge base initialized successfully")
        except Exception as e:
            error_msg = str(e)
            if "metadata" in error_msg and "Extra inputs" in error_msg:
                print("\n⚠ Detected Qdrant version compatibility issue")
                if not fix_qdrant_storage(config):
                    print("\nPlease manually handle Qdrant storage compatibility issues and retry")
                return
            else:
                raise
        
        # Test 1: Get knowledge base information
        test_knowledge_base_info()
        
        # Test 2: Search papers (using English queries works better)
        test_queries = [
            ("machine learning", {"top_k": 3}),
            ("climate change", {"top_k": 3}),
            ("artificial intelligence", {"top_k": 3}),
        ]
        
        for query, kwargs in test_queries:
            test_search_papers(query, **kwargs)
        
        # If search results are found, can test getting paper details
        # Note: Need to get paper_id from search results
        # test_get_paper_details("your-paper-id-here")
        
        print("\n" + "=" * 60)
        print("Testing completed!")
        print("=" * 60)
        print("\nTip: If you need to use via MCP protocol, please use MCP client (e.g., Claude Desktop)")
        print("     to connect to the configured MCP server.")
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()