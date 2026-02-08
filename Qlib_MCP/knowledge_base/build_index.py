"""
Index building script
Read papers from Articles1219 directory and build vector index
"""

import sys
from pathlib import Path

# Add project path (consistent with test.py)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config import KnowledgeBaseConfig
from knowledge_base.indexer import PaperIndexer

def main():
    # Paper library address (consistent with test.py)
    data_dir = r"C:\edge_download\python_project\Articles1219"
    
    print("=" * 60)
    print("Paper Index Building Script")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    
    # Check if directory exists
    if not Path(data_dir).exists():
        print(f"\n✗ Error: Data directory does not exist: {data_dir}")
        return
    
    # Create configuration
    config = KnowledgeBaseConfig()
    config.data_dir = data_dir
    
    # Try to find metadata file
    metadata_file = Path(data_dir) / "crawl_info_filter.json"
    if metadata_file.exists():
        config.metadata_file = str(metadata_file)
    
    print(f"\nQdrant storage path: {config.qdrant.local_path}")
    print(f"Collection name: {config.qdrant.collection_name}")
    
    # Create indexer
    print("\nInitializing indexer...")
    indexer = PaperIndexer(config)
    
    # Build index (clear existing index)
    print("\nStarting to build index...")
    print("Note: This will clear existing index and rebuild")
    
    stats = indexer.build_index(
        clear_existing=True,  # Clear existing index
        batch_size=32,        # Batch size
    )
    
    print("\n" + "=" * 60)
    print("Index building completed!")
    print("=" * 60)
    print(f"Total papers: {stats.total_papers}")
    print(f"Total documents: {stats.total_documents}")
    print(f"Successfully indexed: {stats.indexed_documents}")
    print(f"Failed count: {stats.failed_documents}")
    print(f"Time elapsed: {stats.elapsed_seconds:.1f} seconds")
    print(f"Speed: {stats.docs_per_second:.1f} documents/second")
    
    # Close resources
    indexer.close()

if __name__ == "__main__":
    main()