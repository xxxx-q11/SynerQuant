"""MCP client service wrapper"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Try to import MCP client
try:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    SyncMCPClient = None


class MCPService:
    """MCP client service wrapper"""
    
    def __init__(self, server_path: Optional[str] = None):
        """
        Initialize MCP service
        
        Args:
            server_path: MCP server script path, defaults to Qlib_MCP/mcp_server_inline.py
        """
        self._client = None
        self._available = MCP_AVAILABLE
        
        if not self._available:
            print("Warning: MCP client module unavailable")
            return
        
        # Determine server path
        if server_path is None:
            server_path = self._get_default_server_path()
        
        # Initialize client
        self._init_client(server_path)
    
    def _get_default_server_path(self) -> Path:
        """Get default MCP server path"""
        current_dir = Path(__file__).parent.parent.parent
        return current_dir / "Qlib_MCP" / "mcp_server_inline.py"
    
    def _init_client(self, server_path: str) -> None:
        """Initialize MCP client"""
        server_path = Path(server_path)
        
        if not server_path.exists():
            print(f"Warning: MCP server script does not exist: {server_path}")
            return
        
        try:
            self._client = SyncMCPClient(str(server_path))
            print(f"MCP client initialized successfully: {server_path}")
        except Exception as e:
            print(f"Warning: MCP client initialization failed: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        """Check if MCP service is available"""
        return self._client is not None
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get available MCP tool list
        
        Returns:
            Tool list, each tool contains name, description, inputSchema
        """
        if not self.is_available:
            return []
        
        try:
            return self._client.list_tools()
        except Exception as e:
            print(f"Failed to get tool list: {e}")
            return []
    
    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        Call MCP tool
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments dictionary
            
        Returns:
            Tool execution result (text format)
            
        Raises:
            RuntimeError: MCP client not initialized or call failed
        """
        if not self.is_available:
            raise RuntimeError("MCP client not initialized")
        
        try:
            return self._client.call_tool(tool_name, arguments)
        except Exception as e:
            error_msg = f"Failed to call tool {tool_name}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def run_benchmark(self, yaml_path: str, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Qlib benchmark
        
        Args:
            yaml_path: Workflow configuration file path
            experiment_name: Experiment name (optional)
            
        Returns:
            Benchmark result dictionary
        """
        import json
        
        result = self.call_tool(
            "qlib_benchmark_runner",
            {
                "yaml_path": str(yaml_path),
                "experiment_name": experiment_name
            }
        )
        
        if isinstance(result, str):
            return json.loads(result)
        return result

