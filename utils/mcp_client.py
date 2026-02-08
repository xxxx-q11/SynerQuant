"""MCP client utility class
Used to call tools provided by MCP server through MCP protocol
"""
import asyncio
import json
import os
import subprocess
import threading  # New
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import method
        from mcp.client import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        import mcp.types as types
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False
        print("Warning: mcp library not installed, MCP client functionality unavailable. Please install: pip install mcp")


class MCPClient:
    """MCP client, used to call tools provided by MCP server"""
    
    def __init__(self, server_script_path: str, working_dir: Optional[str] = None):
        """
        Initialize MCP client
        
        Args:
            server_script_path: MCP server script path (e.g. mcp_server_inline.py)
            working_dir: Working directory, defaults to server script directory
        """
        if not MCP_AVAILABLE:
            raise ImportError("mcp library not installed, cannot use MCP client")
        
        self.server_script_path = Path(server_script_path).resolve()
        if not self.server_script_path.exists():
            raise FileNotFoundError(f"MCP server script does not exist: {server_script_path}")
        
        self.working_dir = working_dir or str(self.server_script_path.parent)
        self.session: Optional[ClientSession] = None
        self._tools_cache: Optional[List[types.Tool]] = None
    
    async def connect(self):
        """Connect to MCP server"""
        try:
            # Create server parameters
            if isinstance(StdioServerParameters, type):
                server_params = StdioServerParameters(
                    command="python",
                    args=[str(self.server_script_path)],
                    env=os.environ.copy()
                )
            else:
                # If StdioServerParameters is a function
                server_params = {
                    "command": "python",
                    "args": [str(self.server_script_path)],
                    "env": os.environ.copy()
                }
            
            # stdio_client returns async context manager, need to use async with
            # But we need to maintain connection, so enter context manager first to get stream
            stdio_context = stdio_client(server_params)
            stdio_transport = await stdio_context.__aenter__()
            
            # Save context manager for proper exit in disconnect
            self._stdio_context = stdio_context
            
            # Handle based on return type
            if isinstance(stdio_transport, tuple):
                read_stream, write_stream = stdio_transport
            else:
                # If returns single object, try to get read and write attributes
                read_stream = getattr(stdio_transport, 'read', stdio_transport)
                write_stream = getattr(stdio_transport, 'write', stdio_transport)
            
            self.session = ClientSession(read_stream, write_stream)
            await self.session.initialize()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass
            self.session = None
        
        # Exit stdio context manager
        if hasattr(self, '_stdio_context'):
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except:
                pass
            delattr(self, '_stdio_context')
    
    async def list_tools(self, use_cache: bool = True) -> List[types.Tool]:
        """
        Get available tool list
        
        Args:
            use_cache: Whether to use cache
            
        Returns:
            Tool list
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache
        
        if not self.session:
            await self.connect()
        
        result = await self.session.list_tools()
        self._tools_cache = result.tools
        return result.tools
    
    async def call_tool(
        self, 
        name: str, 
        arguments: Optional[Dict[str, Any]] = None
    ) -> List[types.TextContent]:
        """
        Call tool
        
        Args:
            name: Tool name
            arguments: Tool parameter dictionary
            
        Returns:
            Tool execution result
        """
        if not self.session:
            await self.connect()
        
        result = await self.session.call_tool(name, arguments or {})
        return result.content
    
    def get_tool_schema(self, tool: types.Tool) -> Dict[str, Any]:
        """
        Get tool's JSON Schema
        
        Args:
            tool: Tool object
            
        Returns:
            JSON Schema dictionary
        """
        return tool.inputSchema if hasattr(tool, 'inputSchema') else {}
    
    def format_tool_info(self, tool: types.Tool) -> str:
        """
        Format tool information as string
        
        Args:
            tool: Tool object
            
        Returns:
            Formatted tool information string
        """
        schema = self.get_tool_schema(tool)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        info = f"Tool name: {tool.name}\n"
        info += f"Description: {tool.description}\n"
        
        if properties:
            info += "Parameters:\n"
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                default = param_info.get("default", None)
                
                info += f"  - {param_name} ({param_type})"
                if is_required:
                    info += " [Required]"
                if default is not None:
                    info += f" [Default: {default}]"
                if param_desc:
                    info += f": {param_desc}"
                info += "\n"
        
        return info
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# class SyncMCPClient:
#     """Synchronous version of MCP client wrapper - runs event loop in separate thread"""
    
#     def __init__(self, server_script_path: str, working_dir: Optional[str] = None):
#         """
#         Initialize synchronous MCP client
        
#         Args:
#             server_script_path: MCP server script path
#             working_dir: Working directory
#         """
#         self.client = MCPClient(server_script_path, working_dir)
#         self._loop: Optional[asyncio.AbstractEventLoop] = None
#         self._thread: Optional[threading.Thread] = None
#         self._ensure_loop()
    
#     def _ensure_loop(self):
#         """Ensure event loop runs in separate thread"""
#         if self._loop is None or not self._loop.is_running():
#             def run_loop(loop):
#                 """Run event loop in thread"""
#                 asyncio.set_event_loop(loop)
#                 loop.run_forever()
            
#             self._loop = asyncio.new_event_loop()
#             self._thread = threading.Thread(
#                 target=run_loop, 
#                 args=(self._loop,), 
#                 daemon=True,
#                 name="MCPClientLoop"
#             )
#             self._thread.start()
#             # Wait for loop to start
#             import time
#             time.sleep(0.1)
    
#     def _run_async(self, coro, timeout=30.0):
#         """
#         Run async coroutine in event loop of separate thread
        
#         Args:
#             coro: Coroutine object
#             timeout: Timeout in seconds
            
#         Returns:
#             Return value of coroutine
#         """
#         if self._loop is None or not self._loop.is_running():
#             raise RuntimeError("Event loop not running")
        
#         future = asyncio.run_coroutine_threadsafe(coro, self._loop)
#         try:
#             return future.result(timeout=timeout)
#         except TimeoutError:
#             raise RuntimeError(f"Operation timed out ({timeout} seconds)")
#         except Exception as e:
#             raise RuntimeError(f"Failed to execute async operation: {e}") from e
    
#     def list_tools(self, use_cache: bool = True) -> List[Dict[str, Any]]:
#         """
#         Synchronously get tool list
        
#         Args:
#             use_cache: Whether to use cache
            
#         Returns:
#             Tool dictionary list
#         """
#         tools = self._run_async(self.client.list_tools(use_cache), timeout=30.0)
#         # Convert to dictionary format for easier use
#         return [
#             {
#                 "name": tool.name,
#                 "description": tool.description,
#                 "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
#             }
#             for tool in tools
#         ]
    
#     def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
#         """
#         Synchronously call tool
        
#         Args:
#             name: Tool name
#             arguments: Tool arguments
            
#         Returns:
#             Tool execution result (text format)
#         """
#         result = self._run_async(self.client.call_tool(name, arguments), timeout=60.0)
        
#         # Extract text content
#         texts = []
#         for content in result:
#             if hasattr(content, 'text'):
#                 texts.append(content.text)
#             elif isinstance(content, dict) and 'text' in content:
#                 texts.append(content['text'])
        
#         return "\n".join(texts) if texts else ""
    
#     def get_tool_info(self, tool_name: str) -> Optional[str]:
#         """
#         Get detailed information of specified tool
        
#         Args:
#             tool_name: Tool name
            
#         Returns:
#             Tool information string, returns None if tool doesn't exist
#         """
#         tools = self.list_tools()
#         for tool_dict in tools:
#             if tool_dict["name"] == tool_name:
#                 # Create temporary Tool object for formatting
#                 tool = types.Tool(
#                     name=tool_dict["name"],
#                     description=tool_dict["description"],
#                     inputSchema=tool_dict.get("inputSchema", {})
#                 )
#                 return self.client.format_tool_info(tool)
#         return None
    
#     def close(self):
#         """Close client connection"""
#         try:
#             if self.client.session:
#                 self._run_async(self.client.disconnect(), timeout=5.0)
#         except Exception as e:
#             print(f"Warning: Error disconnecting: {e}", file=sys.stderr)
        
#         # Stop event loop
#         if self._loop and self._loop.is_running():
#             self._loop.call_soon_threadsafe(self._loop.stop)
        
#         # Wait for thread to end
#         if self._thread and self._thread.is_alive():
#             self._thread.join(timeout=2.0)
    
#     def __del__(self):
#         """Destructor, ensure resource cleanup"""
#         try:
#             self.close()
#         except:
#             pass
class SyncMCPClient:
    """Synchronous version of MCP client wrapper - creates new connection for each call"""
    
    def __init__(self, server_script_path: str, working_dir: Optional[str] = None):
        """
        Initialize synchronous MCP client
        
        Args:
            server_script_path: MCP server script path
            working_dir: Working directory
        """
        if not MCP_AVAILABLE:
            raise ImportError("mcp library not installed, cannot use MCP client")
        
        self.server_script_path = Path(server_script_path).resolve()
        if not self.server_script_path.exists():
            raise FileNotFoundError(f"MCP server script does not exist: {server_script_path}")
        
        self.working_dir = working_dir or str(self.server_script_path.parent)
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
    
    async def _async_list_tools(self) -> List[Dict[str, Any]]:
        """Asynchronously get tool list"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        server_params = StdioServerParameters(
            command="python",
            args=[str(self.server_script_path)],
            env=os.environ.copy()
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                
                # Convert to dictionary format
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    }
                    for tool in result.tools
                ]
    
    async def _async_call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronously call tool"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        server_params = StdioServerParameters(
            command="python",
            args=[str(self.server_script_path)],
            env=os.environ.copy()
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments or {})
                
                # Extract text content
                texts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        texts.append(content.text)
                    elif isinstance(content, dict) and 'text' in content:
                        texts.append(content['text'])
                
                return "\n".join(texts) if texts else ""
    
    def list_tools(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Synchronously get tool list
        
        Args:
            use_cache: Whether to use cache
            
        Returns:
            Tool dictionary list
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache
        
        tools = asyncio.run(self._async_list_tools())
        self._tools_cache = tools
        return tools
    
    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        Synchronously call tool
        
        Args:
            name: Tool name
            arguments: Tool parameters
            
        Returns:
            Tool execution result (text format)
        """
        return asyncio.run(self._async_call_tool(name, arguments))
    
    def get_tool_info(self, tool_name: str) -> Optional[str]:
        """
        Get detailed information for specified tool
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool information string, returns None if tool doesn't exist
        """
        tools = self.list_tools()
        for tool_dict in tools:
            if tool_dict["name"] == tool_name:
                schema = tool_dict.get("inputSchema", {})
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                info = f"Tool name: {tool_dict['name']}\n"
                info += f"Description: {tool_dict['description']}\n"
                
                if properties:
                    info += "Parameters:\n"
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "")
                        is_required = param_name in required
                        default = param_info.get("default", None)
                        
                        info += f"  - {param_name} ({param_type})"
                        if is_required:
                            info += " [Required]"
                        if default is not None:
                            info += f" [Default: {default}]"
                        if param_desc:
                            info += f": {param_desc}"
                        info += "\n"
                
                return info
        return None
    
    def close(self):
        """Close client connection (no need to do anything in this implementation)"""
        pass

