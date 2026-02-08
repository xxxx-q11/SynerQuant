# ===== Important: MCP stdio protocol requires stdout only for protocol messages =====
# Save stdout and redirect stderr
import os
import sys
import warnings
import asyncio  # Add this line
from types import SimpleNamespace  # Add this line
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# Disable all warning output
warnings.filterwarnings("ignore")

# Redirect stderr to log file (optional)
# This allows debugging without affecting MCP protocol
log_file = open('/tmp/mcp_server_debug.log', 'w', buffering=1)
sys.stderr = log_file

# Set before importing modules to avoid warnings from pydantic and other libraries
os.environ['PYTHONWARNINGS'] = 'ignore'

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

import json

from modules import train_qcm_mcp as mod_train_qcm_mcp
from modules import calculate_ic as mod_calculate_ic
from modules import train_AFF_MCP as mod_train_AFF
from modules import train_gfn_AlphaSAGE_MCP as mod_train_gfn_AlphaSAGE
from modules import train_GP_AlphaSAGE_MCP as mod_train_GP_AlphaSAGE
from modules import train_PPO_AlphaSAGE_MCP as mod_train_PPO_AlphaSAGE
from modules import qlib_benchmark_runner_fastapi as mod_qlib_benchmark_runner
from modules import knowledge_base_mcp as mod_knowledge_base


server = Server("qlib-mcp")


def ns(d):
	return SimpleNamespace(**d)

def load_tools_config():
	"""Load tool configuration from tools.json"""
	tools_config_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), 
		"configs", 
		"tools.json"
	)
	
	if not os.path.exists(tools_config_path):
		raise FileNotFoundError(f"Tools config not found: {tools_config_path}")
	
	with open(tools_config_path, 'r', encoding='utf-8') as f:
		config = json.load(f)
	
	# Extract all tools from categories
	tools = []
	for category in config.get("categories", []):
		tools.extend(category.get("tools", []))
	
	return tools

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
	"""Load tool definitions from tools.json"""
	try:
		tools_config = load_tools_config()
		tools = []
		for tool in tools_config:
			tools.append(
				types.Tool(
					name=tool["name"],
					description=tool["description"],
					inputSchema=tool.get("inputSchema", {})
				)
			)
		return tools
	except Exception as e:
		# If loading fails, return basic tool list (fallback)
		return [
			types.Tool(name="train_qcm", description="Train AlphaQCM model", inputSchema={}),
			types.Tool(name="train_AFF", description="Train AlphaForge model", inputSchema={}),
			types.Tool(name="train_gfn_AlphaSAGE", description="Train AlphaSAGE model", inputSchema={}),
			types.Tool(name="train_GP_AlphaSAGE", description="Train GP AlphaSAGE model", inputSchema={}),
			types.Tool(name="train_PPO_AlphaSAGE", description="Train PPO AlphaSAGE model", inputSchema={}),
			types.Tool(name="qlib_benchmark_runner", description="Run Qlib Benchmark model", inputSchema={}),
			types.Tool(name="qlib_benchmark_list_models", description="List all available Qlib Benchmark models", inputSchema={}),
		]


@server.call_tool(validate_input=False)
async def handle_call_tool(name: str, arguments: dict | None):
	args = ns(arguments or {})

	# Execute blocking synchronous calls in thread to avoid running directly in async event loop
	async def run_in_thread(fn, *f_args):
		return await asyncio.to_thread(fn, *f_args)
	if  name == "calculate_ic":
		await run_in_thread(mod_calculate_ic.run, args)
		return [types.TextContent(type="text", text="calculate_ic done")]
	elif name == "train_qcm":
		result = await run_in_thread(mod_train_qcm_mcp.run, args)
		# result should be the path of the finally generated factor file
		if result:
			return [types.TextContent(type="text", text=result)]
		return [types.TextContent(type="text", text="train_qcm done")]
	elif name == "train_AFF":
		result = await run_in_thread(mod_train_AFF.run, args)
		if result:
			return [types.TextContent(type="text", text=result)]
	elif name == "train_gfn_AlphaSAGE":
		await run_in_thread(mod_train_gfn_AlphaSAGE.run, args)
		return [types.TextContent(type="text", text="train_gfn_AlphaSAGE done")]
	elif name == "train_GP_AlphaSAGE":
		result = await run_in_thread(mod_train_GP_AlphaSAGE.run, args)
		# result should be the path of the finally generated factor file
		if result:
			return [types.TextContent(type="text", text=result)]
		return [types.TextContent(type="text", text="train_GP_AlphaSAGE done")]
	elif name == "train_PPO_AlphaSAGE":
		result = await run_in_thread(mod_train_PPO_AlphaSAGE.run, args)
		# result should be the path of the finally generated factor file
		if result:
			return [types.TextContent(type="text", text=result)]
		return [types.TextContent(type="text", text="train_PPO_AlphaSAGE done")]
	elif name == "qlib_benchmark_runner":
		result = await run_in_thread(mod_qlib_benchmark_runner.run, args)
		if result:
			return [types.TextContent(type="text", text=result)]
		return [types.TextContent(type="text", text="qlib_benchmark_runner done")]
	elif name == "qlib_benchmark_list_models":
			result = await run_in_thread(mod_qlib_benchmark_runner.list_models, args)
			# If result is returned, format output
			if result:
				if isinstance(result, dict):
					result_text = json.dumps(result, indent=2, ensure_ascii=False)
				else:
					result_text = str(result)
				return [types.TextContent(type="text", text=result_text)]
			return [types.TextContent(type="text", text="qlib_benchmark_list_models done")]
	elif name == "search_papers":
		result = await run_in_thread(mod_knowledge_base.run, args)
		# Format output result
		if result:
			if isinstance(result, dict):
				result_text = json.dumps(result, indent=2, ensure_ascii=False)
			else:
				result_text = str(result)
			return [types.TextContent(type="text", text=result_text)]
		return [types.TextContent(type="text", text="search_papers done")]
	else:
		return [types.TextContent(type="text", text=f"unknown tool: {name}")]


async def main():
	async with stdio_server() as streams:
		await server.run(
			streams[0],
			streams[1],
			server.create_initialization_options(notification_options=NotificationOptions()),
		)


if __name__ == "__main__":
	asyncio.run(main())

