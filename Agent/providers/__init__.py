"""LLM 提供商适配器模块

使用延迟/安全导入，避免因缺少某些依赖而导致整个模块不可用
"""

_IMPORT_ERRORS = {}

try:
    from .qwen_agent import QwenAgent
except ImportError as e:
    QwenAgent = None
    _IMPORT_ERRORS['QwenAgent'] = str(e)

try:
    from .claude_agent import ClaudeAgent
except ImportError as e:
    ClaudeAgent = None
    _IMPORT_ERRORS['ClaudeAgent'] = str(e)

try:
    from .google_agent import GoogleAgent
except ImportError as e:
    GoogleAgent = None
    _IMPORT_ERRORS['GoogleAgent'] = str(e)

try:
    from .openai_agent import OpenAIAgent
except ImportError as e:
    OpenAIAgent = None
    _IMPORT_ERRORS['OpenAIAgent'] = str(e)

try:
    from .third_party_agent import ThirdPartyAgent
except ImportError as e:
    ThirdPartyAgent = None
    _IMPORT_ERRORS['ThirdPartyAgent'] = str(e)

__all__ = [
    "QwenAgent",
    "ClaudeAgent",
    "GoogleAgent",
    "OpenAIAgent",
    "ThirdPartyAgent",
    "get_import_errors",
]


def get_import_errors():
    """获取导入错误信息，用于调试"""
    return _IMPORT_ERRORS.copy()

