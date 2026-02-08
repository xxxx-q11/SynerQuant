"""
Claude (Anthropic) Agent 实现
"""
from typing import List, Union, Any, Optional
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class ClaudeAgent(BaseAgent):
    """Claude Agent"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        **kwargs
    ):
        """
        初始化 Claude Agent
        
        Args:
            api_key: Anthropic API Key
            model: 模型名称，如 claude-3-5-sonnet-20241022, claude-3-opus-20240229
            base_url: API 基础 URL（可选，用于代理）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        if Anthropic is None:
            raise ImportError("请安装 anthropic 包: pip install anthropic")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = Anthropic(**client_kwargs)
    
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            stream: 是否流式返回
            **kwargs: 其他参数
            
        Returns:
            LLMResponse 或流式响应对象
        """
        # Claude 需要分离 system 消息
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        user_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages if msg.role != "system"
        ]
        
        # 构建参数
        params = {
            "model": self.model,
            "messages": user_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens or 4096),
        }
        
        if system_messages:
            params["system"] = " ".join(system_messages)
        
        # 合并额外参数
        params.update({k: v for k, v in self.extra_params.items() if k not in params})
        params.update({k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]})
        
        if stream:
            # 流式调用
            with self.client.messages.stream(**params) as stream:
                return stream
        else:
            # 非流式调用
            response = self.client.messages.create(**params)
            
            content = ""
            if response.content:
                for block in response.content:
                    if block.type == "text":
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={"id": response.id}
            )
    
    def get_provider(self) -> LLMProvider:
        """获取提供商类型"""
        return LLMProvider.CLAUDE

