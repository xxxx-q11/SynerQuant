"""
OpenAI Agent 实现
"""
from typing import List, Union, Any, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class OpenAIAgent(BaseAgent):
    """OpenAI Agent"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化 OpenAI Agent
        
        Args:
            api_key: OpenAI API Key
            model: 模型名称，如 gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview
            base_url: API 基础 URL（可选，用于代理或兼容 API）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        if OpenAI is None:
            raise ImportError("请安装 openai 包: pip install openai")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
    
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
        # 格式化消息
        formatted_messages = self.format_messages(messages)
        
        # 构建参数
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        # 合并额外参数
        params.update(self.extra_params)
        params.update(kwargs)
        
        if stream:
            # 流式调用
            response = self.client.chat.completions.create(**params, stream=True)
            return response
        else:
            # 非流式调用
            response = self.client.chat.completions.create(**params)
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={"finish_reason": choice.finish_reason}
            )
    
    def get_provider(self) -> LLMProvider:
        """获取提供商类型"""
        return LLMProvider.OPENAI

