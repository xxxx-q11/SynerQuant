"""
Google Gemini Agent 实现
"""
from typing import List, Union, Any, Optional
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class GoogleAgent(BaseAgent):
    """Google Gemini Agent"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化 Google Agent
        
        Args:
            api_key: Google API Key
            model: 模型名称，如 gemini-pro, gemini-pro-vision
            base_url: 不使用（Google 固定 URL）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        if genai is None:
            raise ImportError("请安装 google-generativeai 包: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model)
    
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
        # 格式化消息（Gemini 使用不同的格式）
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content]
            })
        
        # 构建生成配置
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens:
            generation_config["max_output_tokens"] = self.max_tokens
        
        # 合并额外参数
        generation_config.update({k: v for k, v in self.extra_params.items() if k in ["top_p", "top_k"]})
        generation_config.update({k: v for k, v in kwargs.items() if k in ["top_p", "top_k", "temperature", "max_output_tokens"]})
        
        if stream:
            # 流式调用
            response = self.client.generate_content(
                formatted_messages,
                generation_config=generation_config,
                stream=True
            )
            return response
        else:
            # 非流式调用
            response = self.client.generate_content(
                formatted_messages,
                generation_config=generation_config
            )
            
            content = ""
            if response.text:
                content = response.text
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") else None,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") else None,
                },
                metadata={"finish_reason": response.candidates[0].finish_reason if response.candidates else None}
            )
    
    def get_provider(self) -> LLMProvider:
        """获取提供商类型"""
        return LLMProvider.GOOGLE

