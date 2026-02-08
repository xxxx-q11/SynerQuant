"""
通义千问 (Qwen) Agent 实现
"""
from typing import List, Union, Any, Optional

# 延迟导入 dashscope，避免在未安装时导致整个模块不可用
try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class QwenAgent(BaseAgent):
    """通义千问 Agent"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-turbo",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化千问 Agent
        
        Args:
            api_key: DashScope API Key
            model: 模型名称，如 qwen-turbo, qwen-plus, qwen-max
            base_url: 不使用（DashScope 固定 URL）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        if dashscope is None:
            raise ImportError(
                "dashscope 包未安装。请运行: pip install dashscope"
            )
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        dashscope.api_key = api_key
    
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
            responses = Generation.call(**params, stream=True)
            return responses
        else:
            # 非流式调用
            response = Generation.call(**params)
            
            if response.status_code == 200:
                result = response.output
                
                # 检查 result 是否为 None
                if result is None:
                    raise Exception(f"Qwen API 返回空响应: {response}")
                
                # 兼容不同的响应格式
                # DashScope 原生格式: result.choices[0].message.content
                # OpenAI 兼容格式: result["choices"][0]["message"]["content"]
                if hasattr(result, 'choices'):
                    # DashScope 原生格式
                    choices = result.choices
                    if choices and len(choices) > 0:
                        content = getattr(choices[0].message, 'content', '')
                        usage = getattr(result, 'usage', None)
                    else:
                        content = ''
                        usage = None
                elif isinstance(result, dict):
                    # OpenAI 兼容格式或字典格式
                    choices = result.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                    else:
                        content = ""
                    usage = result.get("usage")
                else:
                    raise Exception(f"未知的响应格式: {type(result)}, 内容: {result}")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                    metadata={"request_id": getattr(response, 'request_id', None)}
                )
            else:
                error_msg = getattr(response, 'message', f'状态码: {response.status_code}')
                raise Exception(f"Qwen API 调用失败: {error_msg}")
    
    def get_provider(self) -> LLMProvider:
        """获取提供商类型"""
        return LLMProvider.QWEN

