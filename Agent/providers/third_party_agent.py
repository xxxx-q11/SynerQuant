# """
# 第三方平台 API Agent 实现（通用接口）
# """
# from typing import List, Union, Any, Optional, Dict
# import requests
# import json

# from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


# class ThirdPartyAgent(BaseAgent):
#     """第三方平台 API Agent（兼容 OpenAI 格式）"""
    
#     def __init__(
#         self,
#         api_key: str,
#         model: str,
#         base_url: str,
#         temperature: float = 0.7,
#         max_tokens: Optional[int] = None,
#         headers: Optional[Dict[str, str]] = None,
#         **kwargs
#     ):
#         """
#         初始化第三方平台 Agent
        
#         Args:
#             api_key: API 密钥
#             model: 模型名称
#             base_url: API 基础 URL（必需）
#             temperature: 温度参数
#             max_tokens: 最大生成 token 数
#             headers: 自定义请求头
#         """
#         if not base_url:
#             raise ValueError("第三方平台 Agent 需要提供 base_url")
        
#         super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
#         self.headers = headers or {}
#         if "Authorization" not in self.headers:
#             self.headers["Authorization"] = f"Bearer {api_key}"
#         if "Content-Type" not in self.headers:
#             self.headers["Content-Type"] = "application/json"
    
#     def chat(
#         self,
#         messages: List[Message],
#         stream: bool = False,
#         **kwargs
#     ) -> Union[LLMResponse, Any]:
#         """
#         发送聊天请求（兼容 OpenAI 格式）
        
#         Args:
#             messages: 消息列表
#             stream: 是否流式返回（第三方平台可能不支持）
#             **kwargs: 其他参数
            
#         Returns:
#             LLMResponse 或流式响应对象
#         """
#         # 格式化消息
#         formatted_messages = self.format_messages(messages)
        
#         # 构建请求体（OpenAI 兼容格式）
#         payload = {
#             "model": self.model,
#             "messages": formatted_messages,
#             "temperature": kwargs.get("temperature", self.temperature),
#         }
        
#         if self.max_tokens:
#             payload["max_tokens"] = self.max_tokens
        
#         # 合并额外参数
#         payload.update({k: v for k, v in self.extra_params.items() if k not in payload})
#         payload.update({k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]})
        
#         # 构建完整 URL
#         url = f"{self.base_url.rstrip('/')}/chat/completions"
        
#         if stream:
#             # 流式调用
#             payload["stream"] = True
#             response = requests.post(
#                 url,
#                 headers=self.headers,
#                 json=payload,
#                 stream=True
#             )
#             response.raise_for_status()
#             return response
#         else:
#             # 非流式调用
#             response = requests.post(
#                 url,
#                 headers=self.headers,
#                 json=payload,
#                 timeout=kwargs.get("timeout", 300)
#             )
#             response.raise_for_status()
#             data = response.json()
            
#             # 解析响应（兼容 OpenAI 格式）
#             choice = data.get("choices", [{}])[0]
#             message = choice.get("message", {})
            
#             return LLMResponse(
#                 content=message.get("content", ""),
#                 model=data.get("model", self.model),
#                 usage=data.get("usage"),
#                 metadata={
#                     "id": data.get("id"),
#                     "finish_reason": choice.get("finish_reason"),
#                 }
#             )
    
#     def get_provider(self) -> LLMProvider:
#         """获取提供商类型"""
#         return LLMProvider.THIRD_PARTY

"""
第三方平台 API Agent 实现（通用接口）
"""
from typing import List, Union, Any, Optional, Dict
import requests
import json
import time
import random

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class ThirdPartyAgent(BaseAgent):
    """第三方平台 API Agent（兼容 OpenAI 格式）"""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs
    ):
        """
        初始化第三方平台 Agent
        
        Args:
            api_key: API 密钥
            model: 模型名称
            base_url: API 基础 URL（必需）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            headers: 自定义请求头
            max_retries: 最大重试次数（针对429和503错误）
            retry_delay: 初始重试延迟（秒）
        """
        if not base_url:
            raise ValueError("第三方平台 Agent 需要提供 base_url")
        
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        self.headers = headers or {}
        if "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        发送聊天请求（兼容 OpenAI 格式），带429和503错误重试机制
        
        Args:
            messages: 消息列表
            stream: 是否流式返回（第三方平台可能不支持）
            **kwargs: 其他参数
            
        Returns:
            LLMResponse 或流式响应对象
        """
        # 格式化消息
        formatted_messages = self.format_messages(messages)
        
        # 构建请求体（OpenAI 兼容格式）
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        # 合并额外参数
        payload.update({k: v for k, v in self.extra_params.items() if k not in payload})
        payload.update({k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]})
        
        # 构建完整 URL
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        # 重试逻辑
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    # 流式调用
                    payload["stream"] = True
                    response = requests.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        stream=True
                    )
                    response.raise_for_status()
                    return response
                else:
                    # 非流式调用
                    response = requests.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=kwargs.get("timeout", 300)
                    )
                    
                    # 处理429和503错误（都是临时性错误，应该重试）
                    if response.status_code in [429, 503]:
                        if attempt < self.max_retries:
                            # 指数退避：2秒, 4秒, 8秒...
                            wait_time = self.retry_delay * (2 ** attempt)
                            # 添加随机抖动，避免多个请求同时重试
                            jitter = random.uniform(0, 1)
                            wait_time += jitter
                            
                            error_name = "429错误" if response.status_code == 429 else "503错误"
                            print(f"[ThirdPartyAgent] {error_name}，等待 {wait_time:.2f}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1})")
                            time.sleep(wait_time)
                            continue
                        else:
                            response.raise_for_status()
                    else:
                        response.raise_for_status()
                    
                    data = response.json()
                    
                    # 解析响应（兼容 OpenAI 格式）
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    
                    return LLMResponse(
                        content=message.get("content", ""),
                        model=data.get("model", self.model),
                        usage=data.get("usage"),
                        metadata={
                            "id": data.get("id"),
                            "finish_reason": choice.get("finish_reason"),
                        }
                    )
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code in [429, 503] and attempt < self.max_retries:
                    # 429 和 503 都是临时性错误，应该重试
                    wait_time = self.retry_delay * (2 ** attempt)
                    jitter = random.uniform(0, 1)
                    wait_time += jitter
                    error_name = "429错误" if e.response.status_code == 429 else "503错误"
                    print(f"[ThirdPartyAgent] {error_name}，等待 {wait_time:.2f}秒后重试: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    # 其他HTTP错误直接抛出
                    raise
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[ThirdPartyAgent] 请求失败，等待 {wait_time:.2f}秒后重试: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        # 如果所有重试都失败了
        if last_exception:
            raise last_exception
    
    def get_provider(self) -> LLMProvider:
        """获取提供商类型"""
        return LLMProvider.THIRD_PARTY