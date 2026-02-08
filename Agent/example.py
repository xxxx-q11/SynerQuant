"""
Agent 使用示例
"""
from Agent.agent_factory import create_agent
from Agent.base_agent import Message


def example_qwen():
    """千问 Agent 使用示例"""
    agent = create_agent(
        provider="qwen",
        api_key="your-dashscope-api-key",
        model="qwen-turbo",
        temperature=0.7
    )
    
    response = agent.call("你好，请介绍一下自己")
    print(f"千问回复: {response.content}")


def example_openai():
    """OpenAI Agent 使用示例"""
    agent = create_agent(
        provider="openai",
        api_key="sk-your-openai-api-key",
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    messages = [
        Message(role="system", content="你是一个专业的量化交易助手"),
        Message(role="user", content="什么是因子挖掘？")
    ]
    
    response = agent.chat(messages)
    print(f"OpenAI 回复: {response.content}")
    print(f"Token 使用: {response.usage}")


def example_claude():
    """Claude Agent 使用示例"""
    agent = create_agent(
        provider="claude",
        api_key="your-anthropic-api-key",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7
    )
    
    response = agent.call(
        prompt="解释一下机器学习在量化交易中的应用",
        system_prompt="你是一个量化交易专家"
    )
    print(f"Claude 回复: {response.content}")


def example_google():
    """Google Gemini Agent 使用示例"""
    agent = create_agent(
        provider="google",
        api_key="your-google-api-key",
        model="gemini-pro",
        temperature=0.7
    )
    
    response = agent.call("什么是风险控制？")
    print(f"Google 回复: {response.content}")


def example_third_party():
    """第三方平台 Agent 使用示例"""
    agent = create_agent(
        provider="third_party",
        api_key="your-api-key",
        model="custom-model",
        base_url="https://api.example.com/v1",
        temperature=0.7
    )
    
    response = agent.call("你好")
    print(f"第三方平台回复: {response.content}")


if __name__ == "__main__":
    # 注意：需要设置实际的 API Key 才能运行
    print("请设置 API Key 后运行示例")
    # example_qwen()
    # example_openai()
    # example_claude()
    # example_google()
    # example_third_party()

