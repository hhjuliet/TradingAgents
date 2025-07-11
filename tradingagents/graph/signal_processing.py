# 交易代理/图/信号处理.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """处理交易信号，提取可执行决策。"""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """用LLM初始化信号处理器。"""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        处理完整交易信号，提取核心决策。

        参数：
            full_signal: 完整的交易信号文本

        返回：
            提取的决策（买入、卖出或持有）
        """
        messages = [
            (
                "system",
                "你是一个高效助手，专门分析分析师团队提供的段落或财报。你的任务是提取投资决策：卖出、买入或持有。只输出提取的决策（卖出、买入或持有），不要添加任何额外文本。",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content
