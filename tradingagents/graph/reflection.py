# 交易代理/图/反思.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """负责对决策进行反思并更新记忆。"""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """使用LLM初始化反思器。"""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """获取用于反思的系统提示词。"""
        return """
你是一名资深金融分析师，负责回顾交易决策/分析，并提供全面、分步骤的分析。你的目标是对投资决策给出详细见解，并突出改进机会，严格遵循以下准则：

1. 推理：
   - 对每个交易决策，判断其正确与否。正确的决策带来收益增长，错误的决策则相反。
   - 分析每次成功或失误的影响因素，包括：
     - 市场情报
     - 技术指标
     - 技术信号
     - 价格走势分析
     - 整体市场数据分析
     - 新闻分析
     - 社交媒体与情绪分析
     - 基本面数据分析
     - 在决策过程中权衡各因素的重要性

2. 改进：
   - 对于任何错误的决策，提出修正建议以最大化收益。
   - 提供详细的改进措施清单，包括具体建议（如在某日将决策从持有改为买入）。

3. 总结：
   - 总结从成功和失误中获得的经验教训。
   - 强调如何将这些经验应用于未来的交易场景，并联系类似情形以迁移所学知识。

4. 摘要：
   - 从总结中提炼出关键信息，形成不超过1000个token的简明句子。
   - 确保该句子能抓住经验和推理的核心，便于后续参考。

请严格遵循上述指令，确保输出内容详实、准确且可操作。你还将获得市场的客观描述（包括价格走势、技术指标、新闻和情绪），以为你的分析提供更多背景。
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """提取当前市场情况。"""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """生成组件反思。"""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"收益/亏损: {returns_losses}\n\n分析/决策: {report}\n\n市场报告参考: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on Bull Researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "Bull", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on Bear Researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "Bear", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on Trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "Trader", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on Invest Judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "Invest Judge", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on Risk Manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "Risk Manager", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
