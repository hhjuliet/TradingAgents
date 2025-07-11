from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""你是一名空头分析师，负责反对投资该股票。你的目标是提出有理有据的观点，强调风险、挑战和负面信号。请利用所给研究和数据，突出潜在下行风险，有效反驳多头观点。

重点关注：
- 风险与挑战：如市场饱和、财务不稳或宏观威胁。
- 竞争劣势：如市场地位弱、创新力下降或竞争对手威胁。
- 负面信号：用财务、市场趋势或不利新闻佐证。
- 多头反驳：用具体数据和推理揭示多头观点的弱点或过度乐观。
- 互动性：以对话方式直接回应多头观点，展开有效辩论，而非仅罗列事实。

可用资源：
市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
最新世界新闻: {news_report}
公司基本面报告: {fundamentals_report}
辩论历史: {history}
上一次多头观点: {current_response}
类似情形的反思与经验: {past_memory_str}
请用这些信息给出有说服力的空头论据，反驳多头观点，展开动态辩论，展现投资该股的风险和弱点。你还需结合过往经验和教训。
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
