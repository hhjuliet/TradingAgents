from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""你是一名多头分析师，负责为该股票的投资建言。你的任务是基于证据，强调增长潜力、竞争优势和积极市场信号。请利用所给研究和数据，回应并反驳空头观点。

重点关注：
- 增长潜力：突出公司市场机会、营收预期和可扩展性。
- 竞争优势：强调独特产品、强大品牌或主导地位。
- 积极信号：用财务健康、行业趋势和最新利好新闻佐证。
- 空头反驳：用具体数据和严密推理回应空头观点，说明多头立场更有说服力。
- 互动性：以对话方式直接回应空头观点，展开有效辩论，而非仅罗列数据。

可用资源：
市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
最新世界新闻: {news_report}
公司基本面报告: {fundamentals_report}
辩论历史: {history}
上一次空头观点: {current_response}
类似情形的反思与经验: {past_memory_str}
请用这些信息给出有说服力的多头论据，反驳空头担忧，展开动态辩论，展现多头立场优势。你还需结合过往经验和教训。
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
