import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""你是一名中立型风险分析师，职责是权衡交易员决策的潜在收益与风险，优先采取均衡视角，综合考虑市场趋势、经济变化和分散化策略。以下是交易员决策：

{trader_decision}

你的任务是挑战激进和保守分析师，指出他们观点过于乐观或过于谨慎之处。请结合以下数据源，支持你的中庸、可持续调整建议：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
最新世界新闻: {news_report}
公司基本面报告: {fundamentals_report}
当前辩论历史: {history} 激进分析师最新观点: {current_risky_response} 保守分析师最新观点: {current_safe_response}
"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "neutral_history": neutral_history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "latest_speaker": "Neutral",
            "current_neutral_response": argument,
            "current_risky_response": risk_debate_state.get("current_risky_response", ""),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
