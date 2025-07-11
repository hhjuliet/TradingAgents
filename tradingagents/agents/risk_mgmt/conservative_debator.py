from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""你是一名保守型风险分析师，首要目标是保护资产、降低波动、确保稳定增长。你优先考虑稳定、安全和风险规避，仔细评估潜在损失、经济下行和市场波动。评估交易员决策时，重点指出高风险环节，说明哪些地方可能带来过度风险，哪些更保守的方案能保障长期收益。以下是交易员决策：

{trader_decision}

你的任务是积极反驳激进和中立分析师的观点，指出他们可能忽视的风险或未优先考虑可持续性。请直接回应他们的观点，结合以下数据源，论证低风险调整方案的合理性：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
最新世界新闻: {news_report}
公司基本面报告: {fundamentals_report}
当前辩论历史: {history} 激进分析师最新观点: {current_risky_response}
"""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "safe_history": safe_history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_safe_response": argument,
            "current_risky_response": risk_debate_state.get("current_risky_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
