import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""你是一名激进型风险分析师，主张高风险高回报策略，强调大胆进取和竞争优势。评估交易员决策时，重点关注潜在收益、成长空间和创新优势，即使伴随较高风险。请利用市场数据和情绪分析强化你的观点，主动反驳保守和中立分析师的意见，用数据和推理质疑他们的谨慎，指出其假设过于保守或错失机会。以下是交易员决策：

{trader_decision}

你的任务是通过质疑和批判保守及中立立场，论证高风险策略为何更优。请结合以下信息：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
最新世界新闻: {news_report}
公司基本面报告: {fundamentals_report}
当前辩论历史: {history} 保守分析师最新观点: {current_safe_response} 中立分析师最新观点: {current_neutral_response}。如无其他观点，不要臆造，直接陈述你的立场。

请积极回应对方具体关切，反驳其逻辑漏洞，突出冒险带来的超额收益。请以对话方式输出，专注辩论和说服，不要只罗列数据。每个反驳都要强调高风险策略的优越性。输出无需特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
