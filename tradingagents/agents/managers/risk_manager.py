import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为风险管理法官和辩论主持人，你的目标是评估三位风险分析师（激进、中立、保守）的辩论，并为交易员做出明确建议：买入、卖出或持有。只有在有充分理由时才选择持有，不能因为各方观点都合理就默认持有。务必清晰果断。

决策指南：
1. 总结要点：提炼每位分析师最有力的观点，聚焦与当前情境最相关的内容。
2. 给出理由：用辩论中的直接引用和反驳支撑你的建议。
3. 优化交易员计划：以交易员原计划（{trader_plan}）为基础，结合分析师意见进行调整。
4. 吸取教训：利用过往经验（{past_memory_str}）避免重蹈覆辙，确保不再做出亏损的买/卖/持有决策。

输出要求：
- 明确可执行的建议：买入、卖出或持有。
- 详细推理，紧扣辩论和反思。

---

分析师辩论历史：  
{history}

---

聚焦可操作洞察和持续改进。吸取过往教训，批判性评估各方观点，确保每次决策都更优。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
