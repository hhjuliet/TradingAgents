from typing import Annotated, Sequence
from datetime import date, timedelta, datetime
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from tradingagents.agents import *
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START, MessagesState


# 研究员团队状态
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "多头对话历史"
    ]  # 多头对话历史
    bear_history: Annotated[
        str, "空头对话历史"
    ]  # 空头对话历史
    history: Annotated[str, "对话历史"]  # 对话历史
    current_response: Annotated[str, "最新回复"]  # 最新回复
    judge_decision: Annotated[str, "最终裁决"]  # 最终裁决
    count: Annotated[int, "当前对话长度"]  # 当前对话长度


# 风控团队状态
class RiskDebateState(TypedDict):
    risky_history: Annotated[
        str, "激进风控员对话历史"
    ]  # 对话历史
    safe_history: Annotated[
        str, "保守风控员对话历史"
    ]  # 对话历史
    neutral_history: Annotated[
        str, "中性风控员对话历史"
    ]  # 对话历史
    history: Annotated[str, "对话历史"]  # 对话历史
    latest_speaker: Annotated[str, "上次发言分析师"]
    current_risky_response: Annotated[
        str, "激进风控员最新回复"
    ]  # 最新回复
    current_safe_response: Annotated[
        str, "保守风控员最新回复"
    ]  # 最新回复
    current_neutral_response: Annotated[
        str, "中性风控员最新回复"
    ]  # 最新回复
    judge_decision: Annotated[str, "裁判裁决"]
    count: Annotated[int, "当前对话长度"]  # 当前对话长度


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "关注交易的公司"]
    trade_date: Annotated[str, "交易日期"]

    sender: Annotated[str, "发送消息的智能体"]

    # 研究阶段
    market_report: Annotated[str, "市场分析师报告"]
    sentiment_report: Annotated[str, "社交媒体分析师报告"]
    news_report: Annotated[
        str, "新闻研究员的时事报告"
    ]
    fundamentals_report: Annotated[str, "基本面研究员报告"]

    # 研究员团队讨论阶段
    investment_debate_state: Annotated[
        InvestDebateState, "是否投资的辩论当前状态"
    ]
    investment_plan: Annotated[str, "分析师生成的投资计划"]

    trader_investment_plan: Annotated[str, "交易员生成的投资计划"]

    # 风控团队讨论阶段
    risk_debate_state: Annotated[
        RiskDebateState, "风险评估辩论当前状态"
    ]
    final_trade_decision: Annotated[str, "风控分析师最终决策"]
