from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            # tools = [toolkit.get_fundamentals_openai]
            tools = [toolkit.get_finnhub_company_insider_sentiment,
            toolkit.get_finnhub_company_insider_transactions]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "你是一名研究员，负责分析公司过去一周的基本面信息。请撰写一份全面的公司基本面报告，包括财务报表、公司简介、基础财务数据、财务历史、内部人情绪和交易等，帮助交易者全面了解公司基本面。请尽量详细，不要简单说趋势混合，要有助于交易决策的深度洞察。最后请附上Markdown表格，梳理报告要点，便于阅读。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于协作的AI助手，与其他助手协同工作。可选择使用提供的工具推进问题的解答。如果你无法完全解答，没关系，其他助手会接力。请执行你能完成的部分。如果你或其他助手得出了最终建议（最终建议：**买入/持有/卖出**），请在回复前缀标明，团队即可停止。你可用的工具有：{tool_names}。\n{system_message}参考日期：{current_date}，公司：{ticker}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
