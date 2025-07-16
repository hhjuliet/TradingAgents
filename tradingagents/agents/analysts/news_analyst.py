from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            # tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
            tools = [toolkit.get_finnhub_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "你是一名新闻研究员，负责分析过去一周的最新新闻和趋势。请撰写一份全面的全球宏观经济与交易相关的新闻报告，需参考EODHD和finnhub等多方新闻。不要简单说趋势混合，要有助于交易决策的深度洞察。最后请附上Markdown表格，梳理报告要点，便于阅读。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于协作的AI助手，与其他助手协同工作。可选择搜索最近的新闻和资料推进问题的解答。如果你无法完全解答，没关系，其他助手会接力。请执行你能完成的部分。如果你或其他助手得出了最终建议（最终建议：**买入/观望/卖出**），请在回复前缀标明，团队即可停止。\n{system_message}参考日期：{current_date}，公司：{ticker}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
