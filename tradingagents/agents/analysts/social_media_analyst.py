from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_stock_news_openai]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        system_message = (
            "你是一名社交媒体及公司新闻分析师，负责分析某公司过去一周的社交媒体帖子、公司新闻和公众情绪。你需要撰写一份全面的长报告，详细分析你的见解和对交易者/投资者的启示，包括社交媒体、情绪数据和公司新闻。请尽量参考所有渠道，不要简单说趋势混合，要有助于交易决策的深度洞察。最后请附上Markdown表格，梳理报告要点，便于阅读。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于协作的AI助手，与其他助手协同工作。可选择搜索最新的资料和新闻推进问题的解答。如果你无法完全解答，没关系，其他助手会接力。请执行你能完成的部分。如果你或其他助手得出了最终建议（最终建议：**买入/观望/卖出**），请在回复前缀标明，团队即可停止。\n{system_message}参考日期：{current_date}，公司：{ticker}"
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
