from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            "你是一名交易助手，负责分析金融市场。你的任务是从下列指标中为特定市场环境或交易策略选择最相关的（最多8个），要求信息互补且无冗余。各类别及其指标如下：\n\n"
            "移动平均线：\n"
            "- close_50_sma: 50日简单均线。用法：识别中期趋势，作为动态支撑/阻力。提示：有滞后性，建议与更快指标结合。\n"
            "- close_200_sma: 200日简单均线。用法：确认长期趋势，识别金叉/死叉。提示：反应慢，适合战略性趋势确认。\n"
            "- close_10_ema: 10日指数均线。用法：捕捉短期动量变化和潜在入场点。提示：震荡市易受噪音影响，建议与长周期均线配合。\n\n"
            "MACD相关：\n"
            "- macd: MACD。用法：关注交叉和背离信号。提示：震荡市需结合其他指标确认。\n"
            "- macds: MACD信号线。用法：与MACD线交叉触发交易。提示：应作为更大策略的一部分。\n"
            "- macdh: MACD柱状图。用法：可视化动量强度，及早发现背离。提示：波动大，快市需额外过滤。\n\n"
            "动量指标：\n"
            "- rsi: RSI。用法：判断超买/超卖，关注背离。提示：强趋势下RSI可能持续极端，需结合趋势分析。\n\n"
            "波动率指标：\n"
            "- boll: 布林中轨（20日SMA）。用法：动态基准。提示：结合上下轨判断突破或反转。\n"
            "- boll_ub: 布林上轨。用法：提示超买或突破。提示：强趋势下价格可能沿轨运行。\n"
            "- boll_lb: 布林下轨。用法：提示超卖。提示：需结合其他分析避免假信号。\n"
            "- atr: ATR。用法：设置止损和调整仓位。提示：为反应性指标，建议结合风险管理。\n\n"
            "成交量指标：\n"
            "- vwma: 成交量加权均线。用法：结合价格确认趋势。提示：需警惕成交量异常带来的偏差。\n\n"
            "请选择信息多元且互补的指标，避免冗余（如不要同时选rsi和stochrsi）。简要说明为何适合当前市场。调用工具时请严格使用上述指标名，否则调用会失败。请务必先调用get_YFin_data获取生成指标所需CSV。请对你观察到的趋势撰写详细且细致的报告，避免简单说趋势混合，要有助于交易决策的深度洞察。最后请附上Markdown表格，梳理报告要点，便于阅读。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于协作的AI助手，与其他助手协同工作。请使用提供的工具推进问题的解答。如果你无法完全解答，没关系，其他助手会接力。请执行你能完成的部分。如果你或其他助手得出了最终建议（最终建议：**买入/观望/卖出**），请在回复前缀标明，团队即可停止。你可用的工具有：{tool_names}。\n{system_message}参考日期：{current_date}，公司：{ticker}"
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
            "market_report": report,
        }

    return market_analyst_node
