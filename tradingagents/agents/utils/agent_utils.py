from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import tradingagents.dataflows.interface as interface
from tradingagents.dashscope_default_config import DEFAULT_CONFIG
from langchain_core.messages import HumanMessage


def create_msg_delete():
    def delete_messages(state):
        """
        清空消息并添加Anthropic兼容的占位符。
        """
        messages = state["messages"]
        
        # 移除所有消息
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        
        # 添加最小占位消息
        placeholder = HumanMessage(content="继续")
        
        return {"messages": removal_operations + [placeholder]}
    
    return delete_messages


class Toolkit:
    _config = DEFAULT_CONFIG.copy()

    @classmethod
    def update_config(cls, config):
        """更新类级别配置。"""
        cls._config.update(config)

    @property
    def config(self):
        """访问配置。"""
        return self._config

    def __init__(self, config=None):
        if config:
            self.update_config(config)

    @staticmethod
    @tool
    def get_reddit_news(
        curr_date: Annotated[str, "您想要获取新闻的日期，格式为yyyy-mm-dd"],
    ) -> str:
        """
        从Reddit获取全球新闻，在指定时间范围内。
        Args:
            curr_date (str): 您想要获取新闻的日期，格式为yyyy-mm-dd
        Returns:
            str: 包含最新全球新闻的格式化数据框。
        """
        
        global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)

        return global_news_result

    @staticmethod
    @tool
    def get_finnhub_news(
        ticker: Annotated[
            str,
            "公司搜索查询，例如'AAPL, TSM, 等'",
        ],
        start_date: Annotated[str, "开始日期，格式为yyyy-mm-dd"],
        end_date: Annotated[str, "结束日期，格式为yyyy-mm-dd"],
    ):
        """
        从Finnhub获取给定股票的最新新闻，在日期范围内。
        Args:
            ticker (str): 公司股票代码。例如AAPL, TSM
            start_date (str): 开始日期，格式为yyyy-mm-dd
            end_date (str): 结束日期，格式为yyyy-mm-dd
        Returns:
            str: 包含公司日期范围内的最新新闻的格式化数据框。
        """

        end_date_str = end_date

        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        look_back_days = (end_date - start_date).days

        finnhub_news_result = interface.get_finnhub_news(
            ticker, end_date_str, look_back_days
        )

        return finnhub_news_result

    @staticmethod
    @tool
    def get_reddit_stock_info(
        ticker: Annotated[
            str,
            "公司股票代码。例如AAPL, TSM",
        ],
        curr_date: Annotated[str, "您想要获取新闻的当前日期"],
    ) -> str:
        """
        从Reddit获取给定股票的最新新闻，给定当前日期。
        Args:
            ticker (str): 公司股票代码。例如AAPL, TSM
            curr_date (str): 您想要获取新闻的当前日期，格式为yyyy-mm-dd
        Returns:
            str: 包含公司给定日期的最新新闻的格式化数据框。
        """

        stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)

        return stock_news_results

    @staticmethod
    @tool
    def get_YFin_data(
        symbol: Annotated[str, "公司股票代码"],
        start_date: Annotated[str, "开始日期，格式为yyyy-mm-dd"],
        end_date: Annotated[str, "结束日期，格式为yyyy-mm-dd"],
    ) -> str:
        """
        从Yahoo Finance获取给定股票代码的股价数据。
        Args:
            symbol (str): 公司股票代码，例如AAPL, TSM
            start_date (str): 开始日期，格式为yyyy-mm-dd
            end_date (str): 结束日期，格式为yyyy-mm-dd
        Returns:
            str: 包含指定股票代码在指定日期范围内的股价数据的格式化数据框。
        """

        result_data = interface.get_YFin_data(symbol, start_date, end_date)

        return result_data

    @staticmethod
    @tool
    def get_YFin_data_online(
        symbol: Annotated[str, "公司股票代码"],
        start_date: Annotated[str, "开始日期，格式为yyyy-mm-dd"],
        end_date: Annotated[str, "结束日期，格式为yyyy-mm-dd"],
    ) -> str:
        """
        从Yahoo Finance获取给定股票代码的股价数据。
        Args:
            symbol (str): 公司股票代码，例如AAPL, TSM
            start_date (str): 开始日期，格式为yyyy-mm-dd
            end_date (str): 结束日期，格式为yyyy-mm-dd
        Returns:
            str: 包含指定股票代码在指定日期范围内的股价数据的格式化数据框。
        """

        result_data = interface.get_YFin_data_online(symbol, start_date, end_date)

        return result_data

    @staticmethod
    @tool
    def get_stockstats_indicators_report(
        symbol: Annotated[str, "公司股票代码"],
        indicator: Annotated[
            str, "您想要获取分析和报告的技术指标"
        ],
        curr_date: Annotated[
            str, "您当前交易的日期，YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "回看天数"] = 30,
    ) -> str:
        """
        获取给定股票代码和指标的技术指标。
        Args:
            symbol (str): 公司股票代码，例如AAPL, TSM
            indicator (str): 您想要获取分析和报告的技术指标
            curr_date (str): 您当前交易的日期，YYYY-mm-dd
            look_back_days (int): 回看天数，默认30
        Returns:
            str: 包含指定股票代码和指标的技术指标的格式化数据框。
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, False
        )

        return result_stockstats

    @staticmethod
    @tool
    def get_stockstats_indicators_report_online(
        symbol: Annotated[str, "公司股票代码"],
        indicator: Annotated[
            str, "您想要获取分析和报告的技术指标"
        ],
        curr_date: Annotated[
            str, "您当前交易的日期，YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "回看天数"] = 30,
    ) -> str:
        """
        获取给定股票代码和指标的技术指标。
        Args:
            symbol (str): 公司股票代码，例如AAPL, TSM
            indicator (str): 您想要获取分析和报告的技术指标
            curr_date (str): 您当前交易的日期，YYYY-mm-dd
            look_back_days (int): 回看天数，默认30
        Returns:
            str: 包含指定股票代码和指标的技术指标的格式化数据框。
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, True
        )

        return result_stockstats

    @staticmethod
    @tool
    def get_finnhub_company_insider_sentiment(
        ticker: Annotated[str, "公司股票代码"],
        curr_date: Annotated[
            str,
            "您当前交易的日期，yyyy-mm-dd",
        ],
    ):
        """
        获取公司（从公共SEC信息中检索）的内幕情绪信息，过去30天。
        Args:
            ticker (str): 公司股票代码
            curr_date (str): 您当前交易的日期，yyyy-mm-dd
        Returns:
            str: 从curr_date开始的过去30天的情绪报告。
        """

        data_sentiment = interface.get_finnhub_company_insider_sentiment(
            ticker, curr_date, 30
        )

        return data_sentiment

    @staticmethod
    @tool
    def get_finnhub_company_insider_transactions(
        ticker: Annotated[str, "公司股票代码"],
        curr_date: Annotated[
            str,
            "您当前交易的日期，yyyy-mm-dd",
        ],
    ):
        """
        获取公司（从公共SEC信息中检索）的内幕交易信息，过去30天。
        Args:
            ticker (str): 公司股票代码
            curr_date (str): 您当前交易的日期，yyyy-mm-dd
        Returns:
            str: 公司过去30天的内幕交易/交易信息报告。
        """

        data_trans = interface.get_finnhub_company_insider_transactions(
            ticker, curr_date, 30
        )

        return data_trans

    @staticmethod
    @tool
    def get_simfin_balance_sheet(
        ticker: Annotated[str, "公司股票代码"],
        freq: Annotated[
            str,
            "公司财务历史报告频率：年度/季度",
        ],
        curr_date: Annotated[str, "您当前交易的日期，yyyy-mm-dd"],
    ):
        """
        获取公司最新资产负债表。
        Args:
            ticker (str): 公司股票代码
            freq (str): 公司财务历史报告频率：年度 / 季度
            curr_date (str): 您当前交易的日期，yyyy-mm-dd
        Returns:
            str: 公司最新资产负债表报告。
        """

        data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)

        return data_balance_sheet

    @staticmethod
    @tool
    def get_simfin_cashflow(
        ticker: Annotated[str, "公司股票代码"],
        freq: Annotated[
            str,
            "公司财务历史报告频率：年度/季度",
        ],
        curr_date: Annotated[str, "您当前交易的日期，yyyy-mm-dd"],
    ):
        """
        获取公司最新现金流量表。
        Args:
            ticker (str): 公司股票代码
            freq (str): 公司财务历史报告频率：年度 / 季度
            curr_date (str): 您当前交易的日期，yyyy-mm-dd
        Returns:
                str: 公司最新现金流量表报告。
        """

        data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)

        return data_cashflow

    @staticmethod
    @tool
    def get_simfin_income_stmt(
        ticker: Annotated[str, "公司股票代码"],
        freq: Annotated[
            str,
            "公司财务历史报告频率：年度/季度",
        ],
        curr_date: Annotated[str, "您当前交易的日期，yyyy-mm-dd"],
    ):
        """
        获取公司最新损益表。
        Args:
            ticker (str): 公司股票代码
            freq (str): 公司财务历史报告频率：年度 / 季度
            curr_date (str): 您当前交易的日期，yyyy-mm-dd
        Returns:
                str: 公司最新损益表报告。
        """

        data_income_stmt = interface.get_simfin_income_statements(
            ticker, freq, curr_date
        )

        return data_income_stmt

    @staticmethod
    @tool
    def get_google_news(
        query: Annotated[str, "搜索查询"],
        curr_date: Annotated[str, "当前日期，格式为yyyy-mm-dd"],
    ):
        """
        根据查询和日期范围从Google News获取最新新闻。
        Args:
            query (str): 搜索查询
            curr_date (str): 当前日期，格式为yyyy-mm-dd
            look_back_days (int): 回看天数
        Returns:
            str: 包含根据查询和日期范围从Google News获取的最新新闻的字符串。
        """

        google_news_results = interface.get_google_news(query, curr_date, 7)

        return google_news_results

    @staticmethod
    @tool
    def get_stock_news_openai(
        ticker: Annotated[str, "公司股票代码"],
        curr_date: Annotated[str, "当前日期，格式为yyyy-mm-dd"],
    ):
        """
        使用OpenAI的新闻API获取给定股票的最新新闻。
        Args:
            ticker (str): 公司股票代码。例如AAPL, TSM
            curr_date (str): 当前日期，格式为yyyy-mm-dd
        Returns:
            str: 包含给定日期公司最新新闻的字符串。
        """

        openai_news_results = interface.get_stock_news_openai(ticker, curr_date)

        return openai_news_results

    @staticmethod
    @tool
    def get_global_news_openai(
        curr_date: Annotated[str, "当前日期，格式为yyyy-mm-dd"],
    ):
        """
        使用OpenAI的宏观经济新闻API获取给定日期的最新宏观经济新闻。
        Args:
            curr_date (str): 当前日期，格式为yyyy-mm-dd
        Returns:
            str: 包含给定日期最新宏观经济新闻的字符串。
        """

        openai_news_results = interface.get_global_news_openai(curr_date)

        return openai_news_results

    @staticmethod
    @tool
    def get_fundamentals_openai(
        ticker: Annotated[str, "公司股票代码"],
        curr_date: Annotated[str, "当前日期，格式为yyyy-mm-dd"],
    ):
        """
        使用OpenAI的新闻API获取给定股票在给定日期的最新基本面信息。
        Args:
            ticker (str): 公司股票代码。例如AAPL, TSM
            curr_date (str): 当前日期，格式为yyyy-mm-dd
        Returns:
            str: 包含给定日期公司最新基本面信息的字符串。
        """

        openai_fundamentals_results = interface.get_fundamentals_openai(
            ticker, curr_date
        )

        return openai_fundamentals_results
