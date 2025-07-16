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
    def get_stock_data_online(
        symbol: Annotated[str, "股票代码（支持A股、港股、美股），如AAPL、600519.SS、00700.HK"],
        start_date: Annotated[str, "开始日期，格式为yyyy-mm-dd"],
        end_date: Annotated[str, "结束日期，格式为yyyy-mm-dd"],
    ) -> str:
        """
        通用股票数据查询工具，支持A股、港股、美股。
        Args:
            symbol (str): 股票代码（如AAPL、600519.SS、00700.HK）
            start_date (str): 开始日期，格式为yyyy-mm-dd
            end_date (str): 结束日期，格式为yyyy-mm-dd
        Returns:
            str: 指定股票在指定日期范围内的行情数据（CSV字符串）。
        """
        import re
        try:
            # A股：6位纯数字
            if re.fullmatch(r"\d{6}", symbol):
                try:
                    import akshare as ak
                except ImportError:
                    return "未安装akshare库，无法查询A股行情。请先 pip install akshare"
                stock_df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""), adjust="qfq")
                if stock_df.empty:
                    return f"未找到A股 {symbol} 在 {start_date} 和 {end_date} 之间的数据"
                stock_df = stock_df.rename(columns={"日期": "Date", "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low", "成交量": "Volume"})
                csv_string = stock_df.to_csv(index=False)
                header = f"# 股票数据，{symbol}（A股），时间段：{start_date} 至 {end_date}\n# 总记录数：{len(stock_df)}\n# 数据获取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                return header + csv_string
            # 港股：5位数字，或0开头5位数字
            elif re.fullmatch(r"0?\d{5}", symbol):
                try:
                    import akshare as ak
                except ImportError:
                    return "未安装akshare库，无法查询港股行情。请先 pip install akshare"
                stock_df = ak.stock_hk_daily(symbol=symbol)
                # akshare港股接口返回全量，需按日期过滤
                stock_df = stock_df[(stock_df['date'] >= start_date) & (stock_df['date'] <= end_date)]
                if stock_df.empty:
                    return f"未找到港股 {symbol} 在 {start_date} 和 {end_date} 之间的数据"
                stock_df = stock_df.rename(columns={"date": "Date", "open": "Open", "close": "Close", "high": "High", "low": "Low", "volume": "Volume"})
                csv_string = stock_df.to_csv(index=False)
                header = f"# 股票数据，{symbol}（港股），时间段：{start_date} 至 {end_date}\n# 总记录数：{len(stock_df)}\n# 数据获取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                return header + csv_string
            # 美股/其它
            else:
                try:
                    import yfinance as yf
                except ImportError:
                    return "未安装yfinance库，无法查询美股行情。请先 pip install yfinance"
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if data.empty:
                    return f"未找到美股 {symbol} 在 {start_date} 和 {end_date} 之间的数据"
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = data[col].round(2)
                csv_string = data.to_csv()
                header = f"# 股票数据，{symbol}（美股/国际），时间段：{start_date} 至 {end_date}\n# 总记录数：{len(data)}\n# 数据获取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                return header + csv_string
        except Exception as e:
            return f"查询股票数据时发生错误: {e}"

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
    def get_news_info_online(
        symbol: Annotated[str, "股票代码（支持A股、港股、美股），如AAPL、600519、00700"],
        curr_date: Annotated[str, "当前日期，格式为yyyy-mm-dd"],
        look_back_days: Annotated[int, "回溯天数，默认7天"] = 7,
    ) -> str:
        """
        通用公司新闻资讯查询工具，支持A股、港股、美股。
        Args:
            symbol (str): 股票代码（如AAPL、600519、00700）
            curr_date (str): 当前日期，格式为yyyy-mm-dd
            look_back_days (int): 回溯天数，默认7天
        Returns:
            str: 公司相关新闻资讯。
        """
        import re
        try:
            if re.fullmatch(r"\d{6}", symbol):  # A股
                try:
                    import akshare as ak
                except ImportError:
                    return "未安装akshare库，无法查询A股新闻。请先 pip install akshare"
                df = ak.stock_news_em(symbol=symbol)
                if df.empty:
                    return f"未找到A股 {symbol} 的相关新闻"
                df = df[df['datetime'] >= (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")]
                news_str = "\n".join([f"{row['datetime']} | {row['title']}\n{row['content']}" for _, row in df.iterrows()])
                return f"# {symbol}（A股）新闻资讯（近{look_back_days}天）\n" + news_str
            elif re.fullmatch(r"0?\d{5}", symbol):  # 港股
                try:
                    import akshare as ak
                except ImportError:
                    return "未安装akshare库，无法查询港股新闻。请先 pip install akshare"
                df = ak.stock_hk_news_em(symbol=symbol)
                if df.empty:
                    return f"未找到港股 {symbol} 的相关新闻"
                df = df[df['datetime'] >= (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")]
                news_str = "\n".join([f"{row['datetime']} | {row['title']}\n{row['content']}" for _, row in df.iterrows()])
                return f"# {symbol}（港股）新闻资讯（近{look_back_days}天）\n" + news_str
            else:  # 美股/其它
                try:
                    import yfinance as yf
                except ImportError:
                    return "未安装yfinance库，无法查询美股新闻。请先 pip install yfinance"
                ticker = yf.Ticker(symbol)
                news = ticker.news if hasattr(ticker, 'news') else []
                if not news:
                    # 兜底用Google News
                    from tradingagents.dataflows.interface import get_google_news
                    news_str = get_google_news(symbol, curr_date, look_back_days)
                    return f"# {symbol}（美股/国际）新闻资讯（近{look_back_days}天）\n" + news_str
                news_str = "\n".join([f"{item.get('providerPublishTime', '')} | {item.get('title', '')}\n{item.get('link', '')}" for item in news if 'title' in item])
                return f"# {symbol}（美股/国际）新闻资讯（近{look_back_days}天）\n" + news_str
        except Exception as e:
            return f"查询新闻资讯时发生错误: {e}"

    @staticmethod
    @tool
    def get_fundamentals_info_online(
        symbol: Annotated[str, "股票代码（支持A股、港股、美股），如AAPL、600519、00700"],
        curr_date: Annotated[str, "当前日期，格式为yyyy-mm-dd"],
    ) -> str:
        """
        通用公司资讯查询工具，支持A股、港股、美股。
        Args:
            symbol (str): 股票代码（如AAPL、600519、00700）
            curr_date (str): 当前日期，格式为yyyy-mm-dd
        Returns:
            str: 公司简介、主营、行业、官网等资讯。
        """
        import re
        try:
            if re.fullmatch(r"\d{6}", symbol):  # A股
                try:
                    import akshare as ak
                except ImportError:
                    return "未安装akshare库，无法查询A股公司资讯。请先 pip install akshare"
                try:
                    df = ak.stock_zh_a_profile(symbol=symbol)
                    if df.empty:
                        raise ValueError("empty")
                    info = df.iloc[0].to_dict()
                    return f"# {symbol}（A股）公司资讯\n公司名称: {info.get('公司名称', '')}\n主营业务: {info.get('主营业务', '')}\n所属行业: {info.get('所属行业', '')}\n公司网址: {info.get('公司网址', '')}\n上市日期: {info.get('上市日期', '')}\n"
                except Exception:
                    # 降级用spot接口
                    df_spot = ak.stock_zh_a_spot_em()
                    row = df_spot[df_spot['代码'] == symbol]
                    if row.empty:
                        return f"未找到A股 {symbol} 的公司资讯"
                    info = row.iloc[0].to_dict()
                    return f"# {symbol}（A股）公司资讯\n公司名称: {info.get('名称', '')}\n所属行业: {info.get('行业', '')}\n最新价: {info.get('最新价', '')}\n市盈率: {info.get('市盈率', '')}\n"
            elif re.fullmatch(r"0?\d{5}", symbol):  # 港股
                try:
                    import akshare as ak
                except ImportError:
                    return "未安装akshare库，无法查询港股公司资讯。请先 pip install akshare"
                df = ak.stock_hk_company_profile_em(symbol=symbol)
                if df.empty:
                    return f"未找到港股 {symbol} 的公司资讯"
                info = df.iloc[0].to_dict()
                return f"# {symbol}（港股）公司资讯\n公司名称: {info.get('证券简称', '')}\n主营业务: {info.get('主营业务', '')}\n所属行业: {info.get('所属行业', '')}\n公司网址: {info.get('公司网址', '')}\n上市日期: {info.get('上市日期', '')}\n"
            else:  # 美股/其它
                try:
                    import yfinance as yf
                except ImportError:
                    return "未安装yfinance库，无法查询美股公司资讯。请先 pip install yfinance"
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return f"# {symbol}（美股/国际）公司资讯\n公司名称: {info.get('shortName', '')}\n主营业务: {info.get('longBusinessSummary', '')}\n所属行业: {info.get('industry', '')}\n公司网址: {info.get('website', '')}\n上市日期: {info.get('ipoDate', '') if 'ipoDate' in info else ''}\n"
        except Exception as e:
            return f"查询公司资讯时发生错误: {e}"
