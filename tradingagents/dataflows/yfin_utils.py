# 获取数据/统计信息

import yfinance as yf
from typing import Annotated, Callable, Any, Optional
from pandas import DataFrame
import pandas as pd
from functools import wraps

from .utils import save_output, SavePathType, decorate_all_methods


def init_ticker(func: Callable) -> Callable:
    """装饰器：初始化yf.Ticker并传递给函数。"""

    @wraps(func)
    def wrapper(symbol: Annotated[str, "股票代码"], *args, **kwargs) -> Any:
        ticker = yf.Ticker(symbol)
        return func(ticker, *args, **kwargs)

    return wrapper


@decorate_all_methods(init_ticker)
class YFinanceUtils:

    def get_stock_data(
        symbol: Annotated[str, "股票代码"],
        start_date: Annotated[
            str, "获取股票价格数据的起始日期，YYYY-mm-dd"
        ],
        end_date: Annotated[
            str, "获取股票价格数据的结束日期，YYYY-mm-dd"
        ],
        save_path: SavePathType = None,
    ) -> DataFrame:
        """获取指定股票代码的价格数据"""
        ticker = symbol
        # 结束日期加一天，保证数据区间包含结束日
        end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        stock_data = ticker.history(start=start_date, end=end_date)
        # save_output(stock_data, f"Stock data for {ticker.ticker}", save_path)
        return stock_data

    def get_stock_info(
        symbol: Annotated[str, "股票代码"],
    ) -> dict:
        """获取并返回最新股票信息。"""
        ticker = symbol
        stock_info = ticker.info
        return stock_info

    def get_company_info(
        symbol: Annotated[str, "股票代码"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """获取并返回公司信息（DataFrame）。"""
        ticker = symbol
        info = ticker.info
        company_info = {
            "公司名称": info.get("shortName", "N/A"),
            "行业": info.get("industry", "N/A"),
            "板块": info.get("sector", "N/A"),
            "国家": info.get("country", "N/A"),
            "官网": info.get("website", "N/A"),
        }
        company_info_df = DataFrame([company_info])
        if save_path:
            company_info_df.to_csv(save_path)
            print(f"{ticker.ticker} 的公司信息已保存到 {save_path}")
        return company_info_df

    def get_stock_dividends(
        symbol: Annotated[str, "股票代码"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """获取并返回最新分红数据（DataFrame）。"""
        ticker = symbol
        dividends = ticker.dividends
        if save_path:
            dividends.to_csv(save_path)
            print(f"{ticker.ticker} 的分红数据已保存到 {save_path}")
        return dividends

    def get_income_stmt(symbol: Annotated[str, "股票代码"]) -> DataFrame:
        """获取并返回公司最新利润表（DataFrame）。"""
        ticker = symbol
        income_stmt = ticker.financials
        return income_stmt

    def get_balance_sheet(symbol: Annotated[str, "股票代码"]) -> DataFrame:
        """获取并返回公司最新资产负债表（DataFrame）。"""
        ticker = symbol
        balance_sheet = ticker.balance_sheet
        return balance_sheet

    def get_cash_flow(symbol: Annotated[str, "股票代码"]) -> DataFrame:
        """获取并返回公司最新现金流量表（DataFrame）。"""
        ticker = symbol
        cash_flow = ticker.cashflow
        return cash_flow

    def get_analyst_recommendations(symbol: Annotated[str, "股票代码"]) -> tuple:
        """获取最新分析师评级，返回最常见评级及其数量。"""
        ticker = symbol
        recommendations = ticker.recommendations
        if recommendations.empty:
            return None, 0  # 无评级信息

        # 假设存在'period'列，需要排除
        row_0 = recommendations.iloc[0, 1:]  # 排除'period'列

        # 找到最大投票结果
        max_votes = row_0.max()
        majority_voting_result = row_0[row_0 == max_votes].index.tolist()

        return majority_voting_result[0], max_votes
