import pyodbc
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
from typing import Iterable, Literal, Optional

def make_list(a: Iterable):
    return ','.join(f"'{code}'" for code in a)

# @st.cache_resource
class Get_data():
    """
    初始化获取数据链接。
    
    参数：
    index_code(str): 指定选取的股票所来自的指数的代码，规定为string类，默认为沪深300，即'000300'。
    UID: 连接数据库的用户名（除非你知道自己在做什么，否则不要更改默认值。
    PWD: 连接数据库的密码（除非你知道自己在做什么，否则不要更改默认值。
    SERVER: 连接数据库的服务器（除非你知道自己在做什么，否则不要更改默认值。
    DATABASE: 连接数据库的数据库名称（除非你知道自己在做什么，否则不要更改默认值。
    """
    
    def __init__(self, index_code: str="000300", UID: str="syzx_jydb005",
                 PWD: str="KJGCNKRn", SERVER: str="10.2.47.124", DATABASE: str="JYDB"):
    # def __init__(self, index_code: str = "000300", UID: str = st.secrets["username"],
    #                  PWD: str = st.secrets["password"], SERVER: str = st.secrets["server"], DATABASE: str = st.secrets["database"]):
        self.index_code = index_code
        # self.con = pyodbc.connect(f"SERVER={st.secrets["server"]};UID={st.secrets["username"]};PWD={st.secrets["password"]};DRIVER={{ODBC Driver 17 for SQL Server}};PORT=1433;DATABASE={st.secrets["database"]}")
        self.con = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"])
    
    def get_stock_info(self):
        """
        依据初始化时的index_code来获取当前指数内的成分股信息。

        输出:
        一个dataframe。
        """
        self.stock_info = pd.read_sql(f"""
                            SELECT 
                            stock.SecuCode as stock_code,
                            stock.ChiName as stock_name,
                            idx_cp.SecuInnerCode as stock_inner_code,
                            sm.SecuCode as index_code,
                            sm.ChiName as index_name
                            FROM LC_IndexComponent AS idx_cp 
                            INNER JOIN
                            SecuMain as sm ON idx_cp.IndexInnerCode = sm.InnerCode
                            INNER JOIN
                            SecuMain as stock ON idx_cp.SecuInnerCode = stock.InnerCode
                            WHERE idx_cp.OutDate IS NULL
                            AND
                            sm.SecuCode = '{self.index_code}'
                            """, self.con)
        self.stock_code = self.stock_info["stock_code"].to_list()
        return self.stock_info
    
    def get_stock_return(self, stocks: Iterable, interval: Optional[Literal["三年", "一年", "半年", "一月"]]=None,
                     start_date: date=None, end_date: Optional[date]="今天"):
        """
        获取股票在一段时间内的收益率、成交量等信息。

        参数: 
        stocks: 要获取的股票的代码，应为list等iterable类型，元素应为string类型。
        interval: 获取信息的时间窗口的长度。若为None，则以start_date为标准。
        start_date: 获取信息的起始日期，输入应为'2024-01-01'的日期格式。如果同时指定了interval，则以interval为标准。
        end_date: 获取信息的结束日期，输入应为'2024-01-01'的日期格式。默认值为今天的日期。

        输出: 一个dataframe。
        """

        stocks_list = make_list(stocks)
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

        if self.end_date == "今天":
            self.end_date = datetime.now()
        else:
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        if self.interval is None and self.start_date is not None:
            self.end_date = self.end_date.strftime("%Y-%m-%d")
        elif self.interval is None and self.start_date is None:
            return "请指定数据搜集的时间窗口！"
        else:
            if self.interval is not None and self.start_date is not None:
                print("以给定的interval为标准。")
            if self.interval == "三年":
                self.start_date = self.end_date - timedelta(days=365*3)
            elif self.interval == "一年":
                self.start_date = self.end_date - timedelta(days=365)
            elif self.interval == "半年":
                self.start_date = self.end_date - timedelta(days=180)
            else:
                self.start_date = self.end_date - timedelta(days=30)
            self.start_date = self.start_date.strftime("%Y-%m-%d")
            self.end_date = self.end_date.strftime("%Y-%m-%d")

        self.stock_return = pd.read_sql(f"""
                SELECT
                qt.InnerCode as stock_inner_code,
                stock.SecuCode as stock_code,
                stock.ChiName as stock_name,
                qt.TradingDay as trade_date,
                p.ChangePCT as return_pct,
                qt.PrevClosePrice as prev_close_price,
                qt.ClosePrice as close_price,
                qt.TurnoverVolume as turn_over_volume,
                qt.TurnoverValue as turn_over_value,
                qt.TurnoverDeals AS turn_over_deals
                FROM
                QT_DailyQuote AS qt
                INNER JOIN
                SecuMain AS stock ON qt.InnerCode = stock.InnerCode
                INNER JOIN
                QT_PerformanceData AS p ON qt.InnerCode = p.InnerCode 
                AND qt.TradingDay = p.TradingDay
                WHERE
                stock.SecuCode in ({stocks_list})
                AND
                qt.TradingDay BETWEEN '{self.start_date}' AND '{self.end_date}'
                ORDER BY qt.TradingDay DESC
                """, self.con)
        return self.stock_return
    
    def get_risk_free_rate(self, risk_free: Literal["一年国债", "活期", "三月定期",
                                                            "半年定期", "一年定期", "二年定期",
                                                            "三年定期", "五年定期"]="一年定期"):
        """
        获取无风险利率信息。

        参数:
        risk_free: 无风险利率的标准。

        输出:
        一个dataframe。
        """
        self.risk_free = risk_free
        if self.risk_free != "一年国债":
            self.risk_free_rate = pd.read_sql(f"""
                                            SELECT
                                            ir.EndDate as trade_date,
                                            ir.IndexDD as dd,
                                            ir.IndexTD3M as td3m,
                                            ir.IndexTD6M as td6m,
                                            ir.IndexTD1Y as td1y,
                                            ir.IndexTD2Y as td2y,
                                            ir.IndexTD3Y as td3y,
                                            ir.IndexTD5Y as td5y
                                            FROM QT_InterestRateIndex as ir
                                            WHERE ir.EndDate BETWEEN '{self.start_date}' AND '{self.end_date}'
                                            ORDER BY ir.EndDate DESC
                                            """, self.con)
            if self.risk_free == "活期":
                return self.risk_free_rate[["trade_date", "dd"]]
            elif self.risk_free == "三月定期":
                return self.risk_free_rate[["trade_date", "td3m"]]
            elif self.risk_free == "半年定期":
                return self.risk_free_rate[["trade_date", "td6m"]]
            elif self.risk_free == "一年定期":
                return self.risk_free_rate[["trade_date", "td1y"]]
            elif self.risk_free == "二年定期":
                return self.risk_free_rate[["trade_date", "td2y"]]
            elif self.risk_free == "三年定期":
                return self.risk_free_rate[["trade_date", "td3y"]]
            else:
                return self.risk_free_rate[["trade_date", "td5y"]]
        else:
            pass
        

    def get_market_return(self, market_code: str="000300"):
        """
        获取市场收益率等信息。

        参数:
        market_code: 市场指数的代码，要求输入为string类。默认值为沪深300，即'000300'。
        
        输出:
        一个dataframe
        """
        self.market_code = market_code
        self.market_return = pd.read_sql(f"""
                                        SELECT
                                        main.SecuCode as index_code,
                                        main.ChiName as index_name,
                                        main.InnerCode as index_inner_code,
                                        cs.TradingDay as trade_date,
                                        cs.ChangePCT as return_pct,
                                        cs.PrevClosePrice as prev_close_price,
                                        cs.ClosePrice as close_price
                                        FROM QT_CSIIndexQuote as cs
                                        JOIN SecuMain as main ON cs.IndexCode = main.InnerCode
                                        WHERE main.SecuCode = '{self.market_code}'
                                        AND
                                        cs.TradingDay BETWEEN '{self.start_date}' AND '{self.end_date}'
                                        ORDER BY cs.TradingDay DESC
                                        """, self.con)
        return self.market_return
