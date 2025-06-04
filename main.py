import streamlit as st
from utils import Get_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np
import statsmodels.api as sm
from pylab import plt
from scipy.optimize import minimize


data = Get_data()

if 'stock_list' not in st.session_state:
    st.session_state.stock_list = []
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = {}
if 'market_rate' not in st.session_state:
    st.session_state.market_rate = {}
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'time_span' not in st.session_state:
    st.session_state.time_span = "三年"

st.set_page_config(
    page_title="智能股票系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def explanation_page():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.api as sm

    st.title("📘 说明文档")
    st.markdown("通过本页面，你可以了解三个核心金融分析工具：**布林带、有效前沿、Beta 分析**")

    tab1, tab2, tab3 = st.tabs(["📉 布林带", "📈 Beta分析", "🧮 有效前沿分析", ])

    with tab1:
        st.header("📉 布林带")

        st.markdown("""
        布林带是一种广泛应用于股票技术分析的工具，由技术分析大师约翰·布林格（John Bollinger）在1980年代提出。

        它由三条线组成：

        1. **中轨线（Middle Band）**：一般为20日简单移动平均线（SMA）
        2. **上轨线（Upper Band）**：中轨线加上两倍的标准差
        3. **下轨线（Lower Band）**：中轨线减去两倍的标准差

        布林带的原理是：当价格接近上轨时，表示股价处于相对高位；当价格接近下轨时，表示股价处于相对低位。

        下图是布林带的数学公式：
        """)

        st.latex(r'''
        \begin{aligned}
        \text{上轨} &= \text{SMA}_n + k \cdot \sigma \\
        \text{中轨} &= \text{SMA}_n \\
        \text{下轨} &= \text{SMA}_n - k \cdot \sigma
        \end{aligned}
        ''')

        st.markdown("""
        - \\( $n$ \\)：移动平均的周期，通常取 20 天
        - \\( $\sigma$ \\)：该周期内的价格标准差
        - \\( $k$ \\)：倍数，通常取 2

        布林带不仅反映了价格的趋势（通过中轨），还反映了价格波动性（通过带宽宽度）。带宽扩大，表示波动加剧；带宽收窄，表示市场趋于稳定。

        下图展示了一个布林带的示意图（基于模拟数据）：
        """)


        np.random.seed(42)
        x = pd.date_range(start="2022-01-01", periods=100)
        price = np.cumsum(np.random.randn(100)) + 100
        df = pd.DataFrame({"date": x, "price": price})
        df["SMA20"] = df["price"].rolling(window=20).mean()
        df["STD"] = df["price"].rolling(window=20).std()
        df["Upper"] = df["SMA20"] + 2 * df["STD"]
        df["Lower"] = df["SMA20"] - 2 * df["STD"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["price"], mode='lines', name='价格', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA20"], mode='lines', name='中轨', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["Upper"], mode='lines', name='上轨', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["Lower"], mode='lines', name='下轨', line=dict(color='red')))
        fig.update_layout(title="布林带示意图", xaxis_title="时间", yaxis_title="价格", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📈 Beta分析")

        st.markdown("""
        **Beta（β）系数**是衡量单只股票对整个市场变动的敏感程度的重要指标。

        - 如果 β = 1，表示该股票与市场整体波动完全一致；
        - 如果 β > 1，表示股票波动大于市场，风险更高；
        - 如果 β < 1，表示股票波动小于市场，风险更低；
        - 如果 β < 0，表示股票与市场走势相反（较少见）

        Beta 值的估算通常通过回归模型来实现。我们对**股票的超额收益**（减去无风险利率）与**市场超额收益**进行线性回归：

        """)

        st.latex(r'''
        R_i - R_f = \alpha + \beta \cdot (R_m - R_f) + \epsilon
        ''')

        st.markdown(r"""
        - ( $R_i$ )：股票的收益率  
        - ( $R_m$ )：市场的收益率（如沪深300指数）  
        - ( $R_f$ )：无风险利率（可设为 0）  
        - ( $\alpha$ )：截距（Alpha），表示主动收益  
        - ( $\beta$ )：回归斜率，即 Beta 值  
        - ( $\epsilon$ )：误差项

        下图展示了一个模拟数据的回归分析图，包括散点图与拟合直线：
        """)

        np.random.seed(42)
        market_excess = np.random.normal(0, 0.02, 100)
        beta_true = 1.2
        stock_excess = beta_true * market_excess + np.random.normal(0, 0.01, 100)

        df_beta = pd.DataFrame({
            "市场超额收益": market_excess,
            "股票超额收益": stock_excess
        })

        X = sm.add_constant(df_beta["市场超额收益"])
        model = sm.OLS(df_beta["股票超额收益"], X).fit()
        beta_est = model.params[1]
        alpha_est = model.params[0]

        st.markdown(f"""
        **线性回归结果：**

        - Alpha（截距）≈ {alpha_est:.4f}  
        - **Beta（斜率）≈ {beta_est:.4f}**
        """)

        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=df_beta["市场超额收益"],
            y=df_beta["股票超额收益"],
            mode="markers",
            name="数据点",
            marker=dict(color="skyblue", size=6)
        ))
        fig_beta.add_trace(go.Scatter(
            x=df_beta["市场超额收益"],
            y=model.predict(X),
            mode="lines",
            name="回归线",
            line=dict(color="red")
        ))
        fig_beta.update_layout(
            title="Beta 回归图",
            xaxis_title="市场超额收益",
            yaxis_title="股票超额收益",
            template="plotly_white"
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    with tab3:
        st.header("🧮 有效前沿（Efficient Frontier）")

        st.markdown("""
        在投资组合理论中，有效前沿是由哈里·马克维茨（Harry Markowitz）提出的概念。

        有效前沿是所有风险与收益最优的投资组合所组成的边界。换句话说：

        - 在相同的风险下，有效前沿上的组合提供最高的收益
        - 在相同的收益目标下，有效前沿上的组合具有最低的风险

        投资组合的方差（风险）和期望收益可以通过以下公式计算：
        """)

        st.latex(r'''
        \text{期望收益率} = \mathbf{w}^T \mathbf{\mu} \\
        \text{组合方差（风险）} = \mathbf{w}^T \Sigma \mathbf{w}
        ''')

        st.markdown("""
        - \\( $\mathbf{w}$ \\)：投资组合的权重向量  
        - \\( $\mathbf{\mu}$ \\)：各资产的期望收益率向量  
        - \\( $\Sigma$ \\)：资产的协方差矩阵  

        有效前沿曲线一般呈向上弯曲的抛物线状，代表“高风险高收益”。

        下图展示了一个模拟生成的有效前沿图：
        """)

        # 模拟有效前沿
        np.random.seed(42)
        num_portfolios = 5000
        mean_returns = np.array([0.12, 0.10, 0.08])
        cov_matrix = np.array([
            [0.006, 0.002, 0.001],
            [0.002, 0.005, 0.002],
            [0.001, 0.002, 0.004]
        ])

        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(3)
            weights /= np.sum(weights)
            returns = np.dot(weights, mean_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = returns / volatility
            results[0, i] = volatility
            results[1, i] = returns
            results[2, i] = sharpe

        ef_fig = go.Figure(data=go.Scatter(
            x=results[0, :], y=results[1, :],
            mode="markers", marker=dict(color=results[2, :], colorscale="Viridis", colorbar=dict(title="夏普比率")),
        ))
        ef_fig.update_layout(
            title="模拟有效前沿图",
            xaxis_title="组合风险（标准差）",
            yaxis_title="预期收益率",
            template="plotly_white"
        )
        st.plotly_chart(ef_fig, use_container_width=True)
        



def stock_page():
    st.title("📋 股票信息与选择")
    st.write("您可以在下方浏览所有可用股票，并通过按钮添加或移除它们。")

    new_time_span = st.selectbox("选择时间跨度", ["三年", "一年", "半年"], index=["三年", "一年", "半年"].index(st.session_state.time_span))

    if new_time_span != st.session_state.time_span:
        st.session_state.time_span = new_time_span
        try:
            updated_stock_data = {}
            for stock_code in st.session_state.stock_list:
                stock_data = data.get_stock_return([stock_code], interval=new_time_span, end_date="今天")
                if stock_data is not None:
                    updated_stock_data[stock_code] = stock_data
                else:
                    st.warning(f"无法获取 {stock_code} 的数据")
            st.session_state.stock_data = updated_stock_data

            risk_free_data = data.get_risk_free_rate(new_time_span + "定期").iloc[:, 1]
            st.session_state.risk_free_rate['return_series'] = risk_free_data / risk_free_data.shift(1) - 1

            market_data = data.get_market_return()
            st.session_state.market_rate['return_series'] = market_data['close_price'] / market_data['prev_close_price'] - 1

            st.success(f"时间跨度已更新为 {new_time_span}，所有数据已刷新。")
            st.rerun()
        except Exception as e:
            st.error(f"更新时间跨度数据出错: {str(e)}")

    # ====== 显示已添加的股票集合 ======
    if st.session_state.stock_list:
        st.markdown("## ✅ 已选股票")
        
        try:
            stock_info_df = data.get_stock_info()
        except Exception as e:
            st.warning(f"无法获取股票信息: {str(e)}")
            stock_info_df = pd.DataFrame(columns=['stock_code', 'stock_name'])

        added_cols = st.columns(4)
        for idx, stock_code in enumerate(st.session_state.stock_list):
            with added_cols[idx % 4]:
                stock_name = stock_info_df.loc[
                    stock_info_df['stock_code'] == stock_code, 'stock_name'
                ].values[0] if stock_code in stock_info_df['stock_code'].values else "未知名称"
                
                st.markdown(f"""
                <div class="stock-card">
                    <div class="stock-code">{stock_code}</div>
                    <div class="stock-name">{stock_name}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"➖ 移除 {stock_code}", key=f"remove_top_{stock_code}"):
                    st.session_state.stock_list.remove(stock_code)
                    st.session_state.stock_data.pop(stock_code, None)
                    if st.session_state.selected_stock == stock_code:
                        st.session_state.selected_stock = None
                    st.rerun()
    else:
        st.info("暂无已添加的股票，请从下方列表中添加。")


    try:
        info = pd.read_csv('/Users/apple/Downloads/FBD/stock_available.csv', delimiter=",", dtype={'stock_code': str})
        info = info[['stock_code', 'stock_name']].assign(stock_code=lambda x: x['stock_code'].astype(str))

        st.markdown("""
        <style>
            .stock-card {
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .stock-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            .stock-code {
                font-weight: bold;
                font-size: 1.1rem;
                color: #1f77b4;
            }
            .stock-name {
                color: #2a3f5f;
                margin-top: 5px;
            }
        </style>
        """, unsafe_allow_html=True)

        # 搜索和分页
        col1, col2 = st.columns([2,1])
        with col1:
            search_term = st.text_input("🔍 搜索股票代码或名称", "")
        with col2:
            items_per_page = st.selectbox("每页显示数量", [10, 25, 50, 100], index=1)

        if search_term:
            mask = (info['stock_code'].str.contains(search_term, case=False)) | \
                   (info['stock_name'].str.contains(search_term, case=False))
            filtered_info = info[mask].copy()
        else:
            filtered_info = info.copy()

        total_pages = max(1, (len(filtered_info) + items_per_page - 1) // items_per_page)
        page_number = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_info))
        paginated_info = filtered_info.iloc[start_idx:end_idx]

        st.markdown("## 📌 可选股票")

        cols = st.columns(3)
        for idx, (_, row) in enumerate(paginated_info.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="stock-card">
                    <div class="stock-code">{row['stock_code']}</div>
                    <div class="stock-name">{row['stock_name']}</div>
                </div>
                """, unsafe_allow_html=True)

                stock_code = row['stock_code']
                stock_name = row['stock_name']

                if stock_code in st.session_state.stock_list:
                    if st.button(f"➖ 移除 {stock_code}", key=f"remove_{stock_code}"):
                        st.session_state.stock_list.remove(stock_code)
                        st.session_state.stock_data.pop(stock_code, None)
                        if st.session_state.selected_stock == stock_code:
                            st.session_state.selected_stock = None
                        st.rerun()
                else:
                    if st.button(f"➕ 添加 {stock_code}", key=f"add_{stock_code}"):
                        try:
                            # stock_data = data.get_stock_return([stock_code], interval="三年", end_date="今天")
                            stock_data = data.get_stock_return([stock_code], interval=st.session_state.time_span, end_date="今天")
                            # risk_free_data = data.get_risk_free_rate("三年定期")['td3y']
                            # risk_free_data = data.get_risk_free_rate("三年"+"定期").iloc[:,1]
                            risk_free_data = data.get_risk_free_rate(st.session_state.time_span + "定期").iloc[:, 1]
                            risk_free_return = risk_free_data / risk_free_data.shift(1) - 1
                            market_data = data.get_market_return()
                            market_return = market_data['close_price'] / market_data['prev_close_price'] - 1
                            st.session_state.risk_free_rate['return_series'] = risk_free_return
                            st.session_state.market_rate['return_series'] = market_return
                            if stock_data is not None:
                                st.session_state.stock_list.append(stock_code)
                                st.session_state.stock_data[stock_code] = stock_data
                                st.success(f"成功添加股票 {stock_code}")
                                st.rerun()
                            else:
                                st.error(f"无法获取股票 {stock_code} 的数据")
                        except Exception as e:
                            st.error(f"获取数据出错: {str(e)}")

        st.markdown(f"**显示 {start_idx + 1}-{end_idx} 条，共 {len(filtered_info)} 条股票信息**")

        st.download_button(
            label="📥 下载完整股票列表 (CSV)",
            data=filtered_info.to_csv(index=False).encode('utf-8'),
            file_name='stock_list.csv',
            mime='text/csv'
        )

    except FileNotFoundError:
        st.error("找不到股票信息文件 stock_available.csv")
    except Exception as e:
        st.error(f"加载股票信息出错: {str(e)}")



def history_page():
    st.title("📈 历史行情")
    
    if not st.session_state.stock_list:
        st.warning("请先在'股票选择'页面添加股票")
        return
    
    st.session_state.selected_stock = st.selectbox(
        "选择股票",
        st.session_state.stock_list,
        index=0
    )
    
    if st.session_state.selected_stock:
        selected_stock = st.session_state.selected_stock
        stock_data = st.session_state.stock_data[selected_stock]
        
        try:
            stock_info_df = data.get_stock_info()
            stock_name = stock_info_df.loc[
                stock_info_df['stock_code'] == selected_stock, 'stock_name'
            ].values[0] if selected_stock in stock_info_df['stock_code'].values else "未知名称"
        except:
            stock_name = "未知名称"
        
        st.subheader(f"{selected_stock} {stock_name}")
        
        if 'close_price' in stock_data:
            create_professional_chart(selected_stock, stock_name, stock_data)

def create_professional_chart(stock_code, stock_name, stock_data):
    price_df = pd.DataFrame({
        '日期': stock_data['trade_date'].iloc[::-1],
        '收盘价': stock_data['close_price'].iloc[::-1]
    })
    
    price_df['30日均线'] = price_df['收盘价'].rolling(window=30).mean()
    price_df['60日均线'] = price_df['收盘价'].rolling(window=60).mean()
    price_df['20日均线'] = price_df['收盘价'].rolling(window=20).mean()
    price_df['上轨'] = price_df['20日均线'] + 2*price_df['收盘价'].rolling(window=20).std()
    price_df['下轨'] = price_df['20日均线'] - 2*price_df['收盘价'].rolling(window=20).std()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_df['日期'],
        y=price_df['收盘价'],
        name='收盘价',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{y:.2f}元<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=price_df['日期'],
        y=price_df['30日均线'],
        name='30日均线',
        line=dict(color='#ff7f0e', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_df['日期'],
        y=price_df['60日均线'],
        name='60日均线',
        line=dict(color='#2ca02c', width=1.5)
    ))
    
    # 添加布林带
    fig.add_trace(go.Scatter(
        x=price_df['日期'],
        y=price_df['上轨'],
        name='布林带上轨',
        line=dict(color='rgba(214, 39, 40, 0.5)', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_df['日期'],
        y=price_df['下轨'],
        name='布林带下轨',
        line=dict(color='rgba(214, 39, 40, 0.5)', width=1),
        fillcolor='rgba(214, 39, 40, 0.1)',
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f"{stock_code} {stock_name}",
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=3, label="3月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all", label="全部")
                ])
            )
        ),
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


with st.sidebar:
    selected = option_menu(
        "📚 功能导航",
        ["说明文档", "股票选择", "历史价格", "Beta分析", "有效前沿分析"],
        icons=["book", "database", "bar-chart-line", "activity", "bounding-box"],
        menu_icon="cast",  # 侧边栏标题左边的小图标
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "#00BFFF", "font-size": "20px"},  # 图标颜色
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#00BFFF", "color": "white"},
        }
    )

def Beta_page():
    st.title("⚡️Beta分析")

    if not st.session_state.stock_list:
        st.warning("请先在'股票选择'页面添加股票")
        return
    
    st.session_state.selected_stock = st.selectbox(
        "选择股票",
        st.session_state.stock_list,
        index=0
    )
    
    if st.session_state.selected_stock:
        selected_stock = st.session_state.selected_stock
        stock_data = st.session_state.stock_data[selected_stock]
        
        try:
            stock_info_df = data.get_stock_info()
            stock_name = stock_info_df.loc[
                stock_info_df['stock_code'] == selected_stock, 'stock_name'
            ].values[0] if selected_stock in stock_info_df['stock_code'].values else "未知名称"
        except:
            stock_name = "未知名称"
        
        st.subheader(f"{selected_stock} {stock_name}")

        stock_return = stock_data['close_price'] / stock_data['prev_close_price'] - 1
        stock_excess_return = stock_return - st.session_state.risk_free_rate['return_series']
        market_excess_return = st.session_state.market_rate['return_series']

        df = pd.DataFrame({
            'stock_excess': stock_excess_return,
            'market_excess': market_excess_return
        }).dropna()
        
        if len(df) == 0:
            st.error("无有效数据可供分析")
            return
        
        X = df['market_excess']
        y = df['stock_excess']
        X = sm.add_constant(X)  # 添加截距项
        model = sm.OLS(y, X).fit()
        
        alpha = model.params['const']
        beta = model.params['market_excess']
        
        x_range = np.linspace(df['market_excess'].min(), df['market_excess'].max(), 100)
        y_pred = alpha + beta * x_range
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['market_excess'],
            y=df['stock_excess'],
            mode='markers',
            name='实际数据点',
            marker=dict(
                color='rgba(55, 128, 191, 0.7)',
                size=8,
                line=dict(
                    color='rgba(55, 128, 191, 1)',
                    width=0.5
                )
            ),
            hovertemplate='市场超额: %{x:.2%}<br>股票超额: %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'回归线 (β = {beta:.2f})',
            line=dict(
                color='rgba(214, 39, 40, 0.8)',
                width=3,
                dash='solid'
            ),
            hovertemplate='市场超额: %{x:.2%}<br>预期股票超额: %{y:.2%}<extra></extra>'
        ))
        
        predictions = model.get_prediction(sm.add_constant(x_range)).summary_frame()
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_range, x_range[::-1]]),
            y=np.concatenate([predictions['obs_ci_upper'], predictions['obs_ci_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(214, 39, 40, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% 置信区间',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            xaxis_title="市场超额收益率",
            yaxis_title="股票超额收益率",
            hovermode="closest",
            template="plotly_white",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, b=50, t=80, pad=4),
            annotations=[
                dict(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Alpha: {alpha:.4f}<br>Beta: {beta:.4f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="lightgray",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        fig.add_shape(type="line",
            x0=0, y0=df['stock_excess'].min(), x1=0, y1=df['stock_excess'].max(),
            line=dict(color="gray", width=1, dash="dot")
        )
        fig.add_shape(type="line",
            x0=df['market_excess'].min(), y0=0, x1=df['market_excess'].max(), y1=0,
            line=dict(color="gray", width=1, dash="dot")
        )
        
        st.plotly_chart(fig, use_container_width=True)



def efficient_frontier_page():
    st.title("📊 有效前沿分析")
    st.write("""
    ### 投资组合优化 - 有效前沿
    此页面展示基于马科维茨现代投资组合理论的有效前沿
    """)
    rf = st.slider(
        "设定无风险利率 (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.0, 
        step=0.1
    ) / 100

    if not st.session_state.stock_list:
        st.warning("请先在'股票选择'页面添加股票")
        return

    try:
        from pypfopt import risk_models, expected_returns
        from pypfopt.efficient_frontier import EfficientFrontier
    except ImportError:
        st.error("需要安装 PyPortfolioOpt 包: `pip install PyPortfolioOpt`")
        return

    import numpy as np
    import plotly.graph_objs as go

    price_data = {}
    stock_names = {}

    try:
        stock_info_df = data.get_stock_info()
    except Exception as e:
        st.warning(f"无法获取股票信息: {str(e)}")
        stock_info_df = pd.DataFrame(columns=['stock_code', 'stock_name'])

    for stock_code in st.session_state.stock_list:
        stock_data = st.session_state.stock_data[stock_code]
        stock_return = stock_data['close_price'] / stock_data['prev_close_price'] - 1
        # price_data[stock_code] = stock_data['close_price']
        price_data[stock_code] = stock_return

        stock_name = stock_info_df.loc[
            stock_info_df['stock_code'] == stock_code, 'stock_name'
        ].values[0] if stock_code in stock_info_df['stock_code'].values else "未知名称"
        stock_names[stock_code] = stock_name

    price_df = pd.DataFrame(price_data)

    if len(price_df) < 2:
        st.error("数据不足，无法计算有效前沿")
        return

    with st.spinner("正在计算有效前沿..."):
        try:

            mu = expected_returns.ema_historical_return(price_df, returns_data=True, log_returns=False)
            S = risk_models.sample_cov(price_df, returns_data=True)

            if all(mu <= rf):
                st.warning(
                    f"⚠️ 当前所有股票的年化预期收益率均不高于设定的无风险利率（{rf:.2%}），"
                    f"因此无法进行投资组合优化。\n\n"
                    "请尝试：\n"
                    "- 添加更多具有较高收益潜力的股票\n"
                    "- 或适当降低无风险利率设定"
                )
                return

            # 最优组合
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe(risk_free_rate=rf)
            max_sharpe_weights = ef.clean_weights()
            max_sharpe_perf = ef.portfolio_performance()

            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            min_vol_weights = ef.clean_weights()
            min_vol_perf = ef.portfolio_performance()

            # 随机组合模拟
            np.random.seed(42)
            n_samples = 5000
            weights = np.random.dirichlet(np.ones(len(mu)), n_samples)
            returns = []
            risks = []
            sharpes = []

            for w in weights:
                ret = np.dot(w, mu)
                risk = np.sqrt(np.dot(w.T, np.dot(S, w)))
                sharpe = ret / risk if risk != 0 else 0
                returns.append(ret)
                risks.append(risk)
                sharpes.append(sharpe)

            results_df = pd.DataFrame({
                "Return": returns,
                "Risk": risks,
                "Sharpe": sharpes
            })

            frontier_returns = []
            frontier_risks = []

            target_returns = np.linspace(min(returns), max(returns), 100)
            for tr in target_returns:
                try:
                    ef_tmp = EfficientFrontier(mu, S)
                    ef_tmp.efficient_return(target_return=tr)
                    ret, risk, _ = ef_tmp.portfolio_performance()
                    frontier_returns.append(ret)
                    frontier_risks.append(risk)
                except:
                    continue

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results_df['Risk'],
                y=results_df['Return'],
                mode='markers',
                name='随机组合',
                marker=dict(
                    color=results_df['Sharpe'],
                    colorscale='Viridis',
                    size=8,
                    opacity=0.6,
                    colorbar=dict(title='夏普比率')
                ),
                hovertemplate="风险: %{x:.2%}<br>收益: %{y:.2%}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=frontier_risks,
                y=frontier_returns,
                mode='lines',
                name='有效前沿',
                line=dict(color='#FF4B4B', width=3),
                hovertemplate="风险: %{x:.2%}<br>收益: %{y:.2%}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=[max_sharpe_perf[1]],
                y=[max_sharpe_perf[0]],
                mode='markers',
                name='最大夏普组合',
                marker=dict(color='#00CC96', size=15, symbol='star'),
                hovertemplate=f"最大夏普组合<br>风险: {max_sharpe_perf[1]:.2%}<br>收益: {max_sharpe_perf[0]:.2%}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=[min_vol_perf[1]],
                y=[min_vol_perf[0]],
                mode='markers',
                name='最小风险组合',
                marker=dict(color='#AB63FA', size=15, symbol='star'),
                hovertemplate=f"最小风险组合<br>风险: {min_vol_perf[1]:.2%}<br>收益: {min_vol_perf[0]:.2%}<extra></extra>"
            ))

            fig.update_layout(
                title='投资组合有效前沿',
                xaxis_title='年化风险 (标准差)',
                yaxis_title='年化预期收益',
                hovermode='closest',
                template='plotly_white',
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, b=50, t=80, pad=4)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("最优组合分析")
            col1, col2 = st.columns(2)

            def display_weights(title, weights_dict, perf, container):
                if not weights_dict:
                    st.info("暂无可用的最优组合结果。")
                    return
                with container:
                    st.markdown(f"### {title}")
                    st.write(f"**预期年化收益**: {perf[0]:.2%}")
                    st.write(f"**年化波动率**: {perf[1]:.2%}")
                    st.write(f"**夏普比率**: {perf[2]:.2f}")
                    weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['权重'])
                    weights_df.index.name = '股票代码'
                    weights_df['股票名称'] = weights_df.index.map(stock_names)
                    st.dataframe(
                        weights_df[['股票名称', '权重']].sort_values('权重', ascending=False).style.format({'权重': '{:.2%}'}),
                        height=400
                    )

            display_weights("最大夏普组合", max_sharpe_weights, max_sharpe_perf, col1)
            display_weights("最小波动组合", min_vol_weights, min_vol_perf, col2)

            # st.markdown("---")
            # st.subheader("自定义优化目标")

            # target_return = st.slider(
            #     "选择目标年化收益率 (%)",
            #     min_value=float(0),
            #     max_value=float(round(max_sharpe_perf[0]*100*1.5, 1)),
            #     value=float(round((max_sharpe_perf[0] + min_vol_perf[0])/2*100, 1)),
            #     step=0.1
            # ) / 100

            # try:
            #     ef_custom = EfficientFrontier(mu, S)
            #     ef_custom.efficient_return(target_return=target_return)
            #     custom_weights = ef_custom.clean_weights()
            #     custom_perf = ef_custom.portfolio_performance()

            #     st.markdown(f"### 目标收益 {target_return:.2%} 的最优组合")
            #     st.write(f"**实际年化收益**: {custom_perf[0]:.2%}")
            #     st.write(f"**年化波动率**: {custom_perf[1]:.2%}")
            #     st.write(f"**夏普比率**: {custom_perf[2]:.2f}")

            #     custom_weights_df = pd.DataFrame.from_dict(custom_weights, orient='index', columns=['权重'])
            #     custom_weights_df.index.name = '股票代码'
            #     custom_weights_df['股票名称'] = custom_weights_df.index.map(stock_names)
            #     st.dataframe(
            #         custom_weights_df[['股票名称', '权重']].sort_values('权重', ascending=False).style.format({'权重': '{:.2%}'}),
            #         height=400
            #     )
            # except Exception as e:
            #     st.error(f"优化失败: {str(e)}")

        except Exception as e:
            st.error(f"计算有效前沿时出错: {str(e)}")
            st.exception(e)

if selected == "说明文档":
    explanation_page()
elif selected == "股票选择":
    stock_page()
elif selected == "历史价格":
    history_page()
elif selected == "Beta分析":
    Beta_page()
elif selected == "有效前沿分析":
    efficient_frontier_page()

  