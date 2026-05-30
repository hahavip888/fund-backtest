"""
app.py — 基金组合定投回测系统主程序
运行方式：streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import json
import traceback

from data_fetcher import (
    fetch_fund_nav, fetch_benchmark,
    get_nav_for_backtest, get_nav_type_label,
    check_data_coverage, normalize_fund_code
)
from backtest_engine import run_backtest
from history_manager import (
    load_history, save_history, delete_history,
    clear_history, format_history_record
)

# ──────────────────────────────────────────────
# 页面基础配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="基金组合定投回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# 自定义样式
# ──────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 12px;
    text-align: center;
    margin: 4px 0;
}
.metric-card-green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-card-red {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-card-blue {
    background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-title { font-size: 13px; opacity: 0.9; margin-bottom: 6px; }
.metric-value { font-size: 22px; font-weight: bold; }
.metric-sub   { font-size: 12px; opacity: 0.8; margin-top: 4px; }
.notice-box {
    background: #fff8e1;
    border-left: 4px solid #FFC107;
    padding: 10px 16px;
    border-radius: 4px;
    font-size: 13px;
    color: #5d4037;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 辅助：指标卡片
# ──────────────────────────────────────────────
def metric_card(title, value, sub="", color="purple"):
    color_map = {
        "purple": "metric-card",
        "green":  "metric-card-green",
        "red":    "metric-card-red",
        "blue":   "metric-card-blue",
    }
    cls = color_map.get(color, "metric-card")
    st.markdown(f"""
    <div class="{cls}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 辅助：绘制四张图
# ──────────────────────────────────────────────
def plot_results(daily_df: pd.DataFrame, summary: dict, invest_dates: list):
    daily_df = daily_df.copy()
    daily_df.index = pd.to_datetime(daily_df.index)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "📊 组合市值 vs 累计投入",
            "💰 累计收益率",
            "📉 回撤曲线",
            "🏦 组合 vs 基准对比"
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08
    )

    # ── 图1：市值 + 本金 ──
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["portfolio_value"],
        name="组合市值", line=dict(color="#667eea", width=2),
        hovertemplate="日期：%{x|%Y-%m-%d}<br>市值：%{y:,.2f} 元<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["total_invested"],
        name="累计投入", line=dict(color="#FFC107", width=2, dash="dot"),
        hovertemplate="日期：%{x|%Y-%m-%d}<br>投入：%{y:,.2f} 元<extra></extra>"
    ), row=1, col=1)

    # ── 图2：收益率 ──
    colors_rate = ["#11998e" if v >= 0 else "#eb3349" for v in daily_df["profit_rate"]]
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["profit_rate"],
        name="累计收益率", line=dict(color="#11998e", width=2),
        fill="tozeroy", fillcolor="rgba(17,153,142,0.15)",
        hovertemplate="日期：%{x|%Y-%m-%d}<br>收益率：%{y:.2f}%<extra></extra>"
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # ── 图3：回撤 ──
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["drawdown"],
        name="组合回撤", line=dict(color="#eb3349", width=1.5),
        fill="tozeroy", fillcolor="rgba(235,51,73,0.12)",
        hovertemplate="日期：%{x|%Y-%m-%d}<br>回撤：%{y:.2f}%<extra></extra>"
    ), row=2, col=1)

    # ── 图4：组合 vs 基准 ──
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["portfolio_value"],
        name="组合市值", line=dict(color="#667eea", width=2),
        showlegend=False,
        hovertemplate="日期：%{x|%Y-%m-%d}<br>组合：%{y:,.2f} 元<extra></extra>"
    ), row=2, col=2)
    if "benchmark_value" in daily_df.columns and daily_df["benchmark_value"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=daily_df.index, y=daily_df["benchmark_value"],
            name="基准（沪深300定投）", line=dict(color="#FF9800", width=2, dash="dash"),
            hovertemplate="日期：%{x|%Y-%m-%d}<br>基准：%{y:,.2f} 元<extra></extra>"
        ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["total_invested"],
        name="累计投入", line=dict(color="#FFC107", width=1.5, dash="dot"),
        showlegend=False,
        hovertemplate="日期：%{x|%Y-%m-%d}<br>投入：%{y:,.2f} 元<extra></extra>"
    ), row=2, col=2)

    # 标记定投日
    invest_dt = [d for d in pd.to_datetime(invest_dates) if d in daily_df.index]
    if invest_dt:
        invest_vals = [daily_df.loc[d, "portfolio_value"] for d in invest_dt]
        fig.add_trace(go.Scatter(
            x=invest_dt, y=invest_vals,
            mode="markers", name="定投日",
            marker=dict(color="#667eea", size=5, symbol="circle"),
            hovertemplate="定投日：%{x|%Y-%m-%d}<br>市值：%{y:,.2f} 元<extra></extra>"
        ), row=1, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    # Y轴标签
    fig.update_yaxes(ticksuffix=" 元", row=1, col=1)
    fig.update_yaxes(ticksuffix="%",   row=1, col=2)
    fig.update_yaxes(ticksuffix="%",   row=2, col=1)
    fig.update_yaxes(ticksuffix=" 元", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────
def main():
    st.title("📈 基金组合定投回测系统")
    st.caption("支持多只基金组合 · 自动获取净值 · 定投模拟 · 沪深300基准对比 · 历史记录保存")

    # ── 免责声明 ──
    with st.expander("⚠️ 免责声明与数据口径说明", expanded=False):
        st.markdown("""
- 本工具仅供**历史数据研究和学习**使用，不构成任何投资建议。
- **净值口径**：优先使用复权净值，无复权净值时使用单位净值（可能未完整反映分红再投资）。
- **费用**：当前版本暂不考虑申购费、赎回费、交易佣金、税费等交易成本。基金管理费和托管费通常已反映在基金净值中。
- **基准对比**：基准采用与组合相同的定投日期和每期定投金额进行模拟定投，不代表实际指数涨跌幅。
- **非交易日处理**：非交易日顺延至下一个组合可执行日。
- **历史回测不代表未来**：历史表现不保证未来收益。
        """)

    # ════════════════════════════════════════════
    # 侧边栏：输入参数
    # ════════════════════════════════════════════
    with st.sidebar:
        st.header("⚙️ 回测参数设置")

        # ── 组合名称 ──
        portfolio_name = st.text_input("组合名称（可选）", placeholder="如：稳健组合A", value="")

        st.markdown("---")
        st.subheader("📋 基金组合配置")

        # ── 基金列表 ──
        if "fund_rows" not in st.session_state:
            st.session_state.fund_rows = [
                {"code": "110022", "name": "", "weight": 40},
                {"code": "000001", "name": "", "weight": 30},
                {"code": "000011", "name": "", "weight": 30},
            ]

        fund_rows = st.session_state.fund_rows
        new_rows = []
        for i, row in enumerate(fund_rows):
            cols = st.columns([3, 2, 1])
            code = cols[0].text_input(f"基金代码 {i+1}", value=row["code"], key=f"code_{i}",
                                       placeholder="如 000001")
            weight = cols[1].number_input(f"权重% {i+1}", min_value=0, max_value=100,
                                           value=row["weight"], key=f"weight_{i}")
            if cols[2].button("✕", key=f"del_{i}") and len(fund_rows) > 1:
                continue
            new_rows.append({"code": code.strip(), "name": row["name"], "weight": weight})
        st.session_state.fund_rows = new_rows

        col_add, col_reset = st.columns(2)
        if col_add.button("➕ 添加基金", use_container_width=True):
            if len(st.session_state.fund_rows) < 10:
                st.session_state.fund_rows.append({"code": "", "name": "", "weight": 0})
                st.rerun()
            else:
                st.warning("最多支持10只基金")

        # 权重合计
        total_weight = sum(r["weight"] for r in st.session_state.fund_rows)
        weight_ok = abs(total_weight - 100) < 0.01
        if weight_ok:
            st.success(f"✅ 权重合计：{total_weight}%")
        else:
            st.error(f"❌ 权重合计：{total_weight}%（需等于100%）")

        st.markdown("---")
        st.subheader("📅 回测参数")

        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始日期", value=date(2020, 1, 1),
                                      min_value=date(2005, 1, 1), max_value=date.today())
        end_date   = col2.date_input("结束日期", value=date.today(),
                                      min_value=date(2005, 1, 1), max_value=date.today())

        invest_amount = st.number_input("每期定投金额（元）", min_value=100, max_value=1000000,
                                         value=1000, step=100)
        frequency = st.selectbox("定投频率", options=["monthly", "biweekly"],
                                  format_func=lambda x: "每月（默认10日）" if x == "monthly" else "每两周")

        st.markdown("---")
        st.subheader("📊 基准设置")
        use_benchmark = st.checkbox("与沪深300对比", value=True)

        st.markdown("---")
        run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True,
                             disabled=not weight_ok)

    # ════════════════════════════════════════════
    # Tab 布局
    # ════════════════════════════════════════════
    tab1, tab2 = st.tabs(["📊 回测结果", "🗂️ 历史记录"])

    # ────────────────────────────────────────────
    # Tab1: 回测结果
    # ────────────────────────────────────────────
    with tab1:
        if run_btn:
            # 参数校验
            fund_rows = st.session_state.fund_rows
            if start_date >= end_date:
                st.error("开始日期必须早于结束日期")
                st.stop()
            if not weight_ok:
                st.error("权重合计必须为100%")
                st.stop()
            if invest_amount <= 0:
                st.error("每期定投金额必须大于0")
                st.stop()
            valid_funds = [r for r in fund_rows if r["code"].strip()]
            if not valid_funds:
                st.error("请至少输入一只基金代码")
                st.stop()

            start_str = start_date.strftime("%Y-%m-%d")
            end_str   = end_date.strftime("%Y-%m-%d")

            with st.spinner("⏳ 正在获取数据并执行回测，请稍候..."):
                try:
                    # ── 获取基金净值 ──
                    nav_dict = {}
                    nav_series_dict = {}
                    nav_type_labels = {}
                    weights_map = {}
                    fund_info_list = []

                    progress = st.progress(0)
                    for i, row in enumerate(valid_funds):
                        code_raw = row["code"].strip()
                        w = row["weight"] / 100.0
                        progress.progress((i + 1) / (len(valid_funds) + 2),
                                           text=f"获取基金 {code_raw} 净值...")
                        df_nav = fetch_fund_nav(code_raw, start_str, end_str)
                        info = normalize_fund_code(code_raw)
                        code = info["code"]
                        nav_dict[code] = df_nav
                        nav_series_dict[code] = get_nav_for_backtest(df_nav)
                        nav_type_labels[code] = get_nav_type_label(df_nav)
                        weights_map[code] = w
                        fname = df_nav["fund_code"].iloc[0] if "fund_code" in df_nav.columns else code
                        fund_info_list.append({
                            "code": code, "weight": row["weight"],
                            "nav_type": nav_type_labels[code]
                        })

                    # ── 数据完整性检查 ──
                    coverage = check_data_coverage(nav_dict, start_str, end_str)
                    if not coverage["ok"]:
                        st.warning("⚠️ 部分基金数据覆盖不足：")
                        for issue in coverage["issues"]:
                            st.write(f"  • {issue}")
                        col_a, col_b = st.columns(2)
                        adj = col_a.button(f"📅 调整起始日期至 {coverage['recommended_start']} 并继续")
                        bak = col_b.button("↩️ 返回修改参数")
                        if adj:
                            start_str = coverage["recommended_start"]
                        elif bak:
                            st.stop()

                    # ── 获取基准 ──
                    benchmark_series = None
                    if use_benchmark:
                        progress.progress(len(valid_funds) / (len(valid_funds) + 2),
                                           text="获取沪深300基准数据...")
                        try:
                            bench_df = fetch_benchmark(start_str, end_str, "000300")
                            benchmark_series = bench_df.set_index("date")["close"]
                        except Exception as e:
                            st.warning(f"⚠️ 基准数据获取失败，将跳过基准对比：{e}")

                    progress.progress(1.0, text="正在计算回测结果...")

                    # ── 运行回测 ──
                    result = run_backtest(
                        fund_nav_dict=nav_series_dict,
                        weights=weights_map,
                        start_date=start_str,
                        end_date=end_str,
                        invest_amount=float(invest_amount),
                        frequency=frequency,
                        monthly_day=10,
                        benchmark_series=benchmark_series
                    )
                    progress.empty()

                    daily_df = result["daily"]
                    summary  = result["summary"]
                    invest_dates = result["invest_dates"]

                    # ────────────────────────────────
                    # 展示结果
                    # ────────────────────────────────
                    st.success(f"✅ 回测完成！共执行 {summary['invest_count']} 次定投，回测区间 {start_str} ~ {end_str}")

                    # 净值口径提示
                    for code, label in nav_type_labels.items():
                        if "单位净值" in label:
                            st.markdown(f'<div class="notice-box">⚠️ 基金 {code} 使用{label}，结果可能未完整反映分红再投资影响。</div>',
                                        unsafe_allow_html=True)

                    st.markdown("---")
                    st.subheader("📌 核心指标")

                    # 指标卡片
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        metric_card("累计投入本金",
                                    f"¥{summary['total_invested']:,.0f}",
                                    f"共 {summary['invest_count']} 次定投", "blue")
                    with c2:
                        metric_card("期末组合市值",
                                    f"¥{summary['final_value']:,.0f}",
                                    f"收益 ¥{summary['total_profit']:,.0f}", "purple")
                    with c3:
                        color = "green" if summary['total_return'] >= 0 else "red"
                        metric_card("累计收益率",
                                    f"{summary['total_return']:+.2f}%",
                                    f"年化 {summary['annual_return']:+.2f}%", color)
                    with c4:
                        metric_card("最大回撤",
                                    f"{summary['max_drawdown']:.2f}%",
                                    "", "red")

                    if use_benchmark and benchmark_series is not None:
                        c5, c6, c7, c8 = st.columns(4)
                        with c5:
                            metric_card("基准累计收益率（沪深300定投）",
                                        f"{summary['benchmark_profit_rate']:+.2f}%",
                                        "", "blue")
                        with c6:
                            color = "green" if summary['excess_return'] >= 0 else "red"
                            metric_card("相对基准超额收益",
                                        f"{summary['excess_return']:+.2f}%",
                                        "正值代表跑赢基准", color)

                    st.markdown("---")
                    st.subheader("📈 图表分析")
                    plot_results(daily_df, summary, invest_dates)

                    # ── 定投明细表 ──
                    with st.expander("📋 每日回测数据（可下载）"):
                        show_df = daily_df.reset_index().rename(columns={
                            "date": "日期",
                            "total_invested": "累计投入(元)",
                            "portfolio_value": "组合市值(元)",
                            "profit": "累计收益(元)",
                            "profit_rate": "累计收益率(%)",
                            "drawdown": "当日回撤(%)",
                            "benchmark_value": "基准市值(元)",
                            "is_invest_day": "是否定投日"
                        })
                        st.dataframe(show_df, use_container_width=True)
                        csv = show_df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("⬇️ 下载 CSV", csv,
                                           file_name=f"backtest_{start_str}_{end_str}.csv",
                                           mime="text/csv")

                    # ── 保存历史记录 ──
                    record = {
                        "params": {
                            "portfolio_name": portfolio_name or "未命名组合",
                            "funds": fund_info_list,
                            "start_date": start_str,
                            "end_date": end_str,
                            "invest_amount": invest_amount,
                            "frequency": frequency,
                            "benchmark": "沪深300" if use_benchmark else "无"
                        },
                        "summary": summary
                    }
                    save_history(record)

                except Exception as e:
                    st.error(f"❌ 回测出错：{e}")
                    with st.expander("查看详细错误信息"):
                        st.code(traceback.format_exc())

        else:
            # 未运行时显示引导
            st.info("👈 请在左侧填写基金代码、权重和回测参数，然后点击「🚀 开始回测」")
            st.markdown("""
### 使用指南

**第一步：输入基金组合**
- 在左侧输入每只基金的6位代码（如 `000001`）或带市场后缀代码（如 `510300.SH`）
- 设置每只基金的目标权重（所有基金权重之和必须等于100%）

**第二步：设置回测参数**
- 选择回测开始和结束日期
- 输入每期定投总金额（元）
- 选择定投频率：每月（默认每月10日）或每两周

**第三步：查看结果**
- 系统将自动获取历史净值数据
- 输出组合市值、累计收益、回撤曲线
- 与沪深300定投基准进行对比

---
**常用基金代码参考（仅供示例）**

| 基金代码 | 基金名称 |
|---|---|
| 110022 | 易方达沪深300ETF联接A |
| 000001 | 华夏成长混合 |
| 000011 | 华夏大盘精选混合 |
| 510300 | 沪深300ETF（场内） |
| 163407 | 兴全合润混合 |
| 519674 | 银河稳健混合A |

> ⚠️ 以上基金代码仅供测试，不构成投资建议。
            """)

    # ────────────────────────────────────────────
    # Tab2: 历史记录
    # ────────────────────────────────────────────
    with tab2:
        st.subheader("🗂️ 历史回测记录（最近10次）")
        history = load_history()

        if not history:
            st.info("暂无历史记录，完成一次回测后将自动保存。")
        else:
            col_refresh, col_clear = st.columns([6, 1])
            if col_clear.button("🗑️ 清空记录"):
                clear_history()
                st.rerun()

            for i, rec in enumerate(history):
                formatted = format_history_record(rec)
                params = rec.get("params", {})
                summary = rec.get("summary", {})
                with st.expander(
                    f"📌 {formatted['组合名称']} | {formatted['回测区间']} | "
                    f"收益率 {formatted['累计收益率']} | 最大回撤 {formatted['最大回撤']} "
                    f"（{formatted['创建时间']}）"
                ):
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        disp = {k: v for k, v in formatted.items()
                                if k not in ["创建时间", "组合名称"]}
                        st.table(pd.DataFrame(list(disp.items()), columns=["指标", "值"]))
                    with col_b:
                        if st.button("🗑️ 删除", key=f"del_hist_{i}"):
                            delete_history(i)
                            st.rerun()
                        # 一键复用参数
                        if st.button("♻️ 复用参数", key=f"reuse_{i}"):
                            funds = params.get("funds", [])
                            new_rows = [{"code": f["code"], "name": "",
                                         "weight": f["weight"]} for f in funds]
                            st.session_state.fund_rows = new_rows
                            st.toast("✅ 参数已复用到左侧，请切换到「回测结果」Tab 运行")


if __name__ == "__main__":
    main()
