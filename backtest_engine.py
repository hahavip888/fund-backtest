"""
backtest_engine.py — 定投回测引擎
核心计算逻辑：生成定投日期、买入份额、逐日计算组合市值、指标、基准对比
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# ──────────────────────────────────────────────
# 定投日期生成
# ──────────────────────────────────────────────

def generate_invest_dates(start_date: str, end_date: str,
                           frequency: str, monthly_day: int = 10,
                           valid_dates: pd.DatetimeIndex = None) -> list:
    """
    生成计划定投日期列表，并将非交易日顺延到下一个有效日期
    frequency: 'monthly' | 'biweekly'
    monthly_day: 每月几号（默认10日）
    valid_dates: 所有基金都有数据的日期集合
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    plan_dates = []

    if frequency == "monthly":
        cur = start.replace(day=1)
        while cur <= end:
            try:
                plan = cur.replace(day=monthly_day)
            except ValueError:
                # 月份没有那一天，取月末
                plan = cur + relativedelta(months=1) - timedelta(days=1)
            if plan >= start:
                plan_dates.append(plan)
            cur += relativedelta(months=1)

    elif frequency == "biweekly":
        cur = start
        while cur <= end:
            plan_dates.append(cur)
            cur += timedelta(days=14)

    # 将计划日期顺延到最近有效交易日
    actual_dates = []
    if valid_dates is not None:
        valid_sorted = sorted(valid_dates)
        for d in plan_dates:
            # 找到 >= d 的最近有效日期
            candidates = [v for v in valid_sorted if v >= d and v <= end]
            if candidates:
                actual_dates.append(candidates[0])
    else:
        actual_dates = [d for d in plan_dates if d <= end]

    # 去重并排序
    actual_dates = sorted(set(actual_dates))
    return actual_dates


# ──────────────────────────────────────────────
# 核心回测引擎
# ──────────────────────────────────────────────

def run_backtest(
    fund_nav_dict: dict,       # {fund_code: pd.Series(index=date, values=nav)}
    weights: dict,              # {fund_code: float}  权重之和=1.0
    start_date: str,
    end_date: str,
    invest_amount: float,       # 每期定投金额（组合总额）
    frequency: str = "monthly", # 'monthly' | 'biweekly'
    monthly_day: int = 10,
    benchmark_series: pd.Series = None  # index=date, values=close
) -> dict:
    """
    执行定投回测，返回完整结果字典
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # ── 1. 构建共同日期索引（所有基金都有数据的日期）──
    all_dates = None
    for code, nav_s in fund_nav_dict.items():
        nav_s.index = pd.to_datetime(nav_s.index)
        date_set = set(nav_s[(nav_s.index >= start) & (nav_s.index <= end)].index)
        if all_dates is None:
            all_dates = date_set
        else:
            all_dates = all_dates & date_set

    if benchmark_series is not None:
        benchmark_series.index = pd.to_datetime(benchmark_series.index)
        bench_dates = set(benchmark_series[(benchmark_series.index >= start) &
                                           (benchmark_series.index <= end)].index)
        all_dates = all_dates & bench_dates

    if not all_dates:
        raise ValueError("回测区间内无共同有效数据，请检查基金代码或缩短回测区间")

    common_dates = sorted(all_dates)
    date_index = pd.DatetimeIndex(common_dates)

    # ── 2. 生成实际定投日期 ──
    invest_dates = generate_invest_dates(
        start_date, end_date, frequency, monthly_day, valid_dates=date_index
    )

    if not invest_dates:
        raise ValueError("回测区间内未能生成任何定投日期，请检查区间或频率设置")

    # ── 3. 初始化持仓 ──
    holdings = {code: 0.0 for code in fund_nav_dict}  # 累计份额

    # ── 4. 逐日计算 ──
    records = []
    total_invested = 0.0
    benchmark_holdings = 0.0
    max_value = 0.0

    invest_date_set = set(invest_dates)

    for dt in common_dates:
        # 是否为定投日
        if dt in invest_date_set:
            total_invested += invest_amount
            for code, w in weights.items():
                nav_val = fund_nav_dict[code].get(dt)
                if nav_val and nav_val > 0:
                    shares_bought = (invest_amount * w) / nav_val
                    holdings[code] += shares_bought

            # 基准定投
            if benchmark_series is not None:
                bench_price = benchmark_series.get(dt)
                if bench_price and bench_price > 0:
                    benchmark_holdings += invest_amount / bench_price

        # 计算当日组合市值
        portfolio_value = sum(
            holdings[code] * fund_nav_dict[code].get(dt, 0)
            for code in fund_nav_dict
        )

        # 计算基准市值
        benchmark_value = 0.0
        if benchmark_series is not None:
            bench_price = benchmark_series.get(dt, 0)
            benchmark_value = benchmark_holdings * bench_price

        # 计算回撤
        if portfolio_value > max_value:
            max_value = portfolio_value
        drawdown = ((portfolio_value - max_value) / max_value) if max_value > 0 else 0.0

        profit = portfolio_value - total_invested
        profit_rate = (profit / total_invested) if total_invested > 0 else 0.0

        records.append({
            "date": dt,
            "total_invested": round(total_invested, 2),
            "portfolio_value": round(portfolio_value, 2),
            "profit": round(profit, 2),
            "profit_rate": round(profit_rate * 100, 4),  # 百分比
            "drawdown": round(drawdown * 100, 4),         # 百分比
            "benchmark_value": round(benchmark_value, 2),
            "is_invest_day": dt in invest_date_set
        })

    result_df = pd.DataFrame(records).set_index("date")

    # ── 5. 计算基准收益率 ──
    benchmark_profit_rate = 0.0
    if benchmark_series is not None and total_invested > 0:
        final_bench = result_df["benchmark_value"].iloc[-1]
        benchmark_profit_rate = round((final_bench - total_invested) / total_invested * 100, 4)

    # ── 6. 汇总指标 ──
    final_value = result_df["portfolio_value"].iloc[-1]
    max_drawdown = result_df["drawdown"].min()  # 最小值（负数最大回撤）
    invest_count = len(invest_dates)
    total_profit = final_value - total_invested
    total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0.0
    excess_return = total_return - benchmark_profit_rate

    # 年化收益率（简单估算）
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0.0

    summary = {
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "total_profit": round(total_profit, 2),
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "max_drawdown": round(abs(max_drawdown), 4),
        "invest_count": invest_count,
        "benchmark_profit_rate": round(benchmark_profit_rate, 4),
        "excess_return": round(excess_return, 4),
        "start_date": start_date,
        "end_date": end_date,
    }

    return {
        "daily": result_df,
        "summary": summary,
        "invest_dates": invest_dates,
    }
