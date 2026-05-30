"""
data_fetcher.py — 数据获取模块
支持通过 AkShare 自动获取基金净值和基准指数数据
优先使用复权净值，没有则回退到单位净值
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 基金代码标准化
# ──────────────────────────────────────────────

def normalize_fund_code(raw_code: str) -> dict:
    """
    标准化基金代码
    支持：6位纯数字代码、带市场后缀(510300.SH / 159919.SZ)
    返回：{'code': '510300', 'market': 'SH'/'SZ'/None, 'raw': raw_code}
    """
    raw_code = raw_code.strip().upper()
    if "." in raw_code:
        parts = raw_code.split(".")
        code = parts[0]
        market = parts[1] if len(parts) > 1 else None
        return {"code": code, "market": market, "raw": raw_code}
    elif raw_code.isdigit() and len(raw_code) == 6:
        return {"code": raw_code, "market": None, "raw": raw_code}
    else:
        raise ValueError(f"无法识别的基金代码格式: {raw_code}，请使用6位代码或带市场后缀的代码（如 510300.SH）")


# ──────────────────────────────────────────────
# 基金净值获取
# ──────────────────────────────────────────────

def fetch_fund_nav(fund_code_raw: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取基金历史净值
    优先获取复权净值(adjusted_nav)，失败则回退到单位净值(unit_nav)
    返回标准化 DataFrame：date, fund_code, unit_nav, adjusted_nav, nav_type
    """
    info = normalize_fund_code(fund_code_raw)
    code = info["code"]

    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")

    df = None
    nav_type = "unit_nav"

    # ── 尝试场外公募基金（开放式基金）──
    try:
        raw = ak.fund_open_fund_info_em(fund=code, indicator="累计净值走势")
        if raw is not None and len(raw) > 0:
            raw.columns = ["date", "accumulated_nav"]
            raw["date"] = pd.to_datetime(raw["date"])
            # 再获取单位净值
            raw2 = ak.fund_open_fund_info_em(fund=code, indicator="单位净值走势")
            raw2.columns = ["date", "unit_nav", "daily_return"]
            raw2["date"] = pd.to_datetime(raw2["date"])
            merged = pd.merge(raw2[["date", "unit_nav"]], raw[["date", "accumulated_nav"]], on="date", how="inner")
            merged = merged[(merged["date"] >= pd.to_datetime(start_date)) &
                            (merged["date"] <= pd.to_datetime(end_date))]
            merged = merged.sort_values("date").reset_index(drop=True)
            merged["fund_code"] = code
            merged["unit_nav"] = merged["unit_nav"].astype(float)
            merged["accumulated_nav"] = merged["accumulated_nav"].astype(float)
            # 用累计净值计算复权净值（相对起始日）
            if len(merged) > 0:
                base = merged["accumulated_nav"].iloc[0]
                merged["adjusted_nav"] = merged["accumulated_nav"] / base * merged["unit_nav"].iloc[0]
                nav_type = "adjusted_nav"
            merged["nav_type"] = nav_type
            df = merged[["date", "fund_code", "unit_nav", "accumulated_nav", "adjusted_nav", "nav_type"]]
    except Exception:
        pass

    # ── 尝试 ETF / 场内基金 ──
    if df is None or len(df) == 0:
        try:
            raw = ak.fund_etf_hist_em(symbol=code, period="daily",
                                       start_date=start_str, end_date=end_str,
                                       adjust="qfq")
            if raw is not None and len(raw) > 0:
                raw = raw.rename(columns={"日期": "date", "收盘": "adjusted_nav"})
                raw["date"] = pd.to_datetime(raw["date"])
                raw2 = ak.fund_etf_hist_em(symbol=code, period="daily",
                                            start_date=start_str, end_date=end_str,
                                            adjust="")
                raw2 = raw2.rename(columns={"日期": "date", "收盘": "unit_nav"})
                raw2["date"] = pd.to_datetime(raw2["date"])
                merged = pd.merge(raw[["date", "adjusted_nav"]], raw2[["date", "unit_nav"]], on="date", how="inner")
                merged = merged.sort_values("date").reset_index(drop=True)
                merged["fund_code"] = code
                merged["accumulated_nav"] = merged["adjusted_nav"]
                merged["nav_type"] = "adjusted_nav"
                nav_type = "adjusted_nav"
                df = merged[["date", "fund_code", "unit_nav", "accumulated_nav", "adjusted_nav", "nav_type"]]
        except Exception:
            pass

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取基金 {code} 的净值数据，请检查代码是否正确或缩短回测区间")

    df = df.dropna(subset=["unit_nav"]).copy()
    df["unit_nav"] = df["unit_nav"].astype(float)
    if "adjusted_nav" in df.columns:
        df["adjusted_nav"] = df["adjusted_nav"].astype(float)

    return df.reset_index(drop=True)


def get_nav_for_backtest(df: pd.DataFrame) -> pd.Series:
    """从净值 DataFrame 中提取用于回测的净值序列（优先复权，否则单位）"""
    if "adjusted_nav" in df.columns and df["adjusted_nav"].notna().all():
        return df.set_index("date")["adjusted_nav"]
    return df.set_index("date")["unit_nav"]


def get_nav_type_label(df: pd.DataFrame) -> str:
    """返回净值类型说明文字"""
    if df["nav_type"].iloc[0] == "adjusted_nav":
        return "复权净值"
    return "单位净值（未完整反映分红再投资）"


# ──────────────────────────────────────────────
# 基准数据获取（沪深300）
# ──────────────────────────────────────────────

def fetch_benchmark(start_date: str, end_date: str, benchmark_code: str = "000300") -> pd.DataFrame:
    """
    获取基准指数历史数据（默认沪深300）
    返回：date, close
    """
    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{benchmark_code}")
        df = df.rename(columns={"date": "date", "close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.to_datetime(start_date)) &
                (df["date"] <= pd.to_datetime(end_date))]
        df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except Exception:
        pass

    try:
        df = ak.index_zh_a_hist(symbol=benchmark_code, period="daily",
                                  start_date=start_date.replace("-", ""),
                                  end_date=end_date.replace("-", ""))
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        raise ValueError(f"无法获取基准数据（沪深300），错误：{e}")


# ──────────────────────────────────────────────
# 数据完整性检查
# ──────────────────────────────────────────────

def check_data_coverage(nav_dict: dict, start_date: str, end_date: str) -> dict:
    """
    检查所有基金净值数据是否覆盖回测区间
    返回：{'ok': bool, 'issues': list, 'recommended_start': str}
    """
    issues = []
    max_start = pd.to_datetime(start_date)

    for code, df in nav_dict.items():
        if df is None or len(df) == 0:
            issues.append(f"基金 {code}：无法获取数据")
            continue
        actual_start = df["date"].min()
        actual_end = df["date"].max()
        if actual_start > pd.to_datetime(start_date):
            issues.append(f"基金 {code}：数据从 {actual_start.date()} 开始，晚于回测开始日期 {start_date}")
            if actual_start > max_start:
                max_start = actual_start
        if actual_end < pd.to_datetime(end_date):
            issues.append(f"基金 {code}：数据到 {actual_end.date()} 结束，早于回测结束日期 {end_date}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "recommended_start": max_start.strftime("%Y-%m-%d")
    }
