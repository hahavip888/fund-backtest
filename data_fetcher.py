"""
data_fetcher.py — 数据获取模块 V2
主数据源：天天基金 HTTP API（直接调用，无需 AkShare，稳定支持所有场外基金）
备用数据源：mootdx（通达信接口，用于场内 ETF/指数基金 K 线数据）
基准数据：东方财富行情接口（沪深300历史K线）
"""

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── HTTP 请求头 ──────────────────────────────────
_HEADERS = {
    "Referer": "https://fund.eastmoney.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ══════════════════════════════════════════════════
# 1. 基金代码标准化
# ══════════════════════════════════════════════════

def normalize_fund_code(raw_code: str) -> dict:
    """
    标准化基金代码
    支持：6 位纯数字代码 | 带市场后缀（510300.SH / 159919.SZ）
    返回：{'code': '510300', 'market': 'SH'/'SZ'/None, 'raw': raw_code}
    """
    raw_code = raw_code.strip().upper()
    if "." in raw_code:
        parts = raw_code.split(".")
        code   = parts[0]
        market = parts[1] if len(parts) > 1 else None
        return {"code": code, "market": market, "raw": raw_code}
    elif raw_code.isdigit() and len(raw_code) == 6:
        return {"code": raw_code, "market": None, "raw": raw_code}
    else:
        raise ValueError(
            f"无法识别的基金代码格式: {raw_code}，"
            "请使用 6 位代码（如 000001）或带市场后缀的代码（如 510300.SH）"
        )


# ══════════════════════════════════════════════════
# 2. 天天基金 API — 场外公募基金净值（主数据源）
# ══════════════════════════════════════════════════

def _fetch_eastmoney_nav(code: str, start_date: str, end_date: str,
                          page_size: int = 200) -> pd.DataFrame | None:
    """
    调用天天基金 /f10/lsjz 接口获取历史净值
    返回标准 DataFrame：date, unit_nav, accumulated_nav, adjusted_nav
    """
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    all_rows = []
    page = 1

    while True:
        params = {
            "fundCode": code,
            "pageIndex": page,
            "pageSize": page_size,
            "startDate": start_date,
            "endDate": end_date,
            "callback": "",
        }
        try:
            resp = requests.get(url, params=params, headers=_HEADERS,
                                timeout=20, verify=True)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if page == 1:
                raise ValueError(f"天天基金接口请求失败（基金 {code}）：{e}")
            break  # 已有部分数据则继续处理

        rows  = data.get("Data", {}).get("LSJZList", [])
        total = int(data.get("TotalCount", 0))

        if not rows:
            break
        all_rows.extend(rows)

        if len(all_rows) >= total:
            break
        page += 1
        time.sleep(0.08)   # 礼貌性延迟

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)

    # ── 字段重命名 ──
    df = df.rename(columns={
        "FSRQ": "date",
        "DWJZ": "unit_nav",
        "LJJZ": "accumulated_nav",
    })
    df["date"]           = pd.to_datetime(df["date"])
    df["unit_nav"]       = pd.to_numeric(df["unit_nav"],       errors="coerce")
    df["accumulated_nav"]= pd.to_numeric(df["accumulated_nav"],errors="coerce")

    df = (df[["date", "unit_nav", "accumulated_nav"]]
            .dropna(subset=["unit_nav"])
            .sort_values("date")
            .reset_index(drop=True))

    if len(df) == 0:
        return None

    # ── 用累计净值计算复权净值（相对区间起始日归一化）──
    # 公式：adjusted = accumulated / accumulated[0] * unit[0]
    # 等价于：保留分红再投资效果，起点净值锚定到首日单位净值
    base_accum = df["accumulated_nav"].iloc[0]
    base_unit  = df["unit_nav"].iloc[0]
    if base_accum > 0:
        df["adjusted_nav"] = (
            df["accumulated_nav"] / base_accum * base_unit
        )
    else:
        df["adjusted_nav"] = df["unit_nav"]

    df["fund_code"] = code
    df["nav_type"]  = "adjusted_nav"

    return df[["date", "fund_code", "unit_nav", "accumulated_nav",
               "adjusted_nav", "nav_type"]].reset_index(drop=True)


# ══════════════════════════════════════════════════
# 3. mootdx — 场内 ETF / 指数基金 K 线（备用数据源）
# ══════════════════════════════════════════════════

def _fetch_mootdx_etf(code: str, start_date: str, end_date: str
                       ) -> pd.DataFrame | None:
    """
    用 mootdx 获取场内 ETF / 指数基金日 K 线数据
    适用于：以 .SH / .SZ 结尾的代码，或 510xxx / 159xxx 开头的场内基金
    """
    try:
        from mootdx.quotes import Quotes
    except ImportError:
        return None

    q = None
    try:
        q = Quotes.factory(market="std", heartbeat=False,
                           auto_retry=True, max_retry=3)

        # 计算需要拉取的最大条数（日 K = frequency 9）
        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)
        max_bars = min(int((end_dt - start_dt).days * 1.5) + 60, 8000)

        df = q.bars(symbol=code, frequency=9,
                    start=0, offset=max_bars)

        if df is None or len(df) == 0:
            return None

        # mootdx 返回字段通常是：datetime / open / high / low / close / vol
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl:
                rename_map[c] = "date"
            elif cl == "close":
                rename_map[c] = "close"
        df = df.rename(columns=rename_map)

        if "date" not in df.columns or "close" not in df.columns:
            return None

        df["date"]  = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        # 过滤日期范围
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df[["date", "close"]].sort_values("date").reset_index(drop=True)

        if len(df) == 0:
            return None

        # 转换为标准净值格式（用复权收盘价当净值）
        base = df["close"].iloc[0]
        df["unit_nav"]        = df["close"]
        df["accumulated_nav"] = df["close"]
        df["adjusted_nav"]    = df["close"]   # 场内已经是行情价，直接用
        df["fund_code"]       = code
        df["nav_type"]        = "adjusted_nav"

        return df[["date", "fund_code", "unit_nav", "accumulated_nav",
                   "adjusted_nav", "nav_type"]].reset_index(drop=True)

    except Exception:
        return None
    finally:
        try:
            if q is not None:
                q.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════
# 4. 主入口：获取基金净值
# ══════════════════════════════════════════════════

def fetch_fund_nav(fund_code_raw: str, start_date: str,
                   end_date: str) -> pd.DataFrame:
    """
    获取基金历史净值，自动选择最优数据源。

    策略：
    1. 所有基金先尝试天天基金 API（覆盖场内+场外，数据最全）
    2. 若天天基金返回空（场内 ETF 可能不覆盖），改用 mootdx 获取 K 线
    3. 两路均失败则抛出 ValueError，提示用户检查代码或区间

    返回标准 DataFrame：
        date | fund_code | unit_nav | accumulated_nav | adjusted_nav | nav_type
    """
    info = normalize_fund_code(fund_code_raw)
    code = info["code"]

    errors = []

    # ── 路径1：天天基金 API（主数据源）──────────────
    try:
        df = _fetch_eastmoney_nav(code, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
        errors.append("天天基金 API：返回空数据")
    except Exception as e:
        errors.append(f"天天基金 API：{e}")

    # ── 路径2：mootdx（场内 ETF 备用）──────────────
    try:
        df = _fetch_mootdx_etf(code, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
        errors.append("mootdx：返回空数据")
    except Exception as e:
        errors.append(f"mootdx：{e}")

    # ── 两路均失败 ──────────────────────────────────
    err_detail = " | ".join(errors)
    raise ValueError(
        f"无法获取基金 {code} 的净值数据。\n"
        f"错误详情：{err_detail}\n"
        f"请检查：① 基金代码是否正确；② 回测区间是否在基金成立日之后；"
        f"③ 网络连接是否正常。"
    )


# ══════════════════════════════════════════════════
# 5. 净值序列提取
# ══════════════════════════════════════════════════

def get_nav_for_backtest(df: pd.DataFrame) -> pd.Series:
    """提取用于回测的净值序列（优先复权净值，否则单位净值）"""
    if "adjusted_nav" in df.columns and df["adjusted_nav"].notna().all():
        return df.set_index("date")["adjusted_nav"]
    return df.set_index("date")["unit_nav"]


def get_nav_type_label(df: pd.DataFrame) -> str:
    """返回净值类型说明文字"""
    nav_type = df["nav_type"].iloc[0] if "nav_type" in df.columns else "unit_nav"
    if nav_type == "adjusted_nav":
        return "复权净值（已考虑分红再投资）"
    return "单位净值（未完整反映分红再投资）"


# ══════════════════════════════════════════════════
# 6. 基准数据获取（沪深300 / 中证偏股基金指数）
# ══════════════════════════════════════════════════

_BENCHMARK_SECID = {
    "000300": "1.000300",   # 沪深300（上交所）
    "399300": "0.399300",   # 沪深300（深交所，部分接口用）
    "930950": "1.930950",   # 中证偏股基金指数
}

def fetch_benchmark(start_date: str, end_date: str,
                    benchmark_code: str = "000300") -> pd.DataFrame:
    """
    获取基准指数日 K 线数据
    主数据源：东方财富行情推送接口（免 key，覆盖范围广）
    备用数据源：mootdx
    返回：date, close
    """
    errors = []

    # ── 路径1：东方财富行情接口 ──────────────────────
    try:
        secid = _BENCHMARK_SECID.get(benchmark_code, f"1.{benchmark_code}")
        url   = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid":   secid,
            "ut":      "fa5fd1943c7b386f172d6893dbfba10b",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt":     101,        # 日 K
            "fqt":     1,          # 前复权
            "beg":     start_date.replace("-", ""),
            "end":     end_date.replace("-", ""),
            "cb":      "",
        }
        resp   = requests.get(url, params=params, headers=_HEADERS,
                              timeout=20, verify=True)
        resp.raise_for_status()
        raw    = resp.json()
        klines = raw.get("data", {}).get("klines", [])

        if klines:
            rows = []
            for k in klines:
                parts = k.split(",")
                # parts: 日期,开,收,高,低,成交量,成交额,...
                rows.append({
                    "date":  parts[0],
                    "close": float(parts[2]),
                })
            df = pd.DataFrame(rows)
            df["date"]  = pd.to_datetime(df["date"])
            df["close"] = df["close"].astype(float)
            df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
            if len(df) > 0:
                return df
        errors.append(f"东方财富行情接口：返回空 K 线（secid={secid}）")
    except Exception as e:
        errors.append(f"东方财富行情接口：{e}")

    # ── 路径2：mootdx ──────────────────────────────
    try:
        from mootdx.quotes import Quotes
        q = Quotes.factory(market="std", heartbeat=False,
                           auto_retry=True, max_retry=3)
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt   = pd.to_datetime(end_date)
            max_bars = min(int((end_dt - start_dt).days * 1.5) + 60, 8000)

            df = q.bars(symbol=benchmark_code, frequency=9,
                        start=0, offset=max_bars)
            if df is not None and len(df) > 0:
                rename_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if "time" in cl or "date" in cl:
                        rename_map[c] = "date"
                    elif cl == "close":
                        rename_map[c] = "close"
                df = df.rename(columns=rename_map)
                df["date"]  = pd.to_datetime(df["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
                df = (df[["date", "close"]]
                        .sort_values("date")
                        .reset_index(drop=True))
                if len(df) > 0:
                    return df
            errors.append("mootdx：返回空数据")
        finally:
            try:
                q.close()
            except Exception:
                pass
    except ImportError:
        errors.append("mootdx：未安装")
    except Exception as e:
        errors.append(f"mootdx：{e}")

    err_detail = " | ".join(errors)
    raise ValueError(
        f"无法获取基准数据（{benchmark_code}）。\n"
        f"错误详情：{err_detail}\n"
        "请检查网络连接，或取消勾选「基准对比」继续回测。"
    )


# ══════════════════════════════════════════════════
# 7. 数据完整性检查
# ══════════════════════════════════════════════════

def check_data_coverage(nav_dict: dict, start_date: str,
                         end_date: str) -> dict:
    """
    检查所有基金净值数据是否覆盖回测区间
    返回：{'ok': bool, 'issues': list, 'recommended_start': str}
    """
    issues    = []
    max_start = pd.to_datetime(start_date)

    for code, df in nav_dict.items():
        if df is None or len(df) == 0:
            issues.append(f"基金 {code}：无法获取数据")
            continue

        actual_start = df["date"].min()
        actual_end   = df["date"].max()

        if actual_start > pd.to_datetime(start_date):
            issues.append(
                f"基金 {code}：数据从 {actual_start.date()} 开始，"
                f"晚于回测开始日期 {start_date}"
            )
            if actual_start > max_start:
                max_start = actual_start

        if actual_end < pd.to_datetime(end_date):
            issues.append(
                f"基金 {code}：数据到 {actual_end.date()} 结束，"
                f"早于回测结束日期 {end_date}"
            )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "recommended_start": max_start.strftime("%Y-%m-%d"),
    }
