"""
history_manager.py — 历史回测记录管理
保存最近 10 次回测参数和结果摘要到本地 JSON 文件
"""

import json
import os
from datetime import datetime
from pathlib import Path


HISTORY_FILE = Path("backtest_history.json")
MAX_RECORDS = 10


def load_history() -> list:
    """加载历史记录列表"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(record: dict):
    """
    保存一条新的回测记录
    record 包含 params（参数快照）和 summary（结果摘要）
    """
    history = load_history()
    record["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, record)  # 最新的放最前
    # 只保留最近 MAX_RECORDS 条
    history = history[:MAX_RECORDS]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def delete_history(index: int):
    """删除指定索引的历史记录"""
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


def clear_history():
    """清空所有历史记录"""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


def format_history_record(record: dict) -> dict:
    """格式化历史记录用于显示"""
    params = record.get("params", {})
    summary = record.get("summary", {})
    funds = params.get("funds", [])
    fund_str = " | ".join([f"{f['code']}({f['weight']}%)" for f in funds])
    return {
        "创建时间": record.get("created_at", ""),
        "组合名称": params.get("portfolio_name", "未命名组合"),
        "基金组合": fund_str,
        "回测区间": f"{params.get('start_date', '')} ~ {params.get('end_date', '')}",
        "每期金额(元)": params.get("invest_amount", ""),
        "定投频率": "每月" if params.get("frequency") == "monthly" else "每两周",
        "累计投入(元)": f"{summary.get('total_invested', 0):,.2f}",
        "期末市值(元)": f"{summary.get('final_value', 0):,.2f}",
        "累计收益率": f"{summary.get('total_return', 0):.2f}%",
        "最大回撤": f"{summary.get('max_drawdown', 0):.2f}%",
        "定投次数": summary.get("invest_count", 0),
        "相对基准": f"{summary.get('excess_return', 0):+.2f}%",
    }
