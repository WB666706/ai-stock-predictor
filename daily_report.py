"""每日定时预测脚本：巡检自选股并生成 Markdown + CSV 报告。

用法：
    python daily_report.py            # 巡检 watchlist.json 中的自选股
    python daily_report.py 600519 000001   # 巡检指定股票

报告输出到 reports/YYYY-MM-DD.md / .csv。
配合 Windows 计划任务即可实现每日收盘后自动预测（见 README）。
"""

from __future__ import annotations

import datetime as dt
import os
import sys

from core import watchlist

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def main() -> None:
    codes = [c for c in sys.argv[1:] if c.strip()] or watchlist.load_watchlist()
    today = dt.date.today().isoformat()
    print(f"[{today}] 开始巡检 {len(codes)} 只自选股：{', '.join(codes)}")

    df = watchlist.inspect_all(codes)
    os.makedirs(REPORT_DIR, exist_ok=True)

    csv_path = os.path.join(REPORT_DIR, f"{today}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_lines = [
        f"# 每日 AI 预测报告 · {today}",
        "",
        "> 本报告由 ai-stock-predictor 自动生成，仅供学习参考，不构成投资建议。",
        "",
        df.to_markdown(index=False),
        "",
    ]
    bullish = df[df["AI信号"] == "看多"]
    bearish = df[df["AI信号"] == "看空"]
    md_lines.append(f"**摘要**：看多 {len(bullish)} 只，看空 {len(bearish)} 只。")
    if not bullish.empty:
        md_lines.append(f"- 看多：{'、'.join(bullish['名称'].astype(str))}")
    if not bearish.empty:
        md_lines.append(f"- 看空：{'、'.join(bearish['名称'].astype(str))}")

    md_path = os.path.join(REPORT_DIR, f"{today}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(df.to_string(index=False))
    print(f"\n报告已保存：{md_path}")


if __name__ == "__main__":
    main()
