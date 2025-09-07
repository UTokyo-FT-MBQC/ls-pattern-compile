"""
T48: ruff/mypy のエラー件数を集計し、進捗ログ out.txt に追記するデバッグスクリプト。

目的:
- pyproject.toml の方針に従い、ruff/mypy の診断件数を継続的に可視化（T34手法の発展）
- ルール別・ファイル別の集計CSVも保存

使い方（Windows PowerShell/.venv 前提）:
1) PowerShell を開く
2) `.\.venv\Scripts\Activate.ps1` で仮想環境を有効化
3) `python examples/debug_T48.py` を実行
4) ルート直下の `out.txt` に `RUFF_COUNT=...` と `MYPY_COUNT=...` が追記されます

標準出力例（抜粋、長い場合は省略可）:
    [2025-09-07T19:30:00+09:00] ruff: ruff 0.5.x
    [2025-09-07T19:30:00+09:00] RUFF_COUNT=12
    [2025-09-07T19:30:01+09:00] mypy: mypy 1.10.x
    [2025-09-07T19:30:02+09:00] MYPY_COUNT=7

出力:
- out.txt（ルート）に時刻つきで件数追記
- `.local/ruff_T48.json`（ruff の JSON 出力）
- `.local/mypy_T48.txt`（mypy の生ログ）
- `examples/ruff_stats_T48_by_code.csv`（規則コード別件数）
- `examples/ruff_stats_T48_by_file.csv`（ファイル別件数）
"""

# %%
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
LOCAL = ROOT / ".local"
LOCAL.mkdir(parents=True, exist_ok=True)

REPORT_JSON = LOCAL / "ruff_T48.json"
MYPY_LOG = LOCAL / "mypy_T48.txt"
CSV_BY_CODE = EXAMPLES / "ruff_stats_T48_by_code.csv"
CSV_BY_FILE = EXAMPLES / "ruff_stats_T48_by_file.csv"
OUT_TXT = ROOT / "out.txt"


def _ts() -> str:
    # JSTっぽい表示（ユーザー環境に依存せず固定）
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_module_cli(mod: str, cli_hint: str | None = None) -> list[str]:
    """Return argv to run a module as CLI, installing if missing.

    `python -m <mod> --version` を試し、見つからなければ pip で導入。
    """
    cli = [sys.executable, "-m", mod]
    # 存在確認（--version の終了コードで判定）
    res = subprocess.run(
        cli + ["--version"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if res.returncode == 0:
        return cli

    # try install
    print(f"{mod} が見つかりません。pip でインストールします…", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", mod])  # noqa: S603,S607
    return [sys.executable, "-m", mod]


# %% ruff 実行と集計
def run_ruff_and_save_json() -> list[dict]:
    ruff_cli = ensure_module_cli("ruff")
    # バージョン表示
    ver = subprocess.run(ruff_cli + ["--version"], text=True, capture_output=True, check=False)
    print(f"[{_ts()}] ruff: {ver.stdout.strip() or ver.stderr.strip()}")
    # JSON保存
    args = ruff_cli + [
        "check",
        "--output-format",
        "json",
        "--output-file",
        str(REPORT_JSON),
        # 明示的に examples を除外（pyproject と二重化しても問題なし）
        "--extend-exclude",
        "examples",
        ".",
    ]
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode not in (0, 1):
        # 1 は診断ありを示す
        raise RuntimeError(f"ruff 実行失敗: exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    with REPORT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    count = len(data) if isinstance(data, list) else 0
    print(f"[{_ts()}] RUFF_COUNT={count}")
    with OUT_TXT.open("a", encoding="utf-8") as fo:
        fo.write(f"[{_ts()}] RUFF_COUNT={count}\n")
    return data


def aggregate_stats(data: list[dict]) -> tuple[Counter[str], Counter[str]]:
    by_code: Counter[str] = Counter()
    by_file: Counter[str] = Counter()
    for d in data:
        code = d.get("code", "?")
        filename = d.get("filename", "?")
        try:
            filename = os.path.relpath(filename, ROOT)
        except Exception:
            pass
        by_code[code] += 1
        by_file[filename] += 1
    return by_code, by_file


def save_csv_by_code(by_code: Counter[str]) -> None:
    with CSV_BY_CODE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "count"])  # header
        for code, cnt in by_code.most_common():
            w.writerow([code, cnt])


def save_csv_by_file(by_file: Counter[str]) -> None:
    with CSV_BY_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "count"])  # header
        for path, cnt in by_file.most_common():
            w.writerow([path, cnt])


# %% mypy 実行と件数抽出
def run_mypy_and_save() -> int:
    mypy_cli = ensure_module_cli("mypy")
    ver = subprocess.run(mypy_cli + ["--version"], text=True, capture_output=True, check=False)
    print(f"[{_ts()}] mypy: {ver.stdout.strip() or ver.stderr.strip()}")
    proc = subprocess.run(
        mypy_cli + ["--hide-error-context", "--no-color-output"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    MYPY_LOG.write_text(out, encoding="utf-8")
    # 末尾の "Found N error(s) in ..." から抽出
    import re

    m = re.search(r"Found\s+(\d+)\s+error", out)
    if m:
        n = int(m.group(1))
    else:
        # フォールバック: 行内に 'error:' を含む件数（大雑把）
        n = sum(1 for line in out.splitlines() if "error:" in line)
    print(f"[{_ts()}] MYPY_COUNT={n}")
    with OUT_TXT.open("a", encoding="utf-8") as fo:
        fo.write(f"[{_ts()}] MYPY_COUNT={n}\n")
    return n


# %% エントリポイント
def main() -> None:
    data = run_ruff_and_save_json()
    by_code, by_file = aggregate_stats(data)
    save_csv_by_code(by_code)
    save_csv_by_file(by_file)
    run_mypy_and_save()
    print("Saved:")
    print(f"- JSON: {REPORT_JSON}")
    print(f"- CSV (by code): {CSV_BY_CODE}")
    print(f"- CSV (by file): {CSV_BY_FILE}")
    print(f"- mypy log: {MYPY_LOG}")
    print(f"- counts appended to: {OUT_TXT}")


if __name__ == "__main__":
    main()
