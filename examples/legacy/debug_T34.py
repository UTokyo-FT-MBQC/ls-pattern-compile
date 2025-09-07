"""
T34: Ruffのエラーを列挙して統計を集計するデバッグスクリプト。

実行内容:
- リポジトリ全体に対して `ruff check` を実行
- JSONレポートを `examples/ruff_report_T34.json` に保存
- 規則コード別・ファイル別の件数をCSVに保存
- コンソールにサマリ（総数・上位ルール・上位ファイル）を出力

依存:
- ruff が見つからない場合は自動で `pip install ruff` を試みます

使い方:
    python examples/debug_T34.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REPORT_JSON = EXAMPLES / "ruff_report_T34.json"
CSV_BY_CODE = EXAMPLES / "ruff_stats_T34_by_code.csv"
CSV_BY_FILE = EXAMPLES / "ruff_stats_T34_by_file.csv"


def ensure_ruff() -> str:
    """ruff実行ファイルのパスを返す。なければインストールを試みる。"""
    exe = shutil.which("ruff")
    if exe:
        return exe
    print("ruff が見つかりません。pip でインストールします…", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ruff"])  # noqa: S603,S607
    exe = shutil.which("ruff")
    if not exe:
        raise RuntimeError("ruff のインストールに失敗しました")
    return exe


def run_ruff_and_save_json() -> list[dict]:
    ruff = ensure_ruff()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # JSONで出力をファイル保存
    print("Running: ruff check (json output)…")
    proc = subprocess.run(  # noqa: S603
        [
            ruff,
            "check",
            "--output-format",
            "json",
            "--output-file",
            str(REPORT_JSON),
            ".",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # ruff は違反があると終了コード1を返すので、失敗として扱わない
    if proc.returncode not in (0, 1):
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"ruff 実行エラー: exit={proc.returncode}")

    with REPORT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def aggregate_stats(data: list[dict]) -> tuple[Counter[str], Counter[str]]:
    by_code: Counter[str] = Counter()
    by_file: Counter[str] = Counter()
    for d in data:
        code = d.get("code", "?")
        filename = d.get("filename", "?")
        # ルートからの相対パスに変換
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


def print_summary(by_code: Counter[str], by_file: Counter[str], total: int) -> None:
    print(f"Total diagnostics: {total}")
    print()
    print("Top 20 rules:")
    for code, cnt in by_code.most_common(20):
        print(f"{code}: {cnt}")
    print()
    print("Top 20 files:")
    for path, cnt in by_file.most_common(20):
        print(f"{path}: {cnt}")


def main() -> None:
    data = run_ruff_and_save_json()
    by_code, by_file = aggregate_stats(data)
    save_csv_by_code(by_code)
    save_csv_by_file(by_file)
    print_summary(by_code, by_file, total=sum(by_code.values()))
    print()
    print("Saved:")
    print(f"- JSON: {REPORT_JSON}")
    print(f"- CSV (by code): {CSV_BY_CODE}")
    print(f"- CSV (by file): {CSV_BY_FILE}")


if __name__ == "__main__":
    main()

