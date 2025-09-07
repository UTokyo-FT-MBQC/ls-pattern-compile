# T35 目的/説明:
# - ルート直下の AGENTS.md を読み込み、主要セクション見出しの有無を検査します。
# - 先頭20行をプレビューし、文字化けや体裁を簡易確認します。
#
# 実行結果（stdout）サンプル:（環境により日本語が文字化けして見える場合があります）
# --- AGENTS.md preview (first 20 lines) ---
# 01: # �G�[�W�F���g�^�p�K��iAgents�j
# 02:
# 03: ���̃h�L�������g�́A�{���|�W�g���Ŋ�������AI�G�[�W�F���g�̃��[���Ǝ菇���߂܂��B
# 04:
# 05: ## 0. �S�̃��[��
# 06: - �v�l�͉p��A�o�͓͂��{��ōs���iThink in English, output in Japanese�j�B
# 07: - `.local/issues` �� `.local/changelog.md` �̃R���e�L�X�g��K���m�F����B
# 08: - �ڕW�́A`lspattern/*` �f�B���N�g�����ő���x�[�X�ʎq�v�Z�iMBQC�j�̃p�^�[���R���p�C���������E���P���邱�ƁB
# 09: - Git �֘A�t�@�C���������A���|�W�g�����̃t�@�C���͓ǂݏ����E�ύX�E�폜���\\�B�^�₪����ꍇ�̂݋������߂邱�ƁB
# 10: - Git �֘A�̑���E�t�@�C���ύX�͍s��Ȃ����ƁB
# 11:
# 12: ### Issue�i.local/issues/TXX.md�j�쐬�v��
# 13: GitHub �� Issue �Ɠ����̃t�H�[�}�b�g�ŁA`.local/issues` �� Markdown �Ƃ��č쐬����B�t�@�C������ `TXX.md`�iXX �͔ԍ��j�B���e�͈ȉ����܂߂邱�ƁB
# 14:
# 15: - Issue id: `TXX`
# 16: - Title: �Z���^�C�g��
# 17: - Context: �w�i�E�O��
# 18: - Description and coding plan: �ڍׂȐ����ƍ�ƌv��
# 19: - Scope: �{�^�X�N�Ŏ��{����͈�
# 20: - Acceptance criteria: ��������̊
#
# --- Header checks ---
# [OK] �G�[�W�F���g�^�p�K��
# [OK] 0. �S�̃��[��
# [OK] Issue�i.local/issues/TXX.md�j�쐬�v��
# [OK] Issue �̌��ؗv���i���ʕ��j
# [OK] changelog.md �̏���
# [OK] 1. ��ʎ菇
# [OK] 2. ���s�|���V�[
#
# All required headers found.
"""
T35: AGENTS.md を日本語に整備（polish & 翻訳）した結果の簡易検証スクリプト。

実行内容:
- ルート直下の AGENTS.md を読み込み、主要セクション見出しの有無を検査
- 先頭数行を表示してエンコーディングの乱れがないか確認

使い方:
    python examples/debug_T35.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import os
import platform


def _configure_console_utf8() -> None:
    """WindowsコンソールでUTF-8表示を試みる設定。
    - コードページを 65001 に変更
    - Pythonのstdout/stderrをUTF-8に再設定
    失敗しても例外は投げない（環境依存のため）。
    """
    if platform.system() != "Windows":
        return
    try:
        # Python側出力をUTF-8へ
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        # コンソールのコードページをUTF-8へ（PowerShellでも有効）
        os.system("chcp 65001 > nul")
        # 可能ならWinAPIでも設定
        try:
            import ctypes  # noqa: WPS433 (allow local import)

            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
PREVIEW_TXT = ROOT / "examples" / "AGENTS_preview_T35.txt"
AGENTS_BOM_COPY = ROOT / "examples" / "AGENTS_bom_T35.md"

REQUIRED_HEADERS = [
    "エージェント運用規約",
    "0. 全体ルール",
    "Issue（.local/issues/TXX.md）作成要件",
    "Issue の検証要件（成果物）",
    "changelog.md の書式",
    "1. 一般手順",
    "2. 実行ポリシー",
]


def main() -> None:
    _configure_console_utf8()
    assert AGENTS.exists(), f"AGENTS.md が見つかりません: {AGENTS}"
    text = AGENTS.read_text(encoding="utf-8")

    print("--- AGENTS.md preview (first 20 lines) ---")
    lines = text.splitlines()
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:02d}: {line}")

    print("\n--- Header checks ---")
    ok_all = True
    for h in REQUIRED_HEADERS:
        ok = (h in text)
        print(f"[{'OK' if ok else 'NG'}] {h}")
        ok_all &= ok

    if not ok_all:
        raise SystemExit(1)

    print("\nAll required headers found.")

    # 文字化け対策: プレビューをUTF-8 BOM付きで別ファイルにも保存
    PREVIEW_TXT.write_text("\n".join(lines[:20]) + "\n", encoding="utf-8-sig")
    AGENTS_BOM_COPY.write_text(text, encoding="utf-8-sig")
    print("Saved preview to:", PREVIEW_TXT)
    print("Saved BOM copy to:", AGENTS_BOM_COPY)


if __name__ == "__main__":
    main()
