# エージェント運用規約（Agents）

このドキュメントは、本リポジトリで活動するAIエージェントのルールと手順を定めます。

## 0. 全体ルール
- 思考は英語、出力は日本語で行う（Think in English, output in Japanese）。
- `.local/issues` と `.local/changelog.md` のコンテキストを必ず確認する。
- 目標は、`lspattern/*` ディレクトリ内で測定ベース量子計算（MBQC）のパターンコンパイラを実装・改善すること。
- Git 関連ファイルを除き、リポジトリ内のファイルは読み書き・変更・削除が可能。疑問がある場合のみ許可を求めること。
- Git 関連の操作・ファイル変更は行わないこと。

### Issue（.local/issues/TXX.md）作成要件
GitHub の Issue と同等のフォーマットで、`.local/issues` に Markdown として作成する。ファイル名は `TXX.md`（XX は番号）。内容は以下を含めること。

- Issue id: `TXX`
- Title: 短いタイトル
- Context: 背景・前提
- Description and coding plan: 詳細な説明と作業計画
- Scope: 本タスクで実施する範囲
- step by step checklists: 段階的なチェックリスト
- Acceptance criteria: 完了判定の基準
- Testing plan: 動作確認・検証方法
- Notes: 補足（任意）

### Issue の検証要件（成果物）
- 各タスクは最低限、以下の2つを成果物として `examples/` に含めること。
  - `examples/debug_T**.py`: タスクIDごとのデバッグ用スクリプト（単体実行可能）
  - `examples/visualize_T**.ipynb`: 結果可視化用ノートブック
- `examples/debug_T**.py` は Jupyter Notebook スタイルのセルを用い、Python Interactive Window で実行可能にすること。
- `examples/debug_T**.py`の冒頭には(1)タスクの目的，コードの説明，(2)コードのstd outのコピペをコメントで明記すること。outputが100行を超える場合は省略可
- 両者が実行可能であること（ノートブックはJupyter上で図表やログが確認できること）。
- `debug_T**.py` の先頭に、目的・使い方・入出力（必要なら）を日本語のドキュストリングで明記すること。
- 可能なら、インタラクティブな可視化（例: Plotly）も歓迎（必須ではない）。
- 成果物は必ず `examples/` 配下に保存すること。

### changelog.md の書式
`.local/changelog.md` に、次の形式で追記する（新しいエントリほど上に置く）。
- `$Changelog id: $title`
- `date: $timestamp`
- `author: $author`
- summary 
- Files changed（ls-pattern-compile ルートからのツリービュー）
- User's request summary（長くてもよい）
- What you changed（長くてもよい）
- Remaining task（残作業がなければ「No remaining task」）

## 1. 一般手順
1. ユーザーの要求を正確に把握する。
2. 要求を解釈し、実施内容を詳細に計画する。
3. 計画を日本語で提示し、ユーザーの承認を得る。
4. 承認後、計画に従って段階的に実行する。
5. 完了後、実施内容と残作業を `.local/changelog.md` に記録する（日本語）。

## 2. 実行ポリシー
ユーザーの要求は、概ね次の3分類に分かれる。
1. 会話・文書からの課題化（Issue generation）
2. 課題の特定（Issue finding）
3. 課題の実行（Issue execution）

### a. 会話・文書からの課題化（Issue generation）
会話ログや文書を読み、ユーザーの意図・コンテキストを正確に把握したうえで、必要な Issue を作成する。Issue は上記フォーマットに従うこと。

### b. 課題の特定（Issue finding）
コードが期待通りに動作しない場合、コードベースを徹底的に調査し、必要に応じて `.venv` などの環境で再現確認を行う。原因や影響を整理してユーザーに報告し、承認を得たうえで新規 Issue を作成する。

### c. 課題の実行（Issue execution）
ユーザーから指定された Issue（`.local/issues` の `T**.md` など）を精読し、コードベース・関連 Issue の状況を踏まえたうえで手順に沿って実装・検証を進め、作業完了後に `.local/changelog.md` へ結果を記録する。

