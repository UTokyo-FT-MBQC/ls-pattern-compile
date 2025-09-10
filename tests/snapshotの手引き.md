# スナップショットテスト運用の手引き

本ドキュメントは、CompiledRHGCanvas をスナップショット（JSON）として保存・比較し、回帰を検出するための運用手順をまとめたものです。ノートブックの回路ロジックを一切変更せず、結果だけを安定化して比較する方針です。

## 目的
- ノートブックの回路（例: visualize_T43.ipynb, merge_split_mockup.ipynb）を pytest に移植し、CompiledRHGCanvas を辞書へシリアライズ→JSON に保存
- 次回以降の実行結果を同形式で生成し、JSON の完全一致で検証（回帰テスト）

## スナップショットの場所
- `tests/snapshots/T43_compiled_canvas.json`
- `tests/snapshots/mockup_compiled_canvas.json`

各テストファイルで同名の JSON を参照します。

## 何を保存しているか（フィールド仕様）
- `meta`: メタ情報
  - `layers`: レイヤ数（TemporalLayer 数）
  - `zlist`: 現在までに構成した z のリスト
  - `coord_map`: `coord2node` の件数
  - `nodes`: GraphState 上のノード数
  - `edges`: GraphState 上のエッジ数
- `coords`: すべてのノード座標 `(x,y,z)` を昇順（安定化）で配列化
- `edges_coords`: 各エッジを「両端ノードの座標」に写像し、(min, max) の順に整列して配列化
- `inputs`/`outputs`: GraphState の入出力ノードを「座標文字列 → 論理インデックス」に変換
- `in_ports`/`out_ports`/`cout_ports`: パッチ座標ごとのポートノードを「座標文字列の配列」に変換

注: すべて座標ベースに正規化しているため、ノードIDの非決定性に影響されません。

## 初回生成・更新方法
スナップショットが未生成、または仕様変更が意図的でスナップショットを更新したい場合は、環境変数 `UPDATE_SNAPSHOTS=1` を付与して pytest を実行します。

- PowerShell（Windows）
```
$env:UPDATE_SNAPSHOTS=1; pytest -k test_T43_temporal_and_spatial_snapshot -q
$env:UPDATE_SNAPSHOTS=1; pytest -k test_merge_split_mockup_snapshot -q
Remove-Item Env:UPDATE_SNAPSHOTS
```

- Bash（Linux/macOS, Git Bash 等）
```
UPDATE_SNAPSHOTS=1 pytest -k test_T43_temporal_and_spatial_snapshot -q
UPDATE_SNAPSHOTS=1 pytest -k test_merge_split_mockup_snapshot -q
```

成功すると `tests/snapshots/` に JSON が作成（または上書き）されます。

## 通常実行（比較のみ）
```
pytest -k test_T43_temporal_and_spatial_snapshot -q
pytest -k test_merge_split_mockup_snapshot -q
```
スナップショットと完全一致であることを検証します。不一致であればテスト失敗となり、回路生成の差分が生じていることを示します。

## 失敗時の確認ポイント
- まず `meta.nodes`/`meta.edges`/`coords` の差分が大きいか小さいかを確認
- どの座標が増減したか（`coords` 差分）
- どの接続が変わったか（`edges_coords` 差分）
- 入出力ノードやポートの差（`inputs`/`outputs`/`in_ports`/`out_ports`/`cout_ports`）

JSON 差分の見方例（Bash）:
```
jq . tests/snapshots/T43_compiled_canvas.json > exp.json
jq . /tmp/got.json > got.json
diff -u exp.json got.json | less
```

## 新しいノートをスナップショット化する手順
1. ノートブックの回路構成（ブロック配置・edgespec・パイプ）を“そのまま” pytest に移植
   - 例: `tests/test_temporal_and_spatial.py` / `tests/test_mockup.py`
2. `CompiledRHGCanvas` を辞書へスナップショット化（本リポジトリの関数を流用可）
3. 初回は `UPDATE_SNAPSHOTS=1` でスナップショットを生成
4. 以降は通常実行で完全一致を検証

テンプレート（疑似コード）:
```python
compiled = build_from_notebook_logic()
got = snapshot_compiled_canvas(compiled)
# 初回は UPDATE_SNAPSHOTS=1 で保存、以降は一致をアサート
```

## 運用上の注意・ハマりどころ
- `RHGCanvas.to_temporal_layers()` は一回しか呼べません（重複シフト防止のため）。再構成する場合は、`RHGCanvasSkeleton` から再度 `to_canvas()` してください。
- `materialize()` は、すでにテンプレート座標が入っている場合は再 `to_tiling()` しません（XY シフト上書き防止）。
- 時間層シーム接続では、存在しない座標に対して gid が取れない場合があり得ます。その場合は安全に接続しない（`is_allowed_pair` は None セーフ）実装です。
- 実行環境差（依存パッケージのバージョン/OS）で描画や浮動順序が異なる場合、スナップショット化の段階ですべて座標／整列済みに正規化しているため、基本的に一致します。それでも差分が出た場合は、該当テストの `snapshot` 生成ロジックが決定的か確認してください。

## 既存スナップショットテスト
- T43: `tests/test_temporal_and_spatial.py` → `tests/snapshots/T43_compiled_canvas.json`
- Mockup: `tests/test_mockup.py` → `tests/snapshots/mockup_compiled_canvas.json`

以上。運用で不明点があれば、この手引きを更新してください。
