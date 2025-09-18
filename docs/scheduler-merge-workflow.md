# Scheduler Merge Workflow

## 作業手順
- TemporalLayer 内の `ScheduleAccumulator.compose_parallel` とブロック単位の `_construct_schedule` の前提を確認し、時間スロット重複を検知するガードを設計します。
- `compose_parallel` にタイムスロット衝突検知を追加する前に、各ブロックの `schedule.schedule` が空または非連続なケースを洗い出し、仕様を文書化します。
- Temporal 層結合 (`add_temporal_layer`) で使用する `compose_sequential` のシフト量ロジックを調整し、空スロット・非ゼロ最小時刻を許容するアルゴリズムへ置換します。
- `last_nodes_remapped` の取得を `node_map2.get(int(n), int(n))` 形式に書き換え、恒等マップで KeyError が起きないようにします。
- 層内に複数キューブ／パイプを配置した統合テストを追加し、スケジュール・フローが期待通り連結されるかを検証します。
- 変更点の仕様と制約を `docs/` に追記し、開発者へ共有します。

## 検証方法
- 追加したユニットテストと既存テストを `pytest` で実行し、スケジュールの時間帯が衝突しないことを確認します。
- `ruff check` と `mypy lspattern` を走らせ、静的解析上の問題がないことを保証します。
- 多層構成で `add_temporal_layer` を行い、`schedule.schedule` のキーが昇順かつ重複なしであるかをデバッグ出力またはアサーションで確認します。
- Flow 統合で `Flow merge conflict` が発生しないことを、複数パイプを含むケースを含めて確認し、例外が出た場合は原因をログに記録します。
- 修正点の説明を Pull Request テンプレートに沿ってまとめ、レビューアへ影響範囲（スケジュール、フロー、パリティ）を明示します。
