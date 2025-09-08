# LSPattern MyPy 型付け対応順序と影響範囲分析

## 依存関係の少ない順でのmypy対応順序

### Phase 1: 基盤モジュール（他に依存しないモジュール）
1. ✅**lspattern/mytype.py**
   - 依存: なし
   - 影響範囲: 9モジュール（最も影響が大きい基盤型定義）

2. ✅**lspattern/consts/consts.py**
   - 依存: なし
   - 影響範囲: 8モジュール

3. ✅**lspattern/compile.py**
   - 依存: なし
   - 影響範囲: なし（独立モジュール）

4. ✅**lspattern/geom/tiler.py**
   - 依存: なし
   - 影響範囲: なし（独立モジュール）

5. ✅**lspattern/visualizers/template.py**
   - 依存: なし
   - 影響範囲: なし（独立モジュール）

6. ✅**lspattern/visualizers/compiled_canvas.py**
   - 依存: matplotlib のみ
   - 影響範囲: なし（独立モジュール）

7. ✅**lspattern/visualizers/plotly_compiled_canvas.py**
   - 依存: plotly のみ
   - 影響範囲: なし（独立モジュール）

### Phase 2: 第1レベル依存モジュール
8. ✅**lspattern/consts/__init__.py**
   - 依存: なし（consts.pyをimport）
   - 影響範囲: 8モジュール

9. ✅**lspattern/utils.py**
   - 依存: lspattern, consts, mytype
   - 影響範囲: 6モジュール

10. 👷**lspattern/accumulator.py**
    - 依存: lspattern, mytype
    - 影響範囲: 3モジュール

1.  ✅**lspattern/tiling/base.py**
    - 依存: lspattern, mytype
    - 影響範囲: 2モジュール

2.  🐛**lspattern/rhg.py**
    - 依存: matplotlib のみ
    - 影響範囲: 1モジュール（ops.py）

### Phase 3: 第2レベル依存モジュール
13. ✅**lspattern/tiling/template.py**
    - 依存: lspattern, consts, mytype, tiling.base, utils, matplotlib
    - 影響範囲: 8モジュール（重要な中核モジュール）

14. 🐛**lspattern/ops.py**
    - 依存: lspattern, rhg
    - 影響範囲: なし

15. ✅**lspattern/tiling/visualize.py**
    - 依存: lspattern, tiling, tiling.base, matplotlib
    - 影響範囲: なし

### Phase 4: 第3レベル依存モジュール（ブロック基底）
16. **lspattern/blocks/base.py**
    - 依存: lspattern, accumulator, consts, mytype, tiling.template
    - 影響範囲: 3モジュール（cubes.base, cubes.measure, pipes.base）

17. **lspattern/geom/rhg_parity.py**
    - 依存: なし
    - 影響範囲: 3モジュール（可視化関連）

### Phase 5: 第4レベル依存モジュール（ブロック実装）
18. **lspattern/blocks/pipes/base.py**
    - 依存: lspattern, blocks, blocks.base, consts, mytype, tiling.template, utils
    - 影響範囲: 3モジュール（pipes関連）

19. **lspattern/blocks/cubes/base.py**
    - 依存: lspattern, blocks, blocks.base, tiling.template
    - 影響範囲: 3モジュール（cubes関連とcanvas）

20. **lspattern/visualizers/temporallayer.py**
    - 依存: lspattern, geom, geom.rhg_parity, matplotlib
    - 影響範囲: 1モジュール

21. **lspattern/visualizers/plotly_temporallayer.py**
    - 依存: lspattern, geom, geom.rhg_parity, plotly
    - 影響範囲: 1モジュール

22. **lspattern/visualizers/visualize.py**
    - 依存: lspattern, geom, geom.rhg_parity, matplotlib
    - 影響範囲: なし

### Phase 6: 第5レベル依存モジュール（特化実装）
23. **lspattern/blocks/pipes/initialize.py**
    - 依存: lspattern, blocks, pipes, pipes.base, consts, mytype, tiling.template, utils
    - 影響範囲: なし

24. **lspattern/blocks/pipes/memory.py**
    - 依存: lspattern, blocks, pipes, pipes.base, consts, mytype, tiling.template, utils
    - 影響範囲: なし

25. **lspattern/blocks/cubes/initialize.py**
    - 依存: lspattern, blocks, cubes, cubes.base, tiling.template
    - 影響範囲: なし

26. **lspattern/blocks/cubes/memory.py**
    - 依存: lspattern, blocks, cubes, cubes.base
    - 影響範囲: なし

27. **lspattern/visualizers/accumulators.py**
    - 依存: lspattern, accumulator, visualizers, plotly_temporallayer, temporallayer, matplotlib, plotly
    - 影響範囲: なし

### Phase 7: 高レベル統合モジュール
28. **lspattern/canvas.py**
    - 依存: lspattern, accumulator, blocks, cubes.base, pipes.base, consts, mytype, tiling.template, utils
    - 影響範囲: 1モジュール（cubes.measure）

### Phase 8: 最上位モジュール
29. **lspattern/blocks/cubes/measure.py**
    - 依存: lspattern, blocks, blocks.base, canvas
    - 影響範囲: なし

## 重要な影響関係

### 高影響モジュール（変更時に多くのモジュールに影響）
1. **mytype.py** (9モジュールに影響) - 型定義の中核
2. **consts/consts.py** (8モジュールに影響) - 定数定義
3. **tiling/template.py** (8モジュールに影響) - テンプレート機能
4. **utils.py** (6モジュールに影響) - ユーティリティ関数

### 中影響モジュール
- **accumulator.py** (3モジュールに影響)
- **blocks/base.py** (3モジュールに影響) - ブロック基底クラス
- **blocks/cubes/base.py** (3モジュールに影響)
- **blocks/pipes/base.py** (3モジュールに影響)

### 低影響・独立モジュール
- compile.py, ops.py, 各種visualizer実装など

## 推奨作業順序

1. **Phase 1から順番に進める**: 基盤モジュールから開始
2. **mytype.pyを最優先**: すべての型定義の基盤
3. **テスト駆動**: 各Phaseで型付け後、関連モジュールの型チェックも実行
4. **影響範囲を意識**: 高影響モジュール変更後は依存モジュールも再チェック

## mypy実行コマンド例

```bash
# Phase 1
mypy lspattern/mytype.py
mypy lspattern/consts/consts.py

# 各Phaseで影響範囲も含めてチェック
mypy lspattern/mytype.py lspattern/accumulator.py lspattern/utils.py  # mytype変更後
```

この順序で作業することで、依存関係による型エラーの連鎖を最小限に抑えつつ、効率的に型付け作業を進められます。
