# Merge Split Mockup機能分析とデバッグ計画

## 概要

`examples/merge_split_mockup.py`の動作を実現するために必要な機能の調査と、段階的な実装・デバッグ計画をまとめた。

## 現在のコードベース構造

### 主要コンポーネント

#### 1. Canvas系 (`lspattern/canvas.py`)
- **RHGCanvasSkeleton**: cubeとpipeのスケルトンを管理するコンテナ
- **RHGCanvas**: 実際のブロックを含むコンパイル対象
- **TemporalLayer**: 単一の時間層(z)におけるcubeとpipeの集合
- **CompiledRHGCanvas**: コンパイル済みの最終形態

#### 2. Block系
- **Cube系ブロック**:
  - `InitPlusCubeSkeleton/InitPlus`: 初期化ブロック（入力ポートなし、出力ポートあり）
  - `MemoryCubeSkeleton/MemoryCube`: メモリブロック（入力・出力ポートあり、時間延長）
  - `MeasureXSkeleton/MeasureX`: X測定ブロック（入力ポートあり、出力ポートなし）

- **Pipe系ブロック**:
  - `MemoryPipeSkeleton/MemoryPipe`: メモリパイプ（cube間の接続）

#### 3. Template系 (`lspattern/tiling/template.py`)
- **ScalableTemplate**: タイリングのベースクラス
- **RotatedPlanarCubeTemplate**: cube用のテンプレート
- **RotatedPlanarPipetemplate**: pipe用のテンプレート

### 現在のmockupファイルの状態

```python
# 現在有効な部分
blocks = [
    (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
]

# コメントアウトされている部分
# - MemoryCubeSkeleton (z=1,2の層)
# - MeasureXSkeleton (z=3の層)
# - MemoryPipeSkeletonの接続
```

## 発見された型エラーと課題

### 1. 型エラー (canvas.py)
- **Line 288**: `BaseGraphState`が`GraphState`に代入できない
- **Line 323-329**: `list[int]`が`Iterable[NodeIdLocal]`に代入できない
- **Line 1266**: `BaseGraphState | None`が`BaseGraphState`に渡せない

### 2. 機能の未実装・未検証部分

#### ~~A. Single Cube縦連結機能~~ (実装済み)
**問題**: 同一(x,y)座標での異なるz層のcube接続
**必要な機能**:
- temporal layerの適切な管理
- z方向の接続でのポート整合性
- cubeのfinal_layer設定の正しい処理

#### B. 複数Cube横並び機能
**問題**: 異なる(x,y)座標でのcube配置
**必要な機能**:
- 独立したtiling IDの管理
- 空間的に分離されたcubeの共存
- 各cubeの独立したポート管理

#### C. Pipeを含むLayer作成機能
**問題**: cube間をpipeで接続するlayer構築
**必要な機能**:
- PipeCoordGlobal3D((source, sink))の正しい処理
- pipeのqindexとcubeのqindexの整合性
- spatial pipe compositionの実装

#### D. 複数CubeからなるLayer接続機能
**問題**: temporal方向でのlayer間接続
**必要な機能**:
- 前層outputと次層inputの対応づけ
- 異なるパッチ間でのqubit ID整合性
- layer全体でのflow一貫性

#### E. Merge and Split確認
**問題**: 最終的な量子回路としての正当性
**必要な機能**:
- パターンの正当性検証
- flow特性の確認
- 測定スケジュールの妥当性

## 段階的デバッグ計画

### Phase 1: Single Cube縦連結対応
**目標**: 同一(x,y)位置でのcube z方向接続

**作業項目**:
1. **型エラー修正**: canvas.pyの型エラーを解決
2. **TemporalLayer分離**: z=0,1,2,3それぞれでTemporalLayerを作成
3. **Cube接続検証**: InitPlus → Memory → Memory → MeasureXの接続
4. **ポート整合性**: 各cubeのin_ports/out_ports対応確認

**検証方法**:
```python
# z=0: InitPlusCubeSkeleton
# z=1: MemoryCubeSkeleton
# z=2: MemoryCubeSkeleton
# z=3: MeasureXSkeleton
```

### Phase 2: 複数Cube横並び対応
**目標**: (0,0)と(1,0)でのcube並列配置

**作業項目**:
1. **Tiling ID管理**: 各cubeが独立したtiling IDを持つ
2. **Union-Find処理**: pipeなしでの独立cube処理
3. **座標系整合**: グローバル座標でのnode配置確認
4. **可視化検証**: 2つのcubeが正しく表示される

**検証方法**:
```python
# z=0: 2つのInitPlusCubeSkeleton at (0,0,0), (1,0,0)
```

### Phase 3: Pipe含むLayer作成対応
**目標**: cube間をpipeで接続

**作業項目**:
1. **Pipe composition**: `_compose_pipe_graphs`の実装確認
2. **QIndex整合**: pipeとcubeでのqubit index一貫性
3. **Edge仕様処理**: edgespec_trimmedの適切な適用
4. **接続検証**: source cubeのout_portsとsink cubeのin_ports接続

**検証方法**:
```python
# MemoryPipeSkeleton((0,0,1), (1,0,1))でのcube接続
```

### Phase 4: Layer間接続対応
**目標**: temporal方向でのlayer接続

**作業項目**:
1. **Layer積み重ね**: 複数TemporalLayerの統合
2. **Temporal edge**: 時間方向エッジの適切な生成
3. **Flow継続性**: layer間でのflow制約維持
4. **Schedule統合**: 各layerのscheduleをグローバルに統合

**検証方法**:
```python
# z=0: 2 cubes, z=1: 2 cubes + 1 pipe, z=2: 2 cubes, z=3: 2 measurement cubes
```

### Phase 5: Merge and Split検証
**目標**: 最終パターンの正当性確認

**作業項目**:
1. **パターン完全性**: 全nodeとedgeの妥当性
2. **Flow特性**: gflow/pflowの確認
3. **測定順序**: 因果律に従った測定スケジュール
4. **量子回路等価性**: 期待される量子処理との一致

## 実装上の注意点

### 1. 型安全性
- `NodeIdLocal`と`int`の区別を厳密に
- `BaseGraphState`vs`GraphState`の使い分け明確化
- Optional型の適切な処理

### 2.座標系管理
- Global vs Local座標の変換ルール
- パッチ座標とphysical座標の対応
- Z軸方向のオフセット計算

### 3. QIndex管理
- パッチごとのqindex範囲
- `calculate_qindex_base`の活用
- pipe接続でのqindex継続性

### 4. デバッグ支援
- 各段階での可視化確認
- ログ出力でのデータ構造検証
- ユニットテストでの回帰防止

## 期待される成果

1. **段階的動作確認**: 各フェーズでの部分的動作確認
2. **完全なmerge/split**: 最終的な量子パターン生成
3. **拡張可能性**: より複雑な量子回路への応用
4. **デバッグ手法確立**: 今後の機能追加時の指針

この計画に従って段階的にデバッグを進めることで、merge_split_mockup.pyの完全な動作実現を目指す。
