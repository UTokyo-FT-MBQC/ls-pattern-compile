# LS-Pattern-Compile コードベース分析

## 概要

本ドキュメントでは、以下の3つのサンプルファイルに関連するすべてのPythonファイルの包括的な分析を提供します：
- `examples/merge_split_mockup.py`
- `examples/pipe_visualization.py`
- `examples/compiled_canvas_visualization.py`

**ls-pattern-compile**ライブラリは、測定ベース量子計算（MBQC）のための線形サイズパターンコンパイルを実装しています。Resource-efficient Hybrid Graph（RHG）アーキテクチャを使用した量子グラフ状態の構築、コンパイル、可視化ツールを提供します。

### 高レベルアーキテクチャ

ライブラリは階層化されたアーキテクチャに従います：

1. **テンプレートシステム**: 量子構造のためのスケーラブルな2D/3Dタイリングパターン
2. **ブロックシステム**: 定義された量子操作を持つキューブとパイプコンポーネント
3. **キャンバスシステム**: ブロックの時間的レイヤーへの合成とコンパイル
4. **可視化システム**: 量子構造をレンダリングするための複数のバックエンド
5. **アキュムレータシステム**: コンパイル中のスケジュール、フロー、パリティ追跡

## コアモジュール分析

### キャンバスとコンパイルシステム

#### `lspattern/canvas.py` (1045行)
**主要クラス:**
- `TemporalLayer`: キューブとパイプを含む単一の時間スライスを表現
- `CompiledRHGCanvas`: グローバルグラフ状態とマッピングを持つコンパイル済みキャンバス
- `RHGCanvasSkeleton`: キャンバス構造を構築するための設計図
- `RHGCanvas`: コンパイル準備完了の具体化されたキャンバス

**主要関数:**
- `to_temporal_layer()`: キューブ/パイプのコレクションを時間的レイヤーに変換
- `add_temporal_layer()`: 時間的レイヤーを順次合成
- `compile()`: 時間的構造からグローバルグラフ状態を構築

**依存関係:**
- `graphix_zx.graphstate`: 量子グラフ操作用
- `lspattern.accumulator`: コンパイルメタデータの追跡用
- `lspattern.tiling.template`: 座標変換用
- `lspattern.utils`: グラフユーティリティ用

### ブロックシステム

#### キューブブロック

**`lspattern/blocks/cubes/base.py`**
- `RHGCube`: コアキューブブロック実装
- `RHGCubeSkeleton`: `RotatedPlanarCubeTemplate`を使用するキューブ構造の設計図

**`lspattern/blocks/cubes/initialize.py`**
- `InitPlusCubeSkeleton`: 初期化ブロックを作成
- `InitPlus`: 入力ポートなし、データ出力ポートありの具体化された初期化キューブ

**`lspattern/blocks/cubes/memory.py`**
- `MemoryCubeSkeleton`: メモリ/時間拡張ブロックを作成
- `MemoryCube`: 時間的継続性のためのデータ入出力ポートを持つ具体化されたメモリキューブ

**`lspattern/blocks/cubes/measure.py`**
- 測定ベースのキューブ実装（サンプルでインポートされているが直接使用されていない）

#### パイプブロック

**`lspattern/blocks/pipes/base.py`**
- `RHGPipeSkeleton`: キューブ間の空間的接続の設計図
- `RHGPipe`: ソース/シンク座標と方向を持つ具体化されたパイプ

**`lspattern/blocks/pipes/initialize.py`**
- `InitPlusPipeSkeleton`: 初期化パイプを作成
- `InitPlusPipe`: データ出力ポートのみを持つ具体化された初期化パイプ

**`lspattern/blocks/pipes/memory.py`**
- `MemoryPipeSkeleton`: 時間的接続のためのメモリパイプを作成
- `MemoryPipe`: データ入出力ポートを持つ具体化されたメモリパイプ

**`lspattern/blocks/base.py`**
- `RHGBlock`: すべての量子ブロックのベースクラス
- `RHGBlockSkeleton`: 共通機能を持つベーススケルトンクラス

### タイリングとテンプレートシステム

#### `lspattern/tiling/template.py` (100+行分析済み)
**主要クラス:**
- `ScalableTemplate`: 座標管理を持つベーステンプレートクラス
- `RotatedPlanarCubeTemplate`: 3D キューブ形状の量子構造
- `RotatedPlanarPipetemplate`: キューブ間のパイプ接続

**主要メソッド:**
- `to_tiling()`: テンプレートを座標リストに変換
- `shift_coords()`: 座標系間で座標を変換
- `get_data_indices()`: 座標を量子ビットインデックスにマッピング

**サポートファイル:**
- `lspattern/tiling/base.py`: ベースタイリングインターフェース
- `lspattern/tiling/visualize.py`: テンプレート可視化ユーティリティ

### 型システム

#### `lspattern/mytype.py` (113行)
**座標型:**
- `PatchCoordGlobal3D`: 3Dパッチ位置 (x,y,z)
- `PhysCoordGlobal3D`: 3D物理量子ビット座標
- `PipeCoordGlobal3D`: パイプ端点ペア座標
- `TilingCoord2D`: 2Dタイリング座標

**ID型:**
- `NodeIdLocal/Global`: グラフノード識別子
- `QubitGroupIdGlobal`: ゲーティング規則のための量子ビットグループ識別子
- `TilingId`: テンプレートインスタンス識別子

**エッジ仕様:**
- `SpatialEdgeSpec`: 境界面から量子演算子へのマッピング辞書 ("X"/"Z"/"O")
- `EdgeSpecValue`: 境界仕様のリテラル型

### アキュムレータシステム

#### `lspattern/accumulator.py` (100+行分析済み)
**クラス:**
- `BaseAccumulator`: グラフコンテキスト抽出の共通ユーティリティ
- `ScheduleAccumulator`: 操作の時間的順序を追跡
- `FlowAccumulator`: X/Z補正フローを管理
- `ParityAccumulator`: パリティチェック関係を処理

**主要メソッド:**
- `update_at()`: レイヤー合成中の単調更新
- `remap_nodes()`: グラフ合成中のノードID再マッピング
- `compose_sequential()`: アキュムレータの順次結合

### 可視化システム

#### `lspattern/visualizers/__init__.py`
**モジュールエクスポート:**
- `visualize_temporal_layer`: 2D matplotlib可視化
- `visualize_temporal_layer_plotly`: インタラクティブ3D plotly可視化
- `visualize_compiled_canvas`: 3D matplotlibキャンバス可視化
- `visualize_compiled_canvas_plotly`: インタラクティブ3Dキャンバス可視化

#### `lspattern/visualizers/compiled_canvas.py` (142行)
**主要関数:** `visualize_compiled_canvas()`
- matplotlib 3Dを使用して`CompiledRHGCanvas`をレンダリング
- 時間的レイヤー（z座標）によるノードの色分け
- グラフエッジ、入出力ノードの表示
- 注釈、軸制御、エッジ表示オプションをサポート

#### `lspattern/visualizers/plotly_compiled_canvas.py` (160行)
**主要関数:** `visualize_compiled_canvas_plotly()`
- plotlyを使用したインタラクティブ3D可視化
- カラーバー付きz座標による色マッピング
- ノード詳細を表示するホバー情報
- カメラコントロールと軸設定

#### 追加の可視化ツール
- `temporallayer.py`: 単一レイヤーmatplotlib可視化
- `plotly_temporallayer.py`: 単一レイヤーplotly可視化
- `accumulators.py`: フロー、スケジュール、パリティの専用可視化

### ユーティリティモジュール

#### `lspattern/utils.py`
- `UnionFind`: 連結成分のための素集合データ構造
- `get_direction()`: 空間的/時間的パイプ方向を決定
- `is_allowed_pair()`: 量子ビットグループペアリング規則を検証
- `sort_xy()`: 座標ソートユーティリティ

#### `lspattern/consts/consts.py`
- `DIRECTIONS3D`: 3D近傍方向ベクトル
- `PIPEDIRECTION`: パイプ方向タイプの列挙型

#### `lspattern/geom/`
- `rhg_parity.py`: RHG特有のパリティ計算ユーティリティ
- `tiler.py`: 幾何学的タイリングアルゴリズム

## 依存関係グラフ分析

### インポート関係

```
examples/
├── merge_split_mockup.py
├── pipe_visualization.py
└── compiled_canvas_visualization.py
    ├── lspattern.blocks.cubes.initialize
    ├── lspattern.blocks.cubes.memory
    ├── lspattern.blocks.pipes.memory
    ├── lspattern.canvas
    ├── lspattern.mytype
    └── lspattern.visualizers
        ├── compiled_canvas
        └── plotly_compiled_canvas

lspattern/
├── canvas.py
│   ├── graphix_zx.graphstate
│   ├── .accumulator
│   ├── .mytype
│   ├── .tiling.template
│   └── .utils
├── blocks/
│   ├── cubes/
│   │   ├── base.py → .tiling.template
│   │   ├── initialize.py → .base
│   │   └── memory.py → .base
│   └── pipes/
│       ├── base.py → .blocks.base
│       ├── initialize.py → .base, .tiling.template
│       └── memory.py → .base, .tiling.template
├── tiling/
│   ├── template.py → .consts, .mytype, .base
│   └── base.py
├── visualizers/
│   ├── compiled_canvas.py → matplotlib
│   └── plotly_compiled_canvas.py → plotly
├── mytype.py (基盤となる型)
├── accumulator.py → .mytype
└── utils.py
```

### データフロー

1. **テンプレート作成**: `RotatedPlanarCubeTemplate`/`RotatedPlanarPipetemplate`が量子構造レイアウトを定義
2. **ブロック具体化**: スケルトン → テンプレートとポート定義を持つブロック
3. **キャンバス組み立て**: `RHGCanvasSkeleton`がブロックを結合し、空間境界トリミングを適用
4. **時間的レイヤー化**: キャンバス → テンプレートにXY座標シフトが適用された時間的レイヤー
5. **コンパイル**: 時間的レイヤーの`CompiledRHGCanvas`への順次合成
6. **可視化**: コンパイル済みキャンバス → matplotlib/plotly 3Dレンダリング

### 主要インターフェース

- **スケルトン → ブロック**: `to_block()`がテンプレートを具体化しポート設定を行う
- **キャンバス → レイヤー**: `to_temporal_layers()`がz座標でセグメント化し座標変換を適用
- **レイヤー合成**: `add_temporal_layer()`でグラフ状態合成とアキュムレータ更新
- **可視化インターフェース**: レンダリングオプション付きコンパイル済みキャンバスを受け取る標準化された関数

## サンプルファイル分析

### `examples/merge_split_mockup.py`

**目的**: 空間的パイプ接続を持つメモリキューブ操作のデモンストレーション。

**主要コンポーネント:**
- **6ブロック**: 2つの`InitPlusCubeSkeleton` + 4つの`MemoryCubeSkeleton`をz=0,1,2にわたって2×3グリッドに配置
- **2パイプ**: 隣接キューブ間の`MemoryPipeSkeleton`接続
- **エッジ仕様**: キューブ境界とパイプトリミング用の異なる仕様

**ワークフロー:**
1. 配置されたキューブとパイプで`RHGCanvasSkeleton`を作成
2. `RHGCanvas`に変換（境界トリミングを適用）
3. `CompiledRHGCanvas`にコンパイル
4. matplotlibとplotlyの両バックエンドで可視化

**出力**: 時間的レイヤーで色分けされたノードを持つ量子グラフ構造の3D可視化。

### `examples/pipe_visualization.py`

**目的**: パイプ内部アンカー処理の検証とレイヤー0境界ノードの可視化。

**主要コンポーネント:**
- **水平構成**: 水平パイプで接続された2つのキューブ
- **垂直構成**: 垂直パイプで接続された2つのキューブ
- **エッジ仕様**: 方向固有の境界設定（開放境界にO）

**関数:**
- `build_horizontal()`: RIGHT/LEFTパイプ構成を作成
- `build_vertical()`: TOP/BOTTOMパイプ構成を作成

**出力:**
- 2D投影を表示するPNGファイルをディスクに保存
- インタラクティブplotly可視化
- `node2coord`マッピングの座標一意性検証

### `examples/compiled_canvas_visualization.py`

**目的**: 多様なブロック配置を持つ多層コンパイル済みキャンバス可視化。

**主要コンポーネント:**
- **6ブロック**: 異なる座標の初期化とメモリキューブの混合
- **時間的パイプ**: z=0からz=1レイヤーを接続する単一パイプ
- **テストケース**:
  - 時間的進行（z=0→1）
  - 空間的分離（異なるxy座標）
  - トリミングされた境界
  - 接続のない孤立ブロック

**出力**: 以下を含む多層量子構造の3D可視化：
- 時間的レイヤーで色分けされたノード
- レイヤー内およびレイヤー間のエッジ接続
- 統計サマリー（レイヤー数、ノード数、エッジ数）

## 重要概念とインターフェース

### RHGアーキテクチャ概念

1. **時間的レイヤー化**: 離散時間ステップ（z座標）で整理された量子計算
2. **空間的タイリング**: 各時間的レイヤー内の量子構造の2Dテセレーション
3. **ブロック合成**: 定義された入出力インターフェースを持つモジュール式キューブとパイプコンポーネント
4. **境界管理**: エッジ仕様による開放/閉鎖量子境界の体系的処理

### コンパイル処理

1. **スケルトンフェーズ**: パラメータ付きの抽象ブロック定義
2. **キャンバスフェーズ**: 空間配置と境界トリミング
3. **時間的フェーズ**: 座標変換付きレイヤーセグメント化
4. **コンパイルフェーズ**: アキュムレータ追跡付きグラフ状態合成
5. **可視化フェーズ**: 複数バックエンドオプションでの2D/3Dレンダリング

### 設計パターン

- **スケルトン/ブロックパターン**: コンパイル前のパラメータ変更を可能にする遅延具体化
- **テンプレートシステム**: 変換サポート付き再利用可能な座標パターン
- **アキュムレータパターン**: グラフ合成中の増分メタデータ収集
- **マルチバックエンド可視化**: matplotlibとplotlyの両方をサポートする統一インターフェース

### 重要な依存関係

- **graphix-zx**: 量子グラフ状態操作と合成
- **matplotlib/plotly**: 可視化バックエンド
- **dataclasses**: 設定とデータ構造定義
- **typing**: 座標とID管理の包括的型システム

このアーキテクチャにより、効率的なコンパイルと豊富な可視化機能を持つ量子計算パターンのスケーラブルな構築が可能になります。
