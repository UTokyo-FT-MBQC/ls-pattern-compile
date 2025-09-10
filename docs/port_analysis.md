# lspatternライブラリにおけるポート管理の詳細解析

## 概要

lspatternライブラリでは、RHGブロック間の論理的な接続を管理するために、`in_ports`、`out_ports`、`cout_ports`の3つのポート種別を使用している。本文書では、これらのポートがどのように定義され、処理され、接続されるかを詳細に解析する。

## ポート種別の定義

### 1. in_ports
- **型**: `set[QubitIndexLocal]`
- **役割**: ブロックへの論理入力ポート
- **物理的対応**: z-層（最小z座標）のデータノード

### 2. out_ports
- **型**: `set[QubitIndexLocal]`
- **役割**: ブロックからの論理出力ポート
- **物理的対応**: z+層（最大z座標）のデータノード

### 3. cout_ports
- **型**: `list[set[QubitIndexLocal]]`
- **役割**: 古典的出力ポート（測定結果）
- **構造**: 各`set`は1つの論理的測定結果に対応するクラシカルビットのグループ

## アーキテクチャ詳細

### ポート設定プロセス

各RHGBlockの`materialize()`メソッドで以下の処理が実行される：

```python
# 1. ポート初期化
self.set_in_ports()
self.set_out_ports()
self.set_cout_ports()

# 2. 物理ノードへのマッピング
g, node2coord, coord2node, node2role = self._build_3d_graph()

# 3. GraphStateへの登録
self._register_io_nodes(g, node2coord, coord2node, node2role)
```

### 論理インデックスと物理ノードの対応

```python
# 入力ポートの登録 (base.py:401)
lidx = g.register_input(n_in)  # 物理ノードを入力として登録

# 出力ポートの登録 (base.py:429)
g.register_output(n_out, int(lidx))  # 物理ノードを論理インデックス付きで登録
```

## ブロック別実装

### InitPlus（初期化ブロック）
```python
def set_in_ports(self) -> None:
    idx_map = self.template.get_data_indices()
    self.in_ports = set(idx_map.values())

def set_out_ports(self) -> None:
    idx_map = self.template.get_data_indices()
    self.out_ports = set(idx_map.values())

def set_cout_ports(self) -> None:
    return super().set_cout_ports()  # 空実装
```

### Memory（メモリブロック）
```python
def set_in_ports(self) -> None:
    idx_map = self.template.get_data_indices()
    self.in_ports = set(idx_map.values())

def set_out_ports(self) -> None:
    idx_map = self.template.get_data_indices()
    self.out_ports = set(idx_map.values())

def set_cout_ports(self) -> None:
    return super().set_cout_ports()  # 空実装
```

### Measure（測定ブロック）- **未実装**
```python
class _MeasureBase(RHGBlock):
    """測定ブロックの基底クラス（実装不完全）"""

    # 以下のメソッドが未実装:
    # - set_in_ports(): 前段から受け取る論理ポート
    # - set_out_ports(): 通常は空（論理境界を消費）
    # - set_cout_ports(): 測定結果の古典出力
```

## 時間的合成における接続メカニズム

### 1. compose_sequentially()による自動接続

```python
# graphix_zx/graphstate.py:695-697
if set(graph1.output_node_indices.values()) != set(graph2.input_node_indices.values()):
    raise ValueError("Logical qubit indices must match")

# 前段の出力と後段の入力を物理的に接続
for input_node_index2, q_index in graph2.input_node_indices.items():
    node_map1[output_node_from_graph1] = node_map2[input_node_index2]
```

### 2. TemporalLayerでのポート管理

```python
class TemporalLayer:
    # パッチ座標ごとのポートマッピング
    in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    cout_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]

    # 全体のポートリスト
    in_ports: list[NodeIdLocal]
    out_ports: list[NodeIdLocal]
    cout_ports: list[NodeIdLocal]
```

### 3. 時間的合成でのポートリマッピング

```python
def _remap_temporal_portsets(cgraph, next_layer, node_map1, node_map2):
    # 入力ポート: 次層のポートを新しいノードIDでマッピング
    in_portset = {pos: [node_map2[int(n)] for n in nodes]
                  for pos, nodes in next_layer.in_portset.items()}

    # 出力ポート: 前段のポートを保持
    out_portset = {pos: [node_map1[int(n)] for n in nodes]
                   for pos, nodes in cgraph.out_portset.items()}

    # 古典出力ポート: 両段をマージ
    cout_portset = {
        **{pos: [node_map1[int(n)] for n in nodes] for pos, nodes in cgraph.cout_portset.items()},
        **{pos: [node_map2[int(n)] for n in nodes] for pos, nodes in next_layer.cout_portset.items()},
    }

    return in_portset, out_portset, cout_portset
```

## cout_ports使用状況の詳細

### データ構造と意味

```python
cout_ports: list[set[QubitIndexLocal]]
#          ^^^^  ^^^
#          |     各測定結果に対応するクラシカルビットのセット
#          測定結果のグループリスト（XORで組み合わせる）
```

### canvas.pyでの処理

```python
# ブロックのcout_portsをlayerのcout_portsetに追加
if getattr(blk, "cout_ports", None):
    self.cout_portset[patch_pos] = [
        NodeIdLocal(node_map2[n])
        for s in blk.cout_ports  # list内の各set
        for n in s               # set内の各ノード
        if n in node_map2
    ]
```

## 課題と今後の開発方向

### 1. 測定ブロックの実装不備

**問題**:
- `_MeasureBase`でポート設定メソッドが未実装
- `RHGBlock`の適切な初期化が未実行
- ドキュメントと実装の不一致

**必要な実装**:
```python
class _MeasureBase(RHGBlock):
    def __init__(self, logical: int, basis: Axis) -> None:
        super().__init__()  # RHGBlockの初期化が必要
        self.logical = logical
        self.basis = AxisMeasBasis(basis, Sign.PLUS)

    def set_in_ports(self) -> None:
        # 前段から受け取る論理ポートを設定
        pass

    def set_out_ports(self) -> None:
        # 測定は論理境界を消費するため通常は空
        self.out_ports = set()

    def set_cout_ports(self) -> None:
        # 測定結果の古典出力を設定
        pass
```

### 2. cout_portsの活用

現在、InitPlusとMemoryブロックでは`cout_ports`は設定されていない。将来的に以下の用途での活用が想定される：

- 中間測定結果の管理
- エラー訂正情報の伝播
- デバッグ情報の出力

### 3. 型安全性の向上

NewTypeを活用した型定義により、各種IDの混同を防止している：

```python
NodeIdLocal = NewType('NodeIdLocal', int)
NodeIdGlobal = NewType('NodeIdGlobal', int)
QubitIndexLocal = NewType('QubitIndexLocal', int)
PatchCoordGlobal3D = NewType('PatchCoordGlobal3D', tuple[int, int, int])
```

## まとめ

lspatternライブラリのポート管理システムは、論理的な量子回路接続と物理的なグラフ実装を適切に分離する洗練されたアーキテクチャを持つ。しかし、測定ブロックの実装が不完全であり、`cout_ports`の活用も限定的である。今後これらの課題を解決することで、より完全なMBQC実行基盤となることが期待される。
