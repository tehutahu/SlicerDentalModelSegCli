import bpy
import numpy as np
from pathlib import Path
import colorsys

# TODO
# 一色で出る
def import_ply_file(filepath: str) -> bpy.types.Object:
    """バージョンに応じたPLYファイルのインポート
    
    Args:
        filepath: PLYファイルのパス
    
    Returns:
        インポートされたオブジェクト
    """
    # Blenderのバージョンを取得
    version = bpy.app.version

    # 既存のオブジェクトをクリア
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    if version >= (4, 1, 0):
        # 4.1以降
        bpy.ops.import_mesh.ply(filepath=filepath)
    else:
        # 4.0以前
        bpy.ops.wm.ply_import(filepath=filepath)
    
    # インポートされたオブジェクトを返す
    return bpy.context.selected_objects[0]

def create_material_for_label(label: int) -> bpy.types.Material:
    """ラベルごとに固有の色を持つマテリアルを作成"""
    # ラベルに基づいてユニークな色を生成
    # HSVカラースペースを使用して、視覚的に区別しやすい色を生成
    hue = (label * 0.618033988749895) % 1.0  # 黄金比を使用して色相を分散
    material_name = f"tooth_material_{label}"
    
    # 既存のマテリアルがあれば再利用
    mat = bpy.data.materials.get(material_name)
    if mat is not None:
        return mat
    
    # 新しいマテリアルを作成
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # ノードを初期化
    nodes.clear()
    
    # プリンシプルBSDFノードを追加
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # HSVから RGB に変換
    color = tuple(c for c in colorsys.hsv_to_rgb(hue, 0.8, 0.9)) + (1.0,)
    
    # マテリアルの基本プロパティを設定
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Metallic'].default_value = 0.1
    principled.inputs['Roughness'].default_value = 0.8
    
    # ノードを接続
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def setup_mesh_for_vertex_colors(obj: bpy.types.Object):
    """メッシュにバーテックスカラーレイヤーを設定"""
    if obj.data.vertex_colors:
        vcol = obj.data.vertex_colors.active
    else:
        vcol = obj.data.vertex_colors.new()
    return vcol

def apply_segmentation_colors(ply_path: str):
    """PLYファイルを読み込み、セグメンテーション情報に基づいて色を適用"""
    # PLYファイルをインポート
    obj = import_ply_file(ply_path)
    mesh = obj.data
    
    # カスタムデータレイヤーからラベル情報を取得
    if not mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.new()
    else:
        vcol_layer = mesh.vertex_colors.active
    
    # 頂点ごとのラベル情報を取得
    labels = []
    for vertex in mesh.vertices:
        # PLYファイルから読み込んだラベル情報を取得
        label = vertex.groups[0].weight * 100 if vertex.groups else 0  # グループがない場合は0を使用
        labels.append(int(label))
    
    # マテリアルスロットをクリア
    obj.data.materials.clear()
    
    # ユニークなラベルごとにマテリアルを作成
    unique_labels = set(labels)
    label_materials = {label: create_material_for_label(label) for label in unique_labels}
    
    # 各ラベルに対応するマテリアルをオブジェクトに追加
    for label in unique_labels:
        obj.data.materials.append(label_materials[label])
    
    # 新しいマテリアルインデックスレイヤーを作成
    if not mesh.vertex_colors:
        color_layer = mesh.vertex_colors.new()
    else:
        color_layer = mesh.vertex_colors.active
    
    # 各面に対してマテリアルを割り当て
    for poly in mesh.polygons:
        vertices = poly.vertices
        # 面の頂点のラベルの最頻値を取得
        face_label = max(set([labels[v] for v in vertices]), key=[labels[v] for v in vertices].count)
        # 対応するマテリアルインデックスを設定
        poly.material_index = list(unique_labels).index(face_label)
    
    # シェーディングをスムーズに設定
    bpy.ops.object.shade_smooth()
    
    # エッジスプリットモディファイアを追加して、セグメント境界をシャープに
    edge_split = obj.modifiers.new(name="Edge Split", type='EDGE_SPLIT')
    edge_split.split_angle = 30  # 角度のしきい値（度）
    
    # ビューの更新
    bpy.context.view_layer.update()

def main():
    """メイン実行関数"""
    import sys
    import argparse
    
    # アドオンとして実行されている場合は、sys.argvを使用できないため
    if bpy.context.space_data is not None:
        ply_path = bpy.path.abspath("G:/study/SlicerDentalModelSeg/out_ply.ply")  # Blenderファイルと同じディレクトリのPLYファイル
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("ply_path", help="Path to the PLY file")
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
        ply_path = args.ply_path
    
    apply_segmentation_colors(ply_path)

if __name__ == "__main__":
    main() 