import vtk
import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from monai.transforms import ToTensor
import torch
import __CrownSegmentation.LinearSubdivisionFilter as lsf
from .post_process import Threshold
from typing import List, Tuple, Union, Optional
from pathlib import Path


# ReadSurfの前に追加
SUPPORTED_MESH_EXTENSIONS = {
    '.vtk': vtk.vtkPolyDataReader,
    '.vtp': vtk.vtkXMLPolyDataReader,
    '.stl': vtk.vtkSTLReader,
    '.off': 'OFFReader',  # カスタムリーダー
    '.obj': vtk.vtkOBJReader,
    '.gii': 'GIIReader'  # nibabelを使用
}

def get_supported_extensions() -> List[str]:
    """サポートされているメッシュファイルの拡張子を取得

    Returns:
        List[str]: サポートされているファイル拡張子のリスト
    """
    return list(SUPPORTED_MESH_EXTENSIONS.keys())

def Write(vtkdata: 'vtk.vtkPolyData', output_name: str, print_out: bool = True) -> None:
    """VTKデータをファイルに書き出す

    Args:
        vtkdata (vtk.vtkPolyData): 書き出すVTKデータ
        output_name (str): 出力ファイル名
        print_out (bool, optional): 進捗表示の有無. Defaults to True.
    """
    outfilename = output_name
    if print_out:
        print("Writing:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()






def ReadSurf(fileName: Union[str, Path]) -> 'vtk.vtkPolyData':
    """メッシュファイルを読み込む

    様々な形式のメッシュファイル（.vtk, .vtp, .stl, .off, .obj, .gii）を
    vtkPolyDataオブジェクトとして読み込みます。

    Args:
        fileName (Union[str, Path]): 読み込むファイルのパス

    Returns:
        vtk.vtkPolyData: 読み込まれたメッシュデータ

    Raises:
        ValueError: サポートされていないファイル形式の場合
    """
    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()

    if extension not in SUPPORTED_MESH_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}. "
                        f"Supported extensions are: {', '.join(get_supported_extensions())}")

    # .objファイルの特殊処理
    if extension == '.obj' and os.path.exists(fname + ".mtl"):
        obj_import = vtk.vtkOBJImporter()
        obj_import.SetFileName(fileName)
        obj_import.SetFileNameMTL(fname + ".mtl")
        textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
        if os.path.exists(textures_path):
            textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))
            obj_import.SetTexturePath(textures_path)
        else:
            textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))                
            obj_import.SetTexturePath(textures_path)
                    

        obj_import.Read()

        actors = obj_import.GetRenderer().GetActors()
        actors.InitTraversal()
        append = vtk.vtkAppendPolyData()

        for i in range(actors.GetNumberOfItems()):
            surfActor = actors.GetNextActor()
            append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
        
        append.Update()
        surf = append.GetOutput()
        
    else:
        # 標準的なVTKリーダーの処理
        reader_class = SUPPORTED_MESH_EXTENSIONS[extension]
        if isinstance(reader_class, str):
            if reader_class == 'OFFReader':
                from readers import OFFReader
                reader = OFFReader()
            elif reader_class == 'GIIReader':
                # .giiファイルの特殊処理
                import nibabel as nib
                surf = nib.load(fileName)
                coords = surf.agg_data('pointset')
                triangles = surf.agg_data('triangle')

                points = vtk.vtkPoints()

                for c in coords:
                    points.InsertNextPoint(c[0], c[1], c[2])

                cells = vtk.vtkCellArray()

                for t in triangles:
                    t_vtk = vtk.vtkTriangle()
                    t_vtk.GetPointIds().SetId(0, t[0])
                    t_vtk.GetPointIds().SetId(1, t[1])
                    t_vtk.GetPointIds().SetId(2, t[2])
                    cells.InsertNextCell(t_vtk)

                surf = vtk.vtkPolyData()
                surf.SetPoints(points)
                surf.SetPolys(cells)
                return surf
        else:
            reader = reader_class()
        
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()

    return surf


def ComputeNormals(surf: 'vtk.vtkPolyData') -> 'vtk.vtkPolyData':
    """メッシュの法線ベクトルを計算

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ

    Returns:
        vtk.vtkPolyData: 法線が計算されたメッシュ
    """
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf)
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    
    return normals.GetOutput()






def GetUnitSurf(
    surf: 'vtk.vtkPolyData', 
    mean_arr: Optional[np.ndarray] = None, 
    scale_factor: Optional[float] = None, 
    copy: bool = True
) -> 'vtk.vtkPolyData':
    """メッシュを単位空間に正規化

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ
        mean_arr (Optional[np.ndarray], optional): 中心座標. Defaults to None.
        scale_factor (Optional[float], optional): スケール係数. Defaults to None.
        copy (bool, optional): 入力を複製するかどうか. Defaults to True.

    Returns:
        vtk.vtkPolyData: 正規化されたメッシュ
    """
    unit_surf, surf_mean, surf_scale = ScaleSurf(surf, mean_arr, scale_factor, copy)
    return unit_surf




def ScaleSurf(
    surf: 'vtk.vtkPolyData', 
    mean_arr: Optional[np.ndarray] = None, 
    scale_factor: Optional[float] = None, 
    copy: bool = True
) -> Tuple['vtk.vtkPolyData', np.ndarray, float]:
    """メッシュのスケーリングと中心化を行う

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ
        mean_arr (Optional[np.ndarray], optional): 中心座標. Defaults to None.
        scale_factor (Optional[float], optional): スケール係数. Defaults to None.
        copy (bool, optional): 入力を複製するかどうか. Defaults to True.

    Returns:
        Tuple[vtk.vtkPolyData, np.ndarray, float]: 
            - スケーリングされたメッシュ
            - 中心座標
            - スケール係数
    """
    if(copy):
        surf_copy = vtk.vtkPolyData()
        surf_copy.DeepCopy(surf)
        surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = vtk_to_numpy(shapedatapoints.GetData())
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    shapedatapoints.SetData(numpy_to_vtk(shape_points))

    return surf, mean_arr, scale_factor


def RandomRotation(surf: 'vtk.vtkPolyData') -> Tuple['vtk.vtkPolyData', float, np.ndarray]:
    """メッシュにランダムな回転を適用

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ

    Returns:
        Tuple[vtk.vtkPolyData, float, np.ndarray]: 
            - 回転されたメッシュ
            - 回転角度
            - 回転軸ベクトル
    """
    rotationAngle = np.random.random()*360.0
    rotationVector = np.random.random(3)*2.0 - 1.0
    rotationVector = rotationVector/np.linalg.norm(rotationVector)
    return RotateSurf(surf, rotationAngle, rotationVector), rotationAngle, rotationVector


def RotateSurf(surf, rotationAngle, rotationVector):
    transform = GetTransform(rotationAngle, rotationVector)
    return RotateTransform(surf, transform)



def GetTransform(rotationAngle, rotationVector):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
    return transform

def RotateTransform(surf, transform):
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()



def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

def PolyDataToNumpy(surf: 'vtk.vtkPolyData') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VTKメッシュデータをNumPy配列に変換

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - 頂点座標配列
            - 面の接続情報配列
            - エッジの接続情報配列
    """
    edges_filter = vtk.vtkExtractEdges()
    edges_filter.SetInputData(surf)
    edges_filter.Update()

    verts = vtk_to_numpy(surf.GetPoints().GetData())
    faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
    edges = vtk_to_numpy(edges_filter.GetOutput().GetLines().GetData()).reshape(-1, 3)[:,1:]
    
    return verts, faces, edges


def PolyDataToTensors(
    surf: 'vtk.vtkPolyData', 
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VTKメッシュデータをPyTorchテンソルに変換

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ
        device (str, optional): 出力デバイス. Defaults to 'cpu'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - 頂点座標テンソル
            - 面の接続情報テンソル
            - エッジの接続情報テンソル
    """
    verts, faces, edges = PolyDataToNumpy(surf)
    
    verts = ToTensor(dtype=torch.float32, device=device)(verts)
    faces = ToTensor(dtype=torch.int32, device=device)(faces)
    edges = ToTensor(dtype=torch.int32, device=device)(edges)
    
    return verts, faces, edges

def CreateIcosahedron(radius: float, sl: int = 0) -> 'vtk.vtkPolyData':
    """正二十面体メッシュを生成

    Args:
        radius (float): 球の半径
        sl (int, optional): 細分化レベル. Defaults to 0.

    Returns:
        vtk.vtkPolyData: 生成された正二十面体メッシュ
    """
    icosahedronsource = vtk.vtkPlatonicSolidSource()
    icosahedronsource.SetSolidTypeToIcosahedron()
    icosahedronsource.Update()
    icosahedron = icosahedronsource.GetOutput()
    
    subdivfilter = lsf.LinearSubdivisionFilter()
    subdivfilter.SetInputData(icosahedron)
    subdivfilter.SetNumberOfSubdivisions(sl)
    subdivfilter.Update()

    icosahedron = subdivfilter.GetOutput()
    icosahedron = normalize_points(icosahedron, radius)

    return icosahedron



def normalize_points(poly: 'vtk.vtkPolyData', radius: float) -> 'vtk.vtkPolyData':
    """メッシュの頂点を指定された半径の球面上に正規化

    Args:
        poly (vtk.vtkPolyData): 入力メッシュ
        radius (float): 目標半径

    Returns:
        vtk.vtkPolyData: 正規化されたメッシュ
    """
    polypoints = poly.GetPoints()
    for pid in range(polypoints.GetNumberOfPoints()):
        spoint = polypoints.GetPoint(pid)
        spoint = np.array(spoint)
        norm = np.linalg.norm(spoint)
        spoint = spoint/norm * radius
        polypoints.SetPoint(pid, spoint)
    poly.SetPoints(polypoints)
    return poly


def ConvertFDI(surf: 'vtk.vtkPolyData', scal: str) -> 'vtk.vtkPolyData':
    """UniversalからFDI表記に歯のラベルを変換

    Args:
        surf (vtk.vtkPolyData): 入力メッシュ
        scal (str): スカラー配列の名前

    Returns:
        vtk.vtkPolyData: 変換後のメッシュ
    """
    LUT = np.array([0,18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                  38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48,0])
    # extract UniversalID array
    labels = vtk_to_numpy(surf.GetPointData().GetScalars(scal))
    
    # convert to their numbering system
    labels = LUT[labels]
    vtk_id = numpy_to_vtk(labels)
    vtk_id.SetName(scal)
    surf.GetPointData().AddArray(vtk_id)
    return surf

def SeparateLabels(surf, predicted_id: str, labels=None, output_path=None, print_out=True):
    """指定されたラベルをまとめて分離する

    Args:
        surf: vtkPolyDataオブジェクトまたはファイルパス
        predicted_id (str): 予測ラベルの配列名
        labels (list, optional): 分離したいラベルのリスト。Noneの場合は全ラベルを処理
        output_path (str, optional): 出力ファイルのパス。指定された場合はファイルを保存

    Returns:
        vtkPolyData: 指定されたラベルが分離されたsurfオブジェクト（output_pathが指定されていない場合）

    Raises:
        ValueError: 指定されたフィールド名のスカラーデータが見つからない場合
    """
    # 入力がパスの場合はファイルを読み込む
    if isinstance(surf, str):
        surf = ReadSurf(surf)

    # ラベル配列の取得
    surf_point_data = surf.GetPointData().GetScalars(predicted_id)
    if surf_point_data is None:
        available_arrays = [surf.GetPointData().GetArrayName(i) 
                          for i in range(surf.GetPointData().GetNumberOfArrays())]
        raise ValueError(
            f"フィールド名 '{predicted_id}' のスカラーデータが見つかりません。\n"
            f"利用可能なフィールド: {', '.join(available_arrays) if available_arrays else 'なし'}"
        )

    # 処理するラベルの決定
    if print_out:
        print(f"Found Labels: {np.unique(vtk_to_numpy(surf_point_data))}")
    if labels is None:
        labels = np.unique(vtk_to_numpy(surf_point_data))
    
    # 各ラベルに対してThresholdを適用
    separated_surf = vtk.vtkPolyData()
    separated_surf.DeepCopy(surf)
    result = None
    for label in labels:
        if print_out:
            print(f"Remove: {label}")
        if result is None:
            result = Threshold(separated_surf, predicted_id, label-0.5, label+0.5, invert=True)
        else:
            # 追加のラベルを除去
            result = Threshold(result, predicted_id, label-0.5, label+0.5, invert=True)
  
    if output_path:
        Write(result, output_path, print_out=False)
        return None
        
    return result

def WritePLY(vtkdata, output_name, predicted_id='PredictedID', print_out=True):
    """vtkPolyDataをPLYファイルとして保存する
    
    Args:
        vtkdata: vtkPolyDataオブジェクト
        output_name: 出力ファイル名
        predicted_id: セグメンテーションIDの配列名
        print_out: 進捗表示の有無
    """
    if print_out:
        print("Writing PLY:", output_name)
        
    # ポイントとポリゴンデータの取得
    points = vtk_to_numpy(vtkdata.GetPoints().GetData())
    polys = vtk_to_numpy(vtkdata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    
    # セグメンテーションIDの取得
    seg_ids = vtk_to_numpy(vtkdata.GetPointData().GetScalars(predicted_id))
    
    # PLYファイルの書き出し
    with open(output_name, 'w') as f:
        # ヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int label\n")  # セグメンテーションID
        f.write(f"element face {len(polys)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # 頂点データ
        for point, label in zip(points, seg_ids):
            f.write(f"{point[0]} {point[1]} {point[2]} {int(label)}\n")
            
        # ポリゴンデータ
        for poly in polys:
            f.write(f"3 {poly[0]} {poly[1]} {poly[2]}\n")

def WriteSegmentedSTL(vtkdata, output_base, predicted_id='PredictedID', print_out=True):
    """各セグメントを個別のSTLファイルとして保存
    
    Args:
        vtkdata: vtkPolyDataオブジェクト
        output_base: 出力ファイルのベース名（拡張子なし）
        predicted_id: セグメンテーションIDの配列名
        print_out: 進捗表示の有無
    """
    if print_out:
        print("Writing segmented STL files...")
    
    # セグメンテーションIDの取得
    seg_ids = np.unique(vtk_to_numpy(vtkdata.GetPointData().GetScalars(predicted_id)))
    
    # STL書き出し用のwriter
    stl_writer = vtk.vtkSTLWriter()
    
    # 各セグメントの処理
    for label in seg_ids:
        if label == 0:  # 歯肉は除外する場合
            continue
            
        # セグメントの抽出
        segment = Threshold(vtkdata, predicted_id, label-0.5, label+0.5)
        
        # STLファイルとして保存
        output_name = f"{output_base}_tooth_{int(label)}.stl"
        if print_out:
            print(f"Writing: {output_name}")
        
        stl_writer.SetFileName(output_name)
        stl_writer.SetInputData(segment)
        stl_writer.Write()
