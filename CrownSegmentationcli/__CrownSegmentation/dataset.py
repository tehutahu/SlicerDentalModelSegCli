import os
import glob
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from vtk.util.numpy_support import vtk_to_numpy

from __CrownSegmentation.utils import ReadSurf, ComputeNormals, GetColorArray, GetUnitSurf, RandomRotation, get_supported_extensions



class TeethDataset(Dataset):
    """歯科用メッシュデータのデータセットクラス

    単一のメッシュファイル、ディレクトリ内の複数のメッシュファイル、
    またはCSVファイルによる複数ファイルの指定に対応します。

    CSVファイルフォーマット:
    ```
    input_path,patient_id,jaw_type
    /path/to/data/001_upper.stl,001,upper
    /path/to/data/001_lower.stl,001,lower
    ```

    必須列:
    - input_path: メッシュファイルへの絶対パスまたは相対パス
    
    オプション列:
    - patient_id: 患者ID（出力ディレクトリ構造の作成に使用）
    - jaw_type: 顎の種類（'upper' または 'lower'）

    Args:
        path: 入力パス。以下のいずれかを指定:
            - 単一のメッシュファイルパス (.vtk, .stl, etc.)
            - メッシュファイルを含むディレクトリパス
            - 入力ファイル情報を含むCSVファイルパス
        transform: メッシュに適用する前処理変換（オプション）

    Raises:
        Exception: 入力パスが不正な場合
    """

    def __init__(self, path, transform=None):
        self.df = self.setup_df(path)
        self.transform = transform

    def setup_df(self, path):
        """入力パスに基づいてデータフレームをセットアップ

        Args:
            path: 入力パス（ファイル、ディレクトリ、またはCSV）

        Returns:
            list または pandas.DataFrame: 処理対象ファイルのリストまたはデータフレーム

        Raises:
            Exception: パスが不正な場合
        """
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.suffix.lower() == '.csv':
                # CSVからデータを読み込む
                df = pd.read_csv(path, dtype={'patient_id': str})
                if 'input_path' not in df.columns:
                    raise Exception("CSV must contain 'input_path' column")
                return df
            elif path.is_dir():
                # ディレクトリから対応する全ての拡張子のファイルを読み込む
                mesh_files = []
                for ext in get_supported_extensions():
                    mesh_files.extend(path.glob(f"*{ext}"))
                    mesh_files.extend(path.glob(f"*{ext.upper()}"))  # 大文字の拡張子にも対応
                
                if not mesh_files:
                    raise Exception(f'No supported mesh files found in directory. '
                                  f'Supported formats: {", ".join(get_supported_extensions())}')
                
                return [str(f) for f in sorted(mesh_files)]  # ソートして予測可能な順序に
            elif path.is_file():
                # 単一ファイル
                return [str(path)]
            else:
                raise Exception('Incorrect input.')
        else:
            raise Exception('Path must be string or Path object')

    def __len__(self):
        """データセットの長さを返す"""
        return len(self.df)

    def __getitem__(self, idx):
        """指定されたインデックスのデータを取得

        Args:
            idx: データのインデックス

        Returns:
            tuple: (頂点座標, 面情報, 法線カラー情報)
        """
        surf_path = self.df[idx] if isinstance(self.df, list) else self.df.iloc[idx]['input_path']
        surf = ReadSurf(surf_path)

        if self.transform:
            surf = self.transform(surf)

        surf = ComputeNormals(surf)
        color_normals = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))).to(torch.float32)/255.0
        verts = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        faces = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)

        return verts, faces, color_normals

    def getSurf(self, idx):
        """指定されたインデックスの生のサーフェスデータを取得"""
        surf_path = self.df[idx] if isinstance(self.df, list) else self.df.iloc[idx]['input_path']
        return ReadSurf(surf_path)
    
    def getName(self, idx):
        """指定されたインデックスのファイル名を取得"""
        surf_path = self.df[idx] if isinstance(self.df, list) else self.df.iloc[idx]['input_path']
        return Path(surf_path).name

    def getPatientInfo(self, idx):
        """CSVから読み込んだ場合の患者情報を取得

        Args:
            idx: データのインデックス

        Returns:
            dict: 患者情報を含む辞書。CSVからの読み込みでない場合はNone
            {
                'patient_id': 患者ID,
                'jaw_type': 顎の種類（'upper' または 'lower'）
            }
        """
        if isinstance(self.df, pd.DataFrame):
            row = self.df.iloc[idx]
            return {
                'patient_id': row.get('patient_id', ''),
                'jaw_type': row.get('jaw_type', '')
            }
        return None



class UnitSurfTransform:
    """メッシュデータの正規化と回転を行う変換クラス

    メッシュを単位空間に正規化し、オプションでランダムな回転を適用します。
    データ拡張やモデルの学習前の前処理として使用されます。

    Args:
        random_rotation (bool): ランダムな回転を適用するかどうか。
            True: ランダムな回転を適用
            False: 回転を適用しない（デフォルト）

    Example:
        ```python
        transform = UnitSurfTransform(random_rotation=True)
        normalized_mesh = transform(input_mesh)
        ```
    """

    def __init__(self, random_rotation=False):
        self.random_rotation = random_rotation

    def __call__(self, surf):
        """変換を実行する

        Args:
            surf (vtkPolyData): 入力メッシュデータ

        Returns:
            vtkPolyData: 正規化（および必要に応じて回転）されたメッシュデータ
        """
        surf = GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = RandomRotation(surf)
        return surf


def create_dataset_csv(root_dir: Union[str, Path], output_csv: Union[str, Path]) -> None:
    """歯科データディレクトリからデータセット用のCSVファイルを作成

    指定されたディレクトリ構造から*_LowerJawScan.stlと*_UpperJawScan.stlファイルを探し、
    データセット用のCSVファイルを生成します。

    ディレクトリ構造の例:
    root_dir/
    ├── teeth_data1/
    │   ├── 001/
    │   │   └── 001_ios/
    │   │       ├── 001_LowerJawScan.stl
    │   │       └── 001_UpperJawScan.stl
    │   ├── 002/
    │   │   └── 002_ios/
    │   │       ├── 002_LowerJawScan.stl
    │   │       └── 002_UpperJawScan.stl
    ...

    Args:
        root_dir: 歯科データのルートディレクトリ
        output_csv: 出力するCSVファイルのパス

    生成されるCSVの形式:
    ```
    input_path,patient_id,jaw_type
    /path/to/001_ios/001_LowerJawScan.stl,001,lower
    /path/to/001_ios/001_UpperJawScan.stl,001,upper
    ```
    """
    root_dir = Path(root_dir)
    output_csv = Path(output_csv)

    # 結果を格納するリスト
    rows = []

    # ルートディレクトリ以下を再帰的に探索
    for stl_file in root_dir.glob("**/*JawScan.stl"):
        # ファイル名から情報を抽出
        file_name = stl_file.name
        if not (file_name.endswith('LowerJawScan.stl') or file_name.endswith('UpperJawScan.stl')):
            continue

        # patient_idを抽出（例: 001_LowerJawScan.stl -> 001）
        patient_id = file_name.split('_')[0]
        
        # jaw_typeを決定
        jaw_type = 'lower' if 'Lower' in file_name else 'upper'

        # 結果を追加
        rows.append({
            'input_path': str(stl_file.absolute()),
            'patient_id': patient_id,
            'jaw_type': jaw_type
        })

    # 結果をDataFrameに変換
    df = pd.DataFrame(rows)
    
    # patient_idでソート
    df = df.sort_values(['patient_id', 'jaw_type'])

    # CSVとして保存
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Created dataset CSV at: {output_csv}")
    print(f"Found {len(df)} files from {len(df['patient_id'].unique())} patients")
