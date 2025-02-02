#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Union
import vtk
from tqdm import tqdm

from __CrownSegmentation.utils import ReadSurf, WritePLY

def convert_vtk_to_ply(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    predicted_id: str = 'PredictedID'
) -> None:
    """
    VTKファイルをPLYファイルに変換する

    Args:
        input_path: 入力vtkファイルのパス
        output_path: 出力plyファイルのパス
        predicted_id: セグメンテーションIDの配列名
    """
    # 入力ファイルの読み込み
    surf = ReadSurf(str(input_path))
    
    # PLYファイルとして保存
    WritePLY(surf, str(output_path), predicted_id=predicted_id)

def process_directory(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    predicted_id: str = 'PredictedID'
) -> None:
    """
    ディレクトリ内のすべてのVTKファイルを処理する

    Args:
        input_dir: 入力ディレクトリのパス
        output_dir: 出力ディレクトリのパス
        predicted_id: セグメンテーションIDの配列名
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # VTKファイルを検索
    vtk_files = list(input_dir.glob("**/*.vtk"))
    
    if not vtk_files:
        print(f"警告: {input_dir} 内にVTKファイルが見つかりません")
        return

    print(f"{len(vtk_files)}個のVTKファイルを処理します...")
    
    # 各ファイルを処理
    for vtk_file in tqdm(vtk_files, desc="Converting"):
        # 出力パスの作成（相対パス構造を維持）
        rel_path = vtk_file.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.ply')
        
        # 出力ディレクトリの作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            convert_vtk_to_ply(vtk_file, output_path, predicted_id)
        except Exception as e:
            print(f"エラー: {vtk_file} の処理中にエラーが発生しました: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='VTKファイルをPLYファイルに変換するツール')
    
    parser.add_argument(
        'input_path',
        type=str,
        help='入力VTKファイルまたはディレクトリのパス'
    )
    
    parser.add_argument(
        'output_path',
        type=str,
        help='出力PLYファイルまたはディレクトリのパス'
    )
    
    parser.add_argument(
        '--predicted-id',
        type=str,
        default='PredictedID',
        help='セグメンテーションIDの配列名（デフォルト: PredictedID）'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # 入力パスの存在確認
    if not input_path.exists():
        raise FileNotFoundError(f"入力パスが見つかりません: {input_path}")
    
    # ディレクトリかファイルかを判断して処理
    if input_path.is_dir():
        process_directory(input_path, output_path, args.predicted_id)
    else:
        # 出力ディレクトリの作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_vtk_to_ply(input_path, output_path, args.predicted_id)
        print(f"変換完了: {output_path}")

if __name__ == "__main__":
    main() 