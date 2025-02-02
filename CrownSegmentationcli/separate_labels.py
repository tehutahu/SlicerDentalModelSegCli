#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import List

from __CrownSegmentation.utils import SeparateLabels


def main(input_path: str, output_path: str, labels: List[int], field_name: str = "PredictedID"):
    """
    指定されたラベルに基づいてメッシュを分離する

    Args:
        input_path: 入力vtkファイルのパス
        output_path: 出力vtkファイルのパス
        labels: 分離したいラベルのリスト
        field_name: ラベル情報が格納されているフィールド名
    """
    SeparateLabels(input_path, field_name, labels, output_path, print_out=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='メッシュから指定されたラベルを分離するツール')
    
    parser.add_argument(
        'input_path',
        type=str,
        help='入力vtkファイルのパス'
    )
    
    parser.add_argument(
        'output_path',
        type=str,
        help='出力vtkファイルのパス'
    )
    
    parser.add_argument(
        'labels',
        type=int,
        nargs='+',
        help='分離したいラベルのリスト（スペース区切りで複数指定可能）'
    )
    
    parser.add_argument(
        '--field-name',
        type=str,
        default='PredictedID',
        help='ラベル情報が格納されているフィールド名（デフォルト: predict_id）'
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not Path(args.input_path).exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.input_path}")
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args.input_path, args.output_path, args.labels, args.field_name) 