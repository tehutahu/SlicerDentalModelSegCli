#!/usr/bin/env python-real

import argparse
import sys
from pathlib import Path
from typing import NamedTuple
import logging
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from __CrownSegmentation import (
    MonaiUNet, 
    TeethDataset, 
    UnitSurfTransform, 
    Write, 
    RemoveIslands,
    DilateLabel, 
    ErodeLabel, 
    Threshold, 
    ConvertFDI
)

# 歯のセグメンテーションに関する定数
NUM_CLASSES = 34  # クラス数（33本の歯 + 歯肉）
GUM_LABEL_UNIVERSAL = 33  # Universal表記での歯肉のラベル
GUM_LABEL_FDI = 0  # FDI表記での歯肉のラベル

# 後処理のパラメータ
ISLAND_REMOVAL_THRESHOLD_MAIN = 500  # メインの島除去の閾値
ISLAND_REMOVAL_THRESHOLD_SECONDARY = 200  # 二次的な島除去の閾値
CLOSING_ITERATIONS = 2  # クロージング処理の反復回数

# データローダーの設定
BATCH_SIZE = 1
NUM_WORKERS = 4

class SegmentationArgs(NamedTuple):
    input: str
    output: str
    subdivision_level: int
    resolution: int
    model: str
    predictedId: str
    sepOutputs: int
    chooseFDI: int
    logPath: str

def setup_logger(log_path: Path):
    """ロガーの設定"""
    logger = logging.getLogger('CrownSegmentation')
    logger.setLevel(logging.INFO)
    
    # ファイルハンドラの設定
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    
    # コンソールハンドラの設定
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # フォーマットの設定
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def setup_model(args: SegmentationArgs):
    """モデルのセットアップを行う"""
    class_weights = None

    model = MonaiUNet(
        out_channels=NUM_CLASSES, 
        class_weights=class_weights, 
        image_size=args.resolution, 
        subdivision_level=args.subdivision_level
    )
    model.model.module.load_state_dict(torch.load(args.model))
    return model.to(torch.device('cuda'))

def process_predictions(surf, predictions: torch.Tensor, args: SegmentationArgs):
    """予測結果の後処理を実行"""
    predictions = numpy_to_vtk(predictions.cpu().numpy())
    predictions.SetName(args.predictedId)
    surf.GetPointData().AddArray(predictions)

    # Remove islands
    RemoveIslands(surf, predictions, GUM_LABEL_UNIVERSAL, ISLAND_REMOVAL_THRESHOLD_MAIN, ignore_neg1=True)
    for label in tqdm(range(NUM_CLASSES), desc='Remove island'):
        RemoveIslands(surf, predictions, label, ISLAND_REMOVAL_THRESHOLD_SECONDARY, ignore_neg1=True)

    # Closing operation
    for label in tqdm(range(1, NUM_CLASSES), desc='Closing operation'):
        DilateLabel(surf, predictions, label, iterations=CLOSING_ITERATIONS, dilateOverTarget=False, target=None)
        ErodeLabel(surf, predictions, label, iterations=CLOSING_ITERATIONS, target=None)

    return surf

def save_outputs(surf, ds_name: str, args: SegmentationArgs):
    """結果の保存処理を実行"""
    out_root = Path(args.output)
    out_fname = Path(ds_name).with_suffix('.vtk')
    out_dir = out_root / out_fname.name.split('_')[0] # 001_LowerJawScan.stl -> 001
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / out_fname

    if args.chooseFDI:
        surf = ConvertFDI(surf, args.predictedId)
        gum_label = GUM_LABEL_FDI
    else:
        gum_label = GUM_LABEL_UNIVERSAL

    if args.sepOutputs:
        out_basename = str(output_path.with_suffix(''))  # 文字列として扱う
        
        surf_point_data = surf.GetPointData().GetScalars(args.predictedId)
        labels = vtk_to_numpy(surf_point_data)
        
        # 各ラベルの保存
        for label in tqdm(np.unique(labels), desc='Isolating labels'):
            thresh_label = Threshold(surf, args.predictedId, label-0.5, label+0.5, invert=True) # 1本抜け
            suffix = '_gum.vtk' if label == gum_label else f'_id_{label}.vtk'
            Write(thresh_label, out_basename + suffix, print_out=False)  # 単純な文字列連結を使用
        
        # 歯全体の保存
        no_gum = Threshold(surf, args.predictedId, gum_label-0.5, gum_label+0.5, invert=True)
        Write(no_gum, out_basename + '_all_teeth.vtk', print_out=False)  # 単純な文字列連結を使用

    Write(surf, str(output_path), print_out=False)

def main(args: SegmentationArgs):
    start_time = datetime.now()
    logger = setup_logger(Path(args.logPath))
    
    logger.info("Crown segmentation started")
    logger.info(f"Input file: {Path(args.input).absolute()}")
    logger.info(f"Output directory: {Path(args.output).absolute()}")
    logger.info(f"Model path: {Path(args.model).absolute()}")
    logger.info(f"Settings - Subdivision level: {args.subdivision_level}")
    logger.info(f"Settings - Resolution: {args.resolution}")
    logger.info(f"Settings - Predicted ID: {args.predictedId}")
    logger.info(f"Settings - Separate outputs: {args.sepOutputs}")
    logger.info(f"Settings - FDI notation: {args.chooseFDI}")

    try:
        model = setup_model(args)
        logger.info("Model loaded successfully")
        
        ds = TeethDataset(args.input, transform=UnitSurfTransform())
        dataloader = DataLoader(
            ds, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            persistent_workers=True, 
            pin_memory=True
        )
        logger.info(f"Dataset loaded with {len(ds)} files")
        
        model.eval()
        softmax = torch.nn.Softmax(dim=2)

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                current_file = ds.getName(idx)
                logger.info(f"Processing file {idx + 1}/{len(ds)}: {current_file}")

                V, F, CN = [b.cuda(non_blocking=True) for b in batch]
                CN = CN.to(torch.float32)

                x, X, PF = model((V, F, CN))
                x = softmax(x*(PF>=0))

                # 予測処理
                P_faces = torch.zeros(NUM_CLASSES, F.shape[1], device=V.device)
                V_labels_prediction = torch.zeros(V.shape[1], device=V.device, dtype=torch.int64)

                PF = PF.squeeze()
                x = x.squeeze()

                for pf, pred in zip(PF, x):
                    P_faces[:, pf] += pred

                P_faces = torch.argmax(P_faces, dim=0)
                V_labels_prediction[F[0,:,0]] = P_faces

                # 後処理と保存
                surf = ds.getSurf(idx)
                surf = process_predictions(surf, V_labels_prediction, args)
                save_outputs(surf, current_file, args)
                logger.info(f"Completed processing {current_file}")

        processing_time = datetime.now() - start_time
        logger.info(f"All processing completed. Total time: {processing_time}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crown Segmentation CLI tool for dental mesh processing'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input mesh file path (.vtk format)'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output directory path for segmented results'
    )
    parser.add_argument(
        'subdivision_level',
        type=int,
        default=4,
        help='Subdivision level for mesh processing (default: 4)'
    )
    parser.add_argument(
        'resolution',
        type=int,
        default=320,
        help='Resolution for image processing (default: 320)'
    )
    parser.add_argument(
        'model',
        type=str,
        help='Path to the trained model weights file'
    )
    parser.add_argument(
        'predictedId',
        type=str,
        default='PredictedID',
        help='Name of the predicted label array (default: PredictedID)'
    )
    parser.add_argument(
        'sepOutputs',
        type=int,
        default=0,
        choices=[0, 1],
        help='Whether to separate output files by tooth (0: No, 1: Yes, default: 0)'
    )
    parser.add_argument(
        'chooseFDI',
        type=int,
        default=1,
        choices=[0, 1],
        help='Whether to use FDI notation (0: Universal, 1: FDI, default: 1)'
    )
    parser.add_argument(
        'logPath',
        type=str,
        default='crown_segmentation.log',
        help='Path to the log file (default: crown_segmentation.log)'
    )

    args = SegmentationArgs(**vars(parser.parse_args()))
    main(args)

