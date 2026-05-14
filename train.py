"""
GradeGuard — 학습 실행 스크립트
사용법: python train.py
실제 GRAC/GCRB 데이터가 있으면 train_data.csv를 교체하고 실행
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from data_generator import make_sample
from model import train

DATA_PATH = Path(__file__).parent / "train_data.csv"

if __name__ == "__main__":
    if DATA_PATH.exists():
        print(f"기존 데이터 로드: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("데이터 없음 → 합성 데이터 3,000건 생성")
        df = make_sample(3000)
        df.to_csv(DATA_PATH, index=False)

    print(f"학습 데이터: {len(df)}건 (재조정 비율: {df['reclassified'].mean():.1%})")
    train(df)
