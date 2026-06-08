"""
grac_data.py — GRAC API에서 실제 데이터 수집 후 모델 재학습
코랩에서 실행: python grac_data.py
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
from pathlib import Path

API_URL = "https://www.grac.or.kr/WebService/GameSearchSvc.asmx/game"

# GRAC 장르 → 우리 모델 장르 매핑
GENRE_MAP = {
    "격투게임": "격투", "액션": "액션", "FPS/TPS": "슈팅", "비행슈팅": "슈팅",
    "롤플레잉": "RPG", "MMORPG": "RPG", "퍼즐": "퍼즐", "캐주얼": "퍼즐",
    "전략시뮬레이션": "전략", "시뮬레이션": "시뮬레이션", "스포츠": "스포츠",
    "어드벤처": "어드벤처", "교육용": "기타", "레이싱": "기타",
    "보드게임": "기타", "보드게임(베팅성)": "기타", "크레인": "기타", "기타": "기타",
}

# GRAC 플랫폼 → 우리 모델 플랫폼 매핑
PLATFORM_MAP = {
    "PC/온라인 게임": "PC (Steam)",
    "모바일 게임": "구글플레이",
    "비디오 게임": "콘솔",
    "아케이드 게임": "기타",
}

# 등급 매핑
GRADE_MAP = {
    "전체이용가": "전체이용가",
    "12세이용가": "12세이용가",
    "15세이용가": "15세이용가",
    "청소년이용불가": "청소년이용불가",
    "등급거부": "청소년이용불가",
    "등급취소": "청소년이용불가",
}


def fetch_page(pageno: int, display: int = 1000) -> list:
    """한 페이지 데이터 수집"""
    params = {"op": "game", "display": display, "pageno": pageno}
    try:
        res = requests.get(API_URL, params=params, timeout=30)
        res.encoding = "utf-8"
        root = ET.fromstring(res.text)

        if root.tag == "error":
            print(f"API 오류: {root.find('message').text}")
            return []

        items = []
        for item in root.findall("item"):
            def get(tag):
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            items.append({
                "gametitle":    get("gametitle"),
                "genre_raw":    get("genre"),
                "platform_raw": get("platform"),
                "entname":      get("entname"),
                "givenrate":    get("givenrate"),
                "rateddate":    get("rateddate"),
                "descriptors":  get("descriptors"),
                "cancelstatus": get("cancelstatus"),
                "orgname":      get("orgname"),
            })
        return items

    except Exception as e:
        print(f"페이지 {pageno} 수집 실패: {e}")
        return []


def fetch_all(max_pages: int = 20) -> pd.DataFrame:
    """전체 데이터 수집 (max_pages * 1000건)"""
    print("GRAC API 데이터 수집 시작...")

    # 총 건수 확인
    res = requests.get(API_URL, params={"op":"game","display":1,"pageno":1}, timeout=30)
    root = ET.fromstring(res.text)
    total = int(root.findtext("tcount") or 0)
    total_pages = min((total // 1000) + 1, max_pages)
    print(f"전체 {total:,}건 / 수집 예정: {total_pages}페이지 ({total_pages*1000:,}건)")

    all_items = []
    for page in range(1, total_pages + 1):
        items = fetch_page(page)
        all_items.extend(items)
        print(f"  페이지 {page}/{total_pages} — 누적 {len(all_items):,}건")
        time.sleep(0.5)  # API 부하 방지

    return pd.DataFrame(all_items)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """GRAC 원본 데이터 → 모델 학습용 포맷 변환"""
    out = pd.DataFrame()

    # 장르 변환
    out["genre"] = df["genre_raw"].map(GENRE_MAP).fillna("기타")

    # 플랫폼 변환
    out["platform"] = df["platform_raw"].map(PLATFORM_MAP).fillna("기타")

    # 기관유형 추정 (entname 기반 — 실제 데이터 보고 조정 필요)
    def infer_org(name):
        big = ["넥슨","엔씨소프트","넷마블","카카오","크래프톤","스마일게이트",
               "구글","애플","펄어비스","위메이드"]
        if any(b in str(name) for b in big):
            return "대형사"
        elif len(str(name)) <= 2 or str(name) in ("", "개인"):
            return "개인"
        return "중소"
    out["org_type"] = df["entname"].apply(infer_org)

    # 등급 변환
    out["grade"] = df["givenrate"].map(GRADE_MAP).fillna("15세이용가")

    # 출시 연도
    out["year"] = pd.to_datetime(df["rateddate"], errors="coerce").dt.year.fillna(2020).astype(int)

    # 재조정 이력 (동일 업체 기준 집계)
    history_count = df.groupby("entname")["cancelstatus"].apply(
        lambda x: (x.str.lower() == "true").sum()
    ).reset_index()
    history_count.columns = ["entname", "cancel_count"]
    df2 = df.copy()
    df2 = df2.merge(history_count, on="entname", how="left")
    df2["cancel_count"] = df2["cancel_count"].fillna(0)

    def count_to_history(n):
        if n == 0: return "없음"
        elif n == 1: return "1회"
        elif n <= 3: return "2~3회"
        return "4회 이상"
    out["dev_history"] = df2["cancel_count"].apply(count_to_history)

    # 콘텐츠 기술서 (descriptors → 텍스트)
    out["description"] = df["descriptors"].fillna("")

    # Y 레이블: 등급취소 = 재조정
    out["reclassified"] = (df["cancelstatus"].str.lower() == "true").astype(int)

    # 원본 게임명 보존
    out["gametitle"] = df["gametitle"]

    return out.dropna(subset=["genre", "grade"]).reset_index(drop=True)


def save_and_train(df_clean: pd.DataFrame):
    """전처리 데이터 저장 + 모델 재학습"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ml_model import train

    save_path = Path(__file__).parent / "train_data.csv"
    df_clean.to_csv(save_path, index=False)
    print(f"\n전처리 완료 — {len(df_clean):,}건 저장 ({save_path})")
    print(f"재조정 비율: {df_clean['reclassified'].mean():.1%}")

    print("\n모델 재학습 시작...")
    train(df_clean)


if __name__ == "__main__":
    # 1. 데이터 수집
    df_raw = fetch_all(max_pages=20)

    if df_raw.empty:
        print("데이터 수집 실패")
    else:
        # 2. 전처리
        df_clean = preprocess(df_raw)
        print(df_clean.head(3).to_string())

        # 3. 저장 + 재학습
        save_and_train(df_clean)
