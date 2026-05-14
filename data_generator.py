"""
GradeGuard — 합성 학습 데이터 생성기
실제 GRAC/GCRB 데이터 수집 전까지 도메인 지식 기반으로 사용
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

RNG = np.random.default_rng(42)

# ── 도메인 지식 기반 위험도 가중치 ──────────────────────────────────────────
GENRE_RISK = {
    "액션": 0.55, "슈팅": 0.60, "격투": 0.65,
    "RPG": 0.45, "어드벤처": 0.40, "전략": 0.30,
    "스포츠": 0.25, "시뮬레이션": 0.25, "퍼즐": 0.15, "기타": 0.35,
}

PLATFORM_RISK = {
    "구글플레이": 0.40, "애플 앱스토어": 0.35,
    "PC (Steam)": 0.45, "PC (자체)": 0.60, "콘솔": 0.30, "기타": 0.55,
}

ORG_RISK = {"대형사": 0.20, "중소": 0.45, "개인": 0.70}

GRADE_RISK = {
    "전체이용가": 0.50,   # 전체로 신청했는데 실제론 더 높을 수 있음
    "12세이용가": 0.35,
    "15세이용가": 0.30,
    "청소년이용불가": 0.15,  # 이미 최고 등급
}

HISTORY_RISK = {"없음": 0.10, "1회": 0.35, "2~3회": 0.65, "4회 이상": 0.85}

# 콘텐츠 기술서 키워드 → 위험도 부스터
CONTENT_KEYWORDS = {
    "폭력": 0.20, "살인": 0.25, "혈액": 0.20, "gore": 0.25,
    "선정": 0.20, "노출": 0.18, "성적": 0.22,
    "도박": 0.25, "베팅": 0.25, "사행": 0.25, "확률형": 0.15,
    "욕설": 0.10, "혐오": 0.12,
    "캐릭터 성장": -0.05, "퍼즐": -0.08, "교육": -0.12, "가족": -0.10,
}


def keyword_score(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    score = 0.0
    text_lower = text.lower()
    for kw, val in CONTENT_KEYWORDS.items():
        if kw in text_lower:
            score += val
    return np.clip(score, -0.30, 0.60)


def make_sample(n: int = 3000) -> pd.DataFrame:
    genres    = RNG.choice(list(GENRE_RISK), n)
    platforms = RNG.choice(list(PLATFORM_RISK), n)
    org_types = RNG.choice(list(ORG_RISK), n, p=[0.15, 0.50, 0.35])
    grades    = RNG.choice(list(GRADE_RISK), n, p=[0.30, 0.25, 0.30, 0.15])
    years     = RNG.integers(2018, 2027, n)
    histories = RNG.choice(list(HISTORY_RISK), n, p=[0.60, 0.20, 0.12, 0.08])

    # 콘텐츠 기술서 키워드 시뮬레이션
    kw_pool = list(CONTENT_KEYWORDS.keys())
    descriptions = []
    for _ in range(n):
        k = RNG.integers(0, 5)
        words = RNG.choice(kw_pool, k, replace=False).tolist() if k > 0 else []
        descriptions.append(" ".join(words))

    rows = []
    for i in range(n):
        base = (
            GENRE_RISK[genres[i]] * 0.25
            + PLATFORM_RISK[platforms[i]] * 0.20
            + ORG_RISK[org_types[i]] * 0.20
            + GRADE_RISK[grades[i]] * 0.15
            + HISTORY_RISK[histories[i]] * 0.20
        )
        kw = keyword_score(descriptions[i]) * 0.15
        year_penalty = max(0, (2022 - years[i]) * 0.01)  # 오래된 게임 소폭 페널티

        prob = np.clip(base + kw + year_penalty + RNG.normal(0, 0.06), 0.02, 0.98)
        label = int(prob > 0.50)   # 1 = 재조정 발생

        rows.append({
            "genre": genres[i],
            "platform": platforms[i],
            "org_type": org_types[i],
            "grade": grades[i],
            "year": years[i],
            "dev_history": histories[i],
            "description": descriptions[i],
            "reclassify_prob": round(prob, 4),
            "reclassified": label,
        })

    df = pd.DataFrame(rows)
    return shuffle(df, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = make_sample(3000)
    df.to_csv("/home/claude/gradeguard/train_data.csv", index=False)
    print(f"데이터 생성 완료: {len(df)}건")
    print(f"재조정 비율: {df['reclassified'].mean():.1%}")
    print(df.head(3).to_string())
