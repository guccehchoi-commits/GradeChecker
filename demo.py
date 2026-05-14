"""
GradeGuard — 데모 및 B2C 신뢰도 조회
사용법: python demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model import load_model, predict_one

# ── B2C: 게임명 → 신뢰도 조회 (규칙 기반 + 모델) ───────────────────────────
# 실제 서비스에서는 GCRB 등록 DB와 연계
KNOWN_GAMES = {
    "배틀그라운드": {"genre":"슈팅","platform":"PC (Steam)","org_type":"대형사",
                    "grade":"청소년이용불가","year":2017,"dev_history":"없음",
                    "description":"총기 폭력 혈액"},
    "마인크래프트": {"genre":"어드벤처","platform":"구글플레이","org_type":"대형사",
                    "grade":"전체이용가","year":2011,"dev_history":"없음",
                    "description":"블록 건설 교육"},
    "브롤스타즈":   {"genre":"액션","platform":"애플 앱스토어","org_type":"대형사",
                    "grade":"12세이용가","year":2018,"dev_history":"없음",
                    "description":"캐릭터 전투 경쟁"},
    "리니지":       {"genre":"RPG","platform":"PC (자체)","org_type":"대형사",
                    "grade":"15세이용가","year":1998,"dev_history":"2~3회",
                    "description":"폭력 확률형 사행"},
}


def b2c_lookup(game_name: str, pipeline) -> dict:
    """게임명으로 신뢰도 조회 (등록 DB 연계 전 시뮬레이션)."""
    # 등록된 게임이면 실제 메타데이터 사용
    meta = KNOWN_GAMES.get(game_name)
    if meta is None:
        # 미등록 게임은 중간 위험도로 기본값 설정
        meta = {"genre":"기타","platform":"구글플레이","org_type":"중소",
                "grade":"15세이용가","year":2023,"dev_history":"없음",
                "description":""}

    result = predict_one(pipeline, meta)
    risk = result["risk_score"]

    # 신뢰도 = 위험도의 반전
    trust_score = 100 - risk
    if trust_score >= 70:
        trust_level = "상"
    elif trust_score >= 45:
        trust_level = "중"
    else:
        trust_level = "하"

    concerns = [f["factor"] + ": " + f["description"]
                for f in result["top_factors"] if f["impact"] > 30]

    return {
        "game_name": game_name,
        "trust_level": trust_level,
        "trust_score": trust_score,
        "current_grade": meta["grade"],
        "concerns": concerns[:3],
        "review_flag": trust_level == "하",
        "summary": (
            f"현재 '{meta['grade']}' 등급에 대한 AI 신뢰도는 {trust_score}점({trust_level})입니다. "
            + ("등급 정보를 추가로 확인하시기 바랍니다." if trust_level == "하"
               else "일반적으로 적절한 등급으로 판단됩니다.")
        ),
    }


def run_b2b_demo(pipeline):
    print("\n" + "═"*55)
    print("  B2B 데모 — 개발사 위험도 예측")
    print("═"*55)

    cases = [
        {
            "name": "고위험 케이스",
            "input": {
                "genre": "액션", "platform": "PC (자체)", "org_type": "개인",
                "grade": "전체이용가", "year": 2020, "dev_history": "2~3회",
                "description": "폭력 혈액 선정 노출",
            },
        },
        {
            "name": "중위험 케이스",
            "input": {
                "genre": "RPG", "platform": "구글플레이", "org_type": "중소",
                "grade": "15세이용가", "year": 2023, "dev_history": "1회",
                "description": "캐릭터 성장 전투 확률형",
            },
        },
        {
            "name": "저위험 케이스",
            "input": {
                "genre": "퍼즐", "platform": "애플 앱스토어", "org_type": "대형사",
                "grade": "전체이용가", "year": 2024, "dev_history": "없음",
                "description": "블록 교육 가족 퍼즐",
            },
        },
    ]

    for c in cases:
        r = predict_one(pipeline, c["input"])
        bar = "█" * (r["risk_score"] // 5) + "░" * (20 - r["risk_score"] // 5)
        print(f"\n[{c['name']}]")
        print(f"  입력: {c['input']['genre']} / {c['input']['platform']} / {c['input']['org_type']} / {c['input']['grade']}")
        print(f"  위험도: {r['risk_score']:3d}점 ({r['risk_level']})  {bar}")
        print(f"  주요요인: " + " | ".join(f['factor'] for f in r['top_factors']))
        print(f"  권고: {r['recommendations'][0]}")


def run_b2c_demo(pipeline):
    print("\n" + "═"*55)
    print("  B2C 데모 — 이용자 등급 신뢰도 조회")
    print("═"*55)

    for game in ["배틀그라운드", "마인크래프트", "리니지", "알 수 없는 게임X"]:
        r = b2c_lookup(game, pipeline)
        flag = " ⚠ 등급 검토 권고" if r["review_flag"] else ""
        print(f"\n  {r['game_name']:15s} | 신뢰도 {r['trust_level']} ({r['trust_score']:3d}점) | {r['current_grade']}{flag}")
        print(f"  → {r['summary']}")


if __name__ == "__main__":
    print("GradeGuard 프로토타입 — 모델 로드 중...")
    try:
        pipeline = load_model()
    except FileNotFoundError:
        print("모델 없음. 학습 먼저 실행합니다...\n")
        from train import *
        import pandas as pd
        from data_generator import make_sample
        from model import train as _train
        df = make_sample(3000)
        pipeline = _train(df)

    run_b2b_demo(pipeline)
    run_b2c_demo(pipeline)
    print("\n" + "═"*55)
