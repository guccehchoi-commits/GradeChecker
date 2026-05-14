"""
GradeGuard — ML 예측 모델
전처리 파이프라인 + RandomForest (표본 1000~5000건 구간 전략)
실제 데이터로 교체 시 model.py만 재학습하면 됨
"""

import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)

MODEL_PATH = Path(__file__).parent / "gradeguard_model.pkl"

CATEGORICAL_COLS = ["genre", "platform", "org_type", "grade", "dev_history"]
NUMERIC_COLS     = ["year"]
TEXT_COL         = "description"

GRADE_ORDER = [["전체이용가", "12세이용가", "15세이용가", "청소년이용불가"]]
HISTORY_ORDER = [["없음", "1회", "2~3회", "4회 이상"]]
ORG_ORDER = [["대형사", "중소", "개인"]]


def build_preprocessor():
    cat_enc = OrdinalEncoder(
        categories=[
            sorted(["액션","슈팅","격투","RPG","어드벤처","전략","스포츠","시뮬레이션","퍼즐","기타"]),
            sorted(["구글플레이","애플 앱스토어","PC (Steam)","PC (자체)","콘솔","기타"]),
            ["대형사", "중소", "개인"],
            ["전체이용가", "12세이용가", "15세이용가", "청소년이용불가"],
            ["없음", "1회", "2~3회", "4회 이상"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_enc, CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
            ("txt", TfidfVectorizer(max_features=60, analyzer="char_wb", ngram_range=(2,4)), TEXT_COL),
        ]
    )
    return preprocessor


def train(df: pd.DataFrame, n_samples: int | None = None):
    """학습 실행. 표본 볼륨에 따라 모델 자동 선택."""
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    n = len(df) if n_samples is None else n_samples

    if n < 1000:
        print(f"표본 {n}건 → 의사결정트리 (해석 우선 모드)")
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
    elif n < 5000:
        print(f"표본 {n}건 → RandomForest")
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    else:
        print(f"표본 {n}건 → GradientBoosting (앙상블)")
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, random_state=42
        )

    preprocessor = build_preprocessor()
    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + [TEXT_COL]]
    y = df["reclassified"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]

    print("\n─── 평가 결과 ───────────────────────────────")
    print(classification_report(y_te, y_pred, target_names=["정상", "재조정"]))
    print(f"AUC-ROC : {roc_auc_score(y_te, y_prob):.4f}")
    print(f"F1 (재조정) : {f1_score(y_te, y_pred):.4f}")
    cm = confusion_matrix(y_te, y_pred)
    print(f"혼동행렬:\n  정상→정상 {cm[0,0]}  정상→재조정 {cm[0,1]}\n"
          f"  재조정→정상 {cm[1,0]}  재조정→재조정 {cm[1,1]}")

    # 교차검증
    cv = cross_val_score(pipeline, X, y, cv=5, scoring="f1", n_jobs=-1)
    print(f"CV F1 (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n모델 저장 완료 → {MODEL_PATH}")
    return pipeline


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("모델 파일 없음. python train.py 먼저 실행하세요.")
    return joblib.load(MODEL_PATH)


def predict_one(pipeline: Pipeline, input_dict: dict) -> dict:
    """
    단건 예측.
    input_dict 예시:
      {"genre":"액션", "platform":"구글플레이", "org_type":"중소",
       "grade":"15세이용가", "year":2024, "dev_history":"없음",
       "description":"폭력 혈액"}
    """
    df_in = pd.DataFrame([input_dict])
    df_in[TEXT_COL] = df_in[TEXT_COL].fillna("").astype(str)

    prob = pipeline.predict_proba(df_in)[0, 1]
    risk_score = int(round(prob * 100))

    if risk_score >= 65:
        risk_level = "높음"
    elif risk_score >= 40:
        risk_level = "보통"
    else:
        risk_level = "낮음"

    # SHAP 기반 피처 중요도
    shap_factors = get_shap_factors(pipeline, df_in, risk_score)

    # 권고사항 생성
    recommendations = make_recommendations(input_dict, risk_score, shap_factors)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "top_factors": shap_factors[:3],
        "recommendations": recommendations,
        "summary": make_summary(input_dict, risk_score, risk_level),
    }


def get_shap_factors(pipeline: Pipeline, df_in: pd.DataFrame, risk_score: int) -> list:
    """RandomForest feature_importances를 SHAP 대용으로 활용."""
    try:
        import shap
        prep = pipeline.named_steps["prep"]
        clf  = pipeline.named_steps["clf"]
        X_transformed = prep.transform(df_in)

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_transformed)

        # 이진분류: shap_vals[1] = 재조정 클래스
        vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

        feature_names = _get_feature_names(prep)
        pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)

        factors = []
        for name, val in pairs[:6]:
            human = _humanize_feature(name, df_in)
            if human:
                impact = int(min(abs(val) * 300, 95))
                factors.append({"factor": human["label"],
                                 "impact": max(impact, 5),
                                 "description": human["desc"]})
            if len(factors) >= 3:
                break
        return factors

    except Exception:
        # SHAP 실패 시 규칙 기반 폴백
        return _rule_based_factors(df_in.iloc[0].to_dict(), risk_score)


def _get_feature_names(preprocessor):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            names += cols
        elif name == "num":
            names += cols
        elif name == "txt":
            names += [f"txt_{v}" for v in trans.get_feature_names_out()]
    return names


def _humanize_feature(name: str, df_in: pd.DataFrame) -> dict | None:
    row = df_in.iloc[0]
    mapping = {
        "genre":       {"label": f"장르 ({row.get('genre','')})",       "desc": "장르별 콘텐츠 위험 특성"},
        "platform":    {"label": f"플랫폼 ({row.get('platform','')})",   "desc": "플랫폼 자체심사 엄격도"},
        "org_type":    {"label": f"기관유형 ({row.get('org_type','')})", "desc": "신청기관 규모별 심사 정밀도"},
        "grade":       {"label": f"신청등급 ({row.get('grade','')})",    "desc": "실제 콘텐츠 수위 대비 등급 적절성"},
        "dev_history": {"label": f"재조정이력 ({row.get('dev_history','')})", "desc": "동일 개발사 과거 오류 패턴"},
        "year":        {"label": f"출시연도 ({row.get('year','')})",     "desc": "심사 기준 변경 이전 출시 여부"},
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("txt_"):
        kw = name.replace("txt_", "")
        return {"label": f"콘텐츠 키워드", "desc": f"'{kw}' 관련 기술서 내용"}
    return None


def _rule_based_factors(row: dict, risk_score: int) -> list:
    from data_generator import GENRE_RISK, PLATFORM_RISK, ORG_RISK, HISTORY_RISK, GRADE_RISK
    scored = [
        ("장르 특성",     GENRE_RISK.get(row.get("genre","기타"), 0.35),    f"{row.get('genre','')} 장르 콘텐츠 위험도"),
        ("개발사 이력",   HISTORY_RISK.get(row.get("dev_history","없음"),0), f"재조정이력 {row.get('dev_history','')}"),
        ("기관 유형",     ORG_RISK.get(row.get("org_type","중소"),0),        f"{row.get('org_type','')} 기관 심사 정밀도"),
        ("플랫폼",        PLATFORM_RISK.get(row.get("platform","기타"),0),   f"{row.get('platform','')} 플랫폼"),
        ("신청 등급",     GRADE_RISK.get(row.get("grade","15세이용가"),0),   f"{row.get('grade','')} 적절성"),
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"factor": s[0], "impact": int(s[1]*100), "description": s[2]}
            for s in scored[:3]]


def make_recommendations(row: dict, risk_score: int, factors: list) -> list:
    recs = []
    if risk_score >= 65:
        recs.append("등록 전 GRAC 사전 자문을 받는 것을 권고합니다.")
    if row.get("org_type") == "개인":
        recs.append("개인 개발사의 경우 GCRB 자체등급분류 가이드라인을 재검토하세요.")
    if row.get("dev_history") not in ("없음", None):
        recs.append("과거 재조정 이력이 있는 경우 콘텐츠 기술서를 더욱 상세히 작성하세요.")
    if row.get("grade") == "전체이용가" and row.get("genre") in ("액션","슈팅","격투"):
        recs.append(f"{row['genre']} 장르를 전체이용가로 신청 시 폭력성 기준 재확인이 필요합니다.")
    if not recs:
        recs.append("현재 입력된 정보 기준으로 등급 오류 가능성이 낮습니다. 콘텐츠 기술서를 상세히 작성하세요.")
    recs.append("등급 부여 후에도 업데이트 시 콘텐츠 변동 여부를 재검토하세요.")
    return recs[:3]


def make_summary(row: dict, risk_score: int, risk_level: str) -> str:
    return (
        f"{row.get('genre','')} 장르 / {row.get('platform','')} 플랫폼 기준으로 "
        f"등급재조정 위험도는 {risk_score}점({risk_level})으로 분석됩니다. "
        f"{'즉각적인 사전 검토가 권고됩니다.' if risk_score >= 65 else '일반적인 수준의 검토가 필요합니다.' if risk_score >= 40 else '현재 등급 분류가 적절할 가능성이 높습니다.'}"
    )
