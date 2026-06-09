"""
GradeChecker — ML 예측 모델 (v0.2)
NLP 확장: summary + descriptors 결합 → TF-IDF
"""

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

MODEL_PATH = Path(__file__).parent / "gradechecker_model.pkl"

CATEGORICAL_COLS = ["genre", "platform", "org_type", "grade", "dev_history"]
NUMERIC_COLS     = ["year"]
TEXT_COL         = "description"   # summary + descriptors 결합 컬럼


def build_preprocessor():
    cat_enc = OrdinalEncoder(
        categories=[
            sorted(["액션","슈팅","격투","RPG","어드벤처","전략",
                    "스포츠","시뮬레이션","퍼즐","기타"]),
            sorted(["구글플레이","애플 앱스토어","PC (Steam)",
                    "PC (자체)","콘솔","기타"]),
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
            # NLP: summary + descriptors 결합 텍스트 → TF-IDF
            ("txt", TfidfVectorizer(
                max_features=100,        # v0.1(60)에서 확장
                analyzer="char_wb",
                ngram_range=(2, 4),
                sublinear_tf=True,       # 빈도 로그 스케일링
            ), TEXT_COL),
        ]
    )
    return preprocessor


def _merge_text(df: pd.DataFrame) -> pd.Series:
    """summary 컬럼이 있으면 descriptors와 결합, 없으면 description만 사용."""
    desc = df.get("description", pd.Series([""] * len(df))).fillna("")
    if "summary" in df.columns:
        summ = df["summary"].fillna("")
        return (summ + " " + desc).str.strip()
    return desc


def train(df: pd.DataFrame):
    df = df.copy()
    # NLP: summary + descriptors 결합
    df[TEXT_COL] = _merge_text(df)

    n = len(df)
    if n < 1000:
        print(f"표본 {n}건 → 의사결정트리")
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                     random_state=42)
    elif n < 5000:
        print(f"표본 {n}건 → RandomForest")
        clf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                     class_weight="balanced", random_state=42,
                                     n_jobs=-1)
    else:
        print(f"표본 {n}건 → GradientBoosting (앙상블)")
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                         learning_rate=0.05, random_state=42)

    preprocessor = build_preprocessor()
    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + [TEXT_COL]]
    y = df["reclassified"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
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

    from sklearn.model_selection import StratifiedKFold
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring="f1", n_jobs=-1)
    print(f"CV F1 (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n모델 저장 → {MODEL_PATH}")
    return pipeline


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("모델 없음. train.py 먼저 실행하세요.")
    return joblib.load(MODEL_PATH)


def predict_one(pipeline: Pipeline, input_dict: dict) -> dict:
    df_in = pd.DataFrame([input_dict])
    # 예측 시에도 summary 있으면 결합
    df_in[TEXT_COL] = _merge_text(df_in)

    prob       = pipeline.predict_proba(df_in)[0, 1]
    risk_score = int(round(prob * 100))
    risk_level = ("높음" if risk_score >= 65 else
                  "보통" if risk_score >= 40 else "낮음")

    factors = _get_shap_factors(pipeline, df_in, risk_score)
    recs    = _make_recommendations(input_dict, risk_score)

    return {
        "risk_score":    risk_score,
        "risk_level":    risk_level,
        "top_factors":   factors[:3],
        "recommendations": recs,
        "summary":       _make_summary(input_dict, risk_score, risk_level),
    }


def _get_shap_factors(pipeline, df_in, risk_score):
    try:
        import shap
        prep = pipeline.named_steps["prep"]
        clf  = pipeline.named_steps["clf"]
        X_t  = prep.transform(df_in)
        exp  = shap.TreeExplainer(clf)
        sv   = exp.shap_values(X_t)
        vals = sv[1][0] if isinstance(sv, list) else sv[0]
        names = _get_feature_names(prep)
        pairs = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)
        factors = []
        for name, val in pairs[:8]:
            h = _humanize(name, df_in)
            if h:
                factors.append({"factor": h["label"],
                                 "impact": max(int(min(abs(val)*300, 95)), 5),
                                 "description": h["desc"]})
            if len(factors) >= 3:
                break
        return factors or _rule_factors(df_in.iloc[0].to_dict(), risk_score)
    except Exception:
        return _rule_factors(df_in.iloc[0].to_dict(), risk_score)


def _get_feature_names(preprocessor):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name in ("cat", "num"):
            names += list(cols)
        elif name == "txt":
            names += [f"txt_{v}" for v in trans.get_feature_names_out()]
    return names


def _humanize(name, df_in):
    row = df_in.iloc[0]
    m = {
        "genre":       {"label": f"장르 ({row.get('genre','')})",
                        "desc": "장르별 콘텐츠 위험 특성"},
        "platform":    {"label": f"플랫폼 ({row.get('platform','')})",
                        "desc": "플랫폼 자체심사 엄격도"},
        "org_type":    {"label": f"기관유형 ({row.get('org_type','')})",
                        "desc": "신청기관 규모별 심사 정밀도"},
        "grade":       {"label": f"신청등급 ({row.get('grade','')})",
                        "desc": "실제 수위 대비 등급 적절성"},
        "dev_history": {"label": f"재조정이력 ({row.get('dev_history','')})",
                        "desc": "동일 개발사 과거 오류 패턴"},
        "year":        {"label": f"출시연도 ({row.get('year','')})",
                        "desc": "심사 기준 변경 이전 출시 여부"},
    }
    if name in m:
        return m[name]
    if name.startswith("txt_"):
        return {"label": "콘텐츠·개요 키워드", "desc": "게임물 개요/기술서 내 위험 표현"}
    return None


def _rule_factors(row, risk_score):
    GENRE_RISK    = {"액션":0.55,"슈팅":0.60,"격투":0.65,"RPG":0.45,
                     "어드벤처":0.40,"전략":0.30,"스포츠":0.25,
                     "시뮬레이션":0.25,"퍼즐":0.15,"기타":0.35}
    PLATFORM_RISK = {"구글플레이":0.40,"애플 앱스토어":0.35,"PC (Steam)":0.45,
                     "PC (자체)":0.60,"콘솔":0.30,"기타":0.55}
    ORG_RISK      = {"대형사":0.20,"중소":0.45,"개인":0.70}
    HISTORY_RISK  = {"없음":0.10,"1회":0.35,"2~3회":0.65,"4회 이상":0.85}
    GRADE_RISK    = {"전체이용가":0.50,"12세이용가":0.35,
                     "15세이용가":0.30,"청소년이용불가":0.15}
    scored = [
        ("장르 특성",   GENRE_RISK.get(row.get("genre","기타"),0.35),
                        f"{row.get('genre','')} 장르 위험도"),
        ("개발사 이력", HISTORY_RISK.get(row.get("dev_history","없음"),0),
                        f"재조정이력 {row.get('dev_history','')}"),
        ("기관 유형",   ORG_RISK.get(row.get("org_type","중소"),0),
                        f"{row.get('org_type','')} 기관 심사 정밀도"),
        ("플랫폼",      PLATFORM_RISK.get(row.get("platform","기타"),0),
                        f"{row.get('platform','')} 플랫폼"),
        ("신청 등급",   GRADE_RISK.get(row.get("grade","15세이용가"),0),
                        f"{row.get('grade','')} 적절성"),
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"factor": s[0], "impact": int(s[1]*100), "description": s[2]}
            for s in scored[:3]]


def _make_recommendations(row, risk_score):
    recs = []
    if risk_score >= 65:
        recs.append("등록 전 GRAC 사전 자문을 받는 것을 권고합니다.")
    if row.get("org_type") == "개인":
        recs.append("개인 개발사의 경우 GCRB 가이드라인을 재검토하세요.")
    if row.get("dev_history") not in ("없음", None):
        recs.append("재조정 이력 있는 경우 콘텐츠 기술서를 더욱 상세히 작성하세요.")
    if row.get("grade") == "전체이용가" and row.get("genre") in ("액션","슈팅","격투"):
        recs.append(f"{row['genre']} 장르 전체이용가 신청 시 폭력성 기준 재확인 필요.")
    if not recs:
        recs.append("현재 입력 기준 오류 가능성 낮음. 콘텐츠 기술서를 상세히 작성하세요.")
    recs.append("업데이트 시 콘텐츠 변동 여부를 재검토하세요.")
    return recs[:3]


def _make_summary(row, risk_score, risk_level):
    return (
        f"{row.get('genre','')} 장르 / {row.get('platform','')} 플랫폼 기준 "
        f"등급재조정 위험도는 {risk_score}점({risk_level})입니다. "
        + ("즉각적인 사전 검토가 권고됩니다." if risk_score >= 65
           else "일반적 수준의 검토가 필요합니다." if risk_score >= 40
           else "현재 등급 분류가 적절할 가능성이 높습니다.")
    )
