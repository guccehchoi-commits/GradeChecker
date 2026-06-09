"""
GradeChecker — 게임 자체등급분류 오류 사전 예측 서비스
실행: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_generator import make_sample
from ml_model import train, predict_one


_SELF_PLAT_MAP = {
    "AAPL": "애플 앱스토어", "GOOG": "구글플레이", "NTDO": "콘솔", "SONY": "콘솔",
    "MSFT": "기타", "OCUL": "기타", "SSCP": "기타", "SECL": "기타",
    "SGHS": "기타", "ONIA": "기타", "FORT": "기타", "ONST": "기타",
}


def preprocess_self(df: pd.DataFrame) -> pd.DataFrame:
    def extract_year(rateno):
        try:
            date_part = str(rateno).split("-")[2]
            year = int("20" + date_part[:2])
            return year if 2015 <= year <= 2030 else 2022
        except:
            return 2022

    def extract_platform(rateno):  # rateno 접두(AAPL/GOOG/…) → 플랫폼 복원
        pre = str(rateno).split("-")[0].upper()
        return _SELF_PLAT_MAP.get(pre, "기타")

    out = pd.DataFrame()
    out["genre"]        = df["genre"].fillna("기타")
    out["platform"]     = df["rateno"].apply(extract_platform)
    out["org_type"]     = "민간"
    out["grade"]        = df["grade"].fillna("15세이용가")
    out["year"]         = df["rateno"].apply(extract_year)
    out["dev_history"]  = "없음"
    out["description"]  = df["content"].fillna("")
    out["reclassified"] = 0
    out["gametitle"]    = df["gametitle"].fillna("")
    return out.reset_index(drop=True)


# ── 등급재분류 데이터 설정 (도메인 판단 — 자유롭게 수정) ──────────────────────
INCLUDE_CANCELLATION = True   # '등급취소'(991건)도 reclassified=1 로 볼지

# 자체등급 원표기 → 한국식 4단계 (Apple/구글 글로벌 표기 흡수)
_RERATE_GRADE_MAP = {
    "전체이용가": "전체이용가", "4+": "전체이용가", "3+": "전체이용가", "만3세이상": "전체이용가",
    "9+": "12세이용가", "7세이상": "12세이용가", "12세이용가": "12세이용가",
    "12+": "12세이용가", "만12세이상": "12세이용가",
    "15세이용가": "15세이용가", "15+": "15세이용가",
    "17+": "청소년이용불가", "청소년이용불가": "청소년이용불가",
}


def preprocess_rerating(df: pd.DataFrame) -> pd.DataFrame:
    """게임위 직권등급재분류·등급취소·등급조정 목록 → 학습 스키마(전부 reclassified=1).

    · 자체등급(처음 부여된 등급)이 실제로 적혀 있는 행만 사용한다.
      빈칸은 추정해서 채우지 않고 제외한다 — '처음등급 → 재조정등급'이
      사실관계여야 하므로, 지어낸 등급은 데이터에 섞지 않는다.
    · 제작자(개발사) 등장 빈도를 dev_history(동일 개발사 재조정 이력)로 환산한다.
    """

    def norm_platform(x):
        if pd.isna(x):
            return "기타"
        s = str(x).strip().lower()
        if "google" in s:                                       return "구글플레이"
        if "apple" in s:                                        return "애플 앱스토어"
        if "sony" in s or "닌텐도" in s or "nintendo" in s:        return "콘솔"
        return "기타"  # 삼성 / 원스토어 / Microsoft / Oculus / - 등

    def norm_grade(x):
        if pd.isna(x):
            return None
        return _RERATE_GRADE_MAP.get(str(x).strip().replace(" ", ""))  # 매핑 밖이면 None

    def extract_year(d):
        y = pd.to_datetime(d, errors="coerce")
        return int(y.year) if pd.notna(y) and 2015 <= y.year <= 2030 else 2022

    def dev_bucket(n):  # 개발사 등장 횟수 -> 이력 구간 (n회 등장 ~= n건 재조정)
        return "없음" if n == 1 else "1회" if n == 2 else "2~3회" if n <= 4 else "4회 이상"

    work = df.copy()
    if not INCLUDE_CANCELLATION:
        work = work[work["구분"] != "등급취소"]

    # 자체등급이 실제로 있는 행만 (빈칸·매핑밖 제외)
    work["_g"] = work["자체등급"].apply(norm_grade)
    work = work[work["_g"].notna()].reset_index(drop=True)

    # 제작자 빈도 -> dev_history
    freq = work["제작자"].map(work["제작자"].value_counts()).fillna(1).astype(int)

    out = pd.DataFrame()
    out["genre"]        = "기타"                                       # 장르 정보 없음
    out["platform"]     = work["자체등급분류사업자"].apply(norm_platform)
    out["org_type"]     = "민간"                                       # preprocess_self 와 동일
    out["grade"]        = work["_g"]                                   # 처음 부여된 자체등급(팩트)
    out["year"]         = work["결정일자"].apply(extract_year)
    out["dev_history"]  = freq.apply(dev_bucket)
    out["description"]  = ""                                           # 콘텐츠 기술서 없음
    out["reclassified"] = 1                                            # 전부 재조정 발생 사례
    out["gametitle"]    = work["게임물명"].fillna("")
    return out.reset_index(drop=True)


st.set_page_config(
    page_title="GradeChecker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.gg-header { background: #0F6E56; padding: 1.5rem 2rem; border-radius: 12px;
             color: white; margin-bottom: 1.5rem; }
.gg-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
.gg-header p  { font-size: 0.85rem; opacity: 0.85; margin: 0.3rem 0 0; }
.risk-card { border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.risk-high { background: #FCEBEB; border-left: 4px solid #E24B4A; }
.risk-med  { background: #FAEEDA; border-left: 4px solid #EF9F27; }
.risk-low  { background: #E1F5EE; border-left: 4px solid #1D9E75; }
.trust-badge { display:inline-block; padding:0.3rem 1rem;
               border-radius:20px; font-weight:700; font-size:1.1rem; }
.trust-high { background:#E1F5EE; color:#085041; }
.trust-mid  { background:#FAEEDA; color:#633806; }
.trust-low  { background:#FCEBEB; color:#501313; }
.rec-item { padding:0.4rem 0.6rem; background:#f1f8f5; border-radius:6px;
            margin-bottom:0.4rem; font-size:0.88rem; border-left:3px solid #1D9E75; }
.warn-box { background:#FFF3CD; border:1px solid #EF9F27; border-radius:8px;
            padding:0.8rem 1rem; font-size:0.85rem; color:#633806; margin-top:0.8rem; }
.info-box { background:#E6F1FB; border:1px solid #378ADD; border-radius:8px;
            padding:0.8rem 1rem; font-size:0.85rem; color:#042C53; margin-top:0.8rem; }
div[data-testid="stTabs"] button { font-size:0.95rem; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 로딩 공통 헬퍼 ───────────────────────────────────────────────────────
def _load_full_dataset():
    """train_data + grac_self_data + grac_rerating_data 를 합쳐 반환.
    train_data.csv 가 없으면 (None, 합성데이터) 를 반환."""
    base = Path(__file__).parent
    csv_path = base / "train_data.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    self_path = base / "grac_self_data.csv"
    if self_path.exists():
        df_self = pd.read_csv(self_path, encoding="utf-8-sig")
        df = pd.concat([df, preprocess_self(df_self)], ignore_index=True)

    rerate_path = base / "grac_rerating_data.csv"
    if rerate_path.exists():
        df_re = pd.read_csv(rerate_path, encoding="utf-8-sig")
        df = pd.concat([df, preprocess_rerating(df_re)], ignore_index=True)

    return df


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    """train_data.csv 있으면 실데이터, 없으면 합성데이터로 학습."""
    df = _load_full_dataset()
    if df is not None:
        src_msg = f"실데이터 {len(df):,}건으로 모델 초기화 완료"
    else:
        df = make_sample(3000)
        src_msg = "합성 데이터로 모델 초기화 완료"
    return train(df), src_msg


# ── 통계 데이터 ────────────────────────────────────────────────────────────────
@st.cache_data
def get_stats_data():
    df = _load_full_dataset()
    if df is not None:
        return df
    return make_sample(3000)


# ── B2C 더미 DB ────────────────────────────────────────────────────────────────
GAME_DB = {
    "배틀그라운드": {"genre":"슈팅","platform":"PC (Steam)","org_type":"대형사",
                    "grade":"청소년이용불가","year":2017,"dev_history":"없음",
                    "description":"총기 폭력 혈액 생존"},
    "마인크래프트": {"genre":"어드벤처","platform":"구글플레이","org_type":"대형사",
                    "grade":"전체이용가","year":2011,"dev_history":"없음",
                    "description":"블록 건설 교육 가족"},
    "브롤스타즈":   {"genre":"액션","platform":"애플 앱스토어","org_type":"대형사",
                    "grade":"12세이용가","year":2018,"dev_history":"없음",
                    "description":"캐릭터 전투 경쟁"},
    "리니지M":      {"genre":"RPG","platform":"구글플레이","org_type":"대형사",
                    "grade":"15세이용가","year":2017,"dev_history":"2~3회",
                    "description":"폭력 확률형 사행 도박"},
    "로블록스":     {"genre":"시뮬레이션","platform":"구글플레이","org_type":"대형사",
                    "grade":"전체이용가","year":2006,"dev_history":"없음",
                    "description":"블록 건설 가족 교육"},
    "오딘":         {"genre":"RPG","platform":"구글플레이","org_type":"대형사",
                    "grade":"청소년이용불가","year":2021,"dev_history":"없음",
                    "description":"폭력 혈액 선정 확률형"},
    "쿠키런 킹덤":  {"genre":"시뮬레이션","platform":"애플 앱스토어","org_type":"대형사",
                    "grade":"전체이용가","year":2021,"dev_history":"없음",
                    "description":"캐릭터 성장 퍼즐 가족"},
    "로스트아크":   {"genre":"RPG","platform":"PC (자체)","org_type":"대형사",
                    "grade":"15세이용가","year":2018,"dev_history":"1회",
                    "description":"폭력 혈액 전투 확률형"},
}

# ── 자체등급분류 현황 테이블 (기획안 1p 수치) ──────────────────────────────────
SELF_RATING_STATS = pd.DataFrame({
    "연도":          [2022, 2023, 2024],
    "전체 게임물":   [1021996, 893064, 1562028],
    "모니터링 게임물": [130000, 110000, 117000],
    "시정 요청":     [11000, 9600, 7661],
    "직권등급재분류": [1838, 1548, 929],
})


# ── 차트 함수 ──────────────────────────────────────────────────────────────────
def gauge_chart(score: int, title: str = "위험도"):
    color = "#E24B4A" if score >= 65 else "#EF9F27" if score >= 40 else "#1D9E75"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.6},
            "steps": [
                {"range": [0,  40], "color": "#E8F8F2"},
                {"range": [40, 65], "color": "#FDF6E3"},
                {"range": [65,100], "color": "#FDF0F0"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": score},
        },
        number={"font": {"size": 36, "color": color}, "suffix": "점"},
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def shap_bar_chart(factors: list):
    names   = [f["factor"] for f in factors]
    impacts = [f["impact"] for f in factors]
    colors  = ["#E24B4A" if v >= 60 else "#EF9F27" if v >= 35 else "#1D9E75"
               for v in impacts]
    fig = go.Figure(go.Bar(
        x=impacts, y=names, orientation="h",
        marker_color=colors, text=impacts, textposition="outside",
    ))
    fig.update_layout(
        height=160, margin=dict(t=10, b=10, l=10, r=40),
        xaxis=dict(range=[0, 110], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def trust_donut(score: int):
    color = "#1D9E75" if score >= 70 else "#EF9F27" if score >= 45 else "#E24B4A"
    fig = go.Figure(go.Pie(
        values=[score, 100 - score], hole=0.72,
        marker_colors=[color, "#f0f0f0"], textinfo="none", hoverinfo="skip",
    ))
    fig.add_annotation(text=f"<b>{score}</b>", x=0.5, y=0.5,
                       font_size=28, showarrow=False, font_color=color)
    fig.update_layout(height=160, margin=dict(t=10, b=10, l=10, r=10),
                      showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── 헤더 ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gg-header">
  <h1>🛡️ GradeChecker</h1>
  <p>게임 자체등급분류 오류 사전 예측 및 이용자 신뢰도 조회 서비스 — 프로토타입</p>
  <p style="opacity:0.6; font-size:0.75rem;">게임이용자보호센터 (게임문화재단) | 2026 문화 디지털혁신 및 데이터 활용 공모전</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("AI 예측 모델 초기화 중... (최초 실행 시 2~3분 소요)"):
    pipeline, src_msg = get_model()
st.toast(src_msg)

tab_b2b, tab_b2c, tab_stats = st.tabs([
    "🏢 개발사·플랫폼 도구", "👥 이용자 등급 조회", "📊 통계 대시보드"
])


# ══════════════════════════════════════════════════════════════════════════════
# B2B TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_b2b:
    st.markdown("#### 게임 메타데이터 입력")
    st.caption("등록 전 게임 정보를 입력하면 등급재조정 위험도를 사전 예측합니다.")

    col1, col2, col3 = st.columns(3)
    with col1:
        name     = st.text_input("게임명", placeholder="예) 배틀히어로즈 2")
        genre    = st.selectbox("장르", ["액션","RPG","퍼즐","전략","스포츠",
                                         "시뮬레이션","어드벤처","슈팅","격투","기타"])
        platform = st.selectbox("플랫폼", ["구글플레이","애플 앱스토어",
                                            "PC (Steam)","PC (자체)","콘솔","기타"])
    with col2:
        org_type    = st.selectbox("등급신청 기관 유형", ["대형사","중소","개인"])
        grade       = st.selectbox("신청 등급", ["전체이용가","12세이용가",
                                                  "15세이용가","청소년이용불가"])
        year        = st.selectbox("출시 연도", list(range(2026, 2017, -1)))
    with col3:
        dev_history = st.selectbox("동일 개발사 재조정 이력", ["없음","1회","2~3회","4회 이상"])
        description = st.text_area("콘텐츠 기술서 / 게임물 개요", height=118,
                                   placeholder="폭력성·선정성·사행성·언어 관련 내용 + 게임물 개요를 함께 기술하세요.\n예) 총기 전투, 혈액 표현, 확률형 아이템 포함, 판타지 배경의 RPG")

    run = st.button("🔍 위험도 분석", type="primary", use_container_width=True)

    if run:
        inp = {"genre": genre, "platform": platform, "org_type": org_type,
               "grade": grade, "year": year, "dev_history": dev_history,
               "description": description}

        with st.spinner("분석 중..."):
            result = predict_one(pipeline, inp)

        score = result["risk_score"]
        level = result["risk_level"]
        css   = "risk-high" if level=="높음" else "risk-med" if level=="보통" else "risk-low"
        icon  = "🔴" if level=="높음" else "🟡" if level=="보통" else "🟢"

        st.markdown("---")
        st.markdown(f"""
        <div class="risk-card {css}">
          <b style="font-size:1.05rem">{icon} {name or '입력 게임'} — 등급재조정 위험도 {level}</b>
          <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.8">{result['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1.4, 1.4])
        with c1:
            st.plotly_chart(gauge_chart(score), use_container_width=True, key="gauge")
        with c2:
            st.markdown("**주요 위험 요인 (SHAP)**")
            st.plotly_chart(shap_bar_chart(result["top_factors"]),
                            use_container_width=True, key="shap")
            for f in result["top_factors"]:
                st.caption(f"• **{f['factor']}**: {f['description']}")
        with c3:
            st.markdown("**권고 사항**")
            for r in result["recommendations"]:
                st.markdown(f'<div class="rec-item">✅ {r}</div>', unsafe_allow_html=True)
            if level == "높음":
                st.markdown("""<div class="warn-box">⚠️ <b>사전 검토 권고</b><br>
                등록 전 GRAC 사전 자문 또는 GCRB 가이드라인 재검토를 권고합니다.</div>""",
                            unsafe_allow_html=True)
            else:
                st.markdown("""<div class="info-box">ℹ️ 콘텐츠 기술서와 게임물 개요를
                상세히 작성하면 예측 정확도가 높아집니다.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# B2C TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_b2c:
    st.markdown("#### 게임 등급 신뢰도 조회")
    st.caption("게임명을 검색하면 현재 자체분류 등급의 신뢰도를 확인할 수 있습니다.")

    c_search, c_btn = st.columns([5, 1])
    with c_search:
        query = st.text_input("게임명 검색", placeholder="예) 마인크래프트, 리니지M, 브롤스타즈...",
                              label_visibility="collapsed")
    with c_btn:
        search = st.button("조회", type="primary", use_container_width=True)

    st.markdown("**등록 게임 바로 조회**")
    quick_cols = st.columns(len(GAME_DB))
    for i, gname in enumerate(GAME_DB):
        with quick_cols[i]:
            if st.button(gname, key=f"quick_{gname}", use_container_width=True):
                query = gname
                search = True

    if search and query:
        meta = GAME_DB.get(query)
        if meta is None:
            meta = {"genre":"기타","platform":"구글플레이","org_type":"중소",
                    "grade":"15세이용가","year":2023,"dev_history":"없음","description":""}
            st.info(f"'{query}'은(는) 데이터베이스에 없는 게임입니다. 기본값으로 예측합니다.")

        with st.spinner("분석 중..."):
            r = predict_one(pipeline, meta)

        trust = 100 - r["risk_score"]
        level = "상" if trust >= 70 else "중" if trust >= 45 else "하"
        badge_css = "trust-high" if level=="상" else "trust-mid" if level=="중" else "trust-low"
        grade_emoji = {"전체이용가":"🟢","12세이용가":"🔵","15세이용가":"🟡","청소년이용불가":"🔴"}

        st.markdown("---")
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            st.markdown(f"### {query}")
            st.markdown(f'<span class="trust-badge {badge_css}">신뢰도 {level}</span>',
                        unsafe_allow_html=True)
            ge = meta["grade"]
            st.markdown(f"**현재 등급:** {grade_emoji.get(ge,'')} {ge}")
            st.markdown(f"**장르:** {meta['genre']} | **플랫폼:** {meta['platform']}")
        with col_b:
            st.plotly_chart(trust_donut(trust), use_container_width=True, key="donut")
            st.caption(f"신뢰도 점수: {trust}/100")
        with col_c:
            if level == "하":
                st.error("⚠️ **현재 등급 정보 검토 중**\n\n이 게임의 등급 신뢰도가 낮습니다. 구매 또는 이용 전 공식 채널에서 최신 등급을 확인하세요.")
            elif level == "중":
                st.warning("🔍 **일부 항목 검토 권고**\n\n등급 부여 기준 일부 항목에서 재확인이 필요할 수 있습니다.")
            else:
                st.success("✅ **등급 신뢰도 양호**\n\n현재 등급이 적절한 것으로 판단됩니다.")
            if r["top_factors"]:
                st.markdown("**검토 항목**")
                for f in r["top_factors"]:
                    if f["impact"] > 25:
                        st.caption(f"• {f['factor']}: {f['description']}")


# ══════════════════════════════════════════════════════════════════════════════
# 통계 대시보드
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    df_stats = get_stats_data()
    base      = Path(__file__).parent
    csv_path  = base / "train_data.csv"
    self_path = base / "grac_self_data.csv"
    rerate_path = base / "grac_rerating_data.csv"

    if csv_path.exists():
        parts = []
        if self_path.exists():
            self_n = len(pd.read_csv(self_path, encoding="utf-8-sig"))
            parts.append(f"민간자율분류 {self_n:,}건")
        else:
            self_n = 0
        if rerate_path.exists():
            df_re_n = len(preprocess_rerating(pd.read_csv(rerate_path, encoding="utf-8-sig")))
            parts.append(f"직권재분류 {df_re_n:,}건")
        else:
            df_re_n = 0
        grac_n = len(df_stats) - self_n - df_re_n
        src_note = " + ".join([f"GRAC 실데이터 {grac_n:,}건"] + parts)
    else:
        src_note = "GRAC 실데이터"

    st.markdown("#### 자체등급분류 오류 현황 분석")
    st.caption(f"※ 데이터 출처: {src_note} ({len(df_stats):,}건)")

    total   = len(df_stats)
    reratio = df_stats["reclassified"].mean()
    high_n  = int(df_stats["reclassified"].sum())
    rep_n   = (df_stats["dev_history"] != "없음").sum()  # 재조정 이력 보유 건수

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("전체 분석 건수", f"{total:,}건")
    with m2: st.metric("재조정 발생률", f"{reratio:.1%}")
    with m3: st.metric("재조정 발생 건수", f"{high_n:,}건", f"전체의 {high_n/total:.1%}")
    with m4: st.metric("재조정 이력 보유 비율", f"{rep_n/total:.1%}")

    st.markdown("---")

    # ── 차트 1+2: 장르별 / 등급별 재조정 비율 ──────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**장르별 재조정 비율**")
        genre_df = df_stats.groupby("genre")["reclassified"].mean().reset_index()
        genre_df.columns = ["장르", "재조정비율"]
        genre_df = genre_df.sort_values("재조정비율", ascending=True)
        fig1 = px.bar(genre_df, x="재조정비율", y="장르", orientation="h",
                      color="재조정비율",
                      color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"],
                      text=genre_df["재조정비율"].apply(lambda x: f"{x:.1%}"))
        fig1.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=10,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", xaxis_tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        st.markdown("**등급별 재조정 비율**")
        grade_order = ["전체이용가","12세이용가","15세이용가","청소년이용불가"]
        grade_df = df_stats.groupby("grade")["reclassified"].mean().reset_index()
        grade_df.columns = ["등급", "재조정비율"]
        grade_df["등급"] = pd.Categorical(grade_df["등급"], categories=grade_order, ordered=True)
        grade_df = grade_df.sort_values("등급")
        fig2 = px.bar(grade_df, x="등급", y="재조정비율",
                      color="재조정비율",
                      color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"],
                      text=grade_df["재조정비율"].apply(lambda x: f"{x:.1%}"))
        fig2.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=10,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # ── 차트 3: 플랫폼별 재조정 비율 ──────────────────────────────────────────
    st.markdown("**플랫폼별 재조정 비율**")
    platform_df = df_stats.groupby("platform")["reclassified"].mean().reset_index()
    platform_df.columns = ["플랫폼", "재조정비율"]
    platform_df = platform_df.sort_values("재조정비율", ascending=True)
    fig_p = px.bar(platform_df, x="재조정비율", y="플랫폼", orientation="h",
                   color="재조정비율",
                   color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"],
                   text=platform_df["재조정비율"].apply(lambda x: f"{x:.1%}"))
    fig_p.update_layout(height=260, showlegend=False, coloraxis_showscale=False,
                        margin=dict(t=10,b=10,l=10,r=10),
                        paper_bgcolor="rgba(0,0,0,0)", xaxis_tickformat=".0%")
    st.plotly_chart(fig_p, use_container_width=True)

    # ── 차트 4: 연도별 재조정 건수 추이 ───────────────────────────────────────
    st.markdown("**연도별 재조정 건수 추이**")
    year_df = df_stats.groupby("year")["reclassified"].agg(["sum","count"]).reset_index()
    year_df.columns = ["연도","재조정건수","전체건수"]
    year_df["비율"] = year_df["재조정건수"] / year_df["전체건수"]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=year_df["연도"], y=year_df["재조정건수"],
                          name="재조정건수", marker_color="#EF9F27"))
    fig3.add_trace(go.Scatter(x=year_df["연도"], y=year_df["비율"],
                              name="재조정비율", yaxis="y2",
                              line=dict(color="#E24B4A", width=2), mode="lines+markers"))
    fig3.update_layout(
        height=260, margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="건수"),
        yaxis2=dict(title="비율", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── 표: 자체등급분류 게임물 연간 현황 ─────────────────────────────────────
    st.markdown("---")
    st.markdown("**자체등급분류 게임물 연간 현황**")
    st.caption("※ 출처: 게임물관리위원회, 2024-2025 게임물 등급분류 및 사후관리 연감")
    display_df = SELF_RATING_STATS.copy()
    display_df["전체 게임물"]    = display_df["전체 게임물"].apply(lambda x: f"{x:,}")
    display_df["모니터링 게임물"] = display_df["모니터링 게임물"].apply(lambda x: f"{x:,}")
    display_df["시정 요청"]      = display_df["시정 요청"].apply(lambda x: f"{x:,}")
    display_df["직권등급재분류"]  = display_df["직권등급재분류"].apply(lambda x: f"{x:,}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("GradeChecker v0.3 | 게임이용자보호센터 (게임문화재단) | 2026 문화 디지털혁신 및 데이터 활용 공모전 출품작 | 본 서비스는 프로토타입이며 실제 등급 판정 효력이 없습니다.")
