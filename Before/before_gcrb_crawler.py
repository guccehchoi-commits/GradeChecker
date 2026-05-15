"""
GradeChecker — GCRB 자체등급분류 크롤러 + GRAC NLP 확장
코랩에서 실행: python gcrb_crawler.py

수집 순서:
1. GRAC API → 정식 등급 + 게임물 개요(summary) 포함
2. GCRB 홈페이지 크롤링 → 자체등급 게임 목록 (타이틀·장르·연령가)
3. 실패 시 → 구글플레이 크롤링으로 장르·등급 보완
4. GRAC ↔ GCRB 타이틀 매칭 → Y 레이블 생성
5. 결과 저장 → train_data.csv
"""

import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from difflib import SequenceMatcher

# ── 패키지 확인 ──────────────────────────────────────────────────────────────
try:
    from bs4 import BeautifulSoup
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "beautifulsoup4", "--break-system-packages", "-q"])
    from bs4 import BeautifulSoup

try:
    from google_play_scraper import search as gp_search, app as gp_app
    HAS_GPS = True
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "google-play-scraper", "--break-system-packages", "-q"])
    try:
        from google_play_scraper import search as gp_search, app as gp_app
        HAS_GPS = True
    except ImportError:
        HAS_GPS = False
        print("⚠️ google-play-scraper 설치 실패. 스토어 크롤링 불가.")


# ── 상수 ─────────────────────────────────────────────────────────────────────
GRAC_API = "https://www.grac.or.kr/WebService/GameSearchSvc.asmx/game"
GCRB_BASE = "https://www.gcrb.or.kr"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

GENRE_MAP = {
    "격투게임":"격투","액션":"액션","FPS/TPS":"슈팅","비행슈팅":"슈팅",
    "롤플레잉":"RPG","MMORPG":"RPG","퍼즐":"퍼즐","캐주얼":"퍼즐",
    "전략시뮬레이션":"전략","시뮬레이션":"시뮬레이션","스포츠":"스포츠",
    "어드벤처":"어드벤처","교육용":"기타","레이싱":"기타",
    "보드게임":"기타","보드게임(베팅성)":"기타","크레인":"기타","기타":"기타",
}
PLATFORM_MAP = {
    "PC/온라인 게임":"PC (Steam)","모바일 게임":"구글플레이",
    "비디오 게임":"콘솔","아케이드 게임":"기타",
}
GRADE_MAP = {
    "전체이용가":"전체이용가","12세이용가":"12세이용가",
    "15세이용가":"15세이용가","청소년이용불가":"청소년이용불가",
    "등급거부":"청소년이용불가","등급취소":"청소년이용불가",
}


# ══════════════════════════════════════════════════════════════════════════════
# ① GRAC 수집 (summary 포함, NLP 확장)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_grac(max_pages: int = 34) -> pd.DataFrame:
    print("① GRAC API 수집 시작...")
    all_items = []
    for page in range(1, max_pages + 1):
        res = requests.get(GRAC_API,
                           params={"op":"game","display":1000,"pageno":page},
                           timeout=30)
        root = ET.fromstring(res.text)
        if root.tag == "error":
            break
        for item in root.findall("item"):
            def g(tag):
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""
            all_items.append({
                "gametitle":    g("gametitle"),
                "genre_raw":    g("genre"),
                "platform_raw": g("platform"),
                "entname":      g("entname"),
                "givenrate":    g("givenrate"),
                "rateddate":    g("rateddate"),
                "descriptors":  g("descriptors"),
                "summary":      g("summary"),       # ← NLP 확장
                "cancelstatus": g("cancelstatus"),
                "orgname":      g("orgname"),
            })
        print(f"  {page}/{max_pages} — {len(all_items):,}건")
        time.sleep(0.3)
    df = pd.DataFrame(all_items)
    print(f"  GRAC 수집 완료: {len(df):,}건\n")
    return df


def preprocess_grac(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["gametitle"] = df["gametitle"]
    out["genre"]     = df["genre_raw"].map(GENRE_MAP).fillna("기타")
    out["platform"]  = df["platform_raw"].map(PLATFORM_MAP).fillna("기타")
    out["grade"]     = df["givenrate"].map(GRADE_MAP).fillna("15세이용가")
    out["year"]      = pd.to_datetime(df["rateddate"], errors="coerce").dt.year.fillna(2020).astype(int)

    # NLP: summary + descriptors 결합 → description 컬럼
    out["description"] = (df["summary"].fillna("") + " " + df["descriptors"].fillna("")).str.strip()

    out["reclassified"] = (df["cancelstatus"].str.lower() == "true").astype(int)

    def infer_org(name):
        big = ["넥슨","엔씨소프트","넷마블","카카오","크래프톤",
               "스마일게이트","구글","애플","펄어비스","위메이드"]
        return ("대형사" if any(b in str(name) for b in big)
                else "개인" if len(str(name)) <= 2 else "중소")
    out["org_type"] = df["entname"].apply(infer_org)

    hist = df.groupby("entname")["cancelstatus"].apply(
        lambda x: (x.str.lower() == "true").sum())
    df2 = df.join(hist.rename("cnt"), on="entname")

    def cnt2h(n):
        if n == 0: return "없음"
        elif n == 1: return "1회"
        elif n <= 3: return "2~3회"
        return "4회 이상"
    out["dev_history"] = df2["cnt"].fillna(0).apply(cnt2h)
    return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# ② GCRB 홈페이지 크롤링
# ══════════════════════════════════════════════════════════════════════════════
def fetch_gcrb_web(max_pages: int = 50) -> pd.DataFrame:
    """
    GCRB 홈페이지에서 자체등급분류 게임 목록 크롤링.
    타이틀, 장르, 연령가, 사업자 추출.
    """
    print("② GCRB 홈페이지 크롤링 시작...")
    items = []

    # GCRB 자체등급분류 검색 URL (실제 URL은 확인 후 수정 필요)
    SEARCH_URL = f"{GCRB_BASE}/game/selfRating/selfRatingList.do"

    for page in range(1, max_pages + 1):
        try:
            params = {"pageIndex": page, "searchGubun": "all"}
            res = requests.get(SEARCH_URL, params=params,
                               headers=HEADERS, timeout=20)

            if res.status_code != 200:
                print(f"  ⚠️ HTTP {res.status_code} — GCRB 접근 실패")
                print("  → 스토어 크롤링으로 전환합니다.")
                return pd.DataFrame()

            soup = BeautifulSoup(res.text, "html.parser")

            # 테이블에서 게임 목록 파싱 (실제 HTML 구조에 맞게 조정)
            rows = soup.select("table tbody tr")
            if not rows:
                print(f"  페이지 {page}: 데이터 없음 (마지막 페이지)")
                break

            for row in rows:
                cols = row.select("td")
                if len(cols) < 4:
                    continue
                items.append({
                    "gametitle_self": cols[1].get_text(strip=True),
                    "genre_self":     cols[2].get_text(strip=True),
                    "grade_self":     cols[3].get_text(strip=True),
                    "entname_self":   cols[4].get_text(strip=True) if len(cols) > 4 else "",
                })

            print(f"  페이지 {page} — 누적 {len(items):,}건")
            time.sleep(1.0)

        except Exception as e:
            print(f"  ⚠️ 크롤링 오류 ({page}페이지): {e}")
            break

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    print(f"  GCRB 크롤링 완료: {len(df):,}건\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ③ 스토어 크롤링 (GCRB 실패 시 fallback)
# ══════════════════════════════════════════════════════════════════════════════
GENRE_KR_MAP = {
    "ACTION": "액션", "ROLE_PLAYING": "RPG", "PUZZLE": "퍼즐",
    "STRATEGY": "전략", "SPORTS": "스포츠", "SIMULATION": "시뮬레이션",
    "ADVENTURE": "어드벤처", "ARCADE": "액션", "CASUAL": "퍼즐",
    "RACING": "기타", "BOARD": "기타", "EDUCATIONAL": "기타",
    "CARD": "기타", "WORD": "퍼즐", "MUSIC": "기타", "TRIVIA": "기타",
}

CONTENT_RATING_MAP = {
    "Everyone": "전체이용가", "Everyone 10+": "12세이용가",
    "Teen": "15세이용가", "Mature 17+": "청소년이용불가",
    "Adults only 18+": "청소년이용불가",
    "전체이용가": "전체이용가", "12세이용가": "12세이용가",
    "15세이용가": "15세이용가", "청소년이용불가": "청소년이용불가",
}


def fetch_store(game_titles: list, batch_size: int = 100) -> pd.DataFrame:
    """
    구글플레이 스토어에서 게임 정보 수집.
    game_titles: GCRB에서 수집된 게임명 리스트.
    """
    if not HAS_GPS:
        print("  ⚠️ google-play-scraper 없음. 스토어 크롤링 생략.")
        return pd.DataFrame()

    print(f"③ 스토어 크롤링 — {len(game_titles):,}개 게임 조회...")
    items = []

    for i, title in enumerate(game_titles[:batch_size]):
        try:
            results = gp_search(title, lang="ko", country="kr", n_hits=1)
            if not results:
                continue
            r = results[0]
            genre_raw = r.get("genre", "")
            genre_en  = r.get("genreId", "")
            items.append({
                "gametitle_self": title,
                "genre_self":     GENRE_KR_MAP.get(genre_en, genre_raw or "기타"),
                "grade_self":     CONTENT_RATING_MAP.get(
                                      r.get("contentRating",""), "15세이용가"),
                "score":          r.get("score", 0),
                "installs":       r.get("installs", ""),
            })
            if i % 20 == 0:
                print(f"  {i+1}/{min(len(game_titles), batch_size)}건 처리...")
            time.sleep(0.5)
        except Exception:
            continue

    df = pd.DataFrame(items)
    print(f"  스토어 크롤링 완료: {len(df):,}건\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ④ 타이틀 매칭 (GRAC ↔ GCRB)
# ══════════════════════════════════════════════════════════════════════════════
def normalize_title(title: str) -> str:
    """타이틀 정규화: 소문자, 특수문자 제거, 공백 통일."""
    title = str(title).lower()
    title = re.sub(r"[^\w\s가-힣]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def fuzzy_match(title_a: str, title_b: str) -> float:
    """타이틀 유사도 (0~1). 0.7 이상이면 동일 게임으로 판단."""
    return SequenceMatcher(None,
                           normalize_title(title_a),
                           normalize_title(title_b)).ratio()


def match_datasets(df_grac: pd.DataFrame,
                   df_self: pd.DataFrame,
                   threshold: float = 0.75) -> pd.DataFrame:
    """
    GRAC 정식 데이터 ↔ 자체등급 데이터를 타이틀 기반 매칭.
    Y 레이블: 자체등급 ≠ 정식등급 → 오류(1), 일치 → 정상(0)
    """
    if df_self.empty:
        print("④ 매칭 대상 자체등급 데이터 없음 — GRAC 단독 학습으로 진행")
        return df_grac

    print(f"④ 타이틀 매칭 시작 — GRAC {len(df_grac):,}건 × 자체등급 {len(df_self):,}건")

    grac_norm = df_grac["gametitle"].apply(normalize_title).tolist()
    self_norm = df_self["gametitle_self"].apply(normalize_title).tolist()

    matched_rows = []
    for i, (grac_row, grac_t) in enumerate(zip(df_grac.itertuples(), grac_norm)):
        best_score, best_j = 0, -1
        for j, self_t in enumerate(self_norm):
            s = fuzzy_match(grac_t, self_t)
            if s > best_score:
                best_score, best_j = s, j

        row = grac_row._asdict()
        row.pop("Index", None)

        if best_score >= threshold and best_j >= 0:
            self_row = df_self.iloc[best_j]
            self_grade = GRADE_MAP.get(self_row.get("grade_self",""), "15세이용가")
            grac_grade = row.get("grade", "15세이용가")
            # Y 레이블 재정의: 자체등급 ≠ 정식등급
            row["reclassified"] = int(self_grade != grac_grade)
            row["match_score"]  = round(best_score, 3)
            row["grade_self"]   = self_grade
            row["genre_self"]   = self_row.get("genre_self", "")
        else:
            row["match_score"] = 0.0
            row["grade_self"]  = ""
            row["genre_self"]  = ""

        matched_rows.append(row)

        if i % 1000 == 0:
            print(f"  매칭 진행 {i:,}/{len(df_grac):,}...")

    result = pd.DataFrame(matched_rows)
    matched_n = (result["match_score"] >= threshold).sum()
    print(f"  매칭 완료 — {matched_n:,}건 매칭 ({matched_n/len(result):.1%})")
    print(f"  재조정 비율(재정의): {result['reclassified'].mean():.1%}\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 저장 + 재학습
# ══════════════════════════════════════════════════════════════════════════════
def save_and_train(df: pd.DataFrame, out_path: str = None):
    from ml_model import train as ml_train

    out = out_path or str(Path(__file__).parent / "train_data.csv")
    # 필수 컬럼만 저장
    keep = ["genre","platform","org_type","grade","year",
            "dev_history","description","reclassified"]
    df_save = df[[c for c in keep if c in df.columns]].copy()
    df_save.to_csv(out, index=False)
    print(f"⑤ 데이터 저장 완료 → {out} ({len(df_save):,}건)")
    print(f"   재조정 비율: {df_save['reclassified'].mean():.1%}")
    print("\n모델 재학습 시작...")
    ml_train(df_save)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ① GRAC 수집
    df_grac_raw = fetch_grac(max_pages=34)
    df_grac     = preprocess_grac(df_grac_raw)

    # ② GCRB 크롤링 시도
    df_self = fetch_gcrb_web(max_pages=50)

    # ③ GCRB 실패 시 스토어 fallback
    if df_self.empty and HAS_GPS:
        print("  → 구글플레이 스토어 크롤링으로 전환...")
        titles = df_grac["gametitle"].tolist()
        df_self = fetch_store(titles, batch_size=200)

    # ④ 매칭 + Y 레이블
    df_final = match_datasets(df_grac, df_self)

    # ⑤ 저장 + 재학습
    save_and_train(df_final)

    # 코랩 다운로드
    try:
        from google.colab import files
        files.download("train_data.csv")
        print("\n✅ train_data.csv 다운로드 완료")
    except ImportError:
        print("\n✅ 완료. train_data.csv를 GitHub에 업로드하세요.")
