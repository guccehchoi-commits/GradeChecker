"""
GradeChecker — GRAC 자체등급분류게임물 통계 크롤러
대상: https://www.grac.or.kr/Statistics/SelfRateGameStatistics.aspx

코랩에서 실행:
    !pip install beautifulsoup4 requests lxml -q
    !python /content/files/grac_self_crawler.py

수집 항목: 게임물명, 장르, 등급, 플랫폼, 사업자명, 등록일
"""

import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.grac.or.kr/Statistics/SelfRateGameStatistics.aspx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": URL,
}

# 수집할 등급 목록 (드롭다운 값)
GRADES = [
    ("전체이용가",      "1"),
    ("12세이용가",      "2"),
    ("15세이용가",      "3"),
    ("청소년이용불가",  "4"),
    ("등급거부",        "5"),
]


def get_hidden_fields(soup: BeautifulSoup) -> dict:
    """ASP.NET hidden field 추출."""
    fields = {}
    for name in ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION",
                 "__VIEWSTATEENCRYPTED"]:
        el = soup.find("input", {"name": name})
        if el:
            fields[name] = el.get("value", "")
    return fields


def parse_table(soup: BeautifulSoup) -> list:
    """결과 테이블에서 게임 목록 파싱."""
    items = []

    # 일반적인 결과 테이블 탐색
    table = (soup.find("table", {"id": re.compile(r"Grid|List|Result", re.I)})
             or soup.find("table", {"class": re.compile(r"list|grid|board", re.I)})
             or soup.find("table"))

    if not table:
        return items

    rows = table.find_all("tr")
    # 첫 행은 헤더 — 컬럼 순서 파악
    if not rows:
        return items

    # 헤더로 컬럼 인덱스 추정
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]
    print(f"    헤더: {header}")

    col_map = {}
    keywords = {
        "gametitle": ["게임물명","게임명","명칭","타이틀"],
        "genre":     ["장르"],
        "grade":     ["등급","이용가"],
        "platform":  ["플랫폼","유통망"],
        "entname":   ["사업자","업체","신청자"],
        "rateddate": ["등록일","결정일","분류일"],
    }
    for key, kws in keywords.items():
        for i, h in enumerate(header):
            if any(kw in h for kw in kws):
                col_map[key] = i
                break

    for row in rows[1:]:
        cols = row.find_all("td")
        if not cols or len(cols) < 2:
            continue
        item = {}
        for key, idx in col_map.items():
            if idx < len(cols):
                item[key] = cols[idx].get_text(strip=True)
        if item.get("gametitle"):
            items.append(item)

    return items


def get_total_pages(soup: BeautifulSoup) -> int:
    """총 페이지 수 파악."""
    # 페이지네이션 영역 탐색
    pager = (soup.find(class_=re.compile(r"pager|pagination|page", re.I))
             or soup.find("div", {"id": re.compile(r"pager|page", re.I)}))

    if pager:
        page_links = pager.find_all("a")
        nums = []
        for a in page_links:
            t = a.get_text(strip=True)
            if t.isdigit():
                nums.append(int(t))
        if nums:
            return max(nums)

    # 텍스트에서 "총 N건" 파악
    text = soup.get_text()
    m = re.search(r"총\s*([\d,]+)\s*건", text)
    if m:
        total = int(m.group(1).replace(",",""))
        return max(1, (total + 9) // 10)  # 10건/페이지 가정

    return 1


def fetch_grade(session: requests.Session, grade_name: str,
                grade_val: str, start_date: str = "2018-01-01",
                end_date: str = "2026-12-31") -> list:
    """특정 등급의 전체 데이터 수집."""
    print(f"\n  [{grade_name}] 수집 시작...")
    items = []

    # ── 초기 GET ──────────────────────────────────────────────────────────────
    r = session.get(URL, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"    ⚠️ GET 실패: {r.status_code}")
        return items

    soup = BeautifulSoup(r.text, "html.parser")
    hidden = get_hidden_fields(soup)

    # ── 검색 POST ────────────────────────────────────────────────────────────
    # 폼 필드명은 페이지 실제 구조에 따라 조정 필요
    # (디버그 모드에서 실제 필드명 출력 후 수정)
    form_data = {
        **hidden,
        "__EVENTTARGET":   "",
        "__EVENTARGUMENT": "",
        # 아래 필드명은 실제 HTML 확인 후 수정하세요
        "ctl00$ContentPlaceHolder1$ddlGrade":     grade_val,
        "ctl00$ContentPlaceHolder1$txtStartDate": start_date,
        "ctl00$ContentPlaceHolder1$txtEndDate":   end_date,
        "ctl00$ContentPlaceHolder1$btnSearch":    "검색",
    }

    r = session.post(URL, data=form_data, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")

    # 디버그: 실제 폼 필드 출력 (첫 실행 시 확인)
    if not items:
        _debug_form(soup)

    total_pages = get_total_pages(soup)
    print(f"    총 {total_pages}페이지")

    page_items = parse_table(soup)
    items.extend(page_items)
    print(f"    1/{total_pages} — {len(page_items)}건")

    # ── 페이지네이션 ──────────────────────────────────────────────────────────
    for page in range(2, total_pages + 1):
        hidden = get_hidden_fields(soup)
        form_data = {
            **hidden,
            "__EVENTTARGET":   "ctl00$ContentPlaceHolder1$GridView1",  # GridView ID
            "__EVENTARGUMENT": f"Page${page}",
            "ctl00$ContentPlaceHolder1$ddlGrade":     grade_val,
            "ctl00$ContentPlaceHolder1$txtStartDate": start_date,
            "ctl00$ContentPlaceHolder1$txtEndDate":   end_date,
        }
        r = session.post(URL, data=form_data, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        page_items = parse_table(soup)
        if not page_items:
            print(f"    {page}/{total_pages} — 데이터 없음 (종료)")
            break
        items.extend(page_items)
        print(f"    {page}/{total_pages} — {len(page_items)}건 (누적 {len(items):,}건)")
        time.sleep(0.8)

    print(f"  [{grade_name}] 완료: {len(items):,}건")
    return items


def _debug_form(soup: BeautifulSoup):
    """실제 폼 필드명 출력 (첫 실행 시 확인용)."""
    print("\n    ── 폼 필드 디버그 ──")
    for el in soup.find_all(["input", "select"]):
        name = el.get("name","")
        if name and not name.startswith("__"):
            val = el.get("value","") or el.get("id","")
            print(f"      name='{name}' | type={el.name} | value='{val[:40]}'")
    print("    ─────────────────────\n")


def normalize_grade(grade_str: str) -> str:
    mapping = {
        "전체이용가":"전체이용가", "12세이용가":"12세이용가",
        "15세이용가":"15세이용가", "청소년이용불가":"청소년이용불가",
        "등급거부":"청소년이용불가",
        "4+":"전체이용가", "9+":"전체이용가",
        "12+":"12세이용가", "17+":"청소년이용불가",
    }
    for k, v in mapping.items():
        if k in str(grade_str):
            return v
    return "15세이용가"


def normalize_genre(genre_str: str) -> str:
    mapping = {
        "격투":"격투","액션":"액션","슈팅":"슈팅","FPS":"슈팅","TPS":"슈팅",
        "RPG":"RPG","롤플레잉":"RPG","MMORPG":"RPG",
        "퍼즐":"퍼즐","캐주얼":"퍼즐",
        "전략":"전략","시뮬레이션":"시뮬레이션","스포츠":"스포츠",
        "어드벤처":"어드벤처","레이싱":"기타","보드":"기타","교육":"기타",
    }
    for k, v in mapping.items():
        if k in str(genre_str):
            return v
    return "기타"


def run_crawl(start_date="2018-01-01", end_date="2026-12-31"):
    print("=" * 55)
    print("  GRAC 자체등급분류게임물 크롤러")
    print(f"  기간: {start_date} ~ {end_date}")
    print("=" * 55)

    session = requests.Session()
    session.headers.update({"User-Agent": HEADERS["User-Agent"]})

    all_items = []
    for grade_name, grade_val in GRADES:
        items = fetch_grade(session, grade_name, grade_val,
                            start_date, end_date)
        for item in items:
            item["grade_label"] = grade_name
        all_items.extend(items)
        time.sleep(1.0)

    if not all_items:
        print("\n⚠️ 수집된 데이터 없음.")
        print("  → 폼 필드명 디버그 출력 결과를 확인하고")
        print("    form_data 내 필드명을 실제 값으로 수정하세요.")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    print(f"\n전체 수집: {len(df):,}건")

    # 정규화
    if "grade" in df.columns:
        df["grade"] = df["grade"].apply(normalize_grade)
    if "genre" in df.columns:
        df["genre"] = df["genre"].apply(normalize_genre)

    return df


if __name__ == "__main__":
    df = run_crawl(start_date="2018-01-01", end_date="2026-12-31")

    if not df.empty:
        out = Path(__file__).parent / "grac_self_data.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n저장 완료 → {out}")
        print(df.head(3).to_string())

        try:
            from google.colab import files
            files.download(str(out))
        except ImportError:
            pass
    else:
        print("\n데이터 없음 — 아래 내용 확인 필요:")
        print("1. 디버그 출력에서 실제 폼 필드명 확인")
        print("2. fetch_grade() 내 form_data 필드명 수정")
        print("3. 재실행")
