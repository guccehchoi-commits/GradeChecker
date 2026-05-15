"""
GradeChecker — GRAC 자체등급분류게임물 크롤러 v2
대상: https://www.grac.or.kr/Statistics/SelfRateGameStatistics.aspx
코랩 실행: !python /content/files/grac_self_crawler.py
"""
import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.grac.or.kr"
URL      = f"{BASE_URL}/Statistics/SelfRateGameStatistics.aspx"
HEADERS  = {
    "User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":      "text/html,application/xhtml+xml",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Content-Type":"application/x-www-form-urlencoded",
    "Referer":     URL,
}

GRADES = [
    ("전체이용가",     "01"),
    ("12세이용가",     "04"),
    ("15세이용가",     "03"),
    ("청소년이용불가", "02"),
    ("등급거부",       "05"),
    ("4+",             "06"),
    ("9+",             "07"),
    ("12+",            "08"),
    ("17+",            "09"),
]

GRADE_NORM = {
    "전체이용가":"전체이용가","4+":"전체이용가","9+":"전체이용가",
    "12세이용가":"12세이용가","12+":"12세이용가",
    "15세이용가":"15세이용가",
    "청소년이용불가":"청소년이용불가","17+":"청소년이용불가",
    "등급거부":"청소년이용불가",
}

GENRE_NORM = {
    "격투":"격투","액션":"액션","슈팅":"슈팅","FPS":"슈팅","TPS":"슈팅",
    "롤플레잉":"RPG","RPG":"RPG","MMORPG":"RPG",
    "퍼즐":"퍼즐","캐주얼":"퍼즐","전략":"전략",
    "시뮬레이션":"시뮬레이션","스포츠":"스포츠","어드벤처":"어드벤처",
}


def get_hidden(soup):
    out = {}
    for name in ["__VIEWSTATE","__VIEWSTATEGENERATOR","__EVENTVALIDATION"]:
        el = soup.find("input", {"name": name})
        if el:
            out[name] = el.get("value","")
    return out


def build_form(soup, grade_val, start, end, ev_target="ctl00$ContentHolder$lbtnSearch", ev_arg=""):
    return {
        **get_hidden(soup),
        "__EVENTTARGET":   ev_target,
        "__EVENTARGUMENT": ev_arg,
        "__LASTFOCUS":     "",
        "ctl00$ContentHolder$tbGameTitle":    "",
        "ctl00$ContentHolder$ddlGrade":       grade_val,
        "ctl00$ContentHolder$tbRatingNbr":    "",
        "ctl00$ContentHolder$CalendarPicker$txtCalStartDate": start,
        "ctl00$ContentHolder$CalendarPicker$txtCalEndDate":   end,
        "ctl00$ContentHolder$CalendarPicker$ajxMaskStartDate_ClientState": "",
        "ctl00$ContentHolder$CalendarPicker$ajxMaskEndDate_ClientState":  "",
    }


def parse_table(soup):
    table, grid_id = None, None
    for t in soup.find_all("table"):
        tid = t.get("id","")
        if any(k in tid for k in ["Grid","List","Result","grid","list"]):
            table, grid_id = t, tid
            break
    if table is None:
        for t in soup.find_all("table"):
            if len(t.find_all("tr")) > 2:
                table = t
                break
    if table is None:
        return [], None

    rows = table.find_all("tr")
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]

    COL_KEYS = {
        "gametitle": ["게임물명","게임명","명칭"],
        "genre":     ["장르"],
        "grade":     ["등급"],
        "platform":  ["플랫폼","유통"],
        "entname":   ["사업자","업체","신청자"],
        "rateddate": ["등록일","결정일","분류일"],
    }
    col_map = {}
    for key, kws in COL_KEYS.items():
        for i, h in enumerate(header):
            if any(kw in h for kw in kws):
                col_map[key] = i
                break

    items = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if not cols or len(cols) < 2:
            continue
        if col_map:
            item = {k: cols[i].get_text(strip=True)
                    for k, i in col_map.items() if i < len(cols)}
        else:
            item = {
                "gametitle": cols[1].get_text(strip=True) if len(cols) > 1 else "",
                "genre":     cols[2].get_text(strip=True) if len(cols) > 2 else "",
                "grade":     cols[3].get_text(strip=True) if len(cols) > 3 else "",
                "platform":  cols[4].get_text(strip=True) if len(cols) > 4 else "",
                "entname":   cols[5].get_text(strip=True) if len(cols) > 5 else "",
                "rateddate": cols[6].get_text(strip=True) if len(cols) > 6 else "",
            }
        if item.get("gametitle"):
            items.append(item)
    return items, grid_id


def get_total_pages(soup):
    text = soup.get_text()
    m = re.search(r"총\s*([\d,]+)\s*건", text)
    if m:
        return max(1, (int(m.group(1).replace(",","")) + 9) // 10)
    for cls in ["pager","pagination"]:
        p = soup.find(class_=re.compile(cls, re.I))
        if p:
            nums = [int(a.get_text()) for a in p.find_all("a") if a.get_text().strip().isdigit()]
            if nums: return max(nums)
    return 1


def fetch_year(session, grade_val, grade_name, year):
    start, end = f"{year}-01-01", f"{year}-12-31"

    r    = session.get(URL, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    form = build_form(soup, grade_val, start, end)
    r    = session.post(URL, data=form, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")

    items, grid_id = parse_table(soup)
    if not items and "검색된 내용이 없습니다" in soup.get_text():
        return []

    total = get_total_pages(soup)
    for page in range(2, total + 1):
        gt = grid_id.replace("_","$") if grid_id else "ctl00$ContentHolder$GridView1"
        f  = build_form(soup, grade_val, start, end, ev_target=gt, ev_arg=f"Page${page}")
        r  = session.post(URL, data=f, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        pg, grid_id = parse_table(soup)
        if not pg: break
        items.extend(pg)
        time.sleep(0.5)
    return items


def norm_genre(s):
    for k, v in GENRE_NORM.items():
        if k in str(s): return v
    return "기타"

def norm_grade(s):
    for k, v in GRADE_NORM.items():
        if k in str(s): return v
    return "15세이용가"


if __name__ == "__main__":
    print("="*55)
    print("  GRAC 자체등급분류게임물 크롤러 v2")
    print("="*55)

    session = requests.Session()
    all_items = []

    for grade_name, grade_val in GRADES:
        print(f"\n[{grade_name}]")
        grade_items = []
        for year in range(2018, 2027):
            items = fetch_year(session, grade_val, grade_name, year)
            if items:
                print(f"  {year}년: {len(items):,}건")
                grade_items.extend(items)
            time.sleep(0.5)
        print(f"  소계: {len(grade_items):,}건")
        for item in grade_items:
            item["grade_label"] = grade_name
        all_items.extend(grade_items)

    if not all_items:
        print("\n데이터 없음 — GridView ID 확인 필요")
        print("코랩에서 실행: soup.find_all('table') 로 ID 확인")
    else:
        df = pd.DataFrame(all_items)
        if "genre" in df.columns: df["genre"] = df["genre"].apply(norm_genre)
        if "grade" in df.columns: df["grade"] = df["grade"].apply(norm_grade)

        out = Path(__file__).parent / "grac_self_data.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n저장 완료: {len(df):,}건 → {out}")
        print(df.head(3).to_string())

        try:
            from google.colab import files
            files.download(str(out))
        except ImportError:
            pass
