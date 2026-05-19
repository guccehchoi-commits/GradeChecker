"""
GRAC 자체등급분류게임물 크롤러 — 로컬 실행용
실행: python grac_self_crawler_local.py
"""

import asyncio, re, calendar
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright

URL = "https://www.grac.or.kr/Statistics/SelfRateGameStatistics.aspx"

GRADES = [
    ("전체이용가","01"),("12세이용가","04"),("15세이용가","03"),
    ("청소년이용불가","02"),("등급거부","05"),
    ("4+","06"),("9+","07"),("12+","08"),("17+","09"),
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

def norm_genre(s):
    for k, v in GENRE_NORM.items():
        if k in str(s): return v
    return "기타"

def norm_grade(s):
    for k, v in GRADE_NORM.items():
        if k in str(s): return v
    return "15세이용가"

def parse_html(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    items = []

    # data-label 속성 기반 파싱
    rows = soup.select("table.statistics tr, table[class*='board'] tr")
    if not rows:
        rows = soup.find_all("tr", id=lambda x: x and "rptGradeDoc" in x)

    for row in rows:
        cols = row.find_all("td")
        if not cols:
            continue

        item = {}
        for td in cols:
            label = td.get("data-label", "").strip()
            if "번호" in label:
                item["rateno"] = td.get_text(strip=True)
            elif "게임물명" in label or "게임명" in label:
                item["gametitle"] = td.get_text(strip=True)
            elif "일자" in label or "분류일" in label:
                item["rateddate"] = td.get_text(strip=True)
            elif "장르" in label:
                item["genre"] = td.get_text(strip=True)
            elif "등급" in label and "번호" not in label and "분류" not in label:
                img = td.find("img")
                item["grade"] = img.get("alt", "") if img else td.get_text(strip=True)
            elif "내용" in label:
                item["content"] = td.get_text(strip=True)

        if item.get("gametitle"):
            items.append(item)

    return items

def get_total_pages(html):
    soup = BeautifulSoup(html, "html.parser")
    m = re.search(r"총\s*([\d,]+)\s*건", soup.get_text())
    if m:
        return max(1, (int(m.group(1).replace(",","")) + 9) // 10)
    return 1

async def search_and_get(page, grade_val, start, end):
    await page.evaluate(f"""
        document.querySelector("select[name='ctl00$ContentHolder$ddlGrade']").value = '{grade_val}';
        document.querySelector("input[name='ctl00$ContentHolder$CalendarPicker$txtCalStartDate']").value = '{start}';
        document.querySelector("input[name='ctl00$ContentHolder$CalendarPicker$txtCalEndDate']").value = '{end}';
    """)
    await asyncio.sleep(0.5)
    async with page.expect_navigation(wait_until="domcontentloaded", timeout=60000):
        await page.click("#ctl00_ContentHolder_lbtnSearch")
    await asyncio.sleep(2)
    return await page.content()

async def go_to_page(page, page_num):
    try:
        links = await page.query_selector_all("a")
        for link in links:
            txt = (await link.inner_text()).strip()
            if txt == str(page_num):
                async with page.expect_navigation(wait_until="domcontentloaded", timeout=30000):
                    await link.click()
                await asyncio.sleep(1)
                return True
        return False
    except:
        return False

async def fetch_grade(browser, grade_name, grade_val, start_year, end_year, max_items=600):
    all_items = []
    context = await browser.new_context()
    page = await context.new_page()
    page.set_default_timeout(60000)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            start = f"{year}-{month:02d}-01"
            end   = f"{year}-{month:02d}-{last_day:02d}"

            try:
                await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(1)
                html  = await search_and_get(page, grade_val, start, end)
                if "검색된 내용이 없습니다" in html:
                    continue
                items = parse_html(html)
                total = get_total_pages(html)
                all_items.extend(items)
                for pg in range(2, total + 1):
                    ok = await go_to_page(page, pg)
                    if not ok: break
                    pg_items = parse_html(await page.content())
                    if not pg_items: break
                    all_items.extend(pg_items)
                    await asyncio.sleep(0.5)
                if items or total > 1:
                    print(f"  {year}-{month:02d}: {len(items)}건")
            except Exception as e:
                print(f"  {year}-{month:02d}: 오류 — {type(e).__name__}")
                continue
            await asyncio.sleep(1)

    await context.close()
    return all_items

async def main(start_year=2018, end_year=2026):
    print("="*55)
    print(f"  GRAC 자체등급분류 크롤러 | {start_year}~{end_year}년")
    print("="*55)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        all_items = []
        for grade_name, grade_val in GRADES:
            print(f"\n[{grade_name}]")
            items = await fetch_grade(browser, grade_name, grade_val, start_year, end_year)
            for item in items: item["grade_label"] = grade_name
            all_items.extend(items)
            print(f"  완료: {len(items):,}건")
        await browser.close()

    if not all_items:
        print("\n수집 데이터 없음"); return

    df = pd.DataFrame(all_items)
    if "genre" in df.columns: df["genre"] = df["genre"].apply(norm_genre)
    if "grade" in df.columns: df["grade"] = df["grade"].apply(norm_grade)
    df = df.drop_duplicates(subset=["rateno"]).reset_index(drop=True)

    out = Path(__file__).parent / "grac_self_data_2.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n완료: {len(df):,}건 → {out}")

if __name__ == "__main__":
    asyncio.run(main(start_year=2022, end_year=2026))
