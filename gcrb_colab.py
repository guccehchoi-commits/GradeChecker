# ──────────────────────────────────────────────
# 셀 1: 설치 (처음 한 번만 실행)
# ──────────────────────────────────────────────
# !pip install playwright -q
# !playwright install chromium
# !playwright install-deps chromium


# ──────────────────────────────────────────────
# 셀 2: 크롤러 실행
# ──────────────────────────────────────────────
import asyncio
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── 설정 ──────────────────────────────────────
BASE_URL    = "https://www.grac.or.kr/Game/SelfGrading/SelfGradeList.aspx"
START_DATE  = "2026-05-01"   # 원하는 시작일
END_DATE    = "2026-05-15"   # 원하는 종료일
PAGE_TIMEOUT = 90_000        # ms

GRADE_MAP = {
    "rating_all":  "전체이용가",
    "rating_12":   "12세이용가",
    "rating_15":   "15세이용가",
    "rating_18":   "청소년이용불가",
    "rating_test": "등급분류거부",
}

def img_to_grade(src: str) -> str:
    for key, val in GRADE_MAP.items():
        if key in src:
            return val
    return "미확인"

def parse_rows(html: str) -> list:
    from bs4 import BeautifulSoup
    soup  = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    if not tbody:
        return []
    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        cell      = {td.get("data-label", ""): td for td in tds}
        name      = cell.get("게임물명",   tds[1]).get_text(strip=True)
        date_     = cell.get("등급분류",   tds[2]).get_text(strip=True)
        genre     = cell.get("장르",       tds[3]).get_text(strip=True)
        grade_td  = cell.get("등급",       tds[4])
        grade_img = grade_td.find("img")
        grade     = img_to_grade(grade_img["src"]) if grade_img else "미확인"
        rows.append({"게임물명": name, "등급분류일자": date_, "장르": genre, "등급": grade})
    return rows

async def crawl(start=START_DATE, end=END_DATE):
    all_rows = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page    = await browser.new_page(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ))

        # 1. 페이지 열기
        print(f"[1] 페이지 로딩 중...")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=PAGE_TIMEOUT)
        print("    완료")

        # 2. 날짜 입력
        print(f"[2] 날짜 입력: {start} ~ {end}")
        await page.fill("#ctl00_ContentHolder_CalendarPicker_txtCalStartDate", start)
        await page.press("#ctl00_ContentHolder_CalendarPicker_txtCalStartDate", "Tab")
        await asyncio.sleep(0.3)
        await page.fill("#ctl00_ContentHolder_CalendarPicker_txtCalEndDate", end)
        await page.press("#ctl00_ContentHolder_CalendarPicker_txtCalEndDate", "Tab")
        await asyncio.sleep(0.3)

        # 3. 검색 클릭
        print("[3] 검색 중...")
        try:
            async with page.expect_navigation(wait_until="networkidle", timeout=PAGE_TIMEOUT):
                await page.click("#ctl00_ContentHolder_lbtnSearch")
            print("    완료")
        except PWTimeout:
            print("    navigation 타임아웃 → 결과 테이블 직접 대기")
            await page.click("#ctl00_ContentHolder_lbtnSearch", no_wait_after=True)
            try:
                await page.wait_for_selector("tbody tr td[data-label='게임물명']", timeout=PAGE_TIMEOUT)
                print("    결과 확인")
            except PWTimeout:
                print("    ❌ 결과 로딩 실패")
                await browser.close()
                return pd.DataFrame()

        # 4. 전 페이지 수집
        print("[4] 수집 중...")
        page_num = 1
        while True:
            html  = await page.content()
            rows  = parse_rows(html)
            all_rows.extend(rows)
            print(f"    {page_num}페이지: {len(rows)}건 (누적 {len(all_rows)}건)")

            # 다음 페이지 버튼 확인
            next_btn = page.locator("a").filter(has_text="다음")
            if await next_btn.count() == 0:
                break
            try:
                async with page.expect_navigation(wait_until="networkidle", timeout=PAGE_TIMEOUT):
                    await next_btn.first.click()
                page_num += 1
            except PWTimeout:
                print("    마지막 페이지")
                break

        await browser.close()

    if not all_rows:
        print("❌ 수집 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["등급분류일자"] = pd.to_datetime(df["등급분류일자"], errors="coerce")
    df = df.sort_values("등급분류일자").reset_index(drop=True)
    print(f"\n✅ 완료: {len(df)}건")
    return df


# 코랩에서는 await 직접 사용
df = await crawl()
df
