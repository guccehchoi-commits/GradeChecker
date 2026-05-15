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

# ── 설정 (여기만 바꾸면 됨) ───────────────────
START_DATE = "2026-01-01"
END_DATE   = "2026-01-30"

# 수집할 등급 목록
GRADES = [
    "전체",
    "전체이용가",
    "12세이용가",
    "15세이용가",
    "청소년이용불가",
    "등급거부",
    "4+",
    "9+",
    "12+",
    "17+",
]

BASE_URL     = "https://www.grac.or.kr/Game/SelfGrading/SelfGradeList.aspx"
PAGE_TIMEOUT = 90_000

# ── 등급 이미지 → 텍스트 변환 ─────────────────
GRADE_IMG_MAP = {
    "rating_all":  "전체이용가",
    "rating_12":   "12세이용가",
    "rating_15":   "15세이용가",
    "rating_18":   "청소년이용불가",
    "rating_test": "등급거부",
}

def img_to_grade(src: str) -> str:
    for key, val in GRADE_IMG_MAP.items():
        if key in src:
            return val
    return "미확인"

# ── 결과 테이블 파싱 ───────────────────────────
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
        name      = cell.get("게임물명", tds[1]).get_text(strip=True)
        date_     = cell.get("등급분류", tds[2]).get_text(strip=True)
        genre     = cell.get("장르",     tds[3]).get_text(strip=True)
        grade_td  = cell.get("등급",     tds[4])
        grade_img = grade_td.find("img")
        grade     = img_to_grade(grade_img["src"]) if grade_img else "미확인"
        rows.append({"게임물명": name, "등급분류일자": date_, "장르": genre, "등급": grade})
    return rows

# ── 검색 + 페이지 수집 ────────────────────────
async def search_and_collect(page) -> list:
    try:
        async with page.expect_navigation(wait_until="networkidle", timeout=PAGE_TIMEOUT):
            await page.click("#ctl00_ContentHolder_lbtnSearch")
    except PWTimeout:
        await page.click("#ctl00_ContentHolder_lbtnSearch", no_wait_after=True)
        try:
            await page.wait_for_selector(
                "tbody tr td[data-label='게임물명']", timeout=PAGE_TIMEOUT
            )
        except PWTimeout:
            return []

    all_rows = []
    page_num = 1
    while True:
        rows = parse_rows(await page.content())
        all_rows.extend(rows)
        print(f"      {page_num}p: {len(rows)}건")

        next_btn = page.locator("a").filter(has_text="다음")
        if await next_btn.count() == 0:
            break
        try:
            async with page.expect_navigation(wait_until="networkidle", timeout=PAGE_TIMEOUT):
                await next_btn.first.click()
            page_num += 1
        except PWTimeout:
            break

    return all_rows

# ── 메인 크롤러 ───────────────────────────────
async def crawl():
    all_results = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page    = await browser.new_page(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ))

        # 페이지 최초 로딩
        print("[1] 페이지 로딩 중...")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=PAGE_TIMEOUT)

        # 등급 드롭다운 ID 자동 탐지
        grade_select_id = await page.evaluate("""() => {
            const selects = document.querySelectorAll('select');
            for (const sel of selects) {
                const opts = [...sel.options].map(o => o.text.trim());
                if (opts.includes('전체이용가') || opts.includes('전체')) return sel.id;
            }
            return null;
        }""")
        print(f"    등급 드롭다운 ID: {grade_select_id!r}")

        # 날짜 입력
        print(f"[2] 날짜: {START_DATE} ~ {END_DATE}")
        await page.fill("#ctl00_ContentHolder_CalendarPicker_txtCalStartDate", START_DATE)
        await page.press("#ctl00_ContentHolder_CalendarPicker_txtCalStartDate", "Tab")
        await asyncio.sleep(0.2)
        await page.fill("#ctl00_ContentHolder_CalendarPicker_txtCalEndDate", END_DATE)
        await page.press("#ctl00_ContentHolder_CalendarPicker_txtCalEndDate", "Tab")
        await asyncio.sleep(0.2)

        # 등급별 반복 수집
        print(f"[3] 등급별 수집 시작 ({len(GRADES)}개 등급)\n")
        for grade in GRADES:
            print(f"  [{grade}] 검색 중...")

            if grade_select_id:
                try:
                    await page.select_option(f"#{grade_select_id}", label=grade)
                    await asyncio.sleep(0.2)
                except Exception:
                    try:
                        await page.select_option(f"#{grade_select_id}", value=grade)
                        await asyncio.sleep(0.2)
                    except Exception as e:
                        print(f"    ⚠️ 등급 선택 실패: {e}")

            # 날짜 재입력 (포스트백 후 초기화 방지)
            await page.fill("#ctl00_ContentHolder_CalendarPicker_txtCalStartDate", START_DATE)
            await page.fill("#ctl00_ContentHolder_CalendarPicker_txtCalEndDate",   END_DATE)
            await asyncio.sleep(0.2)

            rows = await search_and_collect(page)
            print(f"    → {len(rows)}건\n")
            all_results.extend(rows)

        await browser.close()

    if not all_results:
        print("❌ 수집 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["등급분류일자"] = pd.to_datetime(df["등급분류일자"], errors="coerce")
    df = df.drop_duplicates().sort_values("등급분류일자").reset_index(drop=True)
    print(f"✅ 최종: {len(df)}건 (중복 제거 후)")
    return df



# ── 실행 ──────────────────────────────────────
import nest_asyncio
nest_asyncio.apply()
df = asyncio.run(crawl())
df
