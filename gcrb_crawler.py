"""
게임물관리위원회(GCRB) 자체등급분류 결과 크롤러
- 기간 지정 → 검색 → 결과 수집 → DataFrame 반환
- 수집 컬럼: 게임물명, 등급분류일자, 장르, 등급
"""

import asyncio
import re
import pandas as pd
from datetime import date
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── 설정 ─────────────────────────────────────────────────────────────────────
BASE_URL    = "https://www.grac.or.kr/Game/SelfGrading/SelfGradeList.aspx"
START_DATE  = "2026-05-01"   # YYYY-MM-DD
END_DATE    = "2026-05-15"   # YYYY-MM-DD
PAGE_TIMEOUT = 90_000        # ms (포스트백 응답 대기)

# 등급 이미지 파일명 → 한글 등급 매핑
GRADE_MAP = {
    "rating_all": "전체이용가",
    "rating_12":  "12세이용가",
    "rating_15":  "15세이용가",
    "rating_18":  "청소년이용불가",
    "rating_test":"등급분류거부",
}

def img_to_grade(src: str) -> str:
    """이미지 src에서 등급 추출"""
    for key, val in GRADE_MAP.items():
        if key in src:
            return val
    return "미확인"

def parse_rows(page_html: str) -> list[dict]:
    """tbody의 tr 파싱 → [{'게임물명':..., '등급분류일자':..., '장르':..., '등급':...}, ...]"""
    from bs4 import BeautifulSoup
    soup  = BeautifulSoup(page_html, "html.parser")
    tbody = soup.find("tbody")
    if not tbody:
        return []

    results = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        # data-label 기준으로 안전하게 추출
        cell = {td.get("data-label", ""): td for td in tds}

        name  = cell.get("게임물명", tds[1]).get_text(strip=True)
        date_ = cell.get("등급분류", tds[2]).get_text(strip=True)
        genre = cell.get("장르",     tds[3]).get_text(strip=True)

        grade_td  = cell.get("등급", tds[4])
        grade_img = grade_td.find("img")
        grade     = img_to_grade(grade_img["src"]) if grade_img else "미확인"

        results.append({
            "게임물명":    name,
            "등급분류일자": date_,
            "장르":       genre,
            "등급":       grade,
        })
    return results


async def set_date(page, field_id: str, date_str: str):
    """
    AJAX 달력 대신 텍스트 input에 직접 날짜 입력.
    field_id: 날짜 input의 id (예: ctl00_ContentHolder_CalendarPicker_tbDate)
    date_str: 'YYYY-MM-DD' 형식
    """
    # 입력 필드 클리어 후 날짜 입력
    await page.fill(f"#{field_id}", "")
    await page.fill(f"#{field_id}", date_str)
    await page.press(f"#{field_id}", "Tab")   # 달력 닫기
    await asyncio.sleep(0.3)


async def crawl_all_pages(page, all_rows: list):
    """현재 페이지 파싱 후 '다음 페이지'가 있으면 재귀적으로 수집"""
    html  = await page.content()
    rows  = parse_rows(html)
    all_rows.extend(rows)
    print(f"  현재 페이지 {len(rows)}건 수집 (누적 {len(all_rows)}건)")

    # 다음 페이지 버튼 탐색 (페이지네이션 링크 패턴)
    next_btn = page.locator("a[href*='__doPostBack']").filter(has_text="다음")
    if await next_btn.count() > 0:
        try:
            async with page.expect_navigation(
                wait_until="networkidle", timeout=PAGE_TIMEOUT
            ):
                await next_btn.first.click()
            await crawl_all_pages(page, all_rows)
        except PWTimeout:
            print("  ⚠️ 다음 페이지 타임아웃 — 여기까지 수집")


async def crawl(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    all_rows: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx     = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()

        # ── 1. 페이지 열기 ─────────────────────────────────────────────────
        print(f"[1] 페이지 로딩: {BASE_URL}")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=PAGE_TIMEOUT)

        # ── 2. 날짜 입력 ───────────────────────────────────────────────────
        # 실제 input id는 페이지 소스에서 확인 후 수정
        print(f"[2] 날짜 입력: {start} ~ {end}")
        try:
            # 시작일
            await set_date(page, "ctl00_ContentHolder_CalendarPicker_txtCalStartDate", start)
            # 종료일
            await set_date(page, "ctl00_ContentHolder_CalendarPicker_txtCalEndDate", end)
        except Exception as e:
            print(f"  ⚠️ 날짜 입력 실패: {e}")
            await browser.close()
            return pd.DataFrame()

        # ── 3. 검색 클릭 (__doPostBack 포스트백) ──────────────────────────
        print("[3] 검색 버튼 클릭 (포스트백 대기 중...)")
        try:
            # expect_navigation 으로 포스트백 완료 감지
            async with page.expect_navigation(
                wait_until="networkidle", timeout=PAGE_TIMEOUT
            ):
                await page.click("#ctl00_ContentHolder_lbtnSearch")
            print("  ✅ 검색 완료")
        except PWTimeout:
            # fallback: navigation 없이 클릭만 하고, 결과 테이블 대기
            print("  ⚠️ navigation 타임아웃 → 결과 테이블 직접 대기")
            await page.click(
                "#ctl00_ContentHolder_lbtnSearch",
                no_wait_after=True
            )
            try:
                await page.wait_for_selector(
                    "tbody tr td[data-label='게임물명']",
                    timeout=PAGE_TIMEOUT
                )
                print("  ✅ 결과 테이블 확인")
            except PWTimeout:
                print("  ❌ 결과 로딩 실패")
                await browser.close()
                return pd.DataFrame()

        # ── 4. 전 페이지 수집 ─────────────────────────────────────────────
        print("[4] 결과 수집 시작")
        await crawl_all_pages(page, all_rows)

        await browser.close()

    # ── 5. DataFrame 정리 ─────────────────────────────────────────────────
    if not all_rows:
        print("❌ 수집 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["등급분류일자"] = pd.to_datetime(df["등급분류일자"], errors="coerce")
    df = df.sort_values("등급분류일자").reset_index(drop=True)

    print(f"\n✅ 최종 수집: {len(df)}건")
    print(df.head(10).to_string(index=False))
    return df


# ── 실행 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = asyncio.run(crawl(START_DATE, END_DATE))
    if not df.empty:
        out = f"gcrb_{START_DATE}_{END_DATE}.xlsx"
        df.to_excel(out, index=False)
        print(f"\n💾 저장 완료: {out}")
