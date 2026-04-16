import time
import warnings
from asyncio import wait_for
from http import HTTPStatus
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from playwright.async_api import Page
from playwright_captcha import CaptchaType

from src.consts import CHALLENGE_TITLES
from src.models import (
    HealthcheckResponse,
    LinkRequest,
    LinkResponse,
    Solution,
)
from src.utils import CamoufoxDepClass, TimeoutTimer, get_camoufox, logger

warnings.filterwarnings("ignore", category=SyntaxWarning)


router = APIRouter()

CamoufoxDep = Annotated[CamoufoxDepClass, Depends(get_camoufox)]


@router.get("/", include_in_schema=False)
def read_root():
    """Redirect to /docs."""
    logger.debug("Redirecting to /docs")
    return RedirectResponse(url="/docs", status_code=301)


@router.get("/health")
async def health_check(sb: CamoufoxDep):
    """Health check endpoint."""
    health_check_request = await read_item(
        LinkRequest.model_construct(url="https://google.com"),
        sb,
    )

    if health_check_request.solution.status != HTTPStatus.OK:
        raise HTTPException(
            status_code=500,
            detail="Health check failed",
        )

    return HealthcheckResponse(user_agent=health_check_request.solution.user_agent)


@router.post("/v1")
async def read_item(request: LinkRequest, dep: CamoufoxDep) -> LinkResponse:
    """Handle POST requests."""
    start_time = int(time.time() * 1000)

    timer = TimeoutTimer(duration=request.max_timeout)

    request.url = request.url.replace('"', "").strip()

    if request.cmd == "request.download":
        return await _handle_download(request, dep, timer, start_time)

    return await _handle_get(request, dep, timer, start_time)


async def _solve_challenge(page: Page, dep: CamoufoxDepClass, timer: TimeoutTimer):
    """Solve Cloudflare challenge if detected on current page."""
    title = await page.title()
    if title not in CHALLENGE_TITLES:
        return

    logger.info(f"Challenge detected: {title}")
    try:
        await wait_for(
            dep.solver.solve_captcha(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                captcha_container=page,
                captcha_type=CaptchaType.CLOUDFLARE_INTERSTITIAL,
                wait_checkbox_attempts=3,
                wait_checkbox_delay=0.5,
            ),
            timeout=timer.remaining(),
        )
        logger.info("Challenge solved!")
    except TimeoutError:
        logger.warning("ClickSolver timed out, waiting for JS challenge...")
        # Fallback: wait for JS-only challenges to auto-complete
        for _ in range(15):
            if await page.title() not in CHALLENGE_TITLES:
                logger.info("JS challenge resolved!")
                return
            await page.wait_for_timeout(1000)
        raise


async def _wait_for_js_challenge(page: Page):
    """Wait for non-Cloudflare JS challenges (AWS WAF, etc.)."""
    wait_start = time.time()
    max_wait = 15

    while time.time() - wait_start < max_wait:
        try:
            content = await page.content()
        except Exception:
            content = ''

        has_waf = 'awsWafCookieDomainList' in content
        has_challenge = ('challenge' in content.lower()[:2000]
                         and len(content) < 5000
                         and '<form' not in content.lower()[:2000])

        if has_waf or has_challenge:
            if time.time() - wait_start < 1:
                logger.info(f"JS challenge detected (WAF={has_waf})")
            await page.wait_for_timeout(1000)
        else:
            break

    if time.time() - wait_start >= 2:
        logger.info(f"JS challenge wait completed in {time.time() - wait_start:.1f}s")


async def _handle_get(
        request: LinkRequest, dep: CamoufoxDepClass,
        timer: TimeoutTimer, start_time: int) -> LinkResponse:
    """Handle request.get — navigate, solve challenge, return page content."""
    try:
        page_request = await dep.page.goto(
            request.url, timeout=timer.remaining() * 1000
        )
        status = page_request.status if page_request else HTTPStatus.OK
        await dep.page.wait_for_load_state(
            state="domcontentloaded", timeout=timer.remaining() * 1000
        )
        await dep.page.wait_for_load_state(
            "networkidle", timeout=timer.remaining() * 1000
        )

        await _solve_challenge(dep.page, dep, timer)

    except TimeoutError as e:
        logger.error("Timed out while solving the challenge")
        raise HTTPException(
            status_code=408,
            detail="Timed out while solving the challenge",
        ) from e

    cookies = await dep.context.cookies()

    return LinkResponse(
        message="Success",
        solution=Solution(
            user_agent=await dep.page.evaluate("navigator.userAgent"),
            url=dep.page.url,
            status=status,
            cookies=cookies,
            headers=page_request.headers if page_request else {},
            response=await dep.page.content(),
        ),
        start_timestamp=start_time,
    )


async def _handle_download(
        request: LinkRequest, dep: CamoufoxDepClass,
        timer: TimeoutTimer, start_time: int) -> LinkResponse:
    """Handle request.download — navigate, solve challenge, download file via JS fetch."""
    page = dep.page

    # Step 1: Navigate to URL
    logger.info(f"[Download] Navigating to {request.url[:80]}...")
    try:
        await page.goto(request.url, timeout=min(timer.remaining(), 30) * 1000,
                        wait_until='domcontentloaded')
    except Exception as e:
        logger.debug(f"[Download] Navigation error (may be expected): {str(e)[:80]}")

    # Step 2: Solve Cloudflare challenge if present
    try:
        await _solve_challenge(page, dep, timer)
    except TimeoutError:
        logger.error("[Download] Challenge solving timed out")
        raise HTTPException(status_code=408, detail="Challenge solving timed out")

    # Step 3: Wait for non-Cloudflare JS challenges (AWS WAF, etc.)
    await _wait_for_js_challenge(page)

    # Step 4: Determine fetch URL (handle CDN redirects and native downloads)
    current_url = page.url
    target_origin = f"{urlparse(request.url).scheme}://{urlparse(request.url).netloc}"
    current_origin = f"{urlparse(current_url).scheme}://{urlparse(current_url).netloc}"

    is_internal = current_url.startswith(('chrome://', 'about:', 'data:'))

    if is_internal:
        # Browser triggered native download, tab is empty
        logger.info(f"[Download] Native download detected, navigating to {target_origin}")
        await page.goto(target_origin, wait_until='domcontentloaded')
        await _solve_challenge(page, dep, timer)
        fetch_url = request.url
    elif current_origin != target_origin:
        # Redirected to CDN domain, fetch from there (same-origin)
        fetch_url = current_url
        logger.info(f"[Download] Redirected to {current_origin}, will fetch: {fetch_url[:80]}")
    else:
        fetch_url = request.url

    # Step 5: JS fetch the file
    logger.info("[Download] Downloading via JS fetch...")
    try:
        js_result = await page.evaluate("""
            async (url) => {
                try {
                    const r = await fetch(url);
                    if (!r.ok) return {error: 'HTTP ' + r.status + ' ' + r.statusText};
                    const blob = await r.blob();
                    return new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve({
                            data: reader.result.split(',')[1],
                            size: blob.size,
                            type: blob.type
                        });
                        reader.onerror = () => resolve({error: 'FileReader error'});
                        reader.readAsDataURL(blob);
                    });
                } catch (e) {
                    return {error: e.toString()};
                }
            }
        """, fetch_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JS fetch failed: {e}")

    if not isinstance(js_result, dict) or 'error' in js_result:
        error = js_result.get('error', 'Unknown') if isinstance(js_result, dict) else str(js_result)
        raise HTTPException(status_code=500, detail=f"JS fetch error: {error}")

    file_base64 = js_result.get('data')
    if not file_base64:
        raise HTTPException(status_code=500, detail="JS fetch returned no data")

    logger.info(f"[Download] Success: {js_result.get('size', '?')} bytes, "
                f"type: {js_result.get('type', '?')}")

    cookies = await dep.context.cookies()

    return LinkResponse(
        message="File downloaded successfully",
        solution=Solution(
            user_agent=await page.evaluate("navigator.userAgent"),
            url=current_url,
            status=HTTPStatus.OK,
            cookies=cookies,
            file_base64=file_base64,
        ),
        start_timestamp=start_time,
    )
