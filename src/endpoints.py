import asyncio
import time
import warnings
from asyncio import wait_for
from base64 import b64encode
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
    """Handle request.download — download file using Playwright response interception.

    Strategy:
    1. Navigate to base URL to pass Cloudflare challenge
    2. Try JS fetch from same-origin page (works for most publishers)
    3. If JS fetch fails (CORS for CDN-redirected publishers), use Playwright
       response interception to capture the PDF binary directly from navigation
    """
    page = dep.page
    target_url = request.url
    base_url = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"

    # Step 1: Navigate to base URL to establish session and pass Cloudflare
    logger.info(f"[Download] Navigating to base URL {base_url} ...")
    try:
        await page.goto(base_url, timeout=min(timer.remaining(), 30) * 1000,
                        wait_until='domcontentloaded')
    except Exception as e:
        logger.debug(f"[Download] Base URL navigation error: {str(e)[:80]}")

    # Step 2: Solve Cloudflare challenge if present
    try:
        await _solve_challenge(page, dep, timer)
    except TimeoutError:
        logger.error("[Download] Challenge solving timed out")
        raise HTTPException(status_code=408, detail="Challenge solving timed out")

    # Step 3: Wait for non-Cloudflare JS challenges
    await _wait_for_js_challenge(page)

    # Step 4: Try JS fetch first (same-origin, fast)
    logger.info(f"[Download] Trying JS fetch: {target_url[:80]}...")
    file_base64 = await _js_fetch_file(page, target_url)

    # Step 5: If JS fetch failed, use Playwright response interception
    # This handles CDN-redirected publishers (OUP→silverchair, ScienceDirect→CDN)
    # and avoids Firefox PDF viewer by capturing the response before rendering
    if file_base64 is None:
        logger.info("[Download] JS fetch failed, trying response interception...")
        file_base64 = await _intercept_download(page, dep, timer, target_url)

    if not file_base64:
        raise HTTPException(status_code=500, detail="Failed to download file")

    cookies = await dep.context.cookies()

    return LinkResponse(
        message="File downloaded successfully",
        solution=Solution(
            user_agent=await page.evaluate("navigator.userAgent"),
            url=page.url,
            status=HTTPStatus.OK,
            cookies=cookies,
            file_base64=file_base64,
        ),
        start_timestamp=start_time,
    )


async def _js_fetch_file(page: Page, url: str) -> str | None:
    """Try to download file via JS fetch. Returns base64 string or None on failure."""
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
        """, url)
    except Exception as e:
        logger.warning(f"[Download] JS fetch exception: {e}")
        return None

    if not isinstance(js_result, dict) or 'error' in js_result:
        error = js_result.get('error', 'Unknown') if isinstance(js_result, dict) else str(js_result)
        logger.warning(f"[Download] JS fetch error: {error}")
        return None

    data = js_result.get('data')
    if data:
        logger.info(f"[Download] JS fetch success: {js_result.get('size', '?')} bytes, "
                    f"type: {js_result.get('type', '?')}")
    return data


async def _intercept_download(
        page: Page, dep: CamoufoxDepClass, timer: TimeoutTimer,
        url: str) -> str | None:
    """Download file by intercepting the HTTP response via Playwright.

    Navigates to the URL and captures the response body before Firefox's PDF
    viewer can intercept it. This works for CDN-redirected URLs too.
    """

    captured_body = None

    async def handle_response(response):
        nonlocal captured_body
        # Capture any response that looks like a file (PDF, binary)
        content_type = response.headers.get('content-type', '')
        if ('pdf' in content_type or 'octet-stream' in content_type) and captured_body is None:
            try:
                body = await response.body()
                if len(body) > 1000:  # Skip tiny error pages
                    captured_body = body
                    logger.info(f"[Download] Intercepted response: {len(body)} bytes, "
                                f"type: {content_type}")
            except Exception as e:
                logger.debug(f"[Download] Failed to read response body: {e}")

    page.on("response", handle_response)

    try:
        # Navigate to PDF URL — Playwright captures response before Firefox renders
        try:
            await page.goto(url, timeout=min(timer.remaining(), 30) * 1000,
                            wait_until='domcontentloaded')
        except Exception as e:
            logger.debug(f"[Download] Navigation error (may be expected): {str(e)[:80]}")

        # Solve challenge if we landed on one
        try:
            await _solve_challenge(page, dep, timer)
        except TimeoutError:
            pass

        # Wait a bit for response to be captured
        for _ in range(10):
            if captured_body is not None:
                break
            await asyncio.sleep(1)

    finally:
        page.remove_listener("response", handle_response)

    if captured_body:
        return b64encode(captured_body).decode('ascii')

    return None
