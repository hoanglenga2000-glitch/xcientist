"""Accept Kaggle competition rules via Playwright with API session cookies.

Credentials must come from environment variables or secret files:
KAGGLE_USERNAME / KAGGLE_USERNAME_FILE and KAGGLE_KEY / KAGGLE_KEY_FILE.
Do not hard-code Kaggle credentials in this script.
"""
import asyncio
import os
from pathlib import Path

import requests
from playwright.async_api import async_playwright

def read_secret(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value.strip()
    file_value = os.environ.get(f"{name}_FILE")
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Missing {name} or {name}_FILE")


KAGGLE_USER = read_secret("KAGGLE_USERNAME")
KAGGLE_KEY = read_secret("KAGGLE_KEY")

COMPETITIONS = [
    "playground-series-s3e18",
    "leaf-classification",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
]

def get_auth_cookies():
    """Get session cookies by authenticating with Kaggle API."""
    session = requests.Session()
    session.auth = (KAGGLE_USER, KAGGLE_KEY)
    r = session.get("https://www.kaggle.com/", timeout=15)

    cookies = []
    for cookie in session.cookies:
        cookies.append({
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
            "httpOnly": False,
            "secure": True,
            "sameSite": "Lax",
        })
    print(f"Got {len(cookies)} cookies from API auth")
    return cookies


async def accept_rules():
    cookies = get_auth_cookies()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.add_cookies(cookies)

        for comp in COMPETITIONS:
            print(f"\n=== {comp} ===")
            page = await context.new_page()

            try:
                # First navigate to competition main page to check status
                await page.goto(
                    f"https://www.kaggle.com/competitions/{comp}",
                    wait_until="networkidle",
                    timeout=30000,
                )
                await page.wait_for_timeout(2000)

                # Check if rules are already accepted
                page_text = await page.evaluate("() => document.body.innerText")
                if "already accepted" in page_text.lower():
                    print("  Already accepted!")
                    await page.close()
                    continue

                # Navigate to rules page
                await page.goto(
                    f"https://www.kaggle.com/competitions/{comp}/rules",
                    wait_until="networkidle",
                    timeout=30000,
                )
                await page.wait_for_timeout(3000)

                # Try to find and click accept button
                # Kaggle uses React - look for common button patterns
                clicked = False
                selectors = [
                    "button:has-text('I Understand and Accept')",
                    "button:has-text('Accept')",
                    "button:has-text('I Accept')",
                    "button:has-text('Agree')",
                    "button:has-text('Join Competition')",
                    "button:has-text('Enter')",
                    "button:has-text('Participate')",
                    "[data-testid='accept-rules-button']",
                    ".competition-rules__accept-button button",
                    "button.MuiButton-containedPrimary",
                    "button[type='submit']",
                ]

                for sel in selectors:
                    try:
                        btn = await page.wait_for_selector(sel, timeout=3000)
                        if btn:
                            text = await btn.inner_text()
                            is_disabled = await btn.is_disabled()
                            print(f"  Found: '{text.strip()}' disabled={is_disabled} [{sel}]")
                            if not is_disabled:
                                await btn.click()
                                await page.wait_for_timeout(3000)
                                clicked = True
                                print(f"  Clicked!")
                                break
                    except:
                        continue

                if not clicked:
                    # Print all buttons for debugging
                    buttons = await page.evaluate("""() => {
                        const all = document.querySelectorAll('button, a[role="button"], [class*="btn"]');
                        return Array.from(all).slice(0, 20).map(b => ({
                            text: b.innerText?.substring(0, 80)?.replace(/\\s+/g, ' '),
                            tag: b.tagName,
                            disabled: b.disabled,
                            visible: b.offsetParent !== null,
                            classes: b.className?.substring(0, 60)
                        }));
                    }""")
                    print(f"  Buttons visible on page:")
                    for b in buttons:
                        if b["visible"]:
                            print(f"    [{b['tag']}] '{b['text']}' {b['classes']} disabled={b['disabled']}")

                # Also check for data download page
                await page.goto(
                    f"https://www.kaggle.com/competitions/{comp}/data",
                    wait_until="networkidle",
                    timeout=30000,
                )
                await page.wait_for_timeout(3000)
                data_text = await page.evaluate("() => document.body.innerText")
                if "accept" in data_text.lower() and "rules" in data_text.lower():
                    print(f"  Data page still shows rules not accepted")
                    # Look for accept button on data page too
                    for sel in selectors[:6]:
                        try:
                            btn = await page.wait_for_selector(sel, timeout=2000)
                            if btn and not await btn.is_disabled():
                                await btn.click()
                                await page.wait_for_timeout(2000)
                                print(f"  Clicked accept on data page!")
                                break
                        except:
                            continue
                elif "download" in data_text.lower() or "data" in data_text.lower():
                    print(f"  Data page accessible - rules likely accepted!")
                else:
                    print(f"  Data page: {data_text[:200]}")

                await page.close()
            except Exception as e:
                print(f"  Error: {e}")
                await page.close()

        await browser.close()


asyncio.run(accept_rules())
print("\nDone!")

