#!/usr/bin/env python3
"""
Automated Kaggle Competition Rule Acceptance via Playwright.

Uses storage_state (not persistent context) to avoid lockfile issues.
First run: Opens browser for manual Google OAuth login. After login detected,
processes all competitions automatically. Login state saved for future runs.

Usage:
    python kaggle_accept_rules.py
"""

import asyncio, json, os, sys, time
from pathlib import Path

from playwright.async_api import async_playwright

SCRIPT_DIR = Path(__file__).parent
STATE_FILE = Path.home() / ".kaggle_playwright_state.json"
RESULTS_FILE = SCRIPT_DIR / "rules_acceptance_results.json"

# All 69 blocked competitions
BLOCKED = [
    "aerial-cactus-identification", "aptos2019-blindness-detection",
    "denoising-dirty-documents", "detecting-insults-in-social-commentary",
    "dogs-vs-cats-redux-kernels-edition", "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification", "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction", "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7", "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification", "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "3d-object-detection-for-autonomous-vehicles", "AI4Code",
    "alaska2-image-steganalysis", "billion-word-imputation",
    "bms-molecular-translation", "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge", "chaii-hindi-tamil-question-answering",
    "champs-scalar-coupling", "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019", "google-quest-challenge",
    "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7", "herbarium-2021-fgvc8", "herbarium-2022-fgvc9",
    "hms-harmful-brain-activity-classification", "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation", "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7", "inaturalist-2019-fgvc6",
    "invasive-species-monitoring", "iwildcam-2019-fgvc6", "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification", "kuzushiji-recognition",
    "learning-agency-lab-automated-essay-scoring-2",
    "ml2021spring-hw2", "movie-review-sentiment-analysis-kernels-only",
    "nfl-player-contact-detection",
    "osic-pulmonary-fibrosis-progression", "paddy-disease-classification",
    "petfinder-pawpularity-score", "plant-pathology-2021-fgvc8",
    "plant-seedlings-classification", "playground-series-s3e18",
    "predict-volcanic-eruptions-ingv-oe", "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection", "rsna-miccai-brain-tumor-radiogenomic-classification",
    "seti-breakthrough-listen", "siim-covid19-detection",
    "smartphone-decimeter-2022",
    "stanford-covid-vaccine", "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge", "tensorflow2-question-answering",
    "tgs-salt-identification-challenge", "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching", "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction", "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection", "whale-categorization-playground",
]

ACCESSIBLE = {
    "spaceship-titanic", "dog-breed-identification",
    "tabular-playground-series-dec-2021", "tabular-playground-series-may-2022",
    "lmsys-chatbot-arena", "multi-modal-gesture-recognition",
}

BLOCKED = [s for s in BLOCKED if s not in ACCESSIBLE]


async def accept_rules_for_competition(page, slug: str) -> str:
    """Accept rules for a single competition. Returns status string."""
    url = f"https://www.kaggle.com/c/{slug}/rules"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        content = await page.content()

        if "You have already accepted" in content or "already accepted" in content.lower():
            return "already_accepted"

        if "Competition rules acceptance is closed" in content:
            return "rules_closed"

        btn_clicked = False
        selectors = [
            "button:has-text('I Understand and Accept')",
            "button:has-text('Accept Rules')",
            "button:has-text('Accept')",
            "button:has-text('I Accept')",
            "a:has-text('I Understand and Accept')",
            "[data-testid='accept-rules-button']",
        ]
        for sel in selectors:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0 and await btn.is_visible():
                    await btn.click()
                    btn_clicked = True
                    break
            except Exception:
                continue

        if not btn_clicked:
            try:
                btn = page.get_by_role("button", name="Accept")
                if await btn.count() > 0:
                    await btn.first.click()
                    btn_clicked = True
            except Exception:
                pass

        if not btn_clicked:
            return "no_button"

        await asyncio.sleep(2)
        await page.wait_for_load_state("networkidle", timeout=15000)

        content = await page.content()
        if "accepted" in content.lower() or "thank you" in content.lower():
            return "accepted"
        return "accepted_uncertain"

    except Exception as e:
        return f"error: {e}"


async def check_login_status(page) -> bool:
    """Check if we're logged into Kaggle."""
    await page.goto("https://www.kaggle.com/", wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(3)
    content = await page.content()
    has_user = await page.locator("[data-testid='avatar-dropdown']").count()
    has_signin = await page.locator("button:has-text('Sign In')").count()
    if has_user > 0:
        return True
    if has_signin > 0:
        return False
    # Heuristic: if neither user menu nor Sign In found, check content
    if "Sign In" in content and "Register" in content:
        return False
    return True


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )

        # Try to load saved state
        storage_state = None
        if STATE_FILE.exists():
            try:
                storage_state = json.loads(STATE_FILE.read_text())
                print(f"Loaded saved login state from {STATE_FILE}")
            except Exception:
                print("Could not load saved state, starting fresh")

        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            storage_state=storage_state,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
        )

        page = await context.new_page()

        # Check login status
        print("Checking Kaggle login status...")
        logged_in = await check_login_status(page)

        if not logged_in:
            print("\n" + "="*60)
            print("NOT LOGGED IN — Please log in via Google OAuth")
            print("="*60)
            print("A browser window is open. Please:")
            print("1. Click 'Sign In' on Kaggle")
            print("2. Choose 'Sign in with Google'")
            print("3. Select: eizharobinson@gmail.com")
            print("4. Complete the login")
            print("\nWaiting up to 5 minutes for login...")

            for _ in range(60):
                await asyncio.sleep(5)
                try:
                    logged_in = await check_login_status(page)
                    if logged_in:
                        print("\nLogin detected! Saving state...")
                        state = await context.storage_state()
                        STATE_FILE.write_text(json.dumps(state, indent=2))
                        break
                except Exception:
                    pass
                print(".", end="", flush=True)

            if not logged_in:
                print("\nLogin not detected — exiting. Please run again after login.")
                await browser.close()
                return
        else:
            print("Already logged in.")

        # Load previous results if any
        results = {}
        if RESULTS_FILE.exists():
            results = json.loads(RESULTS_FILE.read_text())

        pending = [s for s in BLOCKED if s not in results]
        print(f"\nNeed to process: {len(pending)} competitions")
        print(f"Already processed: {len(results)} competitions")

        for i, slug in enumerate(pending):
            print(f"[{i+1}/{len(pending)}] {slug}...", end=" ", flush=True)
            status = await accept_rules_for_competition(page, slug)
            results[slug] = status
            print(status)

            RESULTS_FILE.write_text(json.dumps(results, indent=2))
            await asyncio.sleep(1)

        # Summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        counts = {}
        for v in results.values():
            counts[v] = counts.get(v, 0) + 1
        for status, count in sorted(counts.items()):
            print(f"  {status}: {count}")

        # Save final state
        state = await context.storage_state()
        STATE_FILE.write_text(json.dumps(state, indent=2))

        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
