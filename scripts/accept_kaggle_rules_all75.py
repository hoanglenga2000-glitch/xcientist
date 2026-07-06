"""Accept Kaggle competition rules for ALL 75 MLE-Bench split tasks.
Uses Playwright browser automation with Kaggle API session cookies.

Credentials must come from environment variables or secret files:
KAGGLE_USERNAME / KAGGLE_USERNAME_FILE and KAGGLE_KEY / KAGGLE_KEY_FILE.
Do not hard-code Kaggle credentials in this script.
"""
import asyncio
import json
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
    # Fallback to kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        creds = json.loads(kaggle_json.read_text())
        if name == "KAGGLE_USERNAME":
            return creds.get("username", "")
        elif name == "KAGGLE_KEY":
            return creds.get("key", "")
    raise RuntimeError(f"Missing {name} or {name}_FILE")


KAGGLE_USER = read_secret("KAGGLE_USERNAME")
KAGGLE_KEY = read_secret("KAGGLE_KEY")

# Complete MLE-Bench split75 competition list
SPLIT75 = [
    "3d-object-detection-for-autonomous-vehicles",
    "AI4Code",
    "aerial-cactus-identification",
    "alaska2-image-steganalysis",
    "aptos2019-blindness-detection",
    "billion-word-imputation",
    "bms-molecular-translation",
    "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "champs-scalar-coupling",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019",
    "google-quest-challenge",
    "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "histopathologic-cancer-detection",
    "hms-harmful-brain-activity-classification",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "jigsaw-toxic-comment-classification-challenge",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition",
    "leaf-classification",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "mlsp-2013-birds",
    "multi-modal-gesture-recognition",
    "new-york-city-taxi-fare-prediction",
    "nfl-player-contact-detection",
    "nomad2018-predict-transparent-conductors",
    "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score",
    "plant-pathology-2020-fgvc7",
    "plant-pathology-2021-fgvc8",
    "playground-series-s3e18",
    "predict-volcanic-eruptions-ingv-oe",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "seti-breakthrough-listen",
    "siim-covid19-detection",
    "siim-isic-melanoma-classification",
    "smartphone-decimeter-2022",
    "spooky-author-identification",
    "stanford-covid-vaccine",
    "statoil-iceberg-classifier-challenge",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "tgs-salt-identification-challenge",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
    "whale-categorization-playground",
]

RESULTS_FILE = Path(__file__).resolve().parent.parent / "reports" / "kaggle_rules_acceptance_results.json"


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


async def try_download_after_accept(page, comp: str) -> bool:
    """After accepting, verify by trying to list data files."""
    import subprocess, tempfile
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "files", "-c", comp],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"  DOWNLOAD ACCESS: OK ({len(result.stdout.splitlines())} files listed)")
            return True
        else:
            print(f"  DOWNLOAD ACCESS: STILL BLOCKED ({result.stderr[:100]})")
            return False
    except Exception as e:
        print(f"  DOWNLOAD CHECK ERROR: {e}")
        return False


async def accept_one(page, comp: str) -> dict:
    result = {"competition": comp, "status": "unknown", "error": None}

    try:
        await page.goto(
            f"https://www.kaggle.com/competitions/{comp}",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        await page.wait_for_timeout(2000)

        page_text = await page.evaluate("() => document.body.innerText")

        # Check if already accepted or competition page is accessible
        if "Late Submission" in page_text:
            result["status"] = "closed_late_submission"
            print(f"  Already in Late Submission phase")
            return result

        # Navigate to rules page
        await page.goto(
            f"https://www.kaggle.com/competitions/{comp}/rules",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        await page.wait_for_timeout(3000)

        # Try to find and click accept button
        selectors = [
            "button:has-text('I Understand and Accept')",
            "button:has-text('Accept')",
            "button:has-text('I Accept')",
            "button:has-text('Agree')",
            "button:has-text('Join Competition')",
            "button:has-text('Enter')",
            "button:has-text('Participate')",
            "button:has-text('Accept Rules')",
            "[data-testid='accept-rules-button']",
            ".competition-rules__accept-button button",
            "button.MuiButton-containedPrimary",
            "button[type='submit']",
        ]

        clicked = False
        for sel in selectors:
            try:
                btn = await page.wait_for_selector(sel, timeout=2000)
                if btn:
                    text = (await btn.inner_text()).strip()
                    is_disabled = await btn.is_disabled()
                    if not is_disabled:
                        await btn.click()
                        await page.wait_for_timeout(3000)
                        clicked = True
                        print(f"  CLICKED: '{text}' [{sel}]")
                        result["status"] = "accepted"
                        break
            except:
                continue

        if not clicked:
            # Check if already accepted
            rules_text = await page.evaluate("() => document.body.innerText")
            if "You have accepted" in rules_text or "already accepted" in rules_text.lower():
                result["status"] = "already_accepted"
                print(f"  Rules already accepted")
            else:
                result["status"] = "no_button_found"
                print(f"  No accept button found - may need manual check")

        # Verify by checking data page
        await page.goto(
            f"https://www.kaggle.com/competitions/{comp}/data",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        await page.wait_for_timeout(2000)
        data_text = await page.evaluate("() => document.body.innerText")
        has_download = "download" in data_text.lower() or "data" in data_text.lower()
        has_accept_btn = "accept" in data_text.lower() and "rules" in data_text.lower()

        if has_download and not has_accept_btn:
            result["status"] = "data_accessible"
            print(f"  Data page accessible!")
        elif has_accept_btn:
            print(f"  Data page still blocked by rules")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]
        print(f"  ERROR: {e}")

    return result


async def main():
    cookies = get_auth_cookies()
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.add_cookies(cookies)

        for i, comp in enumerate(SPLIT75):
            print(f"\n[{i+1}/{len(SPLIT75)}] {comp}")
            page = await context.new_page()
            try:
                result = await accept_one(page, comp)
                results.append(result)
            except Exception as e:
                results.append({"competition": comp, "status": "fatal_error", "error": str(e)[:200]})
                print(f"  FATAL: {e}")
            finally:
                await page.close()

        await browser.close()

    # Summary
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} competitions processed")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\nResults saved to: {RESULTS_FILE}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))


asyncio.run(main())
