"""Script to capture high-resolution screenshots of the running application."""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


async def capture_all():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 960},
            device_scale_factor=2,  # Retina high resolution
        )
        page = await context.new_page()

        # 1. Overview & Statistics
        print("Capturing 1. Overview & Statistics...")
        await page.goto("http://localhost:8501/?nav=1", wait_until="networkidle")
        await page.wait_for_timeout(3500)
        await page.screenshot(path=str(SCREENSHOT_DIR / "overview.png"))

        # 2. Signal Explorer
        print("Capturing 2. Signal Explorer...")
        await page.goto("http://localhost:8501/?nav=2", wait_until="networkidle")
        await page.wait_for_timeout(3500)
        await page.screenshot(path=str(SCREENSHOT_DIR / "signal-explorer.png"))

        # 3. Spectral Analysis
        print("Capturing 3. Spectral Analysis...")
        await page.goto("http://localhost:8501/?nav=3", wait_until="networkidle")
        await page.wait_for_timeout(4000)
        await page.screenshot(path=str(SCREENSHOT_DIR / "spectral-analysis.png"))

        # 4. Disturbance Diagnosis
        print("Capturing 4. Disturbance Diagnosis...")
        await page.goto("http://localhost:8501/?nav=4", wait_until="networkidle")
        await page.wait_for_timeout(4000)
        await page.screenshot(path=str(SCREENSHOT_DIR / "disturbance-diagnosis.png"))

        # 5. Event Comparison
        print("Capturing 5. Event Comparison...")
        await page.goto("http://localhost:8501/?nav=5", wait_until="networkidle")
        await page.wait_for_timeout(3500)
        await page.screenshot(path=str(SCREENSHOT_DIR / "event-comparison.png"))

        # 6. Event Clustering
        print("Capturing 6. Event Clustering...")
        await page.goto("http://localhost:8501/?nav=6", wait_until="networkidle")
        await page.wait_for_timeout(2500)
        # Click the clustering button
        btn = page.locator("button:has-text('Run Unsupervised Clustering')")
        if await btn.count() > 0:
            await btn.click()
            await page.wait_for_timeout(6000)
        await page.screenshot(path=str(SCREENSHOT_DIR / "event-clustering.png"))

        await browser.close()
        print("All screenshots captured successfully!")


if __name__ == "__main__":
    asyncio.run(capture_all())
