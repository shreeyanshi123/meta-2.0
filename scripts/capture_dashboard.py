#!/usr/bin/env python3
"""Capture a 1440×900 screenshot of the Tribunal dashboard.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python scripts/capture_dashboard.py [--url URL] [--output PATH]

The script waits for the dashboard to fully load, then saves a PNG screenshot.
"""

import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Capture dashboard screenshot")
    parser.add_argument("--url", default="http://localhost:5173", help="Dashboard URL")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "assets" / "dashboard.png"),
        help="Output PNG path",
    )
    parser.add_argument("--width", type=int, default=1440, help="Viewport width")
    parser.add_argument("--height", type=int, default=900, help="Viewport height")
    parser.add_argument("--wait", type=float, default=3.0, help="Seconds to wait after load")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})

        print(f"Navigating to {args.url} ...")
        page.goto(args.url, wait_until="networkidle")

        # Wait for content to render
        time.sleep(args.wait)

        # Take screenshot
        page.screenshot(path=str(output_path), full_page=False)
        print(f"Screenshot saved to {output_path} ({args.width}×{args.height})")

        browser.close()


if __name__ == "__main__":
    main()
