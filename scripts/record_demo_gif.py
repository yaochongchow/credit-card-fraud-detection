"""
Capture a short GIF walkthrough of the Streamlit demo running locally.

Example:
    python scripts/record_demo_gif.py --url http://localhost:8501
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image

try:
    from playwright.sync_api import sync_playwright
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Playwright is required. Install it with:\n"
        "  python3 -m pip install playwright\n"
        "  python3 -m playwright install chromium"
    ) from exc


VIEWPORT = {"width": 1440, "height": 1024}
CSS_CLEANUP = """
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
.modebar {
    display: none;
}
"""


def add_frame(frames: list[Image.Image], page, repeats: int = 1, resize_to: tuple[int, int] | None = None) -> None:
    screenshot = page.screenshot(full_page=False)
    image = Image.open(BytesIO(screenshot)).convert("RGB")
    if resize_to:
        image = image.resize(resize_to, Image.Resampling.LANCZOS)
    for _ in range(repeats):
        frames.append(image.copy())


def smooth_scroll(frames: list[Image.Image], page, start: int, stop: int, steps: int, resize_to: tuple[int, int]) -> None:
    main = page.locator('section[data-testid="stMain"]')
    for i in range(steps):
        progress = i / max(steps - 1, 1)
        position = round(start + (stop - start) * progress)
        main.evaluate("(el, value) => el.scrollTo(0, value)", position)
        page.wait_for_timeout(90)
        add_frame(frames, page, resize_to=resize_to)


def build_demo(url: str, output_path: Path, width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resize_to = (width, round(width * VIEWPORT["height"] / VIEWPORT["width"]))
    frames: list[Image.Image] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(url, wait_until="networkidle", timeout=120_000)
        page.add_style_tag(content=CSS_CLEANUP)
        page.wait_for_timeout(1000)

        # Intro state.
        add_frame(frames, page, repeats=10, resize_to=resize_to)

        # Load a fraud sample to showcase the live update behavior.
        page.get_by_role("button", name="Fraud sample").click()
        page.wait_for_timeout(1500)
        add_frame(frames, page, repeats=10, resize_to=resize_to)

        # Scroll through the model predictions.
        smooth_scroll(frames, page, start=0, stop=1080, steps=18, resize_to=resize_to)
        add_frame(frames, page, repeats=12, resize_to=resize_to)

        # Continue down to feature importance and switch the selected model.
        smooth_scroll(frames, page, start=1080, stop=1750, steps=12, resize_to=resize_to)
        add_frame(frames, page, repeats=6, resize_to=resize_to)

        page.get_by_role("combobox").click()
        page.wait_for_timeout(350)
        add_frame(frames, page, repeats=2, resize_to=resize_to)
        page.get_by_role("option", name="XGBoost (SMOTE)").click()
        page.wait_for_timeout(1200)
        add_frame(frames, page, repeats=12, resize_to=resize_to)

        browser.close()

    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=120,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a short Streamlit demo GIF.")
    parser.add_argument("--url", default="http://localhost:8501", help="Running Streamlit app URL")
    parser.add_argument("--output", default="outputs/streamlit_demo.gif", help="Output GIF path")
    parser.add_argument("--width", type=int, default=1100, help="Output GIF width in pixels")
    args = parser.parse_args()

    build_demo(args.url, Path(args.output), args.width)
    print(f"Saved GIF to {args.output}")


if __name__ == "__main__":
    main()
