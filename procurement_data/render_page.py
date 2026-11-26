#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""playwright 无头渲染页面 → stdout 输出可见文本（由 procurement_data.fetch_render_adapters 子进程调用）。

用法: python render_page.py <url> [max_wait_ms]
要求: 解释器环境已安装 playwright（本项目用 ~/OpenManus/.venv）。
"""

import asyncio
import sys

from playwright.async_api import async_playwright


async def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    wait_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    if not url:
        print("", end="")
        return
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"))
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=wait_ms)
            except Exception:
                pass
            text = await page.inner_text("body")
            print(text)
            await browser.close()
    except Exception as exc:
        sys.stderr.write(f"render error: {type(exc).__name__}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())