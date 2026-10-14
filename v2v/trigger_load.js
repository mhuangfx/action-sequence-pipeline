#!/usr/bin/env node
/*
 * trigger_load.js — fire WanGP's `main.load` Gradio event so it autoloads
 * queue.zip. WanGP has no headless job API; its only autonomous intake is the
 * autoload-on-client-connect path, which requires the SPA to actually mount in
 * a browser (gradio_client does NOT fire load events). We load the page once,
 * hold it open long enough for the mount→backend load round-trip, then close.
 * Generation continues server-side in WanGP's async queue thread after we leave.
 *
 *   node v2v/trigger_load.js http://127.0.0.1:7861 [holdMs]
 */
const { chromium } = require("playwright");

(async () => {
  const url = process.argv[2];
  const holdMs = parseInt(process.argv[3] || "12000", 10);
  if (!url) {
    console.error("usage: node trigger_load.js <url> [holdMs]");
    process.exit(2);
  }
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: "load", timeout: 60000 })
      .catch((e) => console.error("goto warn:", e.message));
    // Give the Gradio SPA real wall-clock time to mount and call main.load.
    await page.waitForTimeout(holdMs);
    console.log("trigger: page held " + holdMs + "ms, closing");
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error("trigger error:", e); process.exit(1); });
