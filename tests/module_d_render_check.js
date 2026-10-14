#!/usr/bin/env node
/*
 * Module D render-check (free, local): loads every fight-style template, substitutes
 * the example spec's slots (computed exactly like make_video.js buildSlots), asserts
 * no {{ }} leftovers + required rule blocks, and validates all Module D JSON configs.
 * Run: node tests/module_d_render_check.js
 *
 * *** EXECUTION PENDING (2026-06-10): the implementing agent was permission-blocked
 * from running node (same harness outage that blocked qa_gate.py calibration).
 * Static checks done instead (slot-name grep inventory = only the 9 contract slots
 * appear; all JSON written + reviewed). Run these two free commands to close out:
 *   node tests/module_d_render_check.js          # must print ALL CHECKS PASSED
 *   node plan_specs.js --count 3 --seed 42 --date 2026-06-10 --out /tmp/specs_a && \
 *   node plan_specs.js --count 3 --seed 42 --date 2026-06-10 --out /tmp/specs_b && \
 *   diff -r /tmp/specs_a /tmp/specs_b            # must be empty (determinism)
 */
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
let failures = 0;
function ok(cond, msg) {
  if (cond) console.log("  PASS  " + msg);
  else { console.log("  FAIL  " + msg); failures++; }
}

const heroines = JSON.parse(fs.readFileSync(path.join(ROOT, "config/heroines.json"), "utf8"));
const pools = JSON.parse(fs.readFileSync(path.join(ROOT, "config/variety_pools.json"), "utf8"));
const opponents = JSON.parse(fs.readFileSync(path.join(ROOT, "config/opponents.json"), "utf8"));
const spec = JSON.parse(fs.readFileSync(path.join(ROOT, "specs/example_duel_nox.json"), "utf8"));
console.log("JSON configs + example spec parse: OK");
ok(opponents.opponents.nox.asset_ts === "0000000000", "nox asset_ts");
ok(spec.opponent.ref_ts === "0000000000", "example spec ref_ts");
ok(typeof spec.hero_desc_zh === "string" && spec.hero_desc_zh.length > 0, "example spec hero_desc_zh present");
ok(!!pools.heroes[spec.hero], "example spec hero in variety_pools");

// Slots exactly as make_video.js buildSlots() computes them
const h = heroines.heroines[spec.hero] || {};
function slots(refMode) {
  return {
    HERO_NAME: h.display_name || spec.hero,
    HERO_DESC_ZH: spec.hero_desc_zh || h.desc_zh || null,
    OPPONENT_DESC_ZH: spec.opponent.desc_zh,
    OPPONENT_DESC_EN: spec.opponent.desc_en,
    ENVIRONMENT_EN: spec.environment.desc_en,
    ENVIRONMENT_ZH: spec.environment.desc_zh,
    PALETTE_HERO: spec.palette.hero,
    PALETTE_ENEMY: spec.palette.enemy,
    OPPONENT_REF_CLAUSE: refMode
      ? "以 @image3 为对手的角色形象基准，严格保持 @image3 的设计完全一致。"
      : "",
  };
}
function render(tpl, s) {
  return tpl.replace(/\{\{([A-Z_]+)\}\}/g, (m, k) => {
    if (!(k in s) || s[k] == null) throw new Error("unknown/missing slot " + k);
    return s[k];
  });
}

const styles = ["1v1_rival_duel", "surrounded_bullettime", "vs_titan_bridge", "one_vs_group_vertical"];
const files = ["sb_clip1_en.txt", "sb_clip2_bridge_en.txt", "video_zh.txt", "video_clip1_zh.txt", "video_clip2_zh.txt"];
let duelClip1Rendered = null;
for (const st of styles) {
  console.log("\n== " + st + " ==");
  for (const f of files) {
    const p = path.join(ROOT, "config/fight_style_templates", st, f);
    const raw = fs.readFileSync(p, "utf8");
    for (const refMode of [true, false]) {
      const out = render(raw, slots(refMode));
      ok(!/\{\{/.test(out) && !/\}\}/.test(out), f + " ref=" + refMode + ": no {{ }} leftovers");
    }
    if (f.startsWith("sb_")) {
      ok(/VERTICAL 4x4/.test(raw), f + ": vertical 4x4 grid");
      ok(/TALL 9:16/.test(raw), f + ": tall 9:16 cells");
      ok(/CENTER COLUMN/.test(raw), f + ": center-column rule");
      ok(/number 1-16/.test(raw), f + ": numbered panels");
      ok(/SILHOUETTE/.test(raw), f + ": flat silhouette style");
      ok(/BIGGEST/.test(raw), f + ": biggest beat called out");
      if (f === "sb_clip2_bridge_en.txt")
        ok(/provided reference image/.test(raw), f + ": panel-1 bridge echo");
    } else {
      ok(/严禁出现简笔火柴人/.test(raw), f + ": style-lock block");
      ok(/最大化视觉特效/.test(raw), f + ": VFX-max block");
      ok(/非慢动作/.test(raw), f + ": slow-mo guard");
      ok(/竖屏构图规则/.test(raw), f + ": vertical composition paragraph");
      ok(/绝不互换配色/.test(raw), f + ": palette-separation guard");
      ok(raw.includes("{{OPPONENT_REF_CLAUSE}}"), f + ": OPPONENT_REF_CLAUSE slot");
      if (f === "video_zh.txt")
        ok(/=== CLIP1 ===/.test(raw) && /=== CLIP2 ===/.test(raw), f + ": CLIP1/CLIP2 sections");
      if (f === "video_clip2_zh.txt")
        ok(/承接上一段/.test(raw), f + ": clip2 opens on continuity");
      if (f === "video_clip1_zh.txt" || f === "video_clip2_zh.txt")
        ok(!/=== CLIP/.test(raw), f + ": no section markers in per-clip file");
    }
    if (st === "1v1_rival_duel" && f === "sb_clip1_en.txt")
      duelClip1Rendered = render(raw, slots(true));
  }
}

const pairSet = new Set(pools.palettes.map((p) => p.hero + "|" + p.enemy));
for (const hk of Object.keys(pools.heroes)) {
  const hp = pools.heroes[hk].palette;
  const opts = pools.enemies.filter((e) => pairSet.has(hp + "|" + e.palette));
  ok(opts.length >= 3, "hero " + hk + " has " + opts.length + " palette-compatible enemies (>=3)");
}
for (const hk of Object.keys(heroines.heroines))
  ok(!!pools.heroes[hk], "heroines.json key '" + hk + "' present in variety_pools.heroes");

console.log("\n--- fully rendered sample: 1v1_rival_duel sb_clip1 (example spec) ---\n");
console.log(duelClip1Rendered);

if (failures) { console.error("\n" + failures + " FAILURE(S)"); process.exit(1); }
console.log("\nALL CHECKS PASSED");
