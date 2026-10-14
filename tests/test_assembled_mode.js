#!/usr/bin/env node
"use strict";
// tests/test_assembled_mode.js — the prompt-only "assembled" video mode (transformation_reveal_v2).
// Pure tests of make_video.js: spec validation (new mode + legacy regression), driver-invocation
// argv construction, and storyboard-skip. No credits spent, no browser, no files generated.
// Run:  node tests/test_assembled_mode.js
const path = require("path");
const fs = require("fs");
const mv = require(path.join(__dirname, "..", "make_video.js"));

let pass = 0, fail = 0;
function ok(cond, name, detail) {
  console.log((cond ? "PASS" : "FAIL"), name, detail ? "-> " + detail : "");
  cond ? pass++ : fail++;
}
function eq(name, got, exp) { ok(JSON.stringify(got) === JSON.stringify(exp), name, JSON.stringify(got)); }

// validateSpec calls process.exit(1) on invalid input. Wrap it so a failure is catchable in-process
// (capture the emitted error lines instead of killing the test runner).
function validateCatching(spec) {
  const realExit = process.exit;
  const realErr = console.error;
  const errs = [];
  console.error = (m) => errs.push(String(m));
  process.exit = (code) => { throw { __validateExit: true, code, errs }; };
  try {
    const out = mv.validateSpec(spec, "<test>");
    return { ok: true, spec: out };
  } catch (e) {
    if (e && e.__validateExit) return { ok: false, code: e.code, errs: e.errs };
    throw e;
  } finally {
    process.exit = realExit;
    console.error = realErr;
  }
}

const FIXTURE = path.join(__dirname, "fixtures", "fixture_transformation_reveal_v2.json");
const baseAssembled = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
const clone = (o) => JSON.parse(JSON.stringify(o));

// ---------------------------------------------------------------------------
// 1. Mode detection
// ---------------------------------------------------------------------------
ok(mv.isAssembledMode({ fight_style: "transformation_reveal_v2" }) === true, "isAssembledMode: by fight_style");
ok(mv.isAssembledMode({ prompt_mode: "assembled" }) === true, "isAssembledMode: by explicit prompt_mode marker");
ok(mv.isAssembledMode({ fight_style: "transformation_reveal" }) === false, "isAssembledMode: legacy style is NOT assembled");
ok(mv.isAssembledMode(null) === false, "isAssembledMode: null-safe");

// ---------------------------------------------------------------------------
// 2. Assembled spec validation
// ---------------------------------------------------------------------------
let r = validateCatching(clone(baseAssembled));
ok(r.ok, "assembled dreamina spec validates (asset_token NOT required)", r.ok ? "" : r.errs.join("; "));
if (r.ok) {
  eq("  normalized prompt_mode", r.spec.prompt_mode, "assembled");
  eq("  aspect -> format mapped", r.spec.format, "9:16");
  eq("  aspect preserved", r.spec.aspect, "9:16");
  eq("  clips forced to 1", r.spec.clips, 1);
  eq("  clip_length", r.spec.clip_length, 15);
  eq("  asset_token null (prompt-only, even for dreamina)", r.spec.asset_token, null);
  eq("  storyboard_images null", r.spec.storyboard_images, null);
  eq("  prompt_zh carried through verbatim", r.spec.prompt_zh, baseAssembled.prompt_zh);
  eq("  generator defaults to gen_path", r.spec.generator, "dreamina");
}

// hailuo assembled spec also validates without any refs
const hailuoAssembled = clone(baseAssembled);
hailuoAssembled.gen_path = "hailuo";
r = validateCatching(hailuoAssembled);
ok(r.ok && r.spec.gen_path === "hailuo" && r.spec.asset_token === null, "assembled hailuo spec validates (no refs)");

// explicit prompt_mode marker with a non-v2 fight_style still routes to assembled validation
const markerSpec = clone(baseAssembled);
markerSpec.fight_style = "some_future_style";
markerSpec.prompt_mode = "assembled";
r = validateCatching(markerSpec);
ok(r.ok && r.spec.prompt_mode === "assembled", "prompt_mode:assembled marker validates on a non-v2 style");

// missing prompt_zh -> INVALID
const noPrompt = clone(baseAssembled); delete noPrompt.prompt_zh;
r = validateCatching(noPrompt);
ok(!r.ok && r.errs.join("\n").includes("prompt_zh"), "missing prompt_zh rejected");

// empty prompt_zh -> INVALID
const emptyPrompt = clone(baseAssembled); emptyPrompt.prompt_zh = "   ";
r = validateCatching(emptyPrompt);
ok(!r.ok && r.errs.join("\n").includes("prompt_zh"), "empty/whitespace prompt_zh rejected");

// clips != 1 -> INVALID (assembled is single-clip only)
const twoClips = clone(baseAssembled); twoClips.clips = 2;
r = validateCatching(twoClips);
ok(!r.ok && r.errs.join("\n").includes("single-clip"), "clips:2 rejected in assembled mode");

// bad aspect -> INVALID
const badAspect = clone(baseAssembled); badAspect.aspect = "banana";
r = validateCatching(badAspect);
ok(!r.ok && r.errs.join("\n").includes("aspect"), "invalid aspect rejected");

// hero need NOT be in heroines.json — a made-up hero label still validates (no char ref)
const oddHero = clone(baseAssembled); oddHero.hero = "Zzephyrine";
r = validateCatching(oddHero);
ok(r.ok, "assembled hero is a free label (not a heroines.json key)");

// ---------------------------------------------------------------------------
// 3. Legacy spec validation REGRESSION — must behave exactly as before
// ---------------------------------------------------------------------------
const legacyHailuo = {
  id: "legacy_hailuo_001", hero: "vesta", fight_style: "transformation_reveal",
  opponent: { mode: "prompt", desc_zh: "暗影巨像", desc_en: "shadow colossus" },
  environment: { desc_en: "ruined temple", desc_zh: "废墟神庙" },
};
r = validateCatching(clone(legacyHailuo));
ok(r.ok, "legacy hailuo spec still validates", r.ok ? "" : (r.errs || []).join("; "));
if (r.ok) {
  eq("  legacy defaults: clips=2", r.spec.clips, 2);
  eq("  legacy defaults: format 9:16", r.spec.format, "9:16");
  eq("  legacy defaults: gen_path hailuo", r.spec.gen_path, "hailuo");
  ok(r.spec.prompt_mode === undefined, "  legacy spec is NOT stamped prompt_mode");
}

// legacy dreamina STILL requires asset_token (the old contract is untouched)
const legacyDreaminaNoToken = clone(legacyHailuo);
legacyDreaminaNoToken.gen_path = "dreamina";
r = validateCatching(legacyDreaminaNoToken);
ok(!r.ok && r.errs.join("\n").includes("asset_token"), "legacy dreamina still REQUIRES asset_token (regression)");

// legacy dreamina WITH asset_token validates + normalizes the @-prefix (unchanged behavior)
const legacyDreamina = clone(legacyHailuo);
legacyDreamina.gen_path = "dreamina"; legacyDreamina.asset_token = "vesta";
r = validateCatching(legacyDreamina);
ok(r.ok && r.spec.asset_token === "@vesta", "legacy dreamina asset_token normalized to @vesta");

// legacy spec missing opponent still rejected (contract unchanged)
const legacyNoOpp = clone(legacyHailuo); delete legacyNoOpp.opponent;
r = validateCatching(legacyNoOpp);
ok(!r.ok && r.errs.join("\n").includes("opponent"), "legacy spec still requires opponent");

// ---------------------------------------------------------------------------
// 4. Driver-invocation argv (dry, no browser)
// ---------------------------------------------------------------------------
const normAssembledDreamina = validateCatching(clone(baseAssembled)).spec;
const normAssembledHailuo = validateCatching(hailuoAssembled).spec;

// assembled HAILUO: prompt-only, no refs, aspect 9:16
const hArgv = mv.videoArgv(normAssembledHailuo, normAssembledHailuo.prompt_zh, /*sbRef*/ "IGNORED", "/tmp/out", 0);
ok(hArgv.some((a) => /generate_hailuo\.js$/.test(a)), "assembled hailuo -> generate_hailuo.js");
ok(hArgv.includes("--prompt") && hArgv[hArgv.indexOf("--prompt") + 1] === normAssembledHailuo.prompt_zh, "assembled hailuo passes verbatim --prompt");
ok(hArgv.includes("--aspect") && hArgv[hArgv.indexOf("--aspect") + 1] === "9:16", "assembled hailuo passes --aspect 9:16");
ok(hArgv.includes("--clip-length") && hArgv[hArgv.indexOf("--clip-length") + 1] === "15", "assembled hailuo passes --clip-length 15");
ok(!hArgv.includes("--favorite-ref"), "assembled hailuo does NOT pass --favorite-ref (no char ref)");
ok(!hArgv.includes("--storyboard-id"), "assembled hailuo does NOT pass --storyboard-id (no board, sbRef ignored)");
ok(!hArgv.includes("--favorite-opponent"), "assembled hailuo does NOT pass --favorite-opponent");

// assembled DREAMINA: prompt-only, no storyboard-image, no asset-token
const dArgv = mv.videoArgv(normAssembledDreamina, normAssembledDreamina.prompt_zh, "IGNORED", "/tmp/out", 0);
ok(dArgv.some((a) => /generate_dreamina\.js$/.test(a)), "assembled dreamina -> generate_dreamina.js");
ok(dArgv.includes("--prompt-only"), "assembled dreamina passes --prompt-only");
ok(dArgv.includes("--aspect") && dArgv[dArgv.indexOf("--aspect") + 1] === "9:16", "assembled dreamina passes --aspect 9:16");
ok(!dArgv.includes("--storyboard-image"), "assembled dreamina does NOT pass --storyboard-image");
ok(!dArgv.includes("--asset-token"), "assembled dreamina does NOT pass --asset-token");

// REGRESSION: legacy hailuo argv still binds the character ref + storyboard id
const lArgv = mv.videoArgv(validateCatching(clone(legacyHailuo)).spec, "prompt body", "1699999999", "/tmp/out", 0);
ok(lArgv.includes("--favorite-ref"), "legacy hailuo STILL passes --favorite-ref (regression)");
ok(lArgv.includes("--storyboard-id") && lArgv[lArgv.indexOf("--storyboard-id") + 1] === "1699999999", "legacy hailuo STILL passes --storyboard-id (regression)");

// ---------------------------------------------------------------------------
// 5. Storyboard-skip signal
// ---------------------------------------------------------------------------
ok(mv.isAssembledMode(normAssembledDreamina) === true, "normalized assembled spec still reads as assembled (sb1 will skip)");
ok(mv.isAssembledMode(validateCatching(clone(legacyHailuo)).spec) === false, "normalized legacy spec is not assembled (sb1 runs)");

// ---------------------------------------------------------------------------
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
