#!/usr/bin/env node
/*
 * make_video.js — Module A orchestrator: job-spec JSON in -> publish-ready 30s package out.
 *
 * One 30s chain = 2x15s clips (9:16 native), joined by the BRIDGE technique
 * (clip 2's storyboard seeded by clip 1's last frame), hard-cut ffmpeg concat.
 * Spec: docs/archive/implementation_plan_phase1.md (Module A). Concat/bridge:
 * docs/archive/combat_gods_30s_chaining.md §4.
 *
 * Usage:
 *   node make_video.js --spec specs/<id>.json [--resume] [--connect 9222] [--skip-qa]
 *   node make_video.js --spec <spec.json> --validate-only   # parse spec + render all
 *       templates to <job>/prompts/ and exit BEFORE any browser stage (safe in dev)
 *
 * Concurrency: make_video.js acquires the CANONICAL Chrome-tab lock
 * (output/.chrome_tab.lock) before any browser stage, so a manual run and a
 * nightly cron run can never both drive the single Chrome tab. If a LIVE holder
 * exists it exits with code 9 (LOCK_BUSY_EXIT), naming the holder pid + age. A
 * stale lock (dead holder pid) is reclaimed automatically. --no-lock disables
 * this (tests only — never in production).
 *
 * Dev-only overrides (testing without touching repo config/output):
 *   --templates-root <dir>   default: config/fight_style_templates
 *   --output-root <dir>      default: ./output  (jobs at <root>/jobs/<id>,
 *                            manifest at <root>/manifest.jsonl)
 *
 * Stages (state.json written to output/jobs/<id>/ after EVERY stage):
 *   preflight -> sb1 -> clip1 -> sb2 -> clip2 -> concat -> qa -> package -> manifest
 *
 * Failure semantics: any stage error -> state.json {stage, status:"error", error},
 * nonzero exit. --resume re-enters at the failed stage. Gen stages (sb1/clip1/sb2/clip2):
 * max 1 automatic retry, verbatim (per feedback_seedance_content_filter).
 * QA verdict is ADVISORY (not a gate): FAIL still packages + manifests so the clip reaches
 * HUMAN review on the review app (qa_verdict carried through for the reviewer). --skip-qa bypasses QA.
 *
 * Browser stages are STRICTLY SEQUENTIAL (single Chrome tab); this script never
 * parallelizes child gen scripts.
 */

const path = require("path");
const fs = require("fs");
const http = require("http");
const os = require("os");
const { execFileSync, spawnSync } = require("child_process");

// --- CLI ---

const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : defaultVal;
}
function hasFlag(name) {
  return args.includes(`--${name}`);
}

const SPEC_PATH = getArg("spec", "");
const RESUME = hasFlag("resume");
const CONNECT = getArg("connect", "9222"); // bare port or URL; passed through to child scripts
const SKIP_QA = hasFlag("skip-qa");
const VALIDATE_ONLY = hasFlag("validate-only");
const NO_LOCK = hasFlag("no-lock"); // escape hatch for tests / validate-only; never use in prod
const TEMPLATES_ROOT = getArg("templates-root", path.join(__dirname, "config", "fight_style_templates"));
const OUTPUT_ROOT = getArg("output-root", path.join(__dirname, "output"));

const CDP_PORT = (CONNECT.match(/(\d+)$/) || [, "9222"])[1];
// Dreamina (CapCut) runs in its own Chrome on a separate CDP port (default 9223). Clip stages
// of a gen_path:"dreamina" spec are driven there; the Hailuo storyboard stages (when they run)
// stay on CDP_PORT (9222). See generate_dreamina.js / docs/dreamina_pipeline.md.
const DREAMINA_CONNECT = getArg("dreamina-connect", "9223");
const DREAMINA_PORT = (DREAMINA_CONNECT.match(/(\d+)$/) || [, "9223"])[1];
const MIN_FREE_GB = 2; // preflight disk sanity floor

// Rough ledger numbers: gen_storyboard_hailuo.js header says ~2 credits/image;
// plan's integration estimate ~600 credits/30s chain => ~295/15s 720p clip.
const CREDITS_PER_STORYBOARD = 2;
const CREDITS_PER_CLIP = 295;

const STAGES = ["preflight", "sb1", "clip1", "sb2", "clip2", "concat", "qa", "delivery", "package", "manifest"];

// Slot contract (Module D templates). Unknown placeholders => hard fail at render.
const KNOWN_SLOTS = [
  "HERO_NAME", "HERO_DESC_ZH",
  "OPPONENT_DESC_ZH", "OPPONENT_DESC_EN",
  "ENVIRONMENT_EN", "ENVIRONMENT_ZH",
  "PALETTE_HERO", "PALETTE_ENEMY",
  "OPPONENT_REF_CLAUSE", // computed: @image3 binding sentence when opponent.mode==="ref", else ""
  "CASUAL_OPENING_EN", "CASUAL_OPENING_ZH", // per-hero ordinary-life opening (transformation_reveal)
  "REVERSAL_CLAUSE", "REVERSAL_CLAUSE_EN", // computed: mid-clip "reversal beat" insert when spec.reversal_beat, else ""
];

// Only enforce the --spec requirement when run as a CLI. When required as a module (tests import
// validateSpec / videoArgv / isAssembledMode) there is no --spec and we must not exit.
if (require.main === module && !SPEC_PATH) {
  console.error("ERROR: --spec <spec.json> is required");
  console.error("Usage: node make_video.js --spec specs/<id>.json [--resume] [--connect 9222] [--skip-qa] [--validate-only]");
  process.exit(1);
}

// --- helpers ---

function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

// --- assembled / prompt-only mode -------------------------------------------------
// The pivot format ("transformation_reveal_v2"): ONE prompt-only 15s 9:16 clip rendered from a
// VERBATIM final Chinese prompt (spec.prompt_zh). It BYPASSES the whole template/storyboard/ref
// machinery — no storyboard stages, no @image ref binding, no asset_token, no 2-clip bridge/concat.
// The prompt is assembled upstream (plan_specs + the LLM assembler) and passed through untouched.
//
// Detected by EITHER an explicit marker `prompt_mode:"assembled"` (the clean, generator-agnostic
// signal — preferred for hand-written/test specs) OR the `fight_style:"transformation_reveal_v2"`
// style name (the contract the planner emits). Both are honored so a future style that reuses the
// prompt-only path only needs `prompt_mode:"assembled"`. See docs/dreamina_pipeline.md.
function isAssembledMode(spec) {
  return Boolean(spec) && (spec.prompt_mode === "assembled" || spec.fight_style === "transformation_reveal_v2");
}

// --- canonical Chrome-tab lock (council fix) ---
//
// ONE canonical lockfile guards the single Chrome tab against any concurrent
// browser-driving entry point — a manual run and the nightly cron run can't both
// drive it. Acquired by make_video.js ITSELF at startup, released on exit
// (incl. SIGINT/SIGTERM). Uses atomic O_EXCL create carrying {pid, mtime}; a
// stale lock whose pid is dead is reclaimed. If a LIVE holder is found we exit
// with the DISTINCT code LOCK_BUSY_EXIT naming the holder pid + age.
//
// (flock is intentionally NOT used: output/ is NTFS/fuseblk where flock can be a
// silent no-op — post_actuator.py documents the same hazard. O_EXCL create on the
// existing ext4-or-NTFS path is atomic enough for the single-box, single-tab case,
// and the pid-liveness reclaim handles crashed holders that O_EXCL alone can't.)

const LOCK_PATH = path.join(OUTPUT_ROOT, ".chrome_tab.lock");
// Dreamina runs in a SEPARATE Chrome (port 9223), so a dreamina-only run and a Hailuo nightly
// run are genuinely independent — they take different locks and may run concurrently.
const DREAMINA_LOCK_PATH = path.join(OUTPUT_ROOT, ".dreamina_tab.lock");
const LOCK_BUSY_EXIT = 9; // distinct from 1 (stage error) and 2 (QA fail)

function pidAlive(pid) {
  if (!pid || pid <= 0) return false;
  try {
    process.kill(pid, 0); // signal 0 = liveness probe, no signal delivered
    return true;
  } catch (e) {
    return e.code === "EPERM"; // exists but not ours
  }
}

function ageStr(iso) {
  const ms = Date.now() - new Date(iso).getTime();
  if (!isFinite(ms)) return "unknown age";
  const s = Math.round(ms / 1000);
  if (s < 90) return `${s}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

// Returns a release() fn, or exits LOCK_BUSY_EXIT if a live holder is present.
// lockPath selects which Chrome tab we contend for (Hailuo vs Dreamina).
function acquireChromeLock(lockPath = LOCK_PATH) {
  if (NO_LOCK) {
    log("WARN: --no-lock set — skipping the canonical Chrome-tab lock (tests only)");
    return () => {};
  }
  fs.mkdirSync(OUTPUT_ROOT, { recursive: true });
  const payload = () =>
    JSON.stringify({ pid: process.pid, host: os.hostname(), acquired_at: new Date().toISOString() });

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const fd = fs.openSync(lockPath, "wx"); // O_CREAT|O_EXCL|O_WRONLY — atomic
      fs.writeSync(fd, payload());
      fs.closeSync(fd);
      let released = false;
      const release = () => {
        if (released) return;
        released = true;
        try {
          // only delete if it's still OURS (don't clobber a successor's lock)
          const held = JSON.parse(fs.readFileSync(lockPath, "utf8"));
          if (held.pid === process.pid) fs.rmSync(lockPath, { force: true });
        } catch { /* gone already / unreadable — nothing to do */ }
      };
      // release on normal exit AND on signals (so a Ctrl-C / cron kill frees the tab)
      process.on("exit", release);
      for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
        process.on(sig, () => { release(); process.exit(130); });
      }
      log(`Chrome-tab lock acquired (${lockPath}, pid ${process.pid})`);
      return release;
    } catch (e) {
      if (e.code !== "EEXIST") throw e;
      // lock exists — inspect the holder
      let held = null;
      try { held = JSON.parse(fs.readFileSync(lockPath, "utf8")); } catch { /* corrupt */ }
      const holderPid = held && held.pid;
      const acquiredAt = held && held.acquired_at;
      if (holderPid && pidAlive(holderPid)) {
        console.error(
          `ERROR: Chrome-tab lock held by a LIVE process (pid ${holderPid}, ` +
          `age ${ageStr(acquiredAt)}, host ${held.host || "?"}) — another make_video / ` +
          `nightly run is driving Chrome. Refusing to start (lock: ${lockPath}).`
        );
        process.exit(LOCK_BUSY_EXIT);
      }
      // stale (dead pid or corrupt) — reclaim and retry the atomic create once
      log(`WARN: reclaiming stale Chrome-tab lock (holder pid ${holderPid || "?"} not alive)`);
      try { fs.rmSync(lockPath, { force: true }); } catch { /* race; retry will resolve */ }
    }
  }
  console.error(`ERROR: could not acquire Chrome-tab lock at ${lockPath} (racing reclaim) — exiting`);
  process.exit(LOCK_BUSY_EXIT);
}

// --- spec validation (inline validator, clear per-field errors) ---

const VALID_ASPECTS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const VALID_RESOLUTIONS = ["480p", "720p", "1080p"];

// Assembled / prompt-only validator (transformation_reveal_v2). Accepts the pivot spec contract:
// prompt_zh (verbatim final Chinese prompt) REQUIRED; single clip; aspect honored via spec.aspect
// (falls back to spec.format); asset_token NOT required even for gen_path:"dreamina" (no refs); and
// NO opponent/environment/storyboard/motion-ref machinery. Legacy specs never reach here (the
// isAssembledMode branch in validateSpec keeps their validation byte-for-byte identical).
function validateAssembledSpec(spec, specPath) {
  const errors = [];
  const isStr = (v) => typeof v === "string" && v.trim().length > 0;

  if (!isStr(spec.id)) errors.push(`"id": required non-empty string (used as output dir name)`);
  else if (!/^[A-Za-z0-9._-]+$/.test(spec.id)) errors.push(`"id": must match [A-Za-z0-9._-]+ (got "${spec.id}")`);

  // hero is a LABEL here (manifest/filename), NOT a config key — assembled mode has no character
  // ref, so the hero need not exist in config/heroines.json (it's described in the prompt prose).
  if (!isStr(spec.hero)) errors.push(`"hero": required non-empty string (label for manifest/series)`);

  if (!isStr(spec.fight_style)) errors.push(`"fight_style": required non-empty string`);
  else if (!/^[A-Za-z0-9._-]+$/.test(spec.fight_style)) errors.push(`"fight_style": must match [A-Za-z0-9._-]+ (got "${spec.fight_style}")`);

  if (!isStr(spec.prompt_zh)) errors.push(`"prompt_zh": required non-empty string (the VERBATIM final Chinese prompt — assembled/prompt-only mode)`);

  // aspect: the pivot contract uses "aspect"; accept "format" as a fallback; default 9:16.
  const aspect = spec.aspect || spec.format || "9:16";
  if (!VALID_ASPECTS.includes(aspect)) errors.push(`"aspect": must be one of ${VALID_ASPECTS.join("|")} (got ${JSON.stringify(spec.aspect ?? spec.format)})`);

  const clips = spec.clips == null ? 1 : spec.clips;
  if (clips !== 1) errors.push(`"clips": assembled/prompt-only mode is single-clip only (must be 1, got ${JSON.stringify(spec.clips)})`);

  const clipLength = spec.clip_length == null ? 15 : spec.clip_length;
  if (!Number.isInteger(clipLength) || clipLength < 4 || clipLength > 15) {
    errors.push(`"clip_length": must be an integer 4..15 (got ${JSON.stringify(spec.clip_length)})`);
  }

  const resolution = spec.resolution == null ? "720p" : spec.resolution;
  if (!VALID_RESOLUTIONS.includes(resolution)) errors.push(`"resolution": must be one of ${VALID_RESOLUTIONS.join("|")} (got ${JSON.stringify(spec.resolution)})`);

  const genPath = spec.gen_path == null ? "hailuo" : spec.gen_path;
  if (genPath !== "hailuo" && genPath !== "dreamina") errors.push(`"gen_path": must be "hailuo" or "dreamina" (got ${JSON.stringify(spec.gen_path)})`);
  // NOTE: asset_token is intentionally NOT required for dreamina in this mode — prompt-only has no
  // character/@image ref to bind. If one is supplied it's normalized below but never used.

  if (errors.length) {
    console.error(`SPEC INVALID (${specPath}):`);
    for (const e of errors) console.error(`  - ${e}`);
    process.exit(1);
  }

  // normalized copy: stamp prompt_mode + map aspect->format so every downstream stage (which reads
  // spec.format) sees the right geometry. All template/ref/storyboard fields are forced null/absent.
  return Object.assign({}, spec, {
    prompt_mode: "assembled",
    aspect,
    format: aspect,
    clips: 1,
    clip_length: clipLength,
    resolution,
    escalation: spec.escalation || "bridge",
    palette: spec.palette || {},
    gen_path: genPath,
    generator: spec.generator || genPath,
    asset_token: null,       // prompt-only: no @image ref binding on either path
    storyboard_images: null,
    storyboard: false,
    clip_roles: null,
    motion_ref: null,
    motion_refs: null,
    opponent: spec.opponent || null,
  });
}

function validateSpec(spec, specPath) {
  if (isAssembledMode(spec)) return validateAssembledSpec(spec, specPath);
  const errors = [];
  const isStr = (v) => typeof v === "string" && v.trim().length > 0;

  if (!isStr(spec.id)) errors.push(`"id": required non-empty string (used as output dir name)`);
  else if (!/^[A-Za-z0-9._-]+$/.test(spec.id)) errors.push(`"id": must match [A-Za-z0-9._-]+ (got "${spec.id}")`);

  if (!isStr(spec.hero)) errors.push(`"hero": required non-empty string (key in config/heroines.json)`);
  if (!isStr(spec.fight_style)) errors.push(`"fight_style": required non-empty string (dir under config/fight_style_templates/)`);
  else if (!/^[A-Za-z0-9._-]+$/.test(spec.fight_style)) errors.push(`"fight_style": must match [A-Za-z0-9._-]+ (got "${spec.fight_style}")`);

  if (!spec.opponent || typeof spec.opponent !== "object") {
    errors.push(`"opponent": required object {mode, desc_zh, desc_en, ref_ts}`);
  } else {
    const mode = spec.opponent.mode;
    if (mode !== "prompt" && mode !== "ref") errors.push(`"opponent.mode": must be "prompt" or "ref" (got ${JSON.stringify(mode)})`);
    if (mode === "ref" && !/^\d{10,}$/.test(String(spec.opponent.ref_ts || ""))) {
      errors.push(`"opponent.ref_ts": mode "ref" requires a numeric asset ts (e.g. "0000000000") (got ${JSON.stringify(spec.opponent.ref_ts)})`);
    }
    if (mode === "prompt" && !isStr(spec.opponent.desc_zh) && !isStr(spec.opponent.desc_en)) {
      errors.push(`"opponent": mode "prompt" requires desc_zh and/or desc_en`);
    }
  }

  if (!spec.environment || typeof spec.environment !== "object") {
    errors.push(`"environment": required object {desc_en, desc_zh}`);
  } else if (!isStr(spec.environment.desc_en) && !isStr(spec.environment.desc_zh)) {
    errors.push(`"environment": requires desc_en and/or desc_zh`);
  }

  if (spec.escalation != null && !isStr(spec.escalation)) errors.push(`"escalation": must be a string template-variant key when present`);

  // Per-clip roles: "closeup_reveal" (storyboard-less, pacing-less macro reveal — the
  // close-up variant opener) | "action" (storyboard + pacing). Optional; absent => all action.
  if (spec.clip_roles != null) {
    if (!Array.isArray(spec.clip_roles)) errors.push(`"clip_roles": must be an array of role strings or null`);
    else spec.clip_roles.forEach((r, i) => {
      if (r !== "closeup_reveal" && r !== "action") errors.push(`"clip_roles[${i}]": must be "closeup_reveal" or "action" (got ${JSON.stringify(r)})`);
    });
  }
  if (spec.storyboard != null && typeof spec.storyboard !== "boolean") errors.push(`"storyboard": must be a boolean or null`);

  const format = spec.format == null ? "9:16" : spec.format;
  if (!VALID_ASPECTS.includes(format)) errors.push(`"format": must be one of ${VALID_ASPECTS.join("|")} (got ${JSON.stringify(spec.format)})`);

  const clips = spec.clips == null ? 2 : spec.clips;
  if (clips !== 1 && clips !== 2) errors.push(`"clips": must be 1 or 2 (got ${JSON.stringify(spec.clips)})`);

  const clipLength = spec.clip_length == null ? 15 : spec.clip_length;
  if (!Number.isInteger(clipLength) || clipLength < 4 || clipLength > 15) {
    errors.push(`"clip_length": must be an integer 4..15 (got ${JSON.stringify(spec.clip_length)})`);
  }

  const resolution = spec.resolution == null ? "720p" : spec.resolution;
  if (!VALID_RESOLUTIONS.includes(resolution)) errors.push(`"resolution": must be one of ${VALID_RESOLUTIONS.join("|")} (got ${JSON.stringify(spec.resolution)})`);

  if (spec.motion_ref != null) {
    if (!isStr(spec.motion_ref)) errors.push(`"motion_ref": must be a path string or null`);
    else if (!fs.existsSync(spec.motion_ref)) errors.push(`"motion_ref": file not found: ${spec.motion_ref}`);
  }

  // Per-clip motion refs (one @video1 per clip for varied pacing). Takes precedence
  // over the single motion_ref when present; null entries fall back to motion_ref.
  if (spec.motion_refs != null) {
    if (!Array.isArray(spec.motion_refs)) errors.push(`"motion_refs": must be an array of path strings or null`);
    else spec.motion_refs.forEach((m, i) => {
      if (m == null) return;
      if (!isStr(m)) errors.push(`"motion_refs[${i}]": must be a path string`);
      else if (!fs.existsSync(m)) errors.push(`"motion_refs[${i}]": file not found: ${m}`);
    });
  }

  if (spec.palette != null && typeof spec.palette !== "object") errors.push(`"palette": must be an object {hero, enemy} when present`);

  // gen_path: which video service renders the clips. "hailuo" (default) = the storyboard+clip
  // stages run on Hailuo. "dreamina" = clip stages run on Dreamina (CapCut); the storyboard is
  // EITHER pre-supplied on disk (spec.storyboard_images, e.g. for tests/manual) OR produced by
  // the Hailuo sb stages and downloaded (production). The character is a saved-asset @-mention.
  const genPath = spec.gen_path == null ? "hailuo" : spec.gen_path;
  if (genPath !== "hailuo" && genPath !== "dreamina") errors.push(`"gen_path": must be "hailuo" or "dreamina" (got ${JSON.stringify(spec.gen_path)})`);
  if (genPath === "dreamina") {
    if (!isStr(spec.asset_token) || !/^@?[a-z0-9_]+$/i.test(spec.asset_token.trim())) {
      errors.push(`"asset_token": dreamina requires a saved-asset mention like "@vesta" (got ${JSON.stringify(spec.asset_token)})`);
    }
    if (spec.storyboard_images != null) {
      if (!Array.isArray(spec.storyboard_images)) errors.push(`"storyboard_images": must be an array of image paths or null`);
      else {
        // Only clip1 needs a board — clip2 BRIDGES off clip1's last frame. So 1 board suffices
        // (a 2nd, if present, is ignored for clip2). storyboard_images[0] must exist.
        if (spec.storyboard_images.length < 1) errors.push(`"storyboard_images": need at least 1 (clip1's board; clip2 bridges from clip1's last frame)`);
        spec.storyboard_images.forEach((p, i) => {
          if (i > 0) return; // clip2's board is unused (bridge)
          if (!isStr(p)) errors.push(`"storyboard_images[${i}]": must be a path string`);
          else if (!fs.existsSync(p)) errors.push(`"storyboard_images[${i}]": file not found: ${p}`);
        });
      }
    }
  }

  if (errors.length) {
    console.error(`SPEC INVALID (${specPath}):`);
    for (const e of errors) console.error(`  - ${e}`);
    process.exit(1);
  }

  // normalized copy with defaults applied
  return Object.assign({}, spec, {
    escalation: spec.escalation || "bridge",
    format,
    clips,
    clip_length: clipLength,
    resolution,
    motion_ref: spec.motion_ref || null,
    motion_refs: Array.isArray(spec.motion_refs) ? spec.motion_refs : null,
    clip_roles: Array.isArray(spec.clip_roles) ? spec.clip_roles : null,
    storyboard: typeof spec.storyboard === "boolean" ? spec.storyboard : null,
    palette: spec.palette || {},
    gen_path: genPath,
    generator: spec.generator || genPath, // for emit_package IG tagging (dreamina vs hailuo)
    asset_token: genPath === "dreamina" ? (spec.asset_token.trim().startsWith("@") ? spec.asset_token.trim() : "@" + spec.asset_token.trim()) : (spec.asset_token || null),
    storyboard_images: Array.isArray(spec.storyboard_images) ? spec.storyboard_images : null,
  });
}

// --- heroines / templates ---

function loadHeroines() {
  const p = path.join(__dirname, "config", "heroines.json");
  if (!fs.existsSync(p)) throw new Error(`config/heroines.json not found at ${p}`);
  return readJson(p).heroines || {};
}

function buildSlots(spec, heroines) {
  const h = heroines[spec.hero] || {};
  return {
    // HERO_NAME lands in ZH video prompts only (EN storyboards describe by palette),
    // so prefer the Chinese name when configured.
    HERO_NAME: h.name_zh || h.display_name || spec.hero,
    // hero_desc_zh: optional spec override; heroines.json desc_zh if present.
    // If a template uses {{HERO_DESC_ZH}} and neither source provides it, render fails
    // with a clear "missing value" error.
    HERO_DESC_ZH: spec.hero_desc_zh || h.desc_zh || null,
    OPPONENT_DESC_ZH: (spec.opponent && spec.opponent.desc_zh) || null,
    OPPONENT_DESC_EN: (spec.opponent && spec.opponent.desc_en) || null,
    ENVIRONMENT_EN: (spec.environment && spec.environment.desc_en) || null,
    ENVIRONMENT_ZH: (spec.environment && spec.environment.desc_zh) || null,
    PALETTE_HERO: spec.palette.hero || null,
    PALETTE_ENEMY: spec.palette.enemy || null,
    // @image3 role binding for video prompts (validated wording from the T1a duel run).
    // Empty string for prompt-only opponents so the slot vanishes cleanly.
    OPPONENT_REF_CLAUSE:
      spec.opponent && spec.opponent.mode === "ref"
        ? "以 @image3 为对手的角色形象基准，严格保持 @image3 的设计完全一致。"
        : "",
    // per-hero ordinary-life opening for the transformation_reveal "casual" beat
    // (Morta = modern city; Vesta/Aurora = in-world fantasy travel). Planner sources
    // these from variety_pools heroes; null if absent (templates that use them fail clear).
    CASUAL_OPENING_EN: spec.casual_opening_en || null,
    CASUAL_OPENING_ZH: spec.casual_opening_zh || null,
    // Reversal-beat A/B (flywheel): a mid-clip pattern-interrupt where the hero briefly appears
    // to be LOSING before turning it — a "wait, what?" spike, distinct from the linear end-clip
    // escalation. Injected into clip-2 templates that carry the slot; empty string (not null, so
    // the slot renders away cleanly) when spec.reversal_beat is unset. Generic across heroes —
    // contains NO other {{slots}} so render order can't leave a nested placeholder.
    REVERSAL_CLAUSE: spec.reversal_beat
      ? "【中段反转节拍（约第8-11秒，必须明显呈现）】：在最后的蓄力总攻之前插入一记“逆境”节拍——"
        + "敌方一记重击正面命中，主角被狠狠击落、单膝跪地、周身能量光芒一度黯淡、仿佛即将落败，"
        + "明显陷入劣势；随即她猛然爆发、打破颓势、逆转战局，直接衔接最后的蓄力总攻与最大爆发。"
      : "",
    REVERSAL_CLAUSE_EN: spec.reversal_beat
      ? "  REVERSAL BEAT — make this panels 9-12 knockdown an UNMISTAKABLE low point (the clip's "
        + "\"wait, is she losing?\" beat): the titan's blow drives the heroine DOWN to one knee, her "
        + "energy glow guttering, the colossus looming over her crouched near-beaten silhouette — "
        + "hold on the near-defeat here. Her eruption and turn come only in panels 13-16."
      : "",
  };
}

function renderTemplate(tplPath, slots) {
  let text = fs.readFileSync(tplPath, "utf8");
  for (const [k, v] of Object.entries(slots)) {
    if (v == null) continue;
    text = text.split(`{{${k}}}`).join(String(v));
  }
  const leftover = [...new Set(text.match(/\{\{[A-Za-z0-9_]+\}\}/g) || [])].map((s) => s.slice(2, -2));
  if (leftover.length) {
    const missingValue = leftover.filter((s) => KNOWN_SLOTS.includes(s));
    const unknown = leftover.filter((s) => !KNOWN_SLOTS.includes(s));
    const parts = [];
    if (missingValue.length) parts.push(`missing values for known slots: ${missingValue.join(", ")} (provide them in the spec/heroines.json)`);
    if (unknown.length) parts.push(`unknown slots (not in contract): ${unknown.join(", ")} (known slots: ${KNOWN_SLOTS.join(", ")})`);
    throw new Error(`Template ${tplPath} has unresolved placeholders — ${parts.join("; ")}`);
  }
  return text;
}

// Template file resolution within config/fight_style_templates/<style>/.
// Plan names sb_clip1_en.txt / sb_clip2_bridge_en.txt / video_zh.txt ("clip 1 + clip 2
// variants ok"); escalation is "a template variant key", so we resolve most-specific first.
function resolveTemplate(styleDir, candidates, label) {
  for (const c of candidates) {
    const p = path.join(styleDir, c);
    if (fs.existsSync(p)) return p;
  }
  throw new Error(`No ${label} template in ${styleDir} (looked for: ${candidates.join(", ")})`);
}

// Like resolveTemplate but searches several dirs in order. Used for clip-2 so a
// mode can keep its own escalation file in its folder (existing modes) OR draw a
// shared escalation variant from config/fight_style_templates/_escalations/ (the
// transformation/elemental modes sample clip-2 from a shared pool). styleDir is
// searched first so a per-mode override always wins.
function resolveTemplateDirs(dirs, candidates, label) {
  for (const d of dirs) {
    for (const c of candidates) {
      const p = path.join(d, c);
      if (fs.existsSync(p)) return p;
    }
  }
  throw new Error(`No ${label} template in [${dirs.join(", ")}] (looked for: ${candidates.join(", ")})`);
}

// Format -> composition tag for template-file selection. The original (9:16-native,
// vertical-composed) templates carry NO tag; horizontal 16:9 variants carry "16x9".
// We resolve the tagged variant FIRST, then fall back to the untagged 9:16 file so a
// style that hasn't been ported yet still renders (and 9:16 templates stay intact for
// A/B). Tag derived from the spec format: "16:9" -> "16x9", everything else -> "" (none).
function compositionTag(format) {
  return format === "16:9" ? "16x9" : "";
}

// Build a tagged-then-untagged candidate list. The tag is inserted before the trailing
// language token (_en/_zh) so it reads naturally and matches the on-disk drafts:
//   withTag("sb_clip1_en.txt","16x9")     -> ["sb_clip1_16x9_en.txt","sb_clip1_en.txt"]
//   withTag("video_clip2_zh.txt","16x9")  -> ["video_clip2_16x9_zh.txt","video_clip2_zh.txt"]
// (If no _en/_zh suffix is present, the tag goes just before the extension.)
function withTag(filename, tag) {
  if (!tag) return [filename];
  const m = filename.match(/^(.*)(_(?:en|zh))(\.[^.]+)$/);
  const tagged = m ? `${m[1]}_${tag}${m[2]}${m[3]}` : filename.replace(/(\.[^.]+)$/, `_${tag}$1`);
  return [tagged, filename];
}

// Per-clip storyboard decision. A clip is storyboard-LESS when it is a close-up reveal
// (clip_roles[idx]==="closeup_reveal") or, for single-clip close-up test specs, when
// spec.storyboard===false on clip 0. Everything else (action clips) keeps the storyboard.
function clipUsesStoryboard(spec, idx) {
  const role = (Array.isArray(spec.clip_roles) && spec.clip_roles[idx]) || "action";
  if (role === "closeup_reveal") return false;
  if (spec.storyboard === false && idx === 0) return false;
  return true;
}

// Strip @image2 (storyboard-grid) clauses from a clip prompt while keeping @image1 (the
// character ref) — for storyboard-less close-up clips that run on the char ref alone. Mirrors
// make_video_atlas.strip_image2: clause-level surgery (the @image2 ref sits mid-sentence next
// to @image1), then collapse the punctuation orphaned by the removed clauses.
// Drop the leading "【16:9 草稿 …】" / "[16:9 DRAFT …]" orchestrator-review line before the prompt
// reaches ANY generator (Hailuo or Dreamina). Mirrors make_video_atlas.strip_draft_header — the
// Atlas path already strips it; this closes the same gap on the CDP path (the header is the literal
// first line of every clip template and must not be submitted to the model).
function stripDraftHeader(text) {
  const lines = text.split("\n");
  if (lines.length && (lines[0].includes("草稿") || lines[0].includes("DRAFT"))) lines.shift();
  return lines.join("\n").trim();
}

function stripImage2(text) {
  const toks = text.split(/([。，；\n])/);
  let out = "";
  for (let i = 0; i < toks.length; i += 2) {
    const clause = toks[i];
    const delim = toks[i + 1] || "";
    out += clause.includes("@image2") ? delim : clause + delim;
  }
  return out
    .replace(/，+([。；])/g, "$1")
    .replace(/([。；])，+/g, "$1")
    .replace(/。{2,}/g, "。")
    .replace(/，{2,}/g, "，")
    .replace(/^\s*[，。；]+/gm, "");
}

function templatePaths(spec) {
  const styleDir = path.join(TEMPLATES_ROOT, spec.fight_style);
  if (!fs.existsSync(styleDir)) {
    throw new Error(`fight_style template dir not found: ${styleDir} (fight_style "${spec.fight_style}")`);
  }
  const esc = spec.escalation;
  const tag = compositionTag(spec.format); // "" for 9:16 (original templates), "16x9" for 16:9
  const out = {
    styleDir,
    sb1: resolveTemplate(styleDir, withTag("sb_clip1_en.txt", tag), "clip-1 storyboard"),
    video1: resolveTemplate(
      styleDir,
      [].concat(withTag("video_clip1_zh.txt", tag), withTag("video_zh.txt", tag)),
      "clip-1 video prompt"
    ),
  };
  if (spec.clips === 2) {
    // clip-2 escalation: the mode's own folder first, then the shared _escalations
    // pool (so transformation/elemental modes draw battle/boss/chase/duel variants).
    const clip2Dirs = [styleDir, path.join(TEMPLATES_ROOT, "_escalations")];
    out.sb2 = resolveTemplateDirs(
      clip2Dirs,
      [].concat(withTag(`sb_clip2_${esc}_en.txt`, tag), withTag("sb_clip2_bridge_en.txt", tag)),
      `clip-2 (escalation "${esc}") storyboard`
    );
    out.video2 = resolveTemplateDirs(
      clip2Dirs,
      [].concat(
        withTag(`video_clip2_${esc}_zh.txt`, tag),
        withTag("video_clip2_zh.txt", tag),
        withTag("video_zh.txt", tag)
      ),
      "clip-2 video prompt"
    );
  }
  return out;
}

// Render all prompts up-front (fail fast, before any paid/browser stage) into
// <job>/prompts/. Gen stages then read the rendered files.
function renderAllPrompts(spec, jobDir, heroines) {
  const tpls = templatePaths(spec);
  const slots = buildSlots(spec, heroines);
  const promptsDir = path.join(jobDir, "prompts");
  fs.mkdirSync(promptsDir, { recursive: true });

  const rendered = {};
  const plan = [
    ["sb1", tpls.sb1, "sb1_en.txt"],
    ["video1", tpls.video1, "video_clip1_zh.txt"],
  ];
  if (spec.clips === 2) {
    plan.push(["sb2", tpls.sb2, "sb2_bridge_en.txt"]);
    plan.push(["video2", tpls.video2, "video_clip2_zh.txt"]);
  }
  for (const [key, tplPath, outName] of plan) {
    let text = renderTemplate(tplPath, slots);
    // A storyboard-less clip (close-up reveal) gets its @image2 storyboard clauses stripped,
    // since it renders from the char ref alone (no grid to follow). video1->clip0, video2->clip1.
    const clipIdx = key === "video1" ? 0 : key === "video2" ? 1 : null;
    if (clipIdx != null && !clipUsesStoryboard(spec, clipIdx)) text = stripImage2(text);
    const outPath = path.join(promptsDir, outName);
    fs.writeFileSync(outPath, text);
    rendered[key] = { template: tplPath, prompt_file: outPath, chars: text.length };
    log(`  rendered ${key}: ${path.basename(tplPath)} -> ${outPath} (${text.length} chars)`);
  }
  fs.writeFileSync(path.join(promptsDir, "render_info.json"), JSON.stringify({ slots, rendered, rendered_at: new Date().toISOString() }, null, 2));
  return rendered;
}

// --- state ---

function statePath(jobDir) {
  return path.join(jobDir, "state.json");
}

function freshState(spec) {
  return {
    job_id: spec.id,
    spec_path: path.resolve(SPEC_PATH),
    stage: null,
    status: "pending",
    error: null,
    stages: {},
    created_at: new Date().toISOString(),
    updated_at: null,
  };
}

function saveState(jobDir, state) {
  state.updated_at = new Date().toISOString();
  fs.writeFileSync(statePath(jobDir), JSON.stringify(state, null, 2));
}

function markStage(jobDir, state, stage, status, data, error) {
  state.stage = stage;
  state.status = status;
  state.error = error || null;
  state.stages[stage] = {
    status,
    data: data || null,
    error: error || null,
    completed_at: new Date().toISOString(),
  };
  saveState(jobDir, state);
}

function stageDone(state, stage) {
  const s = state.stages[stage];
  return s && (s.status === "completed" || s.status === "skipped");
}

// --- preflight checks ---

function checkCdp(port) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: "localhost", port, path: "/json/version", timeout: 5000 }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => {
        if (res.statusCode === 200) {
          try { resolve(JSON.parse(body).Browser || "ok"); }
          catch { resolve("ok"); }
        } else reject(new Error(`CDP /json/version returned HTTP ${res.statusCode}`));
      });
    });
    req.on("timeout", () => { req.destroy(new Error("timeout")); });
    req.on("error", (e) => reject(new Error(`CDP not reachable on localhost:${port} (${e.message}) — start Chrome per docs/archive/SESSION_HANDOFF.md`)));
  });
}

function checkDiskFreeGb(dir) {
  // fs.statfsSync exists on Node >= 18.15; fall back to `df -k` otherwise.
  try {
    if (typeof fs.statfsSync === "function") {
      const s = fs.statfsSync(dir);
      return (s.bavail * s.bsize) / 1024 ** 3;
    }
  } catch { /* fall through */ }
  try {
    const out = execFileSync("df", ["-k", dir], { encoding: "utf8" });
    const fields = out.trim().split("\n").pop().trim().split(/\s+/);
    return parseInt(fields[3], 10) / 1024 ** 2;
  } catch {
    return null; // unknown — warn, don't block
  }
}

// --- gen-stage child runner (max 1 automatic retry, verbatim) ---

function runGenChild(stageName, argv, resultPath, extract) {
  let lastErr = null;
  for (let attempt = 0; attempt <= 1; attempt++) {
    if (attempt > 0) log(`[${stageName}] RETRY (1 of 1, verbatim per content-filter playbook)`);
    try {
      log(`[${stageName}] node ${argv.map((a) => (/\s/.test(a) ? JSON.stringify(a.length > 60 ? a.slice(0, 57) + "..." : a) : a)).join(" ")}`);
      try { fs.rmSync(resultPath, { force: true }); } catch { /* ok */ }
      execFileSync("node", argv, { stdio: "inherit", cwd: __dirname });
      if (!fs.existsSync(resultPath)) throw new Error(`child exited 0 but no result.json at ${resultPath}`);
      const result = readJson(resultPath);
      if (result.status !== "completed") throw new Error(`child result status "${result.status}": ${result.error || "no error message"}`);
      return extract(result);
    } catch (e) {
      lastErr = e;
      log(`[${stageName}] attempt ${attempt + 1} failed: ${e.message.split("\n")[0]}`);
    }
  }
  throw new Error(`${stageName} failed after 1 retry: ${lastErr.message}`);
}

// --- stages ---

async function stagePreflight(ctx) {
  const { spec, jobDir, heroines } = ctx;
  const data = {};
  const assembled = isAssembledMode(spec);

  // 1. hero exists. Assembled/prompt-only mode has NO character ref, so the hero is a free label
  //    (described in the prompt prose) and need not exist in config/heroines.json.
  if (!assembled && !heroines[spec.hero]) {
    throw new Error(`hero "${spec.hero}" not found in config/heroines.json (have: ${Object.keys(heroines).join(", ")})`);
  }
  data.hero = spec.hero;

  // 2. prompts. Assembled mode has no templates — write the VERBATIM prompt_zh to the same
  //    prompts/video_clip1_zh.txt that stageClip1 reads, so the clip stage is uniform across modes.
  if (assembled) {
    log("  Assembled/prompt-only mode: writing verbatim prompt_zh (no templates rendered)...");
    const promptsDir = path.join(jobDir, "prompts");
    fs.mkdirSync(promptsDir, { recursive: true });
    const promptFile = path.join(promptsDir, "video_clip1_zh.txt");
    fs.writeFileSync(promptFile, spec.prompt_zh);
    fs.writeFileSync(
      path.join(promptsDir, "render_info.json"),
      JSON.stringify({ mode: "assembled", prompt_file: promptFile, chars: spec.prompt_zh.length, rendered_at: new Date().toISOString() }, null, 2)
    );
    data.rendered = { video1: { template: null, prompt_file: promptFile, chars: spec.prompt_zh.length, mode: "assembled-verbatim" } };
    log(`  wrote verbatim prompt (${spec.prompt_zh.length} chars) -> ${promptFile}`);
  } else {
    // templates exist + render cleanly (writes <job>/prompts/)
    log("  Rendering all templates...");
    data.rendered = renderAllPrompts(spec, jobDir, heroines);
  }

  // 3. CDP reachable — check only the ports this spec will actually drive. A dreamina spec
  //    drives clips on DREAMINA_PORT (9223) and sources its board from Atlas (nano-banana), NEVER
  //    from Hailuo — so a dreamina spec never needs the Hailuo port (9222), pre-supplied board or
  //    not. Only non-dreamina specs drive Hailuo.
  const needsHailuo = spec.gen_path !== "dreamina";
  const needsDreamina = spec.gen_path === "dreamina";
  if (needsHailuo) { data.cdp_browser = await checkCdp(CDP_PORT); log(`  CDP ok on :${CDP_PORT} (${data.cdp_browser})`); }
  if (needsDreamina) { data.dreamina_cdp_browser = await checkCdp(DREAMINA_PORT); log(`  Dreamina CDP ok on :${DREAMINA_PORT} (${data.dreamina_cdp_browser})`); }

  // 4. disk space sane
  const freeGb = checkDiskFreeGb(jobDir);
  if (freeGb == null) log("  WARN: could not determine free disk space — continuing");
  else if (freeGb < MIN_FREE_GB) throw new Error(`only ${freeGb.toFixed(1)} GB free under ${jobDir} (need >= ${MIN_FREE_GB} GB)`);
  else log(`  Disk ok: ${freeGb.toFixed(1)} GB free`);
  data.disk_free_gb = freeGb == null ? null : Number(freeGb.toFixed(1));

  // 5. snapshot the spec into the job dir
  fs.writeFileSync(path.join(jobDir, "spec.json"), JSON.stringify(spec, null, 2));
  return data;
}

function storyboardArgv(spec, promptFile, outputDir, structurePath) {
  const argv = [
    path.join(__dirname, "gen_storyboard_hailuo.js"),
    "--character", spec.hero,
    "--prompt-file", promptFile,
    "--model", "GPT Image 2",
    "--ratio", spec.format,
    "--quality", "Low",
    "--resolution", "2K",
    "--output-dir", outputDir,
    "--connect", CDP_PORT,
  ];
  if (structurePath) argv.push("--structure", structurePath);
  // Dreamina uploads the board as a local image file, so the Hailuo sb stage must save it to disk.
  if (spec.gen_path === "dreamina") argv.push("--download-image");
  return argv;
}

// Pick the @video1 motion ref for clip N: prefer per-clip motion_refs[clipIndex],
// fall back to the single motion_ref, else none.
function resolveMotionRef(spec, clipIndex) {
  if (Array.isArray(spec.motion_refs) && spec.motion_refs[clipIndex]) return spec.motion_refs[clipIndex];
  return spec.motion_ref || null;
}

// Resolve the storyboard IMAGE PATH a Dreamina clip uploads.
//   clip2 (clipIndex 1) BRIDGES: it uploads clip1's LAST FRAME as its anchor image, so Dreamina
//   continues the motion exactly from where clip1 ended → a seamless cut (mirrors Hailuo's sb2
//   bridge; no separate clip2 storyboard needed). Falls back to storyboard_images[1]/sb2 only if
//   clip1's last frame is somehow missing.
//   clip1 (clipIndex 0): spec.storyboard_images[0] (test/manual) else the Hailuo sb1 downloaded
//   PNG (production; recorded at sb1 result.json -> data.image_path).
function resolveStoryboardImage(spec, state, clipIndex) {
  if (spec.gen_path === "dreamina" && clipIndex === 1) {
    const lastFrame = state.stages.clip1 && state.stages.clip1.data && state.stages.clip1.data.last_frame_path;
    if (lastFrame && fs.existsSync(lastFrame)) return lastFrame; // the bridge anchor
    log("  WARN: clip1 last_frame.jpg missing — clip2 falling back to its own storyboard (no bridge)");
  }
  if (Array.isArray(spec.storyboard_images) && spec.storyboard_images[clipIndex]) return spec.storyboard_images[clipIndex];
  const sbStage = clipIndex === 0 ? state.stages.sb1 : state.stages.sb2;
  const img = sbStage && sbStage.data && sbStage.data.image_path;
  if (!img) throw new Error(`dreamina clip${clipIndex + 1}: no storyboard image (set spec.storyboard_images[${clipIndex}] or have the sb stage download one)`);
  return img;
}

// Prompt-only Hailuo clip: verbatim prompt + geometry, NO refs at all (no --favorite-ref, no
// --storyboard-id, no --favorite-opponent → generate_hailuo.js's USE_FAVORITES stays false → pure
// text-to-video). This is exactly the CLI shape proven by hand 5x on 2026-07-26.
function hailuoPromptOnlyArgv(spec, promptText, outputDir) {
  return [
    path.join(__dirname, "generate_hailuo.js"),
    "--prompt", promptText,
    "--clip-length", String(spec.clip_length),
    "--resolution", spec.resolution,
    "--aspect", spec.format,
    "--output-dir", outputDir,
    "--connect", CDP_PORT,
  ];
}

// Prompt-only Dreamina clip: verbatim prompt + geometry, NO storyboard image and NO @asset ref
// (--prompt-only tells generate_dreamina.js to skip the upload + chip-binding entirely). Driven on
// DREAMINA_PORT. Model/aspect/duration are set inside the driver from config + these flags.
function dreaminaPromptOnlyArgv(spec, promptText, outputDir) {
  return [
    path.join(__dirname, "generate_dreamina.js"),
    "--prompt", promptText,
    "--prompt-only",
    "--clip-length", String(spec.clip_length),
    "--resolution", spec.resolution,
    "--aspect", spec.format,
    "--output-dir", outputDir,
    "--connect", DREAMINA_PORT,
  ];
}

function videoArgv(spec, promptText, storyboardRef, outputDir, clipIndex = 0) {
  if (isAssembledMode(spec)) {
    return spec.gen_path === "dreamina"
      ? dreaminaPromptOnlyArgv(spec, promptText, outputDir)
      : hailuoPromptOnlyArgv(spec, promptText, outputDir);
  }
  if (spec.gen_path === "dreamina") return dreaminaVideoArgv(spec, promptText, storyboardRef, outputDir, clipIndex);
  const argv = [
    path.join(__dirname, "generate_hailuo.js"),
    "--prompt", promptText, // generate_hailuo.js has no --prompt-file; pass inline (argv array = no shell quoting issues)
    "--favorite-ref", spec.hero,
    "--clip-length", String(spec.clip_length),
    "--resolution", spec.resolution,
    "--aspect", spec.format,
    "--output-dir", outputDir,
    "--connect", CDP_PORT,
  ];
  // Storyboard-less close-up clips pass no @image2 (generate from the char ref alone).
  if (storyboardRef) argv.push("--storyboard-id", String(storyboardRef));
  if (spec.opponent.mode === "ref" && spec.opponent.ref_ts) argv.push("--favorite-opponent", String(spec.opponent.ref_ts));
  const motionRef = resolveMotionRef(spec, clipIndex);
  if (motionRef) argv.push("--video-ref", motionRef);
  return argv;
}

// Dreamina clip argv: the character is a saved-asset @-mention (spec.asset_token), the storyboard
// is an uploaded image (storyboardImage), motion ref is an uploaded video. Driven on DREAMINA_PORT.
function dreaminaVideoArgv(spec, promptText, storyboardImage, outputDir, clipIndex = 0) {
  const argv = [
    path.join(__dirname, "generate_dreamina.js"),
    "--prompt", promptText,
    "--storyboard-image", storyboardImage,
    "--asset-token", spec.asset_token,
    "--clip-length", String(spec.clip_length),
    "--resolution", spec.resolution,
    "--aspect", spec.format,
    "--output-dir", outputDir,
    "--connect", DREAMINA_PORT,
  ];
  const motionRef = resolveMotionRef(spec, clipIndex);
  if (motionRef) argv.push("--video-ref", motionRef);
  return argv;
}

function extractStoryboardTs(result) {
  const a = (result.new_assets || [])[0];
  if (!a || !a.ts) {
    throw new Error(`storyboard result has no new_assets[0].ts (new_assets=${JSON.stringify(result.new_assets || null)}) — ts is required for --storyboard-id`);
  }
  // image_path is present only when --download-image ran (dreamina path); it's what the Dreamina
  // clip stage uploads. Hailuo clips ignore it (they use the ts).
  return { ts: String(a.ts), asset_id: a.id || null, image_path: result.image_path || null };
}

function extractClip(result) {
  if (!result.video_path || !fs.existsSync(result.video_path)) {
    throw new Error(`clip result has no usable video_path (${result.video_path || "null"})`);
  }
  return {
    video_path: result.video_path,
    last_frame_path: result.last_frame_path || null,
    first_frame_path: result.first_frame_path || null,
    video_url: result.video_url || null,
    retries: result.retries || 0,
    filter_hits: result.filter_hits || 0,
  };
}

// Dreamina can't generate storyboards. When the boards are pre-supplied on disk
// (spec.storyboard_images), the Hailuo sb stages are skipped entirely (no 9222 use). Otherwise
// (production dreamina) they still run on Hailuo to produce + download the boards.
function dreaminaBoardsPresupplied(spec) {
  return spec.gen_path === "dreamina" && Array.isArray(spec.storyboard_images) && spec.storyboard_images.length > 0;
}

// Atlas nano-banana storyboard board for a Dreamina clip1. Dreamina can't make boards, AND it is
// the fallback tier that runs precisely when HAILUO (our usual board maker) is exhausted — so the
// board must come from Atlas, not gen_storyboard_hailuo.js. Generates the 4x4 silhouette grid via
// atlas_cloud.py from the rendered sb1 prompt (draft header stripped) and saves it to
// <job>/sb1/storyboard.png. Returns { image_path } (Dreamina uploads the image; there is no Hailuo
// ts). Idempotent: a valid existing board is reused (resume-safe). Mirrors make_video_atlas.py's
// SB_MODEL + resolution=2k.
const ATLAS_SB_MODEL = "google/nano-banana-pro/text-to-image";
function stageSb1AtlasBoard(spec, jobDir) {
  const outDir = path.join(jobDir, "sb1");
  fs.mkdirSync(outDir, { recursive: true });
  const boardPath = path.join(outDir, "storyboard.png");
  if (fs.existsSync(boardPath) && fs.statSync(boardPath).size > 10000) {
    log("  [sb1] Atlas board exists — reusing");
    return { image_path: boardPath, ts: null };
  }
  const prompt = stripDraftHeader(fs.readFileSync(path.join(jobDir, "prompts", "sb1_en.txt"), "utf8"));
  const argv = ["atlas_cloud.py", "image",
    "--model", ATLAS_SB_MODEL,
    "--prompt", prompt,
    "--aspect", spec.format,
    "--resolution", "2k",
    "--out", boardPath];
  // Retry once, verbatim: nano-banana's content filter (error 1012000) is partly stochastic on
  // silhouette-combat prompts, so a straight retry often clears (mirrors the Seedance content-filter
  // playbook "retry first"). A genuinely-blocked prompt still throws after the retry → the job is
  // left for human review, same as the pure-Atlas path.
  for (let attempt = 1; attempt <= 2; attempt++) {
    log(`  [sb1] Atlas nano-banana board (${spec.format})${attempt > 1 ? " RETRY" : ""} -> ${boardPath}`);
    const r = spawnSync("python3", argv, { stdio: "inherit", cwd: __dirname });
    if (r.status === 0 && fs.existsSync(boardPath) && fs.statSync(boardPath).size >= 10000) {
      return { image_path: boardPath, ts: null };
    }
    if (attempt === 2) throw new Error(`Atlas board gen failed after retry (python3 exit ${r.status})`);
    log(`  [sb1] board gen failed (exit ${r.status}) — retrying once`);
  }
}

function stageSb1(ctx) {
  const { spec, jobDir } = ctx;
  if (isAssembledMode(spec)) return { skipped: "assembled/prompt-only: no storyboard (verbatim prompt)" };
  if (dreaminaBoardsPresupplied(spec)) return { skipped: "dreamina: storyboard pre-supplied (spec.storyboard_images)" };
  if (!clipUsesStoryboard(spec, 0)) return { skipped: "clip1 is a storyboard-less close-up reveal (no @image2)" };
  // Dreamina without pre-supplied boards: source clip1's board from Atlas (Hailuo is unavailable
  // when Dreamina is the fallback tier). Board-less close-up dreamina clips already returned above.
  if (spec.gen_path === "dreamina") return stageSb1AtlasBoard(spec, jobDir);
  const promptFile = path.join(jobDir, "prompts", "sb1_en.txt");
  const outDir = path.join(jobDir, "sb1");
  const argv = storyboardArgv(spec, promptFile, outDir, null);
  return runGenChild("sb1", argv, path.join(outDir, "result.json"), extractStoryboardTs);
}

function stageClip1(ctx) {
  const { spec, jobDir, state } = ctx;
  const promptText = stripDraftHeader(fs.readFileSync(path.join(jobDir, "prompts", "video_clip1_zh.txt"), "utf8"));
  const outDir = path.join(jobDir, "clip1");
  // Hailuo: storyboard ref = asset ts from sb1. Dreamina: storyboard ref = an image path.
  // A storyboard-less close-up clip1 has no sb1 (stage skipped) -> sbRef null (char ref only).
  // Assembled/prompt-only mode never has a storyboard ref (videoArgv ignores it anyway).
  let sbRef = null;
  if (isAssembledMode(spec)) sbRef = null;
  else if (spec.gen_path === "dreamina") sbRef = resolveStoryboardImage(spec, state, 0);
  else if (clipUsesStoryboard(spec, 0)) sbRef = state.stages.sb1.data.ts;
  const argv = videoArgv(spec, promptText, sbRef, outDir, 0);
  return runGenChild("clip1", argv, path.join(outDir, "result.json"), extractClip);
}

function stageSb2(ctx) {
  const { spec, jobDir, state } = ctx;
  if (spec.clips === 1) return { skipped: "single-clip spec" };
  // Dreamina clip2 BRIDGES off clip1's last frame (see resolveStoryboardImage), so it never needs
  // a clip2 storyboard — skip sb2 for the whole dreamina path (no Hailuo storyboard gen for clip2).
  if (spec.gen_path === "dreamina") return { skipped: "dreamina: clip2 bridges from clip1 last_frame (no sb2)" };
  const lastFrame = state.stages.clip1.data.last_frame_path;
  if (!lastFrame || !fs.existsSync(lastFrame)) {
    throw new Error(`clip1 last_frame.jpg missing (${lastFrame || "null"}) — required to seed the bridge storyboard`);
  }
  const promptFile = path.join(jobDir, "prompts", "sb2_bridge_en.txt");
  const outDir = path.join(jobDir, "sb2");
  const argv = storyboardArgv(spec, promptFile, outDir, lastFrame);
  return runGenChild("sb2", argv, path.join(outDir, "result.json"), extractStoryboardTs);
}

function stageClip2(ctx) {
  const { spec, jobDir, state } = ctx;
  if (spec.clips === 1) return { skipped: "single-clip spec" };
  const promptText = stripDraftHeader(fs.readFileSync(path.join(jobDir, "prompts", "video_clip2_zh.txt"), "utf8"));
  const outDir = path.join(jobDir, "clip2");
  const sbRef = spec.gen_path === "dreamina" ? resolveStoryboardImage(spec, state, 1) : state.stages.sb2.data.ts;
  const argv = videoArgv(spec, promptText, sbRef, outDir, 1);
  return runGenChild("clip2", argv, path.join(outDir, "result.json"), extractClip);
}

function stageConcat(ctx) {
  const { spec, jobDir, state } = ctx;
  const master = path.join(jobDir, "master.mp4");
  const clip1 = state.stages.clip1.data.video_path;

  if (spec.clips === 1) {
    log("  Single-clip spec — copying clip1 to master.mp4 (no concat)");
    fs.copyFileSync(clip1, master);
    return { master, mode: "copy" };
  }

  const clip2 = state.stages.clip2.data.video_path;
  // Exact hard-cut concat from docs/archive/combat_gods_30s_chaining.md §4.
  // Seam FX (default "flash"): classic anime IMPACT FRAMES at the cut — 2 frames of a
  // stylized inverted-monochrome still of clip 1's ending + 1 white frame (~0.125s).
  // Makes the hard cut read as a HIT, not a splice (impact-frame technique; see
  // docs/archive/expansion_test_findings_2026-06-10.md transitions addendum). spec.seam_fx:
  // "flash" | "none".
  const fx = spec.seam_fx == null ? "flash" : spec.seam_fx;
  if (fx === "flash") {
    const lastFrame = state.stages.clip1.data.last_frame_path;
    if (!lastFrame || !fs.existsSync(lastFrame)) throw new Error("seam_fx flash needs clip1 last_frame.jpg");
    // Read clip1's REAL WxH (resolution/aspect-agnostic — works for 16:9 1280x720 and
    // 9:16 720x1280 alike; the seam clip + concat are built at the source's native dims).
    const dim = execFileSync("ffprobe", ["-v", "error", "-select_streams", "v:0",
      "-show_entries", "stream=width,height", "-of", "csv=p=0", clip1], { encoding: "utf8" }).trim().split(",");
    const [w, h] = [parseInt(dim[0], 10), parseInt(dim[1], 10)];
    const inv = path.join(jobDir, "seam_impact.mp4");
    // 2 frames inverted high-contrast mono of the ending + 1 white frame, silent audio
    execFileSync("ffmpeg", ["-y",
      "-loop", "1", "-t", "0.084", "-i", lastFrame,
      "-f", "lavfi", "-t", "0.042", "-i", `color=white:s=${w}x${h}:r=24`,
      "-f", "lavfi", "-t", "0.126", "-i", "anullsrc=r=44100:cl=stereo",
      "-filter_complex",
      `[0:v]scale=${w}:${h},negate,hue=s=0,eq=contrast=1.8,fps=24,format=yuv420p[a];` +
      `[1:v]format=yuv420p[b];[a][b]concat=n=2:v=1:a=0[v]`,
      "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-shortest", inv,
    ], { stdio: ["ignore", "ignore", "inherit"], cwd: __dirname });
    const argv = ["-y", "-i", clip1, "-i", inv, "-i", clip2,
      "-filter_complex",
      "[0:v]format=yuv420p[v0];[1:v]format=yuv420p[v1];[2:v]format=yuv420p[v2];" +
      "[v0][0:a][v1][1:a][v2][2:a]concat=n=3:v=1:a=1[v][a]",
      "-map", "[v]", "-map", "[a]",
      "-c:v", "libx264", "-crf", "18", "-preset", "medium",
      "-c:a", "aac", "-b:a", "192k", master];
    log(`  concat with impact-frame seam fx (${w}x${h})`);
    execFileSync("ffmpeg", argv, { stdio: ["ignore", "ignore", "inherit"], cwd: __dirname });
    if (!fs.existsSync(master)) throw new Error("ffmpeg exited 0 but master.mp4 missing");
    return { master, mode: "concat_impact_flash" };
  }

  const argv = [
    "-y", "-i", clip1, "-i", clip2,
    "-filter_complex",
    "[0:v]format=yuv420p[v0];[1:v]format=yuv420p[v1];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]",
    "-map", "[v]", "-map", "[a]",
    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
    "-c:a", "aac", "-b:a", "192k",
    master,
  ];
  log(`  ffmpeg ${argv.join(" ")}`);
  execFileSync("ffmpeg", argv, { stdio: ["ignore", "ignore", "inherit"], cwd: __dirname });
  if (!fs.existsSync(master)) throw new Error("ffmpeg exited 0 but master.mp4 missing");
  return { master, mode: "concat_hard_cut" };
}

function stageQa(ctx) {
  const { jobDir } = ctx;
  if (SKIP_QA) {
    log("  --skip-qa set — bypassing QA gate");
    return { skipped: true };
  }
  const master = path.join(jobDir, "master.mp4");
  const qaJson = path.join(jobDir, "qa.json");
  // --pick-cover: emit cover_t into qa.json (emit_package.js reads it; falls back to 92%).
  const argv = [path.join(__dirname, "qa_gate.py"), master, "--json", qaJson, "--pick-cover"];
  // 2-clip masters have the hard cut at the midpoint — ask the gate to score the seam there.
  if (ctx.spec.clips === 2) argv.push("--seam-at", "auto");
  log(`  python3 ${argv.join(" ")}`);
  const r = spawnSync("python3", argv, { stdio: "inherit", cwd: __dirname });
  const qa = fs.existsSync(qaJson) ? readJson(qaJson) : null;
  if (r.status === 0) return { verdict: "PASS", qa_json: qaJson, scores: qa && qa.scores };
  if (r.status === 2) {
    // QA FAIL is ADVISORY, not a hard gate: record the verdict + scores and CONTINUE to
    // package/manifest so the clip still reaches HUMAN review on the review app. The human is the
    // authority; the VLM is a proposed verdict. Tracking manual-approve vs VLM-FAIL over time
    // tells us if the VLM criteria are too strict (recalibrate). (per owner directive 2026-06-21)
    log(`  WARN: QA gate verdict FAIL — proceeding to package for HUMAN review (advisory)${qa && qa.notes ? ` — ${qa.notes}` : ""}`);
    return { verdict: "FAIL", qa_json: qaJson, scores: qa && qa.scores, advisory: true };
  }
  throw new Error(`qa_gate.py errored (exit ${r.status === null ? `signal ${r.signal}` : r.status})`);
}

// Experiment manipulation-check (analytics flywheel): for jobs that carry a
// delivery_check (cohort experiments only), ask the QA VLM whether the produced master
// actually embodies the experimental change. PROPOSED verdict only — the human confirms
// in the review app review. Never throws and never gates publishing: a normal job no-ops, and
// an experimental job whose check is inconclusive still proceeds (verdict delivered=null).
function stageDelivery(ctx) {
  const { spec, jobDir } = ctx;
  const dc = spec.delivery_check;
  if (!dc || !dc.question) return { skipped: true };
  const master = path.join(jobDir, "master.mp4");
  if (!fs.existsSync(master)) return { skipped: true, reason: "no master" };
  const outJson = path.join(jobDir, "delivery.json");
  const argv = [path.join(__dirname, "qa_gate.py"), master,
    "--verify-delivery", JSON.stringify(dc), "--json", outJson];
  log(`  python3 qa_gate.py master.mp4 --verify-delivery <${dc.when || "whole"}>`);
  const r = spawnSync("python3", argv, { stdio: "inherit", cwd: __dirname });
  let verdict = null;
  if (fs.existsSync(outJson)) {
    try { verdict = readJson(outJson).delivery; } catch (e) { /* keep null */ }
  }
  const delivery = verdict
    ? { question: dc.question, when: dc.when || "whole", ...verdict }
    : { question: dc.question, when: dc.when || "whole", delivered: null,
        error: `verify-delivery exit ${r.status}` };
  return { delivery, delivery_json: outJson };
}

function stagePackage(ctx) {
  const { spec, jobDir } = ctx;
  const argv = [path.join(__dirname, "emit_package.js"), "--job", spec.id, "--jobs-root", path.join(OUTPUT_ROOT, "jobs")];
  log(`  node ${argv.join(" ")}`);
  execFileSync("node", argv, { stdio: "inherit", cwd: __dirname });
  const pkgDir = path.join(jobDir, "package");
  if (!fs.existsSync(pkgDir)) throw new Error(`emit_package.js exited 0 but ${pkgDir} missing`);
  return { package_dir: pkgDir, files: fs.readdirSync(pkgDir) };
}

function stageManifest(ctx) {
  const { spec, jobDir, state } = ctx;
  const manifestPath = path.join(OUTPUT_ROOT, "manifest.jsonl");
  const qa = state.stages.qa && state.stages.qa.data;
  // Assembled/prompt-only jobs make NO storyboard, so no storyboard credits are spent.
  const sbCount = isAssembledMode(spec) ? 0 : (spec.clips === 2 ? 2 : 1);
  const entry = {
    job_id: spec.id,
    generator: spec.generator || "hailuo",
    gen_path: spec.gen_path || "hailuo",
    spec,
    asset_ts: {
      sb1: state.stages.sb1 && state.stages.sb1.data ? state.stages.sb1.data.ts : null,
      sb2: state.stages.sb2 && state.stages.sb2.data ? state.stages.sb2.data.ts || null : null,
    },
    qa: qa && qa.skipped ? { skipped: true } : qa || null,
    delivery: state.stages.delivery && state.stages.delivery.data
      && !state.stages.delivery.data.skipped
      ? state.stages.delivery.data.delivery : null,
    files: {
      master: path.join(jobDir, "master.mp4"),
      package: state.stages.package && state.stages.package.data
        ? state.stages.package.data.files.map((f) => path.join(jobDir, "package", f))
        : [],
    },
    status: "completed",
    timestamps: {
      started: state.created_at,
      finished: new Date().toISOString(),
    },
    credits_estimate: sbCount * CREDITS_PER_STORYBOARD + spec.clips * CREDITS_PER_CLIP,
  };
  fs.mkdirSync(OUTPUT_ROOT, { recursive: true });
  fs.appendFileSync(manifestPath, JSON.stringify(entry) + "\n");
  log(`  Appended manifest line -> ${manifestPath}`);
  return { manifest: manifestPath };
}

const STAGE_FNS = {
  preflight: stagePreflight,
  sb1: stageSb1,
  clip1: stageClip1,
  sb2: stageSb2,
  clip2: stageClip2,
  concat: stageConcat,
  qa: stageQa,
  delivery: stageDelivery,
  package: stagePackage,
  manifest: stageManifest,
};

// --- main ---

async function main() {
  if (!fs.existsSync(SPEC_PATH)) {
    console.error(`ERROR: spec file not found: ${SPEC_PATH}`);
    process.exit(1);
  }
  let rawSpec;
  try { rawSpec = readJson(SPEC_PATH); }
  catch (e) { console.error(`ERROR: spec is not valid JSON: ${e.message}`); process.exit(1); }

  const spec = validateSpec(rawSpec, SPEC_PATH);
  const heroines = loadHeroines();
  const jobDir = path.join(OUTPUT_ROOT, "jobs", spec.id);
  fs.mkdirSync(jobDir, { recursive: true });

  log(`Job ${spec.id}: hero=${spec.hero} style=${spec.fight_style} clips=${spec.clips}x${spec.clip_length}s ${spec.format} ${spec.resolution}`);
  log(`Job dir: ${jobDir}`);

  if (VALIDATE_ONLY) {
    if (isAssembledMode(spec)) {
      // Assembled/prompt-only: no templates + no heroines.json entry required — validation already
      // passed above, and there is nothing to render. Confirm the verbatim prompt is present.
      log("--validate-only: assembled/prompt-only spec (no templates to render)...");
      if (!(typeof spec.prompt_zh === "string" && spec.prompt_zh.trim())) {
        console.error("VALIDATE FAILED: assembled spec has empty prompt_zh");
        process.exit(1);
      }
      log(`VALIDATE OK — assembled spec valid (prompt_zh ${spec.prompt_zh.length} chars, ${spec.format} ${spec.clip_length}s). Exiting before any browser stage.`);
      process.exit(0);
    }
    log("--validate-only: parsing spec + rendering all templates (NO browser stages)...");
    if (!heroines[spec.hero]) {
      console.error(`ERROR: hero "${spec.hero}" not found in config/heroines.json (have: ${Object.keys(heroines).join(", ")})`);
      process.exit(1);
    }
    try {
      renderAllPrompts(spec, jobDir, heroines);
    } catch (e) {
      console.error(`VALIDATE FAILED: ${e.message}`);
      process.exit(1);
    }
    log(`VALIDATE OK — spec valid, all templates rendered to ${path.join(jobDir, "prompts")}. Exiting before any browser stage.`);
    process.exit(0);
  }

  let state;
  const sp = statePath(jobDir);
  if (RESUME && fs.existsSync(sp)) {
    state = readJson(sp);
    const done = STAGES.filter((s) => stageDone(state, s));
    log(`--resume: state found, completed stages: [${done.join(", ") || "none"}]`);
  } else {
    if (fs.existsSync(sp) && !RESUME) {
      log(`WARN: existing state.json found but --resume not set — starting FRESH (prior stage records reset; artifacts on disk get overwritten as stages rerun)`);
    }
    state = freshState(spec);
    saveState(jobDir, state);
  }

  const ctx = { spec, jobDir, heroines, state };

  // Acquire the Chrome-tab lock before ANY browser-driving stage. A dreamina run that drives
  // ONLY Dreamina (boards pre-supplied → no Hailuo sb stages) takes the separate dreamina lock,
  // so it can run alongside a Hailuo nightly. A run that also drives Hailuo takes the Hailuo lock.
  // (validate-only already exited above; it never touches the lock.)
  // A dreamina run drives ONLY the Dreamina Chrome (9223) — never Hailuo — when its boards are
  // pre-supplied OR it's assembled/prompt-only (no board at all). Those take the separate dreamina
  // lock so they can run alongside a Hailuo nightly; anything that touches Hailuo takes LOCK_PATH.
  const drivesOnlyDreamina = spec.gen_path === "dreamina" && (dreaminaBoardsPresupplied(spec) || isAssembledMode(spec));
  const lockPath = drivesOnlyDreamina ? DREAMINA_LOCK_PATH : LOCK_PATH;
  const releaseLock = acquireChromeLock(lockPath);

  for (const stage of STAGES) {
    if (stageDone(state, stage)) {
      log(`=== ${stage}: already ${state.stages[stage].status}, skipping (resume) ===`);
      continue;
    }
    log(`=== STAGE ${stage} ===`);
    try {
      const data = await STAGE_FNS[stage](ctx);
      const skipped = data && (data.skipped === true || typeof data.skipped === "string");
      markStage(jobDir, state, stage, skipped ? "skipped" : "completed", data, null);
      log(`=== ${stage}: ${skipped ? "SKIPPED" : "OK"} ===`);
    } catch (e) {
      // NOTE: QA FAIL no longer throws — it's advisory and the pipeline continues to package
      // for human review (see stageQa). A throw here is a genuine error (tooling/IO).
      markStage(jobDir, state, stage, "error", null, e.message);
      log(`=== ${stage}: ERROR — ${e.message} ===`);
      log(`Re-run with --resume to re-enter at this stage.`);
      process.exit(1);
    }
  }

  state.status = "completed";
  saveState(jobDir, state);
  releaseLock(); // free the Chrome tab promptly (exit handler is a backstop)
  log(`DONE — job ${spec.id} complete. Package: ${path.join(jobDir, "package")}`);
}

// Exported for tests (validation + argv construction are pure and safe to import). The full
// pipeline only runs when invoked as a CLI.
module.exports = {
  isAssembledMode,
  validateSpec,
  validateAssembledSpec,
  videoArgv,
  hailuoPromptOnlyArgv,
  dreaminaPromptOnlyArgv,
  clipUsesStoryboard,
};

if (require.main === module) {
  main().catch((e) => {
    console.error(`FATAL: ${e.message}`);
    process.exit(1);
  });
}
