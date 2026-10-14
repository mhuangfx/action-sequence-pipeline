#!/usr/bin/env python3
"""
assemble_transform_prompt.py — LLM prompt-assembler for the transformation-reveal
video template (fight_style "transformation_reveal_v2"). PRODUCTION.

Promoted 2026-07-26 from the validated probe harness
(output/probes/pivot_wave1/assembler/assemble.py, kept untouched as history).

Architecture: slot-fill, not freeform. The template owns FIXED Chinese boilerplate
(style law, camera law, identity-lock, pacing skeleton) copied verbatim from
probe_04/spec.md (the gold-standard prompt). The LLM's only job is composing the
variable connective tissue that fills the slots: scene+character intro, incident
beat, element-specific transform choreography along the sampled zoom tour, and the
reveal pose description. The assembler rolls the variation axes, calls a local
Ollama model to fill the slots, stitches everything into one paste-ready Chinese
block, and validates the result against the failure library in WAVE1_RESULTS.md.

Register + dark sub-shape are sampled by WEIGHT from config/transform_template_pools.json
(register_sampling; owner ruling heroic 50 / awe 25 / dark_villain 25) unless overridden.

CLI (pipeline entrypoint — plan_specs.js shells out with --json):
    python3 tools/assemble_transform_prompt.py --seed 42 --json
    python3 tools/assemble_transform_prompt.py --seed 5 --register dark_villain --json
    python3 tools/assemble_transform_prompt.py --seed 1        # human-readable

--json prints ONE JSON object to stdout on success:
    {"ok":true,"prompt_zh":"...","hero_latin":"sirael","hero_cn":"西拉尔",
     "attempts":N,"axes":{register,shape,element,incident_class,zoom_tour,
     demographics_summary,director_flavor,assembler_seed,assembler_model}}
On failure it prints {"ok":false,"error":"...","attempts":N,...} and exits non-zero.

See WAVE1_RESULTS.md and probe_04/spec.md for the ground truth this is built from.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_POOLS = os.path.join(REPO_ROOT, "config", "transform_template_pools.json")
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "output", "transform_assembler_runs")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("TRANSFORM_ASSEMBLER_MODEL", "qwen2.5:14b")

# ---------------------------------------------------------------------------
# Gold-standard few-shot example (probe_04/spec.md, verbatim Chinese prompt).
# Kept as a raw string constant here rather than re-read from spec.md so the
# assembler has zero runtime dependency on other probe directories.
# ---------------------------------------------------------------------------
GOLD_EXAMPLE_CN = (
    "竖屏9:16，720p，全程统一的全实拍级写实电影渲染——真实皮肤、真实布料、真实光影，"
    "能量与铠甲皆为电影级视效大片质感的写实特效、打在一个真实人物身上，写实风格从第一帧"
    "到最后一帧完全一致，包括变身之后。全片一镜到底、连续不间断，镜头始终锁定在她身上、"
    "以她为中心。她的发色与发型自始至终保持不变。0-4秒：雨夜空旷的城市广场，路灯忽明忽暗，"
    "尼莎独自走着、神情平常；广场尽头一道高大的兜帽身影在闪烁的灯光中浮现、气势逼人，"
    "她停步扎地、转身正对那道远处的威胁。4-12秒（核心：快速动态变身，仍为一镜到底）："
    "她猛地踏地，翠绿藤木自她脚下接触点炸裂而出；镜头急速推近掠过她的小腿，荆棘甲片顺她"
    "的双腿向上攀爬成形；顺势上扫掠过她的胸前，胸甲逐层结成；急甩至她紧握的拳头，护手与"
    "一柄缠绕荆棘的长柄战刃在她掌中凝聚；再急推抵达她的面庞，她的双眼骤燃琥珀金光、脸上"
    "发光的树纹符文次第点亮。镜头同时以低仰角快速环绕她旋转上升，她的头发与衣摆在能量狂"
    "风中激烈翻飞，她的脊背后弓、身体微微升离地面，藤蔓能量缠绕她的四肢。峰值时翠金色能"
    "量光华自她周身向外炸开、铺满画面，广场仍透过强光隐约可见，她的身影在光芒中始终清晰"
    "可辨。这是同一个人的力量觉醒蜕变，五官与身份始终保持一致，绝非换成另一个人，她的人"
    "类面容与轮廓自始至终清晰可见。12-15秒：光华骤然向外收束，镜头快速拉远至全身英雄定格"
    "——她居中稳定站立，面向广场尽头那道兜帽身影，双手将缠绕荆棘的长柄战刃平举指向对方，"
    "摆出沉着坚定的战斗姿态，翠金光环缓缓落定，镜头轻微缓移，画面清晰稳定地定格。"
)

# Phrases unique to the gold few-shot example (probe_04's thornwood element /
# Nyssa / plaza). thornwood is permanently excluded from the element pool, so
# any of these appearing in generated fill text is a few-shot-leakage signal,
# not a legitimate word choice.
GOLD_EXAMPLE_LEAK_TOKENS = ["藤木", "荆棘", "尼莎", "翠绿藤", "翠金色", "翠金光", "琥珀金", "树纹"]

NEGATION_TOKENS = ["不", "没有", "别", "勿", "避免", "绝不"]
NEW_NOUN_RE = re.compile(r"(一个|一道|一位)[^，。；;,]{0,8}(存在|生物|身影)")
CJK_RE = re.compile(r"[一-鿿]")
TIMING_RE = re.compile(r"\d+-\d+秒")

# Feedback fixes (probe 05 coordinator review):
#  - hero names must be emitted in Chinese: reject any run of 2+ Latin letters
#    (tech tokens like 9:16 / 720p contain no 2+ letter runs, so they pass).
LATIN_SEQ_RE = re.compile(r"[A-Za-z]{2,}")
#  - skin-micro-detail vocabulary may ONLY bind to the eyes/face beat: any
#    sentence carrying one of these tokens must also carry a face/eye marker.
SKIN_DETAIL_TOKENS = ["雀斑", "毛孔", "绒毛", "汗", "肌理", "皮肤质感", "肌肤质感",
                      "细腻光泽", "皮肤纹理", "皮肤的", "肌肤的"]
FACE_MARKERS = ["眼", "瞳", "眸", "脸", "面颊", "双目", "面容"]
SENTENCE_SPLIT_RE = re.compile(r"[。；！？\n]")


def load_pools(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _weighted_choice(rng, weights):
    """Pick a key from a {key: weight} dict by weight (weights need not sum to 1).
    Falls back to a uniform choice over the keys if the total weight is non-positive."""
    keys = list(weights.keys())
    total = sum(max(0.0, float(weights[k])) for k in keys)
    if total <= 0:
        return rng.choice(keys)
    r = rng.random() * total
    upto = 0.0
    for k in keys:
        upto += max(0.0, float(weights[k]))
        if r <= upto:
            return k
    return keys[-1]


def roll_axes(pools, seed, overrides, zoom_count_override=None, zoom_ids_override=None,
              director_override=None):
    rng = random.Random(seed)

    registers = list(pools["register"].keys())
    # Weighted register sampling (owner ruling heroic 50 / awe 25 / dark_villain 25) —
    # weights live in pools.register_sampling.weights; missing -> uniform. An explicit
    # --register override always wins.
    if overrides.get("register"):
        register = overrides["register"]
    else:
        reg_weights = (pools.get("register_sampling") or {}).get("weights")
        reg_weights = {r: (reg_weights or {}).get(r, 1.0) for r in registers}
        register = _weighted_choice(rng, reg_weights)
    if register not in pools["register"]:
        raise ValueError(f"unknown register '{register}', choices: {registers}")

    elements = pools["elements"]["pool"]
    if overrides.get("element"):
        element = next((e for e in elements if e["id"] == overrides["element"]), None)
        if element is None:
            raise ValueError(f"unknown element id '{overrides['element']}'")
    else:
        element = rng.choice(elements)

    incidents = pools["incident_classes"]["pool"]
    if overrides.get("incident"):
        incident = next((i for i in incidents if i["id"] == overrides["incident"]), None)
        if incident is None:
            raise ValueError(f"unknown incident id '{overrides['incident']}'")
    else:
        incident = rng.choice(incidents)

    # hero name: pool entries carry a Chinese transliteration (cn); the prompt
    # uses the cn form and the validator rejects Latin-script names in output.
    hero_pool = pools["hero_names"]["pool"]
    if overrides.get("hero"):
        ov = overrides["hero"]
        match = next((h for h in hero_pool if h["latin"].lower() == ov.lower()), None)
        if match:
            hero_name, hero_name_latin = match["cn"], match["latin"]
        else:
            # caller passed a name not in the pool — assume it is already Chinese
            hero_name, hero_name_latin = ov, ov
    else:
        h = rng.choice(hero_pool)
        hero_name, hero_name_latin = h["cn"], h["latin"]

    environment = overrides.get("environment") or rng.choice(pools["environments"])

    # zoom tour: sample 2-4 unique targets, order = sample order (already randomized)
    zoom_pool = pools["zoom_tour"]["pool"]
    if zoom_ids_override:
        ids = [z.strip() for z in zoom_ids_override.split(",") if z.strip()]
        zoom_targets = []
        for zid in ids:
            z = next((t for t in zoom_pool if t["id"] == zid), None)
            if z is None:
                raise ValueError(f"unknown zoom target id '{zid}'")
            zoom_targets.append(z)
    else:
        n = zoom_count_override or rng.randint(2, 4)
        n = max(2, min(4, n))
        zoom_targets = rng.sample(zoom_pool, n)

    # demographics
    demo = pools["demographics"]
    clothing = rng.choice(demo["clothing_styles"])
    body_type = rng.choice(demo["body_types"])
    ethnicity = rng.choice(demo["ethnicity_descriptors"])

    # realism details
    realism = pools["realism_details"]
    lens = rng.choice(realism["lenses"])
    lighting = rng.choice(realism["lighting_regimes"])
    face_adjacent = any(z["id"] == "eyes-extreme-closeup" for z in zoom_targets)
    skin_detail = rng.choice(realism["skin_micro_detail_phrases"]) if face_adjacent else None

    # director flavor: forced, or ~50% roll
    if director_override == "none":
        director = None
    elif director_override:
        director = next((d for d in pools["director_flavor"]["pool"] if d["id"] == director_override), None)
        if director is None:
            raise ValueError(f"unknown director id '{director_override}'")
    else:
        director = rng.choice(pools["director_flavor"]["pool"]) if rng.random() < 0.5 else None

    # transform_shape: dark_villain sub-shape axis (owner ruling). Non-dark
    # registers have no shape. Default is deterministic (no rng consumed) so the
    # other axes stay reproducible across seeds.
    shape_id = None
    shape_guidance = None
    is_covering_shape = False
    identity_lock_line = pools["boilerplate"]["identity_lock"]
    if register == "dark_villain":
        ts = pools["register"]["dark_villain"]["transform_shape"]
        variants = ts["variants"]
        # Sub-shape sampled by weight (pools.register_sampling.shape_weights) unless
        # overridden; missing weights -> uniform over the variants (owner ruling: dark
        # sub-shape is sampled, not fixed to the old dark_armored default).
        if overrides.get("shape"):
            shape_id = overrides["shape"]
        else:
            shp_weights = (pools.get("register_sampling") or {}).get("shape_weights")
            shp_weights = {s: (shp_weights or {}).get(s, 1.0) for s in variants}
            shape_id = _weighted_choice(rng, shp_weights)
        if shape_id not in variants:
            raise ValueError(f"unknown transform_shape '{shape_id}', choices: {list(variants)}")
        shape_guidance = variants[shape_id]["guidance"]
        lock_kind = variants[shape_id]["identity_lock"]
        is_covering_shape = lock_kind == "morph_covering"
        identity_lock_line = (
            pools["boilerplate"]["identity_lock_variant"]
            if lock_kind == "morph_covering"
            else pools["boilerplate"]["identity_lock"]
        )
    elif overrides.get("shape"):
        raise ValueError("--shape is only valid for the dark_villain register")

    # weapon: awe is always empty-handed; otherwise filtered by register fit
    weapons = pools["weapons"]["pool"]
    if overrides.get("weapon"):
        weapon = next((w for w in weapons if w["id"] == overrides["weapon"]), None)
        if weapon is None:
            raise ValueError(f"unknown weapon id '{overrides['weapon']}'")
    elif register == "awe":
        weapon = next(w for w in weapons if w["id"] == "none-empty-handed")
    else:
        fit = [w for w in weapons if register in w["register_fit"]]
        weapon = rng.choice(fit)

    return {
        "register": register,
        "register_guidance": pools["register"][register]["guidance"],
        "shape_id": shape_id,
        "shape_guidance": shape_guidance,
        "is_covering_shape": is_covering_shape,
        "identity_lock_line": identity_lock_line,
        "element": element,
        "incident": incident,
        "hero_name": hero_name,
        "hero_name_latin": hero_name_latin,
        "environment": environment,
        "zoom_targets": zoom_targets,
        "clothing": clothing,
        "body_type": body_type,
        "ethnicity": ethnicity,
        "lens": lens,
        "lighting": lighting,
        "skin_detail": skin_detail,
        "director": director,
        "weapon": weapon,
    }


def build_system_prompt():
    return (
        "你是一个专为「变身觉醒」竖屏短视频写作中文分镜连接文字的助手。"
        "你不会生成完整提示词，只负责填充三个变量片段，模板会把固定的中文法律条款"
        "（写实风格声明、运镜规则、身份锁定句、时间骨架）拼接在你的文字前后。"
        "你必须只输出严格的JSON对象，三个键：intro_incident, morph_choreography, reveal_pose，"
        "值全部是简体中文，不要markdown、不要代码块、不要多余的键、不要解释文字。"
        "写作规则（硬性）："
        "1) 全文使用积极正面表达，禁止使用以下任何字/词：不、没有、别、勿、避免、绝不——一个字都不能出现。"
        "2) 她（主角）在整个变身过程中必须是每句话的主语，绝不能被称作「一个/一道/一位...存在/生物/身影」，"
        "她自始至终就是「她」，人类面容与轮廓不允许被替换成另一个角色或另一种生物。"
        "3) morph_choreography 中镜头推近的目标必须全部是她自己身体的部位（按给定顺序），"
        "绝不能推近到她身体以外的物体或武器本身（武器可以在她手中成形，但镜头目标仍是她的手/拳头）。"
        "4) 不要在你的文字里写身份锁定句或时间秒数标记，模板会自动插入。"
        "5) 语言要流畅自然、电影感强，符合给定的register语气；不要逐条罗列人物设定，要自然融入描写。"
        "6) 主角的名字只能使用给定的中文译名，全文严禁出现任何拉丁字母拼写的人名或英文单词。"
        "7) 真实皮肤微观质感的描写（雀斑、毛孔、绒毛、汗珠、肌肤质感等）只能出现在描写她面部/眼睛特写的那一句里，"
        "必须与「双眼/眼眸/瞳/面颊/脸」等面部词出现在同一句中；绝不能把这类皮肤质感描写挂在脚、胸口、发丝等其他部位的句子上。"
        "8) reveal_pose 里不要再写镜头如何拉远（拉远取景由模板固定插入），只写她最终的站姿/收势/光环落定/镜头轻微定格。"
    )


def build_user_prompt(axes, hero_name):
    zoom_list_cn = "、".join(f"{i+1}.{z['cn']}" for i, z in enumerate(axes["zoom_targets"]))
    director_line = ""
    if axes["director"]:
        director_line = f"\n导演风格参考（仅作氛围/运镜/光影措辞参考，绝不能覆盖固定法律条款）：{axes['director']['name']} —— {axes['director']['flavor']}。"
    skin_line = ""
    if axes["skin_detail"]:
        skin_line = (
            f"\n由于运镜包含面部/眼睛特写，请把这类真实皮肤微观质感描写：{axes['skin_detail']}——"
            f"只放进描写她双眼/面部那一句里（与「双眼/眼眸/瞳/面颊/脸」同句），绝不要挂到脚、胸口、发丝等其它部位的句子上。"
        )

    shape_line = ""
    if axes.get("shape_guidance"):
        shape_line = f"\n- 变身子形态（transform_shape = {axes['shape_id']}）：{axes['shape_guidance']}"
        if axes.get("is_covering_shape"):
            shape_line += (
                "\n  ★ 覆盖类形态硬性顺序：整段变身中她始终是主语，物质/暗影沿她的身体依次包裹（按给定镜头部位顺序），"
                "唯独「她的面部被最后吞没/笼罩」必须是变身核心段落的最后一个动作、紧接峰值光华之前；"
                "在此之前她的五官始终清晰可辨。变身全程她保持自身的人形轮廓，任何体型暴涨/尺寸变化只能留到12-15秒定格时或之后。"
                "模板会在这段之后自动插入对应的身份锁定句，你不要自己写。"
            )

    dark_vocab_hint = ""
    if axes["register"] == "dark_villain":
        dark_vocab_hint = (
            "\n提示：营造不祥/诡异氛围时，请改用完全不含禁用字的词，例如「诡谲」「邪异」「森冷」「阴冷」"
            "「骇然」「幽森」「凛冽」「森然」等，而不要用「不祥」「不安」这类带「不」字的常见写法。"
        )

    weapon_cn = axes["weapon"]["cn"]
    weapon_line = (
        f"武器：{weapon_cn}（在 morph_choreography 中让它在她手中/掌心凝聚成形，命名一次、清晰唯一；"
        f"reveal_pose 中要用同样的武器名称呼应，姿态与之匹配）。"
        if axes["weapon"]["id"] != "none-empty-handed"
        else "武器：无——她是空手的，能量凝聚在她的双掌/指尖，reveal_pose 描写她空手的收势姿态。"
    )

    return f"""请为以下轴线组合填充三个中文片段。

【参考范例：一条已验证通过的完整提示词（金标准，仅供文风/结构/尺度参考）——范例中的具体元素材质是"翠绿藤木/荆棘"，人物是"尼莎"，环境是"雨夜广场"，这些具体名词只属于范例本身，绝对不能出现在你的输出里；你必须只使用下面【本次要填充的轴线】中指定的元素材质、人物、环境】
{GOLD_EXAMPLE_CN}

【本次要填充的轴线】
- 主角：{hero_name}，{axes['body_type']}，{axes['ethnicity']}，穿着{axes['clothing']}，年龄18-35岁之间
- register（整体基调）：{axes['register']} —— {axes['register_guidance']}{dark_vocab_hint}{shape_line}
- 元素材质：{axes['element']['label']}（{axes['element']['cn_material']}）——{axes['element']['description']}
- 环境：{axes['environment']}
- 触发事件类别：{axes['incident']['label']} —— {axes['incident']['beat']}
- 镜头推进部位顺序（4-12秒变身核心段落必须按此顺序，全部是她自己的身体部位）：{zoom_list_cn}
- 镜头/光影参考词（可自然融入，不必逐字照抄）：{axes['lens']} 镜头，{axes['lighting']}
- {weapon_line}{skin_line}{director_line}

【需要你填充的三个字段，输出为严格JSON——字数是硬性上限，请严格遵守，宁可短一些也不要超】
1. intro_incident（对应0-4秒设定+触发，硬性上限55个中文字，目标40-55字）：交代环境与主角（自然融入年龄段体态穿着等描写，不要像列表一样罗列)，触发事件如何发生，她如何转身/停步准备迎接觉醒。要精炼，只写关键画面，不要铺陈。
2. morph_choreography（对应4-12秒变身核心，硬性上限170个中文字，目标130-170字）：元素材质如何从触点涌现、按给定镜头部位顺序依次在她身上成形为铠甲/纹路/武器（每个部位用给定的关键词，如"拳头""胸前""发丝""掌心""双眼""脚"等，只需一两处提及即可，不要堆砌形容词），运镜与她的身体动态（发丝衣摆翻飞、微微升空等）如何配合，段落结尾应到达能量光华爆发的峰值（她的身影在光芒中依然清晰可辨，不要写成完全白屏）。不要写身份锁定句，模板会自动加在这段后面。
3. reveal_pose（对应12-15秒定格收势，硬性上限70个中文字，目标50-70字）：镜头拉远的取景由模板固定插入（拉远至中景与全身之间、人物占据画面主要高度），你不要再写镜头怎么拉远；只写她最终的站姿、武器或空手的收势姿态、光环/光华缓缓落定、镜头轻微缓移后定格。

【配色铁律】
- 光环/光芒/瞳色等颜色描写必须自己根据本次的元素材质与register重新设定，绝对不能照抄范例的"翠金色""琥珀金"配色；dark_villain register 尤其要用暗色调/不祥的错色瞳孔，不能出现温暖英雄感的金色光环。

【字数与用字铁律】
- 三段合计中文字数不得超过290字，请自行边写边数，超出就删减形容词而不是删减动作。
- 全文（包括你写的每一个字）绝对不能出现这几个字：不、没有、别、勿、避免——哪怕是成语或固定搭配里带这个字也不行（例如"不离分毫""毫不"这类说法一律禁止，请换成完全不含这些字的表达）。
- 她永远是「她」，绝不能写成「一个/一道/一位」+「存在/生物/身影」这类说法来指代变身后的她。

现在只输出JSON对象，不要输出任何其他文字。"""


def call_ollama(ollama_url, model, system_prompt, user_prompt, temperature, timeout=180):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 900},
    }
    req = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("message", {}).get("content", "")


def stitch(pools, axes, fields):
    bp = pools["boilerplate"]
    style_law = bp["style_law"]
    camera_law = bp["camera_law"]
    # per-shape identity-lock: covering shapes (symbiote/reaper) use the variant
    # line that permits a fully-covered end state; everything else uses default.
    identity_lock = axes.get("identity_lock_line") or bp["identity_lock"]
    reveal_framing = bp["reveal_framing"]

    intro = fields["intro_incident"].strip()
    morph = fields["morph_choreography"].strip()
    reveal = fields["reveal_pose"].strip()

    full = (
        f"{style_law}{camera_law}"
        f"0-4秒：{intro}"
        f"4-12秒（核心：动态变身，仍为一镜到底）：{morph}"
        f"{identity_lock}"
        f"12-15秒：{reveal_framing}，{reveal}"
    )
    return full


def validate(pools, axes, fields, full_text):
    errors = []
    bp = pools["boilerplate"]

    # per-shape identity-lock: covering shapes require the variant line, all
    # other runs require the default line. Whichever this run selected must be
    # present verbatim.
    expected_lock = axes.get("identity_lock_line") or bp["identity_lock"]
    if expected_lock not in full_text:
        which = "variant (covering-shape)" if expected_lock == bp.get("identity_lock_variant") else "default"
        errors.append(f"identity-lock line missing or not verbatim (expected {which} lock)")

    # framing pin: the fixed reveal-framing clause must be present verbatim.
    if bp["reveal_framing"] not in full_text:
        errors.append("reveal-framing pin missing or not verbatim")

    timing_matches = TIMING_RE.findall(full_text)
    if len(timing_matches) < 3:
        errors.append(f"expected 3 timing markers (N-N秒), found {len(timing_matches)}: {timing_matches}")

    cjk_count = len(CJK_RE.findall(full_text))
    if not (320 <= cjk_count <= 540):
        errors.append(f"CJK char count {cjk_count} outside 320-540 range")

    # hero names (and any English) must be emitted in Chinese: reject Latin runs
    latin_hits = LATIN_SEQ_RE.findall(full_text)
    if latin_hits:
        errors.append(f"Latin-script token(s) in output (hero names must be Chinese): {sorted(set(latin_hits))}")

    llm_text = fields.get("intro_incident", "") + fields.get("morph_choreography", "") + fields.get("reveal_pose", "")
    for tok in NEGATION_TOKENS:
        if tok in llm_text:
            errors.append(f"negation token '{tok}' found in LLM-generated fill text")

    # new-noun check restricted to morph + reveal (the post-transform danger zone);
    # intro_incident legitimately describes an external threat figure with 一道/一位.
    danger_zone = fields.get("morph_choreography", "") + fields.get("reveal_pose", "")
    m = NEW_NOUN_RE.search(danger_zone)
    if m:
        errors.append(f"new-noun 'a being of X' pattern found: '{m.group(0)}'")

    morph_text = fields.get("morph_choreography", "")

    # skin-micro-detail must bind ONLY to the eyes/face beat: any sentence in the
    # morph (or reveal) carrying skin-detail vocabulary must also carry a face/eye
    # marker in the same sentence-neighborhood. Catches the probe-05 miss where
    # freckle/skin-texture language was attached to the feet/chest beats.
    skin_scan_text = morph_text + "。" + fields.get("reveal_pose", "")
    for sentence in SENTENCE_SPLIT_RE.split(skin_scan_text):
        skin_hit = next((t for t in SKIN_DETAIL_TOKENS if t in sentence), None)
        if skin_hit and not any(fm in sentence for fm in FACE_MARKERS):
            errors.append(
                f"skin-micro-detail '{skin_hit}' bound to a non-face beat "
                f"(no face/eye marker in its sentence): '{sentence.strip()[:40]}...'"
            )
            break

    for z in axes["zoom_targets"]:
        keywords = z.get("keywords") or [z["cn"]]
        if not any(kw in morph_text for kw in keywords):
            errors.append(f"assigned zoom target not mentioned in morph_choreography: {z['cn']} ({z['id']}, keywords tried: {keywords})")

    # few-shot leakage check: gold example's thornwood/Nyssa/plaza vocabulary
    # must not appear — it signals the model copied the example instead of
    # using the assigned element/hero.
    for tok in GOLD_EXAMPLE_LEAK_TOKENS:
        if tok in llm_text:
            errors.append(f"few-shot leakage: gold-example token '{tok}' found in LLM fill text")

    # element material fidelity: at least one keyword root of the assigned
    # element's material should appear in the morph choreography. Uses a
    # keyword list (not the full compound noun) because the model often
    # paraphrases compounds like 黑曜石玻璃 -> 黑曜石, which is a legitimate
    # partial reference, not a fidelity miss.
    material_keywords = axes["element"].get("keywords") or [axes["element"]["cn_material"]]
    if not any(kw in morph_text for kw in material_keywords):
        errors.append(
            f"assigned element material not mentioned in morph_choreography: "
            f"{axes['element']['cn_material']} (keywords tried: {material_keywords})"
        )

    return errors, cjk_count


def run_assembly(pools, seed, overrides, model, ollama_url, max_retries, zoom_count, zoom_ids,
                  director_override, runs_dir, verbose=True):
    axes = roll_axes(pools, seed, overrides, zoom_count, zoom_ids, director_override)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(axes, axes["hero_name"])

    os.makedirs(runs_dir, exist_ok=True)
    run_id = f"{int(time.time())}_seed{seed}_{model.replace(':', '-').replace('/', '-')}"

    temperatures = [0.5, 0.75, 0.95]
    last_result = None

    for attempt in range(1, max_retries + 1):
        temperature = temperatures[min(attempt - 1, len(temperatures) - 1)]
        attempt_log = {
            "run_id": run_id,
            "attempt": attempt,
            "seed": seed,
            "model": model,
            "temperature": temperature,
            "axes": {
                "register": axes["register"],
                "shape": axes.get("shape_id"),
                "element": axes["element"]["id"],
                "incident": axes["incident"]["id"],
                "hero_name": axes["hero_name"],
                "hero_name_latin": axes.get("hero_name_latin"),
                "environment": axes["environment"],
                "zoom_targets": [z["id"] for z in axes["zoom_targets"]],
                "clothing": axes["clothing"],
                "body_type": axes["body_type"],
                "ethnicity": axes["ethnicity"],
                "lens": axes["lens"],
                "lighting": axes["lighting"],
                "skin_detail": axes["skin_detail"],
                "director": axes["director"]["id"] if axes["director"] else None,
                "weapon": axes["weapon"]["id"],
            },
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        if verbose:
            print(f"[seed {seed} | {model}] attempt {attempt}/{max_retries} (temp={temperature}) ...",
                  file=sys.stderr)

        try:
            raw = call_ollama(ollama_url, model, system_prompt, user_prompt, temperature)
        except (urllib.error.URLError, TimeoutError) as e:
            attempt_log["error"] = f"ollama call failed: {e}"
            attempt_log["passed"] = False
            _write_log(runs_dir, run_id, attempt, attempt_log)
            last_result = attempt_log
            continue

        attempt_log["raw_llm_output"] = raw

        try:
            fields = json.loads(raw)
            assert isinstance(fields, dict)
            for k in ("intro_incident", "morph_choreography", "reveal_pose"):
                if k not in fields or not isinstance(fields[k], str):
                    raise ValueError(f"missing/invalid key '{k}'")
        except Exception as e:
            attempt_log["error"] = f"invalid JSON from model: {e}"
            attempt_log["passed"] = False
            _write_log(runs_dir, run_id, attempt, attempt_log)
            last_result = attempt_log
            continue

        full_text = stitch(pools, axes, fields)
        errors, cjk_count = validate(pools, axes, fields, full_text)

        attempt_log["parsed_fields"] = fields
        attempt_log["assembled_prompt"] = full_text
        attempt_log["cjk_char_count"] = cjk_count
        attempt_log["validation_errors"] = errors
        attempt_log["passed"] = len(errors) == 0

        _write_log(runs_dir, run_id, attempt, attempt_log)
        last_result = attempt_log

        if attempt_log["passed"]:
            if verbose:
                print(f"[seed {seed} | {model}] attempt {attempt} PASSED validation ({cjk_count} CJK chars)",
                      file=sys.stderr)
            return attempt_log, axes
        else:
            if verbose:
                print(f"[seed {seed} | {model}] attempt {attempt} FAILED: {errors}", file=sys.stderr)

    return last_result, axes


def _write_log(runs_dir, run_id, attempt, attempt_log):
    path = os.path.join(runs_dir, f"{run_id}_attempt{attempt}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(attempt_log, f, ensure_ascii=False, indent=2)


def contract_axes(axes, result, seed, model):
    """Flatten the rolled axes into the spec-contract `axes{}` block that
    plan_specs.js stamps onto every transformation_reveal_v2 spec."""
    demo = f"{axes['body_type']}, {axes['ethnicity']}, wearing {axes['clothing']}, age 18-35"
    return {
        "register": axes["register"],
        "shape": axes.get("shape_id"),
        "element": axes["element"]["id"],
        "incident_class": axes["incident"]["id"],
        "zoom_tour": [z["id"] for z in axes["zoom_targets"]],
        "demographics_summary": demo,
        "director_flavor": axes["director"]["id"] if axes.get("director") else None,
        "assembler_seed": seed,
        "assembler_model": model,
    }


def json_result(axes, result, seed, model):
    """The success payload printed to stdout in --json mode."""
    return {
        "ok": True,
        "prompt_zh": result["assembled_prompt"],
        "hero_latin": (axes.get("hero_name_latin") or "").lower(),
        "hero_cn": axes["hero_name"],
        "attempts": result.get("attempt"),
        "cjk_char_count": result.get("cjk_char_count"),
        "axes": contract_axes(axes, result, seed, model),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--pools", default=DEFAULT_POOLS)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--register", choices=["heroic", "awe", "dark_villain"], default=None)
    ap.add_argument("--shape", default=None,
                    help="dark_villain transform_shape: dark_armored | symbiote_engulf | demonic | shadow_reaper")
    ap.add_argument("--element", default=None, help="element id, e.g. obsidian-glass")
    ap.add_argument("--incident", default=None, help="incident id, e.g. accidental-contact")
    ap.add_argument("--hero", default=None)
    ap.add_argument("--environment", default=None)
    ap.add_argument("--weapon", default=None, help="weapon id")
    ap.add_argument("--zoom-count", type=int, default=None, help="2-4, random if omitted")
    ap.add_argument("--zoom", default=None, help="comma-separated zoom target ids, overrides sampling")
    ap.add_argument("--director", default=None, help="director id, or 'none' to force no flavor")
    ap.add_argument("--out-json", default=None, help="write final result JSON to this path")
    ap.add_argument("--json", action="store_true",
                    help="print ONE machine-readable JSON result to stdout (pipeline mode; "
                         "plan_specs.js consumes this). Non-zero exit + error JSON on failure.")
    args = ap.parse_args()

    pools = load_pools(args.pools)

    overrides = {
        "register": args.register,
        "shape": args.shape,
        "element": args.element,
        "incident": args.incident,
        "hero": args.hero,
        "environment": args.environment,
        "weapon": args.weapon,
    }

    result, axes = run_assembly(
        pools, args.seed, overrides, args.model, args.ollama_url, args.max_retries,
        args.zoom_count, args.zoom, args.director, args.runs_dir,
        verbose=not args.json,
    )

    # --json: emit ONE machine-readable object to stdout (pipeline contract). All
    # human/log noise stays on stderr, so stdout is a clean JSON document.
    if args.json:
        if result.get("passed"):
            sys.stdout.write(json.dumps(json_result(axes, result, args.seed, args.model),
                                        ensure_ascii=False) + "\n")
            sys.exit(0)
        err = {
            "ok": False,
            "error": result.get("error") or "validation failed after all retries",
            "attempts": result.get("attempt"),
            "validation_errors": result.get("validation_errors") or [],
            "assembler_seed": args.seed,
            "assembler_model": args.model,
        }
        sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
        sys.exit(1)

    print("\n" + "=" * 78)
    print(f"SEED {args.seed} | MODEL {args.model} | PASSED: {result.get('passed')}")
    print("=" * 78)
    if result.get("passed"):
        print(result["assembled_prompt"])
        print(f"\n[{result['cjk_char_count']} CJK chars]")
    else:
        print("FINAL ATTEMPT FAILED VALIDATION.")
        print("Errors:", result.get("validation_errors") or result.get("error"))
        if result.get("assembled_prompt"):
            print("\nLast assembled attempt (for inspection):")
            print(result["assembled_prompt"])

    if args.out_json:
        out = dict(result)
        out["axes_full"] = {
            "register": axes["register"],
            "shape_id": axes.get("shape_id"),
            "shape_guidance": axes.get("shape_guidance"),
            "identity_lock_line": axes.get("identity_lock_line"),
            "hero_name_latin": axes.get("hero_name_latin"),
            "element": axes["element"],
            "incident": axes["incident"],
            "hero_name": axes["hero_name"],
            "environment": axes["environment"],
            "zoom_targets": axes["zoom_targets"],
            "clothing": axes["clothing"],
            "body_type": axes["body_type"],
            "ethnicity": axes["ethnicity"],
            "lens": axes["lens"],
            "lighting": axes["lighting"],
            "skin_detail": axes["skin_detail"],
            "director": axes["director"],
            "weapon": axes["weapon"],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[wrote {args.out_json}]", file=sys.stderr)

    sys.exit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
