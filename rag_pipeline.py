"""
ArchiTech RAG Pipeline
======================
Stage 1 (NEW): Vector match user prompt → closest archetype JSON
Stage 2 (NEW): MODIFIER LLM — tweak the base JSON to match the prompt
Stage 3 (SAME): Reviewer — same Python geometry checks as before
"""

import os, json
from pathlib import Path

try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from schema import CityLayout, CityEntity, ReviewResult, USER_ALIASES

load_dotenv()

# ── Config ────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.3-70b-versatile"
OLLAMA_MODEL  = "qwen2.5:14b"
OLLAMA_URL    = "http://localhost:11434/v1"
GROQ_KEY_1    = os.environ.get("GROQ_API_KEY_1")
GROQ_KEY_2    = os.environ.get("GROQ_API_KEY_2")
if not GROQ_KEY_1 or not GROQ_KEY_2:
    raise RuntimeError(
        "Missing GROQ_API_KEY_1 / GROQ_API_KEY_2 (add in env)"
    )
ARCHETYPES_DIR = Path("./archetypes")
SHARED_FILE   = "./godot/data/current_city.json"
MAX_REVIEW_RETRIES = 2
MATCH_THRESHOLD = 0.35


# ── Clients ───────────────────────────────────────────────
class KeyPool:
    def __init__(self, keys):
        self._keys = keys
        self._index = 0
    def next_client(self):
        key = self._keys[self._index % len(self._keys)]
        self._index += 1
        return instructor.from_openai(
            OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key),
            mode=instructor.Mode.JSON,
        )

_pool = KeyPool([GROQ_KEY_1, GROQ_KEY_2])

_ollama = instructor.from_openai(
    OpenAI(base_url=OLLAMA_URL, api_key="ollama"),
    mode=instructor.Mode.JSON,
)


# ══════════════════════════════════════════════════════════
# STAGE 1 — VECTOR RETRIEVAL
# ══════════════════════════════════════════════════════════

def _load_archetypes() -> list[dict]:
    archetypes = []
    for path in sorted(ARCHETYPES_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        meta = data.get("_meta", {})
        archetypes.append({
            "name":        meta.get("name", path.stem),
            "description": meta.get("description", ""),
            "triggers":    meta.get("triggers", []),
            "path":        str(path),
            "layout":      {k: v for k, v in data.items() if k != "_meta"},
        })
    return archetypes


def _get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("  [RAG] sentence-transformers not installed. Run: pip install sentence-transformers")
        print("  [RAG] Falling back to keyword matching.")
        return None


# Structural keywords that strongly indicate a specific topology.
# Applied as score boosts AFTER embedding scoring — works on both paths.
STRUCTURAL_OVERRIDES = {
    "island":              ["surrounding", "surrounded by", "moat", "ring of water", "water surrounding", "river surrounding", "encircling", "encircled"],
    "river_bridge":        ["bridge", "bridge over", "cross the river", "bridge across"],
    "double_bridge":       ["two bridges", "double bridge", "multiple bridges"],
    "crossroad":           ["crossroad", "intersection", "crossing", "junction"],
    "both_sides":          ["both sides", "either side", "each side", "opposite sides"],
    "split":               ["separated by river", "divided by river", "river between", "split by"],
    "waterfront":          ["waterfront", "harbour", "harbor", "seafront", "facing water"],
    "farm_square":         ["farms in a square", "four farms", "farms arranged"],
    "skyscraper_district": ["skyscrapers", "tall buildings", "high rise"],
    "stadium_district":    ["stadium", "arena", "sports venue"],
}

def _keyword_score(prompt_lower: str, archetype: dict) -> float:
    score = 0.0
    desc_words = set(archetype["description"].lower().split())
    prompt_words = set(prompt_lower.split())

    arch_name = archetype["name"]
    if arch_name in STRUCTURAL_OVERRIDES:
        for kw in STRUCTURAL_OVERRIDES[arch_name]:
            if kw in prompt_lower:
                score += 0.8
                break

    for trigger in archetype["triggers"]:
        if trigger.lower() in prompt_lower:
            score += 0.3

    overlap = len(prompt_words & desc_words)
    score += overlap * 0.04
    return min(score, 1.0)


def _merge_two(a: dict, b: dict, best: dict) -> dict:
    """
    Merge entity lists of archetypes a and b.
    `best` is whichever of the two should win on position conflicts.
    Returns a new layout dict.
    """
    loser  = b if best is a else a
    winner = best

    merged_cells: dict[tuple, dict] = {}
    for e in loser["layout"].get("entities", []):
        key = (e["position"][0], e["position"][2])
        merged_cells[key] = e
    for e in winner["layout"].get("entities", []):
        key = (e["position"][0], e["position"][2])
        merged_cells[key] = e

    merged_layout = dict(winner["layout"])
    merged_layout["entities"]  = list(merged_cells.values())
    merged_layout["city_name"] = (
        f"{winner['layout'].get('city_name', winner['name'])} + "
        f"{loser['layout'].get('city_name', loser['name'])}"
    )
    result = dict(winner)
    result["layout"] = merged_layout
    result["name"]   = f"{winner['name']} + {loser['name']}"
    return result


def retrieve_archetype(prompt: str) -> tuple[dict, float]:
    """
    Find the closest archetype to the user prompt.
    Returns (archetype_dict, similarity_score).

    - Uses sentence-transformers if available, keyword fallback otherwise.
    - Applies STRUCTURAL_OVERRIDES boost on top of embedding scores.
    - Merges exactly TWO archetypes when the prompt explicitly triggers both
      (e.g. "square formation" + "river surrounding").
    - Score-gap fallback (very tight: 0.05) for cases with no keyword overlap.
    """
    archetypes = _load_archetypes()
    if not archetypes:
        raise RuntimeError(f"No archetype JSONs found in {ARCHETYPES_DIR}")

    embedder = _get_embedder()
    prompt_lower = prompt.lower()

    if embedder is not None:
        corpus = [
            a["description"] + " " + " ".join(a["triggers"])
            for a in archetypes
        ]
        corpus_embs = embedder.encode(corpus, normalize_embeddings=True)
        prompt_emb  = embedder.encode([prompt], normalize_embeddings=True)[0]
        scores = list(corpus_embs @ prompt_emb)
    else:
        scores = [_keyword_score(prompt_lower, a) for a in archetypes]

    # Apply structural keyword boosts on top of whatever scoring method was used
    for i, a in enumerate(archetypes):
        arch_name = a["name"]
        if arch_name in STRUCTURAL_OVERRIDES:
            for kw in STRUCTURAL_OVERRIDES[arch_name]:
                if kw in prompt_lower:
                    scores[i] = min(1.0, float(scores[i]) + 0.45)
                    break

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best_idx   = ranked[0]
    best_score = float(scores[best_idx])
    best       = archetypes[best_idx]

    print(f"  [RAG] Best match: '{best['name']}'  (score={best_score:.3f})")
    if best_score < MATCH_THRESHOLD:
        print(f"  [RAG] ⚠ Low confidence match — consider adding a new archetype for this prompt type.")

    for rank, idx in enumerate(ranked[:3], 1):
        print(f"    #{rank}: {archetypes[idx]['name']:20s}  {float(scores[idx]):.3f}")

    # ── Multi-concept merge (keyword-triggered, max 2 archetypes) ──
    # Collect archetypes whose keywords are literally present in the prompt.
    # Cap at 2 (top scoring) to avoid pulling in road archetypes accidentally.
    triggered_idxs: list[int] = []
    for i, a in enumerate(archetypes):
        arch_name = a["name"]
        matched = False
        if arch_name in STRUCTURAL_OVERRIDES:
            for kw in STRUCTURAL_OVERRIDES[arch_name]:
                if kw in prompt_lower:
                    matched = True
                    break
        if not matched:
            for trigger in a["triggers"]:
                if trigger.lower() in prompt_lower:
                    matched = True
                    break
        if matched:
            triggered_idxs.append(i)

    # Keep only the top 2 by score to avoid accidentally merging road archetypes
    triggered_idxs = sorted(triggered_idxs, key=lambda i: scores[i], reverse=True)[:2]

    if len(triggered_idxs) == 2:
        merge_names = [archetypes[i]["name"] for i in triggered_idxs]
        print(f"  [RAG] Multi-concept detected — merging {merge_names}")
        merged = _merge_two(archetypes[triggered_idxs[0]], archetypes[triggered_idxs[1]], best)
        return merged, best_score

    # ── Fallback: score-gap merge (very tight threshold) ──
    MERGE_THRESHOLD_GAP = 0.05
    if len(ranked) > 1:
        runner_up_idx   = ranked[1]
        runner_up_score = float(scores[runner_up_idx])
        if best_score - runner_up_score < MERGE_THRESHOLD_GAP:
            runner_up = archetypes[runner_up_idx]
            print(f"  [RAG] Score-gap merge — '{best['name']}' + '{runner_up['name']}'")
            merged = _merge_two(best, runner_up, best)
            return merged, best_score

    return best, best_score


# ══════════════════════════════════════════════════════════
# STAGE 2 — MODIFIER LLM
# ══════════════════════════════════════════════════════════

MODIFIER_SYSTEM = """You are the MODIFIER stage of ArchiTech, a 3D voxel-city generator.

You are given a BASE JSON layout that already has correct, valid coordinates.
Your job is to MODIFY it to match the user's request.

RULES:
  1. PRESERVE existing coordinates — do NOT move entities that are already placed correctly.
  2. PRESERVE STRUCTURAL PATTERNS — if the base has a water ring, a road perimeter, or a bridge,
     keep ALL tiles of that structure. Never reduce a ring to just 4 tiles when the base has 12+.
     "surrounding" in the prompt = keep the COMPLETE ring from the base JSON, every tile.
  3. You may ADD new entities to match the user's request (same grid rules: x,z multiples of 10, y=0).
  4. You may REMOVE entities that contradict the user's request (e.g. base has parks but user wants water).
  5. You may SWAP types — e.g. change "buildings" to "skyscrapers" or "parks" to "water".
  6. You may ADD more tiles of the same type to extend (e.g. user wants a longer road).
  7. NEVER add roads unless the user explicitly said "road" or "street".
     If the BASE JSON contains roads but the user did not ask for roads, REMOVE them.
  8. NEVER add bridges unless the user explicitly said "bridge".
  9. COUNT RULE: place EXACTLY the number of each thing the user mentioned.
  10. TYPE MAPPING:
       airballoon / balloon    → "sports"
       farmhouse / barn        → "farm"
       river                   → "water"
  11. RING RULE: if the base JSON has a water ring (tiles on all 4 sides AND corners),
      output ALL of those water tiles. The ring only works if it is complete.

GRID RULES:
  - Each tile is 10m × 10m. x and z MUST be multiples of 10. y = 0 always.
  - ONE entity per (x, z) cell.
  - position format: [x, y, z] — THREE separate numbers, never merged.
    CORRECT:   "position": [10, 0, 20]
    WRONG:     "position": [10020]  or  "position": ["10", "0", "20"]
  - rotation format: [x, y, z] — always output [0, 0, 0] for everything.
    The engine corrects rotations automatically.

OUTPUT: valid CityLayout JSON only. No explanation."""


def _modify(base_layout: dict, prompt: str, archetype_name: str, extra: str = "") -> CityLayout:
    base_json = json.dumps(base_layout, indent=2)
    content = (
        f"{MODIFIER_SYSTEM}\n\n"
        f"BASE ARCHETYPE: {archetype_name}\n"
        f"BASE JSON:\n{base_json}\n\n"
        f"USER REQUEST: {prompt}"
    )
    if extra:
        content += f"\n\nCORRECTION NOTES FROM REVIEWER:\n{extra}"

    label = "Stage 2 -- MODIFIER" + (" (retry)" if extra else "")
    print(f"\n[ArchiTech] -- {label} (Groq) --")

    client = _pool.next_client()
    try:
        return client.chat.completions.create(
            model=GROQ_MODEL,
            response_model=CityLayout,
            max_retries=3,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        print(f"  [MODIFIER] Structured parse failed, trying manual repair...")
        import re
        raw_client = _pool.next_client()
        resp = raw_client.chat.completions.create(
            model=GROQ_MODEL,
            response_model=None,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        def _fix_list(lst, expected_len, default):
            if not isinstance(lst, list):
                lst = [lst]
            if len(lst) == 1:
                parts = re.findall(r"-?\d+", str(lst[0]))
                if len(parts) == expected_len:
                    return [float(p) for p in parts]
                return default
            if len(lst) == 2 and expected_len == 3:
                return [float(lst[0]), 0.0, float(lst[1])]
            return [float(v) for v in lst[:expected_len]]

        good = []
        for ent in data.get("entities", []):
            try:
                ent["position"] = _fix_list(ent.get("position", [0,0,0]), 3, [0.0,0.0,0.0])
                ent["rotation"] = _fix_list(ent.get("rotation",  [0,0,0]), 3, [0.0,0.0,0.0])
                good.append(CityEntity(**ent))
                print(f"    Repaired {ent.get('id','?')}: pos={ent['position']}")
            except Exception as ee:
                print(f"    Skipping bad entity {ent.get('id','?')}: {ee}")
        if not good:
            raise RuntimeError("No valid entities after repair") from e
        return CityLayout(city_name=data.get("city_name", "City"), entities=good)


# ══════════════════════════════════════════════════════════
# STAGE 3 — REVIEWER
# ══════════════════════════════════════════════════════════

def _resolve_aliases(layout: CityLayout) -> CityLayout:
    for e in layout.entities:
        if e.type in USER_ALIASES:
            e.type = USER_ALIASES[e.type]
    return layout


def _auto_rotate(layout: CityLayout) -> CityLayout:
    BUILDING_TYPES = {"buildings","houses","skyscrapers","shops","supermarkets","factory","stadium","gasstation"}
    FLAT_TYPES     = {"parks","landscape","parkingareas","farm","sports","water"}

    road_cells = {(e.position[0], e.position[2]): e for e in layout.entities if e.type == "roads"}

    assigned = {}
    for e in [e for e in layout.entities if e.type == "roads"]:
        x, z = e.position[0], e.position[2]
        has_ew = (x+10,z) in road_cells or (x-10,z) in road_cells
        has_ns = (x,z+10) in road_cells or (x,z-10) in road_cells
        if has_ew and not has_ns:
            e.rotation = [0.0, 0.0, 0.0];  assigned[(x,z)] = 0.0
        elif has_ns and not has_ew:
            e.rotation = [0.0, 90.0, 0.0]; assigned[(x,z)] = 90.0

    dominant = 0.0 if sum(1 for v in assigned.values() if v==0.0) >= sum(1 for v in assigned.values() if v==90.0) else 90.0
    for e in [e for e in layout.entities if e.type == "roads"]:
        if (e.position[0], e.position[2]) not in assigned:
            e.rotation = [0.0, dominant, 0.0]

    for e in layout.entities:
        if e.type in FLAT_TYPES:
            e.rotation = [0.0, 0.0, 0.0]

    for e in layout.entities:
        if e.type not in BUILDING_TYPES:
            continue
        x, z = e.position[0], e.position[2]
        for cell, rot_y in [((x,z+10),180.0),((x,z-10),0.0),((x+10,z),90.0),((x-10,z),270.0)]:
            if cell in road_cells:
                e.rotation = [0.0, rot_y, 0.0]
                break
    return layout


def _review(layout: CityLayout, user_prompt: str) -> ReviewResult:
    import re
    issues = []
    road_cells  = {(e.position[0], e.position[2]) for e in layout.entities if e.type == "roads"}
    water_cells = {(e.position[0], e.position[2]) for e in layout.entities if e.type == "water"}
    BUILDING_TYPES = {"buildings","houses","skyscrapers","shops","supermarkets","factory","stadium"}

    user_mentioned_roads = any(w in user_prompt.lower() for w in ["road","street","highway","avenue","lane"])
    if not user_mentioned_roads:
        for e in layout.entities:
            if e.type == "roads":
                issues.append(f"{e.id} (road) was placed but user did not ask for roads — remove it")

    if user_mentioned_roads and road_cells:
        for e in layout.entities:
            if e.type in BUILDING_TYPES:
                x, z = e.position[0], e.position[2]
                adjacent = any(c in road_cells for c in [(x+10,z),(x-10,z),(x,z+10),(x,z-10)])
                if not adjacent:
                    issues.append(f"{e.id} ({e.type}) at {[int(p) for p in e.position[:3]]} has no adjacent road tile")

    for e in layout.entities:
        if e.type == "bridges":
            if (e.position[0], e.position[2]) not in water_cells:
                issues.append(f"{e.id} (bridge) is not on a water tile — remove it")
            if "bridge" not in user_prompt.lower():
                issues.append(f"{e.id} (bridge) was placed but user did not ask for a bridge — remove it")

    number_words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    COUNTABLE = {
        "building": {"buildings","houses","skyscrapers"},
        "house":    {"houses"},
        "shop":     {"shops"},
        "skyscraper":{"skyscrapers"},
        "road":     {"roads"},
        "park":     {"parks"},
        "tree":     {"trees"},
        "factory":  {"factory"},
        "stadium":  {"stadium"},
        "farm":     {"farm"},
    }
    for keyword, type_set in COUNTABLE.items():
        pattern = r'(\d+|' + '|'.join(number_words) + r')\s+' + keyword + r's?'
        m = re.search(pattern, user_prompt.lower())
        if m:
            raw = m.group(1)
            expected = int(raw) if raw.isdigit() else number_words[raw]
            actual = sum(1 for e in layout.entities if e.type in type_set)
            if actual != expected:
                issues.append(f"User asked for {expected} {keyword}(s) but {actual} placed. Place exactly {expected}.")

    if issues:
        return ReviewResult(
            approved=False, issues=issues,
            correction_prompt="Fix these errors:\n" + "\n".join(f"  - {i}" for i in issues)
        )
    return ReviewResult(approved=True, issues=[], correction_prompt=None)


# ══════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════

def build_city_rag(prompt: str, output_path: str = SHARED_FILE) -> CityLayout:
    print(f"\n{'='*60}")
    print(f"[ArchiTech RAG] Prompt: {prompt}")
    print(f"{'='*60}")

    archetype, score = retrieve_archetype(prompt)
    print(f"  Using archetype: {archetype['name']}  ({score:.3f})")

    layout = None
    correction = ""
    for attempt in range(1, MAX_REVIEW_RETRIES + 2):
        layout = _modify(archetype["layout"], prompt, archetype["name"], extra=correction)
        layout = _resolve_aliases(layout)
        layout = _auto_rotate(layout)

        review = _review(layout, user_prompt=prompt)
        if review.approved:
            print(f"  ✓ Reviewer approved on attempt {attempt}.")
            break
        else:
            print(f"  ✗ Reviewer rejected (attempt {attempt}/{MAX_REVIEW_RETRIES+1}):")
            for issue in review.issues:
                print(f"    - {issue}")
            if attempt <= MAX_REVIEW_RETRIES:
                correction = review.correction_prompt or ""
            else:
                print("  Max retries reached. Using last layout.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(layout.model_dump_json(indent=2))
    print(f"\n[ArchiTech RAG] Saved → {output_path}")
    _print_summary(layout)
    return layout


def _print_summary(layout: CityLayout):
    col = max(len(e.type) for e in layout.entities) + 2
    print(f"\n{'─'*60}")
    print(f"  {layout.city_name}  ({len(layout.entities)} entities)")
    print(f"{'─'*60}")
    for e in layout.entities:
        pos = f"[{int(e.position[0]):>4}, {int(e.position[1])}, {int(e.position[2]):>4}]"
        rot = f"y={int(e.rotation[1]):>3}"
        print(f"  {e.type:<{col}} {e.id:<24} {pos}  {rot}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    build_city_rag(
        prompt="A waterfront with buildings facing a wide stretch of water to the south, a road between buildings and water",
        output_path=SHARED_FILE,
    )