import os

try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from schema import CityLayout, SceneBrief, ReviewResult, USER_ALIASES

load_dotenv()

GROQ_MODEL  = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_URL   = "http://localhost:11434/v1"

GROQ_KEY_1 = os.environ.get("GROQ_API_KEY_1")
GROQ_KEY_2 = os.environ.get("GROQ_API_KEY_2")
if not GROQ_KEY_1 or not GROQ_KEY_2:
    raise RuntimeError(
        "Missing GROQ_API_KEY_1 / GROQ_API_KEY_2. (add in env)"
    )
MAX_REVIEW_RETRIES = 2
SHARED_FILE = "./godot/data/current_city.json"


class KeyPool:
    def __init__(self, keys: list[str]):
        self._keys = keys
        self._index = 0

    def next_client(self) -> instructor.Instructor:
        key = self._keys[self._index % len(self._keys)]
        self._index += 1
        return instructor.from_openai(
            OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key),
            mode=instructor.Mode.JSON,
        )


_pool = KeyPool([GROQ_KEY_1, GROQ_KEY_2])

# Ollama client for Stage 1 — local model, no rate limit
_ollama_client = instructor.from_openai(
    OpenAI(base_url=OLLAMA_URL, api_key="ollama"),
    mode=instructor.Mode.JSON,
)


def _sanitize_layout_json(raw: str) -> str:
    """Fix common LLM JSON corruption before Pydantic validation.
    - [30010] -> can't fix, but we can detect & skip
    - rotation '000' -> 0.0
    - position with wrong item count -> skip entity
    """
    import re
    # Fix rotation/position values that are stringified numbers like '000', '090', '0180'
    raw = re.sub(r'"(\d{2,4})"', lambda m: str(float(m.group(1))), raw)
    # Fix [30010] style merged coordinates - replace with spaces: hard to fix so just log
    return raw


def _call_ollama(response_model, messages: list[dict], label: str):
    print(f"\n[ArchiTech] -- {label} (Ollama) --")
    return _ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        response_model=response_model,
        max_retries=3,
        messages=messages,
        extra_body={"options": {"num_ctx": 4096, "temperature": 0.2}},
    )


def _call_groq(response_model, messages: list[dict], label: str):
    client = _pool.next_client()
    print(f"\n[ArchiTech] -- {label} (Groq) --")
    return client.chat.completions.create(
        model=GROQ_MODEL,
        response_model=response_model,
        max_retries=3,
        messages=messages,
    )


PLANNER_SYSTEM = """You are the PLANNER stage of ArchiTech, a 3D voxel-city generator.

Read the user prompt and output a SceneBrief. No coordinates — only zones and spatial relationships.

CRITICAL — ONLY PLAN WHAT IS ASKED:
  - Count exactly what the user mentioned and plan only those things.
  - "two buildings and a river" → 2 buildings, 1 river. No roads, no parks, no extras.
  - "a road with buildings" → add a road. Otherwise skip roads entirely.
  - NEVER add roads, landscape, trees, or decorations the user did not mention.
  - grid_width_tiles and grid_depth_tiles should be the MINIMUM needed. Simple prompt = 3x3 or less.

COORDINATE SYSTEM:
  x-axis = EAST-WEST  (x increases going east/right)
  z-axis = NORTH-SOUTH (z increases going SOUTH, decreases going NORTH)
  north of z=0 is z=-10, z=-20 ...
  south of z=0 is z=+10, z=+20 ...

Available asset types:
  roads, water, parks, landscape, trees, buildings, houses, skyscrapers,
  shops, supermarkets, factory, farm, gasstation, parkingareas, transport,
  bridges, stadium, sports, street_signs"""


PLACER_SYSTEM = """You are the PLACER stage of ArchiTech, a 3D voxel-city generator.

You receive a SceneBrief and must output a CityLayout with exact coordinates.

CRITICAL RULE — MINIMALISM:
  - Place ONLY what the user explicitly asked for. Nothing else.
  - "two buildings and a river" = 2 buildings + river tiles + 1 road. That is it.
  - Do NOT add extra roads, landscape fillers, parks, trees, or decorations unless asked.
  - Do NOT add bridges unless the user explicitly said "bridge".
  - Small prompts = small layouts. 5-15 entities maximum for simple requests.

COORDINATE SYSTEM:
  x-axis = EAST-WEST  (x increases going east/right)
  z-axis = NORTH-SOUTH (z INCREASES going SOUTH, DECREASES going NORTH)
  north of road at z=0 means buildings at z=-10, z=-20 ...
  south of road at z=0 means water/features at z=+10, z=+20 ...

GRID RULES (absolute):
  - Each tile is 10m x 10m.
  - x and z MUST be exact multiples of 10. y = 0 always.
  - ONE entity per (x,z) cell. Exception: bridge may stack on water.
  - Pack tightly from [0,0,0]. No gaps.
  - Long road/river/park = multiple adjacent tiles.
    Example: 5-tile east-west road at z=0: [0,0,0],[10,0,0],[20,0,0],[30,0,0],[40,0,0]

PLACEMENT RULES:
  1. Lay roads first.
  2. Buildings "next to each other along the road" = a ROW at the same z, x incrementing.
     e.g. buildings north of road at z=0: [0,0,-10],[10,0,-10],[20,0,-10]
  3. SPATIAL RELATIONSHIPS — translate natural language directly to grid coordinates:

     OPPOSITE SIDES (between / separating / dividing / in between):
       "river between two buildings"    → building at z=-10, river at z=0, building at z=+10
       "park separating two roads"      → road at z=-10, park at z=0, road at z=+10
       "road dividing north and south"  → same pattern, road in middle
       "X and Y with Z in between"      → X at -10, Z at 0, Y at +10
       CRITICAL: the two outer things MUST be on opposite sides (one negative, one positive z or x).
       NEVER place both outer things at the same z or x value.

     ADJACENCY (next to / beside / adjacent / neighboring):
       "three shops next to each other" → same z, x=0, x=10, x=20
       "buildings beside the road"      → road at z=0, buildings at z=-10 (or z=+10)
       "park next to the river"         → river at x=0, park at x=10

     CARDINAL DIRECTIONS (north/south/east/west):
       "north of" → smaller z (z=-10 is north of z=0)
       "south of" → larger z  (z=+10 is south of z=0)
       "east of"  → larger x  (x=+10 is east of x=0)
       "west of"  → smaller x (x=-10 is west of x=0)
       e.g. "factory north of road" → road at z=0, factory at z=-10

     FACING / ALONG (facing / along / lining / bordering):
       "buildings facing the road"   → buildings at z=-10 facing south toward road at z=0
       "shops lining the street"     → shops in a row at same z, road tiles at z+10 or z-10
       "houses along the river bank" → river at z=0, houses in a row at z=-10

     BEHIND / IN FRONT OF:
       "behind the buildings" → one z step further from the viewer
       "in front of"          → one z step closer to the viewer (toward +z)

     SURROUNDING / AROUND / ENCLOSING:
       "park surrounding the building" → building at center, park tiles on all 4 sides
       e.g. building at [10,0,10], parks at [0,0,10],[20,0,10],[10,0,0],[10,0,20]

     SQUARE / RING FORMATION WITH CENTER:
       For "4 buildings in a square with park in center", use a 3x3 grid:
         - 4 buildings at the 4 corners
         - park at the exact center cell
         - roads (if asked) on the 4 edge cells between corners

       CANONICAL 3x3 layout (ALWAYS use these EXACT coordinates, no exceptions):
         NW building:  [0, 0,  0]
         N  road/gap:  [10,0,  0]   ← only place if roads explicitly asked
         NE building:  [20,0,  0]
         W  road/gap:  [0, 0, 10]   ← only place if roads explicitly asked
         CENTER park:  [10,0, 10]   ← ALWAYS here, never anywhere else
         E  road/gap:  [20,0, 10]   ← only place if roads explicitly asked
         SW building:  [0, 0, 20]
         S  road/gap:  [10,0, 20]   ← only place if roads explicitly asked
         SE building:  [20,0, 20]

       CRITICAL SPACING RULE: buildings are 20 units apart (not 10).
         NW=[0,0,0], NE=[20,0,0], SW=[0,0,20], SE=[20,0,20] — park at [10,0,10].
         NEVER place buildings at [0,0,0],[10,0,0],[0,0,10],[10,0,10] — that is 10-unit spacing and leaves NO room for center park.
       "square formation" or "4 buildings in a square" = use this exact 3x3 grid, every time.

     OUTER RING OF ROADS (roads surrounding a block):
       "roads surrounding the buildings" = roads OUTSIDE the building footprint, one step beyond.
       The 4 buildings sit at [0,0,0],[20,0,0],[0,0,20],[20,0,20] and park at [10,0,10].
       Roads go OUTSIDE this block — they do NOT go between the buildings.

       CORRECT outer ring for the 3x3 block:
         Top row:    [-10,0,-10],[0,0,-10],[10,0,-10],[20,0,-10],[30,0,-10]
         Bottom row: [-10,0,30], [0,0,30], [10,0,30], [20,0,30], [30,0,30]
         Left col:   [-10,0,0],  [-10,0,10],[-10,0,20]
         Right col:  [30,0,0],   [30,0,10], [30,0,20]

       WRONG (never do this — these are between the buildings, not surrounding them):
         [10,0,0],[0,0,10],[20,0,10],[10,0,20]  ← these are INTERIOR edge cells, not outer ring

       MEMORY AID: if a road cell is adjacent to a building cell, it is BETWEEN buildings.
       Outer ring roads are adjacent to buildings on ONE side and empty ground on the other.

     GRID / BLOCK FORMATION:
       "city block" or "grid of roads" → roads forming a grid, buildings filling interior cells
       "crossroads" or "intersection"  → one east-west road crossing one north-south road at a shared tile

     PARALLEL / OPPOSITE / BOTH SIDES:
       "two roads parallel to each other" → road at z=0, road at z=20 (gap between)
       "buildings on both sides of road"  → buildings at z=-10 AND z=+10, road at z=0
       "shops on either side of river"    → shops at z=-10, river at z=0, shops at z=+10
       CRITICAL: "both sides" always means SOME entities at negative z AND some at positive z.
       Never place all entities at the same z when "both sides" or "either side" is mentioned.

     COUNT RULE: place EXACTLY the number of each thing the user mentioned.
       "two buildings" = 2 entities. "a river" = however many tiles fit. Never add extras.

     AXIS CHOICE: pick the axis that best matches the described layout.
       Left-right descriptions → use x-axis.
       North-south / up-down descriptions → use z-axis.
  4. Each building must share its x OR z with an adjacent road tile (differ by exactly 10) ONLY IF roads exist.
  5. Trees beside parks/roads, never ON a road.
  5. ABSOLUTE RULE: NEVER place bridges. Ever. Unless the user's exact words include 'bridge'.
  6. Use "landscape" to fill empty cells.
  7. TYPE MAPPING — always use these exact type strings:
       airballoon / hot air balloon / balloon → "sports"
       farmhouse / barn / farm field / crops  → "farm"
       NEVER use "landscape" for an airballoon or balloon.

rotation: always output [0, 0, 0] for everything — rotations are corrected automatically after placement."""


def _plan(prompt: str) -> SceneBrief:
    return _call_ollama(
        SceneBrief,
        [{"role": "user", "content": f"{PLANNER_SYSTEM}\n\nUSER REQUEST: {prompt}"}],
        label="Stage 1 -- PLANNER",
    )


def _place(brief: SceneBrief, extra_instructions: str = "") -> CityLayout:
    content = f"{PLACER_SYSTEM}\n\nSCENE BRIEF:\n{brief.model_dump_json(indent=2)}"
    if extra_instructions:
        content += f"\n\nCORRECTION NOTES FROM REVIEWER:\n{extra_instructions}"
    label = "Stage 2 -- PLACER" + (" (retry)" if extra_instructions else "")

    try:
        return _call_groq(CityLayout, [{"role": "user", "content": content}], label=label)
    except Exception as e:
        print(f"  [PLACER] Structured parse failed, trying manual repair...")
        import json
        from schema import CityEntity
        client = _groq_client()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        good_entities = []
        for ent in data.get("entities", []):
            try:
                pos = ent.get("position", [])
                rot = ent.get("rotation", [0, 0, 0])
                if isinstance(pos, list) and len(pos) == 3:
                    ent["position"] = [float(v) for v in pos]
                    ent["rotation"] = [float(v) for v in (rot if isinstance(rot, list) and len(rot) == 3 else [0, 0, 0])]
                    good_entities.append(CityEntity(**ent))
            except Exception as ee:
                print(f"    Skipping bad entity {ent.get('id','?')}: {ee}")
        if not good_entities:
            raise RuntimeError("No valid entities after repair") from e
        return CityLayout(city_name=data.get("city_name", "City"), entities=good_entities)


def _review(brief: SceneBrief, layout: CityLayout, user_prompt: str = "") -> ReviewResult:
    # Pure Python geometry checks — no LLM for math
    issues: list[str] = []

    road_cells = {
        (e.position[0], e.position[2])
        for e in layout.entities if e.type == "roads"
    }
    water_cells = {
        (e.position[0], e.position[2])
        for e in layout.entities if e.type == "water"
    }
    BUILDING_TYPES = {
        "buildings", "houses", "skyscrapers", "shops",
        "supermarkets", "factory", "stadium"
    }

    # Only enforce road-adjacency if user asked for roads OR layout actually has road tiles
    user_wants_roads = (
        any(w in user_prompt.lower() for w in ["road", "street", "highway", "avenue", "lane"])
        or len(road_cells) > 0
    )
    # But if roads exist and user did NOT ask for them, flag the roads themselves as unwanted
    user_mentioned_roads = any(w in user_prompt.lower() for w in ["road", "street", "highway", "avenue", "lane"])
    if not user_mentioned_roads:
        for e in layout.entities:
            if e.type == "roads":
                issues.append(f"{e.id} (road) was placed but user did not ask for roads — remove it")

    def has_adjacent_road(pos):
        x, z = pos[0], pos[2]
        return (
            (x + 10, z) in road_cells or
            (x - 10, z) in road_cells or
            (x, z + 10) in road_cells or
            (x, z - 10) in road_cells
        )

    if user_wants_roads:
        for e in layout.entities:
            if e.type in BUILDING_TYPES:
                if not has_adjacent_road(e.position):
                    issues.append(
                        f"{e.id} ({e.type}) at {[int(p) for p in e.position[:3]]} has no adjacent road tile"
                    )

    for e in layout.entities:
        if e.type == "bridges":
            cell = (e.position[0], e.position[2])
            if cell not in water_cells:
                issues.append(
                    f"{e.id} (bridge) at {[int(p) for p in e.position[:3]]} is not on a water tile"
                )

    # Enforce exact entity counts from the prompt
    import re
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    COUNTABLE_TYPES = {
        "building": {"buildings", "houses", "skyscrapers"},
        "house": {"houses"},
        "shop": {"shops"},
        "skyscraper": {"skyscrapers"},
        "road": {"roads"},
        "park": {"parks"},
        "tree": {"trees"},
        "factory": {"factory"},
        "stadium": {"stadium"},
    }
    prompt_lower = user_prompt.lower()
    for keyword, type_set in COUNTABLE_TYPES.items():
        # match "two buildings", "2 buildings", "2 houses" etc
        pattern = r'(\d+|' + '|'.join(number_words.keys()) + r')\s+' + keyword + r's?'
        match = re.search(pattern, prompt_lower)
        if match:
            raw = match.group(1)
            expected = int(raw) if raw.isdigit() else number_words[raw]
            actual = sum(1 for e in layout.entities if e.type in type_set)
            if actual != expected:
                issues.append(
                    f"User asked for {expected} {keyword}(s) but {actual} were placed. "
                    f"Place exactly {expected}."
                )

    # Check for unsolicited bridges
    for e in layout.entities:
        if e.type == "bridges":
            issues.append(
                f"{e.id} (bridge) was placed but user did not ask for a bridge — remove it"
            )

    # Check square/ring formation — all 4 sides must be present
    square_keywords = ["square", "ring", "loop", "surround", "enclosed", "formation"]
    if any(kw in user_prompt.lower() for kw in square_keywords) and road_cells:
        xs = sorted(set(c[0] for c in road_cells))
        zs = sorted(set(c[1] for c in road_cells))
        if len(xs) >= 2 and len(zs) >= 2:
            min_x, max_x = xs[0], xs[-1]
            min_z, max_z = zs[0], zs[-1]
            # Check all 4 sides have at least one road tile
            top    = any(c[1] == min_z for c in road_cells)
            bottom = any(c[1] == max_z for c in road_cells)
            left   = any(c[0] == min_x for c in road_cells)
            right  = any(c[0] == max_x for c in road_cells)
            missing = []
            if not top:    missing.append("top (north) side")
            if not bottom: missing.append("bottom (south) side")
            if not left:   missing.append("left (west) side")
            if not right:  missing.append("right (east) side")
            if missing:
                issues.append(
                    f"Square formation is incomplete — missing roads on: {', '.join(missing)}. "
                    "Add road tiles to complete all 4 sides of the square."
                )

    # Check "both sides" constraint — if prompt mentions both/either side,
    # buildings should appear on both sides of the central feature
    both_sides_keywords = ["both sides", "either side", "each side", "on both", "opposite sides"]
    if any(kw in user_prompt.lower() for kw in both_sides_keywords):
        BUILDING_TYPES = {"buildings", "houses", "skyscrapers", "shops", "supermarkets", "factory", "stadium"}
        building_z_values = [e.position[2] for e in layout.entities if e.type in BUILDING_TYPES]
        if building_z_values:
            has_negative_z = any(z < 0 for z in building_z_values)
            has_positive_z = any(z > 0 for z in building_z_values)
            if not (has_negative_z and has_positive_z):
                issues.append(
                    "User asked for buildings on both sides but all buildings are on the same side. "
                    "Place some buildings at negative z AND some at positive z."
                )

    if issues:
        correction = (
            "Fix these placement errors:\n" +
            "\n".join(f"  - {i}" for i in issues) +
            "\n\nAdjacency rule: building is adjacent to road only if same x OR same z, "
            "differing by exactly 10 in the other axis. Diagonal does NOT count. "
            "Place buildings directly beside a road tile."
        )
        return ReviewResult(approved=False, issues=issues, correction_prompt=correction)

    return ReviewResult(approved=True, issues=[], correction_prompt=None)



def _auto_rotate(layout: CityLayout) -> CityLayout:
    """
    Override rotations using pure Python geometry — never trust the LLM for this.

    Rules:
      ROADS:
        - If the road row has neighbours at +x/-x (east-west row) -> y=0
        - If the road row has neighbours at +z/-z (north-south col) -> y=90

      BUILDINGS / SHOPS / HOUSES etc:
        - Find the nearest road tile.
        - Rotate to face it (i.e. the building's front points toward the road).
          road is at +z (south) -> building faces south -> y=180
          road is at -z (north) -> building faces north -> y=0
          road is at +x (east)  -> building faces east  -> y=90
          road is at -x (west)  -> building faces west   -> y=270
    """
    BUILDING_TYPES = {
        "buildings", "houses", "skyscrapers", "shops",
        "supermarkets", "factory", "stadium", "gasstation", "supermarkets"
    }

    road_cells = {
        (e.position[0], e.position[2]): e
        for e in layout.entities if e.type == "roads"
    }

    # --- Roads: detect orientation from neighbours ---
    # First pass: assign rotation to roads that have clear neighbours
    road_entities = [e for e in layout.entities if e.type == "roads"]
    assigned = {}
    for e in road_entities:
        x, z = e.position[0], e.position[2]
        has_ew = (x + 10, z) in road_cells or (x - 10, z) in road_cells
        has_ns = (x, z + 10) in road_cells or (x, z - 10) in road_cells
        if has_ew and not has_ns:
            e.rotation = [0.0, 0.0, 0.0]
            assigned[(x, z)] = 0.0
        elif has_ns and not has_ew:
            e.rotation = [0.0, 90.0, 0.0]
            assigned[(x, z)] = 90.0

    # Second pass: isolated/intersection tiles get dominant orientation
    ew_count = sum(1 for r in assigned.values() if r == 0.0)
    ns_count = sum(1 for r in assigned.values() if r == 90.0)
    dominant = 0.0 if ew_count >= ns_count else 90.0
    for e in road_entities:
        x, z = e.position[0], e.position[2]
        if (x, z) not in assigned:
            e.rotation = [0.0, dominant, 0.0]

    # --- Flat tiles: always zero rotation ---
    FLAT_TYPES = {"parks", "landscape", "parkingareas", "farm", "sports", "water"}
    for e in layout.entities:
        if e.type in FLAT_TYPES:
            e.rotation = [0.0, 0.0, 0.0]

    # --- Buildings: face the nearest road ---
    for e in layout.entities:
        if e.type not in BUILDING_TYPES:
            continue
        x, z = e.position[0], e.position[2]
        # Check 4 cardinal neighbours for a road tile, pick closest
        candidates = [
            ((x,     z + 10), 180.0),   # road to south -> face south
            ((x,     z - 10),   0.0),   # road to north -> face north
            ((x + 10, z),      90.0),   # road to east  -> face east
            ((x - 10, z),     270.0),   # road to west  -> face west
        ]
        for cell, rot_y in candidates:
            if cell in road_cells:
                e.rotation = [0.0, rot_y, 0.0]
                break

    return layout


def _resolve_aliases(layout: CityLayout) -> CityLayout:
    for entity in layout.entities:
        if entity.type in USER_ALIASES:
            entity.type = USER_ALIASES[entity.type]  # type: ignore[assignment]
    return layout


def build_city(prompt: str, output_path: str) -> CityLayout:
    print(f"\n{'='*60}")
    print(f"[ArchiTech] Prompt: {prompt}")
    print(f"{'='*60}")

    brief = _plan(prompt)
    print(f"  City name  : {brief.city_name}")
    print(f"  Grid size  : {brief.grid_width_tiles} x {brief.grid_depth_tiles} tiles")
    print(f"  Zones      : {[z.label for z in brief.zones]}")
    print(f"  Features   : {brief.special_features}")

    layout = None
    correction = ""

    for attempt in range(1, MAX_REVIEW_RETRIES + 2):
        layout = _place(brief, extra_instructions=correction)
        layout = _resolve_aliases(layout)
        layout = _auto_rotate(layout)
        print(f"\n  Entities placed: {len(layout.entities)}")

        review = _review(brief, layout, user_prompt=prompt)

        if review.approved:
            print(f"  Reviewer approved on attempt {attempt}.")
            break
        else:
            print(f"  Reviewer rejected (attempt {attempt}/{MAX_REVIEW_RETRIES + 1}):")
            for issue in review.issues:
                print(f"      - {issue}")
            if attempt <= MAX_REVIEW_RETRIES:
                correction = review.correction_prompt or ""
                print("  Sending back to Placer with corrections...")
            else:
                print("  Max retries reached. Using last layout.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(layout.model_dump_json(indent=2))

    print(f"\n[ArchiTech] Saved -> {output_path}")
    _print_summary(layout)
    return layout


def _print_summary(layout: CityLayout) -> None:
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
    build_city(
        prompt="four buildings in a square with a park in the middle",
        output_path=SHARED_FILE,
    )