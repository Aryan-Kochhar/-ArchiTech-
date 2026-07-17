"""
ArchiTech Archetype Generator v3
Focused on TOPOLOGY only — buildings, roads, water.
No decorative filler (trees, parks, landscape) unless structurally essential.
These are templates for HARD spatial patterns the LLM struggles with.
"""
import json
from pathlib import Path

OUT = Path("./archetypes")
OUT.mkdir(exist_ok=True)

# Wipe old archetypes first
for f in OUT.glob("*.json"):
    f.unlink()

def save(name, description, triggers, city_name, entities):
    data = {
        "_meta": {"name": name, "description": description, "triggers": triggers},
        "city_name": city_name,
        "entities": entities,
    }
    path = OUT / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [{len(entities):>2}]  {name}")

def b(id, x, z):   return {"id": id, "type": "buildings", "position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}
def r(id, x, z):   return {"id": id, "type": "roads",     "position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}
def w(id, x, z):   return {"id": id, "type": "water",     "position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}
def br(id, x, z):  return {"id": id, "type": "bridges",   "position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}
def sk(id, x, z):  return {"id": id, "type": "skyscrapers","position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}
def h(id, x, z):   return {"id": id, "type": "houses",    "position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}
def sh(id, x, z):  return {"id": id, "type": "shops",     "position": [float(x), 0.0, float(z)], "rotation": [0.0, 0.0, 0.0]}

print("Generating focused archetypes...\n")

# ══════════════════════════════════════════════════════════
# ROAD TOPOLOGIES
# ══════════════════════════════════════════════════════════

# 1. ROW — buildings on one side of a road
save("row",
    "A straight road running east-west with a row of buildings along the north side.",
    ["row", "along", "street", "beside", "line of buildings", "next to each other", "in a line", "along the road"],
    "Row Street", [
    r("rd_1", 0, 0), r("rd_2",10, 0), r("rd_3",20, 0), r("rd_4",30, 0),
    b("b1",  0,-10), b("b2", 10,-10), b("b3", 20,-10), b("b4", 30,-10),
])

# 2. BOTH SIDES — buildings on both sides of a road
save("both_sides",
    "A road running east-west with buildings on both the north side and the south side.",
    ["both sides", "either side", "each side", "opposite sides", "facing each other", "buildings on both sides", "divided by road"],
    "Both Sides", [
    r("rd_1", 0, 0), r("rd_2",10, 0), r("rd_3",20, 0),
    b("n1",  0,-10), b("n2", 10,-10), b("n3", 20,-10),
    b("s1",  0, 10), b("s2", 10, 10), b("s3", 20, 10),
])

# 3. CROSSROAD — + intersection with buildings at corners
save("crossroad",
    "A plus-shaped crossroads intersection where an east-west road crosses a north-south road, buildings at each of the four corners.",
    ["crossroad", "intersection", "crossing", "junction", "cross street", "four corners", "plus shape", "T junction"],
    "Crossroad", [
    r("ew_w",-10, 0), r("ew_c",  0, 0), r("ew_e", 10, 0),
    r("ns_n",  0,-10), r("ns_s", 0, 10),
    b("nw",-10,-10), b("ne", 10,-10),
    b("sw",-10, 10), b("se", 10, 10),
])

# 4. SQUARE FORMATION — 4 buildings at corners, center free
save("square_formation",
    "Four buildings arranged at the corners of a square, with an empty center tile.",
    ["square", "formation", "four corners", "in the middle", "square formation", "arranged in a square", "4 buildings square", "buildings in a square"],
    "Square Formation", [
    b("nw", 0, 0), b("ne",20, 0),
    b("sw", 0,20), b("se",20,20),
    # center left empty — LLM fills with park/water/whatever user asks
])

# 5. DENSE BLOCK — buildings on grid with perimeter roads
save("dense_block",
    "A dense city block with buildings packed on a 3x3 grid, roads forming a complete perimeter around the outside.",
    ["downtown", "dense", "city center", "city block", "urban", "district", "grid", "packed", "dense buildings", "block"],
    "Dense Block", [
    # Perimeter roads
    r("t1", 0,-10), r("t2",10,-10), r("t3",20,-10),
    r("b1", 0, 30), r("b2",10, 30), r("b3",20, 30),
    r("l1",-10, 0), r("l2",-10,10), r("l3",-10,20),
    r("r1", 30, 0), r("r2", 30,10), r("r3", 30,20),
    # 3x3 interior buildings
    b("b00", 0, 0), b("b10",10, 0), b("b20",20, 0),
    b("b01", 0,10), b("b11",10,10), b("b21",20,10),
    b("b02", 0,20), b("b12",10,20), b("b22",20,20),
])

# 6. PARALLEL ROADS — two roads with buildings on outer sides
save("parallel_roads",
    "Two parallel roads running east-west with buildings on the outer north and south sides.",
    ["parallel", "two roads", "dual road", "two streets", "parallel roads", "two parallel"],
    "Parallel Roads", [
    r("rn1", 0, 0), r("rn2",10, 0), r("rn3",20, 0),
    r("rs1", 0,20), r("rs2",10,20), r("rs3",20,20),
    b("n1",  0,-10), b("n2",10,-10), b("n3",20,-10),
    b("s1",  0, 30), b("s2",10, 30), b("s3",20, 30),
])

# 7. L-SHAPE ROAD — road with a corner turn
save("l_shape",
    "An L-shaped road that runs east then turns south, with buildings along the outer edges of both arms.",
    ["corner", "bend", "turn", "L-shape", "L shape", "right angle", "road corner", "elbow road"],
    "L-Shape Road", [
    r("h1", 0, 0), r("h2",10, 0), r("h3",20, 0), r("corner",30, 0),
    r("v1",30,10), r("v2",30,20),
    b("bh1", 0,-10), b("bh2",10,-10), b("bh3",20,-10),
    b("bv1",40,10), b("bv2",40,20),
])

# ══════════════════════════════════════════════════════════
# WATER TOPOLOGIES
# ══════════════════════════════════════════════════════════

# 8. RIVER SPLIT — two groups of buildings separated by river
save("river_split",
    "Two rows of buildings separated by a river running north-south between them.",
    ["split", "divided", "separated by", "river between", "river split", "divided by river", "buildings separated by river"],
    "River Split", [
    b("w1", 0,-10), b("w2", 0,  0), b("w3", 0, 10),
    w("r1",10,-10), w("r2",10,  0), w("r3",10, 10),
    b("e1",20,-10), b("e2",20,  0), b("e3",20, 10),
])

# 9. ISLAND — COMPLETE water ring surrounding buildings
# This is the canonical "surrounding" template
# Water ring has 4 sides (3 tiles each) + 4 corners = 16 water tiles total
# Buildings sit inside the ring
save("island",
    "A complete closed ring of water surrounding a central block of buildings. All 4 sides AND all 4 corner tiles are water, forming an unbroken moat. Buildings are inside the ring.",
    ["surrounding", "surrounded by", "moat", "island", "encircled", "ring of water", "river surrounding", "water around", "water surrounding", "enclosing", "enclosed by water"],
    "Island", [
    # Inner buildings (2x2 block at center)
    b("bldg_nw",10,10), b("bldg_ne",30,10),
    b("bldg_sw",10,30), b("bldg_se",30,30),
    # North water row (z=0)
    w("wn1",  0, 0), w("wn2",10, 0), w("wn3",20, 0), w("wn4",30, 0), w("wn5",40, 0),
    # South water row (z=40)
    w("ws1",  0,40), w("ws2",10,40), w("ws3",20,40), w("ws4",30,40), w("ws5",40,40),
    # West water col (x=0)
    w("ww1",  0,10), w("ww2", 0,20), w("ww3", 0,30),
    # East water col (x=40)
    w("we1", 40,10), w("we2",40,20), w("we3",40,30),
    # Center tile (between the 4 buildings)
    w("wcenter",20,20),
])

# 10. WATERFRONT — buildings facing water
save("waterfront",
    "A waterfront with buildings facing a wide stretch of water to the south, a road between buildings and water.",
    ["waterfront", "harbour", "harbor", "seafront", "riverside", "lakefront", "facing water", "coastal", "buildings facing river"],
    "Waterfront", [
    b("b1", 0,-10), b("b2",10,-10), b("b3",20,-10), b("b4",30,-10),
    r("rd1", 0,  0), r("rd2",10,  0), r("rd3",20,  0), r("rd4",30,  0),
    w("w11", 0, 10), w("w12",10, 10), w("w13",20, 10), w("w14",30, 10),
    w("w21", 0, 20), w("w22",10, 20), w("w23",20, 20), w("w24",30, 20),
])

# 11. RIVER WITH BRIDGE — river NS, bridge EW, buildings on banks
save("river_bridge",
    "A river running north-south crossed by a bridge going east-west, with roads leading to the bridge and buildings on both banks.",
    ["bridge", "bridge over river", "river and bridge", "cross the river", "bridge across", "spanning the river"],
    "River Bridge", [
    # River (NS)
    w("rv1",20,-10), w("rv2",20,  0), w("rv3",20, 10),
    w("rv4",20, 20), w("rv5",20, 30),
    # Bridge ON the river at z=10
    br("bridge",20,10),
    # Roads leading to bridge
    r("rdw1", 0,10), r("rdw2",10,10),
    r("rde1",30,10), r("rde2",40,10),
    # West bank buildings
    b("bw1", 0,-10), b("bw2", 0,  0), b("bw3", 0, 20),
    # East bank buildings
    b("be1",40,-10), b("be2",40,  0), b("be3",40, 20),
])

# 12. RIVER BEND — L-shaped river
save("river_bend",
    "A river that flows east-west then bends south, forming an L-shape. Buildings are on the outer side of the bend.",
    ["river bend", "river turn", "river corner", "winding river", "L shaped river", "river curve"],
    "River Bend", [
    # EW part of river
    w("rh1", 0,10), w("rh2",10,10), w("rh3",20,10), w("rh4",30,10), w("corner",40,10),
    # NS part turning south
    w("rv1",40,20), w("rv2",40,30), w("rv3",40,40),
    # Buildings along north bank
    b("bn1", 0, 0), b("bn2",10, 0), b("bn3",20, 0), b("bn4",30, 0),
    # Buildings along west bank (south of bend)
    b("bw1", 0,20), b("bw2", 0,30),
])

# ══════════════════════════════════════════════════════════
# MIXED / COMMERCIAL
# ══════════════════════════════════════════════════════════

# 13. SHOPPING STRIP — shops along both sides of road
save("shopping_strip",
    "A commercial strip with shops lining both sides of a road.",
    ["shopping", "shops", "commercial", "retail", "strip", "market", "store", "shopping street"],
    "Shopping Strip", [
    r("rd1", 0, 0), r("rd2",10, 0), r("rd3",20, 0), r("rd4",30, 0),
    sh("s1", 0,-10), sh("s2",10,-10), sh("s3",20,-10), sh("s4",30,-10),
    sh("s5", 0, 10), sh("s6",10, 10), sh("s7",20, 10), sh("s8",30, 10),
])

# 14. SKYSCRAPER GRID — towers on a road grid
save("skyscraper_grid",
    "A skyscraper district with tall towers on a road grid, roads running between each tower.",
    ["skyscrapers", "tall buildings", "high rise", "towers", "financial district", "skyline", "CBD", "skyscraper"],
    "Skyscraper Grid", [
    sk("sk1", 0, 0), r("rd1",10, 0), sk("sk2",20, 0),
    r("rd2",  0,10), r("rd3",10,10), r("rd4", 20,10),
    sk("sk3", 0,20), r("rd5",10,20), sk("sk4",20,20),
])

# 15. RESIDENTIAL STREET — houses on both sides
save("residential",
    "A quiet residential street with houses on both sides of the road.",
    ["residential", "houses", "suburban", "quiet street", "neighbourhood", "neighborhood", "house street", "homes"],
    "Residential Street", [
    r("rd1", 0, 0), r("rd2",10, 0), r("rd3",20, 0), r("rd4",30, 0),
    h("h1",  0,-10), h("h2",10,-10), h("h3",20,-10), h("h4",30,-10),
    h("h5",  0, 10), h("h6",10, 10), h("h7",20, 10), h("h8",30, 10),
])

print(f"\nDone! {len(list(OUT.glob('*.json')))} archetypes saved to {OUT}/")
print("\nSummary:")
for p in sorted(OUT.glob("*.json")):
    with open(p) as f:
        d = json.load(f)
    print(f"  {p.stem:<22} {len(d['entities']):>2} entities  —  {d['_meta']['description'][:60]}...")