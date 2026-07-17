from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Optional

ASSET_CATEGORIES = Literal[
    "bridges", "buildings", "factory", "farm", "gasstation", "houses",
    "landscape", "parkingareas", "parks", "roads", "shops",
    "skyscrapers", "sports", "stadium", "street_signs",
    "supermarkets", "transport", "trees", "water",
]

USER_ALIASES: dict[str, str] = {"river": "water", "airballoon": "sports", "air balloon": "sports", "hot air balloon": "sports"}


# Stage 1 - Planner output
class Zone(BaseModel):
    label: str
    asset_types: List[ASSET_CATEGORIES]
    tile_count: int = Field(..., ge=1)
    position_hint: str

class RoadSegment(BaseModel):
    orientation: Literal["east-west", "north-south"]
    length_tiles: int = Field(..., ge=1)
    role: str

class SceneBrief(BaseModel):
    city_name: str
    grid_width_tiles: int = Field(..., ge=2, le=30)
    grid_depth_tiles: int = Field(..., ge=2, le=30)
    road_plan: List[RoadSegment] = Field(..., min_length=1)
    zones: List[Zone] = Field(..., min_length=1)
    special_features: List[str] = Field(default=[])
    placement_notes: str = Field(default="")


# Stage 2 - Placer output
class CityEntity(BaseModel):
    id: str
    type: ASSET_CATEGORIES
    position: List[float] = Field(..., min_length=3, max_length=3)
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)

    @field_validator("position")
    @classmethod
    def snap_to_grid(cls, v: List[float]) -> List[float]:
        v[0] = round(v[0] / 10.0) * 10.0
        v[2] = round(v[2] / 10.0) * 10.0
        v[1] = max(0.0, v[1])
        return v

class CityLayout(BaseModel):
    city_name: str
    entities: List[CityEntity] = Field(..., min_length=1)

    @model_validator(mode="after")
    def no_overlapping_cells(self) -> "CityLayout":
        STACKING_ALLOWED = {
            ("water", "bridges"), ("bridges", "water"),
            ("roads", "parks"), ("parks", "roads"),
            ("roads", "landscape"), ("landscape", "roads"),
        }
        seen: dict[tuple[float, float], str] = {}
        for e in self.entities:
            cell = (e.position[0], e.position[2])
            if cell in seen:
                pair = (seen[cell], e.type)
                if pair not in STACKING_ALLOWED:
                    raise ValueError(
                        f"Overlap at cell {cell} — '{e.id}' ({e.type}) conflicts with existing '{seen[cell]}'. "
                        "Only bridge stacked on water is allowed."
                    )
            else:
                seen[cell] = e.type
        return self


# Stage 3 - Reviewer output
class ReviewResult(BaseModel):
    approved: bool
    issues: List[str] = Field(default=[])
    correction_prompt: Optional[str] = Field(default=None)