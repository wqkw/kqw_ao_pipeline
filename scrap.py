from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------- Shared primitives ----------

class QuantityHint(BaseModel):
    """Optional numeric or qualitative count hint like '1–2' or 'handful'."""
    min: Optional[int] = Field(
        None,
        description="Minimum count if a numeric range is provided.",
        examples=[1]
    )
    max: Optional[int] = Field(
        None,
        description="Maximum count if a numeric range is provided.",
        examples=[2]
    )
    approx: Optional[Literal[
        "single", "pair", "couple", "few", "several", "handful", "many", "unspecified"
    ]] = Field(
        None,
        description="Qualitative descriptor when counts are not exact (e.g., 'handful').",
        examples=["handful"]
    )
    note: Optional[str] = Field(
        None,
        description="Free-text note carrying the original hint/wording.",
        examples=["Locations (1–2)", "Props (handful)"]
    )


class Tag(BaseModel):
    """Lightweight tag for quick filtering/grouping."""
    name: str = Field(..., description="Short tag label.", examples=["lamp", "drone", "warden"])
    note: Optional[str] = Field(None, description="Optional tag note.", examples=["hostile", "portable"])


# ---------- Entities within a scene ----------

class LocationSpec(BaseModel):
    """A specific place used in the scene."""
    name: str = Field(..., description="Human-readable name of the location.", examples=["Bell-Gate 7 façade"])
    description: str = Field(..., description="One or more sentences describing look/feel and salient features.")
    tags: List[Tag] = Field(default_factory=list, description="Optional tags (e.g., lighting, hazards).")


CharacterRole = Literal[
    "named",                # Named character (e.g., 'Sera', 'Jun')
    "group",                # A small group with implied membership (e.g., 'Warden skiff crew')
    "extras",               # Background extras/crowd/silhouettes
    "offscreen_presence"    # Not visible but perceptible (e.g., heard over radio)
]

class CharacterSpec(BaseModel):
    """A character or group of characters present in the scene."""
    name: str = Field(..., description="Character/group name.", examples=["Sera", "Warden skiff crew"])
    role: CharacterRole = Field(..., description="Role type (named, group, extras, offscreen_presence).")
    description: str = Field(..., description="Physical cues, behavior, wardrobe, and intent.")
    quantity: Optional[QuantityHint] = Field(
        None,
        description="Count hint for groups/extras (e.g., (two), (1–4))."
    )
    tags: List[Tag] = Field(default_factory=list, description="Optional tags (e.g., 'silhouette', 'background').")


class PropSpec(BaseModel):
    """Any object/vehicle/creature treated as an on-set prop or interactive element."""
    name: str = Field(..., description="Prop or object name.", examples=["Wrapped brass clapper"])
    description: str = Field(..., description="Function, state, risks, and relevant affordances.")
    tags: List[Tag] = Field(default_factory=list, description="Optional tags (e.g., 'weapon', 'fragile', 'powered').")


SoundType = Literal[
    "ambient", "mechanical", "foley", "dialogue", "voiceover", "nature", "crowd", "signal", "ui"
]

class SoundCue(BaseModel):
    """Atomic sound beat or texture."""
    text: str = Field(..., description="Exact cue language to record/mix.", examples=["Lamp filament flicker-pop"])
    type: Optional[SoundType] = Field(
        None,
        description="Category of sound for post/sound design.",
        examples=["mechanical"]
    )
    source: Optional[str] = Field(
        None,
        description="Likely source/emitter if identifiable (prop, character, location).",
        examples=["indicator lamp", "Warden headset"]
    )
    timing_note: Optional[str] = Field(
        None,
        description="Temporal placement or sequencing hint.",
        examples=["under VO", "as searchlight passes"]
    )


class MusicCue(BaseModel):
    """Musical direction for the moment/scene."""
    description: str = Field(..., description="Texture, motif, dynamics, or structural note.")
    mood: Optional[str] = Field(None, description="Emotional intent in plain language.", examples=["tense", "affirmative"])
    instrumentation: Optional[str] = Field(
        None,
        description="Primary instruments or sound palette.",
        examples=["bowed metal, sub-bass", "music-box chimes and low strings"]
    )
    motif_reference: Optional[str] = Field(
        None,
        description="Reference to a recurring motif/theme if applicable.",
        examples=["quotes the bell’s interval"]
    )
    timing_note: Optional[str] = Field(
        None,
        description="Where/how music should evolve relative to action.",
        examples=["drop-out on engine stutter; resume on restart"]
    )


# ---------- Scene wrapper ----------

class SceneCategoryHints(BaseModel):
    """Optional per-category quantity hints parsed from headings like 'Locations (1–2)'."""
    locations: Optional[QuantityHint] = Field(None, description="Quantity hint for locations in this scene.")
    characters: Optional[QuantityHint] = Field(None, description="Quantity hint for characters in this scene.")
    props: Optional[QuantityHint] = Field(None, description="Quantity hint for props in this scene.")


class SceneSpec(BaseModel):
    """A scene with its assets and audio design."""
    scene_number: int = Field(..., description="1-based scene number.", examples=[1])
    title: str = Field(..., description="Scene title or slug.", examples=["Opening Alarm"])
    synopsis: Optional[str] = Field(None, description="1–3 sentence summary of the scene’s purpose/beat.")
    hints: Optional[SceneCategoryHints] = Field(
        None,
        description="Optional category-level count hints (e.g., 'Locations (1–2)', 'Props (handful)')."
    )

    locations: List[LocationSpec] = Field(
        default_factory=list,
        description="Locations used in the scene (1–2 typical)."
    )
    characters: List[CharacterSpec] = Field(
        default_factory=list,
        description="Named roles, groups, and extras (1–4 typical)."
    )
    props: List[PropSpec] = Field(
        default_factory=list,
        description="Interactive objects, vehicles, creatures, signage, etc."
    )

    sound: List[SoundCue] = Field(
        default_factory=list,
        description="Diegetic and non-diegetic sound cues to capture/mix."
    )
    music: List[MusicCue] = Field(
        default_factory=list,
        description="Score direction for the scene."
    )

    notes: Optional[str] = Field(
        None,
        description="Any scene-specific production notes, uncertainties, or edge cases."
    )


# ---------- Cross-scene lore / consistency ----------

class LoreRuleReference(BaseModel):
    """Reference to a lore/consistency rule applied in one or more scenes."""
    rule_name: str = Field(..., description="Short, stable rule name.", examples=["Hard Rule 1: Eye-bond binds within lamplight"])
    statement: str = Field(..., description="Authoritative phrasing of the rule.")
    scenes: List[int] = Field(..., description="Scene numbers where this rule is invoked/evidenced.", examples=[[2, 8]])
    evidence: Optional[List[str]] = Field(
        None,
        description="Short quotes/extracts backing the application.",
        examples=[["iris 'ping' shimmer when bond takes"]]
    )
    exceptions_or_nullifiers: Optional[str] = Field(
        None,
        description="Conditions that nullify or alter the rule.",
        examples=["salt or mirror present to nullify"]
    )


# ---------- Storyboard container ----------

class StoryboardSpec(BaseModel):
    """
    Root object for a storyboard broken into scenes,
    suitable for Structured Outputs / function calling.
    """
    project_title: Optional[str] = Field(None, description="Human-readable title of the project.")
    overview: Optional[str] = Field(None, description="Optional prose overview for humans.")
    scenes: List[SceneSpec] = Field(..., description="Ordered list of scenes.")
    lore_consistency: List[LoreRuleReference] = Field(
        default_factory=list,
        description="Cross-scene rules and references that ensure internal consistency."
    )
