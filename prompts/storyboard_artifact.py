from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------- Base (forbid unknown keys) ----------

class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep outputs clean."""
    model_config = ConfigDict(extra="forbid")


# ---------- Story Elements ----------

class LocationSpec(StrictModel):
    """A specific place used in the story."""
    name: str = Field(..., description="Unique name for this location (e.g., 'Crimson Foundry', 'Central Plaza').")
    description: Optional[str] = Field(None, description="Visual description including appearance, atmosphere, layout, lighting, notable features, and any hazards or important details that affect the story.")

class CharacterSpec(StrictModel):
    """A character or group in the story."""
    name: str = Field(..., description="Character name or group designation (e.g., 'Captain Thorne', 'Palace Guards').")
    description: Optional[str] = Field(None, description="Physical appearance, personality, motivations, clothing, mannerisms, role in story, and key relationships or abilities.")

class PropSpec(StrictModel):
    """Any object, vehicle, or creature in the story."""
    name: str = Field(..., description="Descriptive name for this prop (e.g., 'Resonance Scanner', 'Broken Crown').")
    description: Optional[str] = Field(None, description="Appearance, materials, condition, function, how characters interact with it, and any special properties or significance.")
    image_url: Optional[str] = Field(None, description="URL or path to the generated image of this prop.")


SoundType = Literal[
    "ambient", "mechanical", "foley", "dialogue", "voiceover", "nature", "crowd", "signal", "ui"
]

class SoundCue(StrictModel):
    """An audio element in the story."""
    name: str = Field(..., description="Descriptive name for this sound (e.g., 'Distant Thunder', 'Machinery Groan').")
    description: Optional[str] = Field(None, description="Sound characteristics (pitch, rhythm, intensity), source, timing, duration, and whether it's diegetic (characters hear it) or non-diegetic (audience only).")


class MusicCue(StrictModel):
    """A musical element in the story."""
    name: str = Field(..., description="Thematic name for this music cue (e.g., 'Hope's Theme', 'Battle March').")
    description: Optional[str] = Field(None, description="Instrumentation, tempo, mood, how it evolves, relationship to characters or themes, and whether it's background score or source music.")


# ---------- Shot specification ----------

ShotType = Literal[
    "establishing", "wide", "two_shot", "medium", "close_up", "extreme_close_up",
    "over_shoulder", "insert", "cutaway", "pov", "tracking", "dolly", "crane",
    "aerial", "handheld", "static", "montage", "transition", "stage_setting"
]

ShotMovement = Literal[
    "static", "pan", "tilt", "push_in", "pull_back", "truck_left", "truck_right",
    "crane_up", "crane_down", "arc", "handheld", "steadicam", "gimbal", "drone"
]

ShotAngle = Literal["eye_level", "high", "low", "canted"]

class ShotSpec(StrictModel):
    """A single camera shot within a scene."""
    name: str = Field(..., description="Descriptive name for this shot (e.g., 'Hero Close-up', 'Establishing Wide').")
    movement: Optional[str] = Field(None, description="Camera movement and purpose (e.g., 'slow push-in for tension', 'handheld for intimacy').")
    location_name: Optional[str] = Field(None, description="Location where this shot takes place (references scene locations).")
    character_names: Optional[List[str]] = Field(None, description="Characters visible in this shot (references scene characters).")
    prop_names: Optional[List[str]] = Field(None, description="Props visible or used in this shot (references scene props).")
    sound_names: Optional[List[str]] = Field(None, description="Important sound cues in this shot (references scene/global sounds).")
    music_names: Optional[List[str]] = Field(None, description="Music playing during this shot (references scene/global music).")
    dialogue: Optional[str] = Field(None, description="All dialogue spoken in this shot, with speaker attribution if needed.")
    duration_seconds: Optional[float] = Field(None, description="Estimated shot duration in seconds.")
    description: Optional[str] = Field(None, description="Camera angle, shot size, composition, lighting, actions, and technical requirements. What the audience sees.")
    image_url: Optional[str] = Field(None, description="URL or path to the generated image of this shot.")


# ---------- Narrative Structure ----------

class SceneSpec(StrictModel):
    """A complete narrative unit within the story."""
    name: str = Field(..., description="Descriptive name for this scene (e.g., 'The Betrayal', 'Underground Chase').")
    description: Optional[str] = Field(None, description="Scene's narrative purpose, emotional arc, character development, conflicts, and how it advances the plot.")
    character_names: Optional[List[str]] = Field(None, description="Characters who appear or are referenced in this scene (from global library).")
    location_names: Optional[List[str]] = Field(None, description="Locations used in this scene (from global library).")
    prop_names: Optional[List[str]] = Field(None, description="Props present or used in this scene (from global library).")
    sound_names: Optional[List[str]] = Field(None, description="Sound cues active in this scene (from global library).")
    music_names: Optional[List[str]] = Field(None, description="Music cues for this scene (from global library).")
    shots: Optional[List[ShotSpec]] = Field(None, description="Ordered sequence of shots that compose this scene.")
    stage_setting_shot: Optional[ShotSpec] = Field(None, description="Establishing shot that sets the visual context for this scene.")


class StoryboardSpec(StrictModel):
    """Complete storyboard for a narrative project."""
    name: str = Field(..., description="Unique identifier for this story (e.g., 'crimson_rebellion', 'neon_dreams').")
    title: Optional[str] = Field(None, description="Official story title as it appears to audiences.")
    lore: Optional[str] = Field(None, description="Background worldbuilding including history, rules, cultures, technologies, and context needed to understand the world.")
    narrative: Optional[str] = Field(None, description="Plot structure, character arcs, themes, conflicts, and how the story progresses from start to finish.")
    moodboard_path: Optional[str] = Field(None, description="File path to visual moodboard showing aesthetic style, color palette, and atmosphere.")
    characters: Optional[List[CharacterSpec]] = Field(None, description="Master library of all characters in the story.")
    locations: Optional[List[LocationSpec]] = Field(None, description="Master library of all locations in the story.")
    props: Optional[List[PropSpec]] = Field(None, description="Master library of all significant objects and elements in the story.")
    sounds: Optional[List[SoundCue]] = Field(None, description="Master library of all sound effects and audio elements in the story.")
    music: Optional[List[MusicCue]] = Field(None, description="Master library of all musical themes and cues in the story.")
    scenes: Optional[List[SceneSpec]] = Field(None, description="Ordered sequence of scenes that compose the complete story.")
