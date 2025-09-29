from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict


# ---------- Base (forbid unknown keys) ----------

class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep outputs clean."""
    model_config = ConfigDict(extra="forbid")


# ---------- Entities within a scene ----------

class LocationSpec(StrictModel):
    """A specific place used in the scene."""
    name: str = Field(..., description="Human-readable name of the location.")
    description: str = Field(..., description="Concise description of look/feel, salient features, hazards, lighting, etc.")


CharacterRole = Literal[
    "named",                # e.g., 'Sera', 'Jun'
    "group",                # e.g., 'Warden skiff crew'
    "extras",               # background/crowd/silhouettes
    "offscreen_presence"    # heard/referenced but not visible
]

class CharacterSpec(StrictModel):
    """A character or group present in the scene."""
    name: str = Field(..., description="Character or group label.")
    role: CharacterRole = Field(..., description="Role type (named, group, extras, offscreen_presence).")
    description: str = Field(..., description="Wardrobe, behavior, intent, or posture relevant to the beat.")


class PropSpec(StrictModel):
    """Any object/vehicle/creature treated as a prop or interactive element."""
    name: str = Field(..., description="Prop/object/creature name.")
    description: str = Field(..., description="Function, state, risks, affordances, and salient details.")


SoundType = Literal[
    "ambient", "mechanical", "foley", "dialogue", "voiceover", "nature", "crowd", "signal", "ui"
]

class SoundCue(StrictModel):
    """Atomic sound cue."""
    text: str = Field(..., description="Exact cue language to record/mix.")
    type: str = Field(..., description="Category tag for post/sound design.")
    source: str = Field(..., description="Likely emitter (prop, character, or location).")
    timing_note: str = Field(..., description="Temporal placement (e.g., 'under VO', 'on cut').")


class MusicCue(StrictModel):
    """Musical direction for the scene."""
    description: str = Field(..., description="Texture/motif/dynamics or structural note.")
    mood: str = Field(..., description="Emotional intent (e.g., 'tense', 'affirmative').")
    instrumentation: str = Field(..., description="Primary instruments/sound palette.")
    timing_note: str = Field(..., description="Evolution relative to action (e.g., 'drop then resume').")


# ---------- Dialogue (optional, per shot) ----------

class DialogueLine(StrictModel):
    """Optional structured dialogue/VO line tied to a shot."""
    text: str = Field(..., description="Spoken line or VO text.")
    speaker: str = Field(..., description="Speaker name if applicable; leave empty for VO/unknown.")
    is_voiceover: bool = Field(..., description="True if delivered as voiceover.")
    timing_note: str = Field(..., description="Beat/cue timing (e.g., 'over cut', 'pre-flare').")


# ---------- Shot specification ----------

ShotType = Literal[
    "establishing", "wide", "two_shot", "medium", "close_up", "extreme_close_up",
    "over_shoulder", "insert", "cutaway", "pov", "tracking", "dolly", "crane",
    "aerial", "handheld", "static", "montage", "transition"
]

ShotMovement = Literal[
    "static", "pan", "tilt", "push_in", "pull_back", "truck_left", "truck_right",
    "crane_up", "crane_down", "arc", "handheld", "steadicam", "gimbal", "drone"
]

ShotAngle = Literal["eye_level", "high", "low", "canted"]

class ShotSpec(StrictModel):
    """A single shot within a scene."""
    shot_number: int = Field(..., description="1-based index within the scene.")
    slug: str = Field(..., description="Short label for quick reference (e.g., 'Alarm on code board').")

    type: str = Field(..., description="Canonical shot type if applicable.")
    angle: str = Field(..., description="Camera angle (eye_level/high/low/canted).")
    movement: str = Field(..., description="Dominant camera movement if any.")
    focal_length_mm: int = Field(..., description="Approximate focal length in millimeters if known.")

    location: str = Field(..., description="Name of a location from this scene that the shot takes place in.")
    subjects: List[str] = Field(..., description="Names of characters/props emphasized in the shot (strings referencing scene lists).")

    framing: str = Field(..., description="Composition/blocking (e.g., foreground/mid/background, leading lines).")
    action: str = Field(..., description="What happens in this shot in one or two sentences.")
    dialogue: List[DialogueLine] = Field(..., description="Optional spoken lines/VO tied to this shot.")
    sound: List[SoundCue] = Field(..., description="Shot-specific sound cues (beyond scene-level).")
    music: List[MusicCue] = Field(..., description="Shot-specific music cues (beyond scene-level).")

    duration_seconds: float = Field(..., description="Approximate duration; omit if unknown.")
    notes: str = Field(..., description="Technical or production notes (e.g., rigging/VFX/continuity).")


# ---------- Scene + Storyboard ----------

class SceneSpec(StrictModel):
    """A scene with its assets, audio, and optional shot breakdown."""
    scene_number: int = Field(..., description="1-based scene number.")
    title: str = Field(..., description="Scene title or slug.")
    synopsis: str = Field(..., description="1–3 sentence summary of the beat/purpose.")

    locations: List[LocationSpec] = Field(..., description="Locations used in the scene.")
    characters: List[CharacterSpec] = Field(..., description="Named roles, groups, and extras.")
    props: List[PropSpec] = Field(..., description="Interactive objects, vehicles, signage, creatures, etc.")

    sound: List[SoundCue] = Field(..., description="Diegetic/non-diegetic sound cues.")
    music: List[MusicCue] = Field(..., description="Score cues/directions.")

    shots: List[ShotSpec] = Field(..., description="Optional ordered list of shots within the scene.")


class StoryboardSpec(StrictModel):
    """Root object for a storyboard broken into scenes."""
    project_title: str = Field(..., description="Human-readable project title.")
    overview: str = Field(..., description="Prose overview for humans.")
    scenes: List[SceneSpec] = Field(..., description="Ordered list of scenes.")


def format_storyboard_spec(storyboard: StoryboardSpec) -> str:
    """Format a StoryboardSpec instance into a nicely structured markdown string."""
    lines = []

    lines.append(f"# {storyboard.project_title}")
    lines.append("")
    lines.append("## Overview")
    lines.append(storyboard.overview)
    lines.append("")

    for scene in storyboard.scenes:
        lines.append(f"## Scene {scene.scene_number}: {scene.title}")
        lines.append("")
        lines.append(f"**Synopsis:** {scene.synopsis}")
        lines.append("")

        if scene.locations:
            lines.append("### Locations")
            for loc in scene.locations:
                lines.append(f"- **{loc.name}**: {loc.description}")
            lines.append("")

        if scene.characters:
            lines.append("### Characters")
            for char in scene.characters:
                lines.append(f"- **{char.name}** ({char.role}): {char.description}")
            lines.append("")

        if scene.props:
            lines.append("### Props")
            for prop in scene.props:
                lines.append(f"- **{prop.name}**: {prop.description}")
            lines.append("")

        if scene.sound:
            lines.append("### Sound Cues")
            for sound in scene.sound:
                lines.append(f"- **{sound.text}** ({sound.type}) - Source: {sound.source}, Timing: {sound.timing_note}")
            lines.append("")

        if scene.music:
            lines.append("### Music Cues")
            for music in scene.music:
                lines.append(f"- **{music.description}** - Mood: {music.mood}, Instrumentation: {music.instrumentation}, Timing: {music.timing_note}")
            lines.append("")

        if scene.shots:
            lines.append("### Shots")
            for shot in scene.shots:
                lines.append(f"#### Shot {shot.shot_number}: {shot.slug}")
                lines.append(f"**Type:** {shot.type} | **Angle:** {shot.angle} | **Movement:** {shot.movement} | **Focal Length:** {shot.focal_length_mm}mm")
                lines.append(f"**Location:** {shot.location}")
                lines.append(f"**Subjects:** {', '.join(shot.subjects)}")
                lines.append(f"**Framing:** {shot.framing}")
                lines.append(f"**Action:** {shot.action}")
                lines.append(f"**Duration:** {shot.duration_seconds}s")

                if shot.dialogue:
                    lines.append("**Dialogue:**")
                    for dialogue in shot.dialogue:
                        speaker_info = f"{dialogue.speaker}: " if dialogue.speaker else ""
                        vo_indicator = " (VO)" if dialogue.is_voiceover else ""
                        lines.append(f"  - {speaker_info}{dialogue.text}{vo_indicator} - {dialogue.timing_note}")

                if shot.sound:
                    lines.append("**Sound:**")
                    for sound in shot.sound:
                        lines.append(f"  - {sound.text} ({sound.type}) - Source: {sound.source}, Timing: {sound.timing_note}")

                if shot.music:
                    lines.append("**Music:**")
                    for music in shot.music:
                        lines.append(f"  - {music.description} - Mood: {music.mood}, Instrumentation: {music.instrumentation}, Timing: {music.timing_note}")

                if shot.notes:
                    lines.append(f"**Notes:** {shot.notes}")

                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
