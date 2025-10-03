from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------- Base (forbid unknown keys) ----------

class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep outputs clean."""
    model_config = ConfigDict(extra="forbid")


# ---------- Shot specification ----------

class ShotSpec(StrictModel):
    """A single camera shot within a scene."""
    name: str = Field(..., description="Descriptive name for this shot (e.g., 'Opening Wide Shot', 'Character Close-up').")
    description: str = Field(..., description="Visual description: what the camera sees, framing, composition, lighting, and actions.")
    camera_angle: Optional[str] = Field(None, description="Camera angle and movement (e.g., 'low angle', 'eye level tracking shot', 'overhead crane down').")
    image_path: Optional[str] = Field(None, description="Path to the generated image of this shot.")


# ---------- Scene specification ----------

class SceneSpec(StrictModel):
    """A scene capturing a moment or vibe."""
    name: str = Field(..., description="Descriptive name for this scene (e.g., 'Urban Chase', 'Quiet Reflection').")
    description: str = Field(..., description="What's happening in the scene: action, mood, atmosphere, and visual elements.")
    shots: Optional[List[ShotSpec]] = Field(None, description="Exactly 10 camera shots that compose this scene.")


# ---------- Main Artifact ----------

class ShortStoryboardSpec(StrictModel):
    """Short storyboard generated from 1-3 input images."""
    name: str = Field(..., description="Unique identifier for this storyboard (e.g., 'test_short_001').")
    image_references: List[str] = Field(..., description="List of file paths to input images (1-3 images).")
    image_descriptions: Optional[List[str]] = Field(None, description="AI-generated descriptions of each input image.")
    scenes: Optional[List[SceneSpec]] = Field(None, description="1-2 scenes generated from the input images.")
    final_image_path: Optional[str] = Field(None, description="Path to the final composite image of the sequence.")
