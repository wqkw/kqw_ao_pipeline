# Storyboard Generation Pipeline - Current State

**Last Updated:** Sept 29, 2025
**Status:** Phase 4 Complete - All 10 generation steps operational

---

## System Overview

Fully functional AI-powered storyboard generation pipeline that transforms a single moodboard image + user prompts into complete visual storyboards with:
- High-level story concepts (loglines)
- World-building (lore)
- Plot structure (narrative)
- Scene breakdowns
- Character/location/prop libraries with concept art
- Stage setting shots for each scene
- Individual shot descriptions and images

---

## Architecture

### Module Structure: `storyboard_core/`

```
storyboard_core/
├── __init__.py               - Public API exports
├── artifact.py               - Data models (StoryboardSpec, SceneSpec, ShotSpec, etc.)
├── artifact_adapters.py      - DTO creation, patching, state evaluation
├── utils.py                  - File I/O (images, checkpoints, multi-option outputs)
└── pipeline.py               - Core orchestration and generation functions
```

**Key Design Principles:**
- Single source of truth: `StoryboardSpec` artifact with Optional fields
- Dynamic DTOs: Context/output models auto-generated from artifact field descriptions
- Dependency-driven execution: Pipeline determines step order automatically
- Batch processing: Parallel LLM calls via `openrouter_wrapper.batch_llm()`

### Data Models

**Core Artifact:** `StoryboardSpec`
- `name`: Project identifier
- `title`: Official story title
- `logline`: High-level concept (25-35 words)
- `lore`: World-building and background
- `narrative`: Plot structure and character arcs
- `moodboard_path`: Visual reference image
- `characters`: List[CharacterSpec] with descriptions + images
- `locations`: List[LocationSpec] with descriptions + images
- `props`: List[PropSpec] with descriptions + images
- `scenes`: List[SceneSpec] with shots and stage settings

**Scene Structure:** `SceneSpec`
- `name`, `description`: Scene metadata
- `character_names`, `location_names`, `prop_names`: Component references
- `stage_setting_shot`: ShotSpec with establishing image
- `shots`: List[ShotSpec] for individual shot breakdown

---

## Current Pipeline (10 Steps)

### ✅ All Steps Operational

| Step | Description | Model | Options | Output |
|------|-------------|-------|---------|--------|
| 1. LOGLINE | High-level concept generation | GPT-5 | 5 | Logline text |
| 2. LORE | World-building | GPT-5 (medium reasoning) | 3 | Lore text |
| 3. NARRATIVE | Plot structure | GPT-5 (medium reasoning) | 3 | Narrative text |
| 4. SCENES | Scene breakdown | Gemini 2.5 Flash | 1 | Scene names + descriptions |
| 5. COMPONENT_DESCRIPTIONS | Characters/locations/props | Gemini 2.5 Flash | 1 | Component descriptions |
| 6. COMPONENT_IMAGES | Concept art generation | Gemini 2.5 Flash Image | - | PNG images (3 batches, 2 retries) |
| 7. STAGE_SHOT_DESCRIPTIONS | Establishing shots | Gemini 2.5 Flash | 1 | Stage shot descriptions |
| 8. STAGE_SHOT_IMAGES | Stage setting images | Gemini 2.5 Flash Image | - | PNG images (2 retries) |
| 9. SHOT_DESCRIPTIONS | Individual shot breakdown | GPT-5 (medium reasoning) | 1 | 3 shots per scene |
| 10. SHOT_IMAGES | Individual shot images | Gemini 2.5 Flash Image | - | PNG images (10/batch, 2 retries) |

**Dependencies:**
```
LOGLINE → LORE → NARRATIVE → SCENES → COMPONENT_DESCRIPTIONS → COMPONENT_IMAGES
                                   ↓                              ↓
                          STAGE_SHOT_DESCRIPTIONS → STAGE_SHOT_IMAGES
                                                        ↓
                                              SHOT_DESCRIPTIONS → SHOT_IMAGES
```

**User Choice:**
- Currently auto-selects option 1 for all multi-option outputs (logline, lore, narrative)
- Options saved to markdown files in `data/{project_name}/`

**Checkpoint System:**
- Artifact saved after each step to `data/{project_name}/artifact_after_{step}.json`
- Pipeline can resume from any checkpoint

---

## File Structure

```
kqw_ao_pipeline/
├── storyboard_core/              # Main pipeline module
│   ├── __init__.py
│   ├── artifact.py
│   ├── artifact_adapters.py
│   ├── utils.py
│   └── pipeline.py
├── prompts/                      # Generation guides
│   ├── logline_guide.md          # Logline generation instructions
│   ├── lore_guide.md             # Lore generation instructions
│   └── narrative_guide.md        # Narrative generation instructions
├── data/                         # Generated outputs
│   └── {project_name}/
│       ├── images/               # All generated images
│       ├── artifact_after_*.json # Checkpoints
│       ├── logline_options_*.md  # Multi-option outputs
│       ├── lore_options_*.md
│       └── narrative_options_*.md
├── openrouter_wrapper.py         # LLM API wrapper with retry logic
├── test_pipeline_step_by_step.py # Interactive debugging script
└── script_v3.py                  # Full pipeline execution script
```

---

## Known Issues & Blockers

### Current Blockers: NONE

All 10 generation steps are fully operational with robust error handling and retry logic.

### Minor Issues

1. **Image Generation Success Rate:** ~85-95% depending on complexity
   - Some abstract props or complex characters occasionally fail
   - Retry logic (2 retries per image) mitigates most failures
   - Failed images logged to `image_generation_debug.txt`

2. **User Choice Interface:** Auto-selects option 1 for now
   - Need interactive selection for logline/lore/narrative options
   - Planned for Phase 5 with web UI

3. **No Visualization:** Generated storyboards not yet viewable
   - Need HTML generator for production review
   - Planned for Phase 5

---

## Developer Guide: Modifying the Pipeline

### How to Add Context to an LLM Call

**Use Case:** You want to pass additional information from the artifact to a specific generation step.

**Example:** Adding `logline` as context to the NARRATIVE step

#### Complete Data Flow:
```
Artifact Field → extract_context_dto() → Populated DTO → context_dto_to_string() → LLM Call
```

#### Step-by-Step Instructions:

**1. Define the field in artifact (if new data)**

*File:* `storyboard_core/artifact.py`
*Location:* Inside `StoryboardSpec` class

```python
class StoryboardSpec(StrictModel):
    logline: Optional[str] = Field(None, description="Brief story outline...")
```

*(Skip if field already exists)*

---

**2. Update the step's block in extract_context_dto()**

*File:* `storyboard_core/artifact_adapters.py`
*Function:* `extract_context_dto(artifact, step, **kwargs)`
*Location:* Lines ~33-163

Find the elif block for your step (e.g., NARRATIVE around line ~65). Each step is self-contained with field definition, model creation, and data extraction in one block:

```python
elif step == GenerationStep.NARRATIVE:
    fields = {
        "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
        "lore": (str, Field(..., description=get_field_desc(StoryboardSpec, "lore"))),
        # ADD THIS LINE:
        "logline": (str, Field(..., description=get_field_desc(StoryboardSpec, "logline")))
    }
    ContextModel = create_model("NarrativeContext", **fields, __base__=StrictModel)
    data = kwargs.copy()
    data["lore"] = artifact.lore
    # ADD THIS LINE:
    data["logline"] = artifact.logline
    return ContextModel(**data)
```

**What this does:** Defines the field in the DTO schema, then directly extracts it from the artifact. If the field is missing, Pydantic will raise a validation error.

---

**3. (Optional) Special LLM call handling**

*File:* `storyboard_core/pipeline.py`
*Function:* `generate_step(artifact, step, prompt_input, **kwargs)`
*Location:* Lines ~97-174

Most steps use standard handling (automatically includes context via `context_dto_to_string()`):
```python
context = extract_context_dto(artifact, step, prompt_input=prompt_input, **kwargs)
context_str = context_dto_to_string(context)
responses, _, _ = await batch_llm(
    model=model,
    texts=[prompt_input],
    context=context_str,
    response_format=OutputModel,
    reasoning_effort=reasoning
)
```

Only add special handling if you need custom prompt formatting (like LOGLINE/LORE/NARRATIVE steps which use guide files).

---

#### Summary Checklist for Adding Context:
- [ ] `artifact.py`: Ensure field exists in StoryboardSpec
- [ ] `artifact_adapters.py` → `extract_context_dto()`: Add field to the step's elif block (both in fields dict and data assignment)
- [ ] `pipeline.py` → `generate_step()`: *(Usually automatic)* Add custom handling only if needed

---

### How to Add Structured Output Fields

**Use Case:** You want the LLM to return additional information in its structured response.

**Example:** Adding `themes` field to NARRATIVE output

#### Complete Data Flow:
```
Output DTO → LLM Response → Patch Artifact
```

#### Step-by-Step Instructions:

**1. Define storage field in artifact (if new)**

*File:* `storyboard_core/artifact.py`
*Location:* Inside `StoryboardSpec` class (or relevant model)

```python
class StoryboardSpec(StrictModel):
    narrative: Optional[str] = Field(None, description="Plot structure...")
    # ADD THIS:
    themes: Optional[List[str]] = Field(None, description="Core themes of the story")
```

---

**2. Add to Output DTO definition**

*File:* `storyboard_core/artifact_adapters.py`
*Function:* `create_output_dto(step)`
*Location:* Lines ~103-164

Find the block for your step (e.g., NARRATIVE at line ~131):

```python
elif step == GenerationStep.NARRATIVE:
    base_desc = get_field_desc(StoryboardSpec, "narrative")
    fields["narrative_option_1"] = (str, Field(..., description=f"First option: {base_desc}"))
    fields["narrative_option_2"] = (str, Field(..., description=f"Second option: {base_desc}"))
    fields["narrative_option_3"] = (str, Field(..., description=f"Third option: {base_desc}"))
    # ADD THESE:
    fields["themes_option_1"] = (List[str], Field(..., description="Themes for option 1"))
    fields["themes_option_2"] = (List[str], Field(..., description="Themes for option 2"))
    fields["themes_option_3"] = (List[str], Field(..., description="Themes for option 3"))
```

**Important:** For multi-option outputs (logline/lore/narrative), you need N versions of each field (one per option).

---

**3. Update patching logic**

*File:* `storyboard_core/artifact_adapters.py`
*Function:* `patch_artifact(artifact, step, output, user_choice)`
*Location:* Lines ~221-290

Find the block for your step (e.g., NARRATIVE at line ~248):

```python
elif step == GenerationStep.NARRATIVE:
    options = [output.narrative_option_1, output.narrative_option_2, output.narrative_option_3]
    save_multi_option_output(artifact.name, step, options, user_choice)
    artifact.narrative = options[user_choice]
    # ADD THIS:
    theme_options = [output.themes_option_1, output.themes_option_2, output.themes_option_3]
    artifact.themes = theme_options[user_choice]
```

**What this does:** Takes the LLM's structured output and writes it to the artifact.

---

**4. Update completion status check (if required field)**

*File:* `storyboard_core/artifact_adapters.py`
*Function:* `get_completion_status(artifact)`
*Location:* Lines ~419-447

If your field is required for the step to be "complete":

```python
GenerationStep.NARRATIVE: artifact.narrative is not None and artifact.themes is not None,
```

---

#### Summary Checklist for Adding Output:
- [ ] `artifact.py`: Add field to StoryboardSpec (or relevant model)
- [ ] `artifact_adapters.py` → `create_output_dto()`: Add field(s) to Output DTO
- [ ] `artifact_adapters.py` → `patch_artifact()`: Add extraction and write logic
- [ ] `artifact_adapters.py` → `get_completion_status()`: *(Optional)* Update completion check

---

### Architecture Overview: Data Flow Patterns

#### Context Flow (Input to LLM):
```
Artifact (data storage)
  ↓
extract_context_dto(artifact, step, **kwargs) [creates DTO, extracts data, returns populated instance]
  ↓
context_dto_to_string(context) [formats DTO fields as readable string]
  ↓
LLM receives as "context" parameter
```

#### Output Flow (LLM Response):
```
LLM returns structured JSON
  ↓
Validated against OutputModel (Pydantic auto-validates)
  ↓
patch_artifact() [writes to artifact fields]
  ↓
Artifact updated with new data
```

#### Two LLM Call Patterns:

**Pattern 1: Standard (most steps)**
```python
context = extract_context_dto(artifact, step, prompt_input=prompt_input, **kwargs)
context_str = context_dto_to_string(context)
responses, _, _ = await batch_llm(
    model=model,
    texts=[prompt_input],
    context=context_str,
    response_format=OutputModel,
    reasoning_effort=reasoning
)
```

**Pattern 2: Guide-based (LOGLINE/LORE/NARRATIVE steps)**
```python
guide = _read_guide("prompts/narrative_guide.md")
responses, _, _ = await batch_llm(
    model=model,
    texts=[prompt_input],
    context=guide,  # Guide content used as context
    image_paths=[moodboard_path],  # For LOGLINE/LORE only
    response_format=OutputModel,
    reasoning_effort=reasoning
)
```

---

### Quick Reference: File Responsibilities

| Task | File | Function/Class | Lines |
|------|------|----------------|-------|
| Define data models | `artifact.py` | `StoryboardSpec`, `SceneSpec` | 96-110 |
| Create and populate context DTO | `artifact_adapters.py` | `extract_context_dto()` | 33-163 |
| Convert context DTO to string | `artifact_adapters.py` | `context_dto_to_string()` | 166-185 |
| Define structured outputs | `artifact_adapters.py` | `create_output_dto()` | 188-250 |
| Write outputs to artifact | `artifact_adapters.py` | `patch_artifact()` | 255-325 |
| Check step completion | `artifact_adapters.py` | `get_completion_status()` | ~400-450 |
| Custom LLM call logic | `pipeline.py` | `generate_step()` | 97-174 |
| Read generation guides | `pipeline.py` | `_read_guide()` | 78-92 |
| Add new generation step | `pipeline.py` | `GenerationStep` enum | 30-41 |
| Step dependencies | `artifact_adapters.py` | `get_dependencies()` | ~380-395 |

---

## Next Steps & Future Work

### Phase 5: Visualization and Video Generation (NEXT PRIORITY)

#### 1. HTML Storyboard Visualization (HIGH PRIORITY)
- [ ] Create `generate_html_storyboard(artifact: StoryboardSpec) -> str`
- [ ] Generate comprehensive HTML with:
  - Logline, lore, narrative sections
  - Component library (characters, locations, props with images)
  - Scene breakdowns with stage settings
  - **Sequential shot display** (main focus):
    - All shots across all scenes in order
    - Shot image + description side-by-side
    - Scene context markers
- [ ] Clean, printable format for production review
- [ ] Navigation: Jump to scenes, components, shots
- [ ] Save to `data/{project_name}/storyboard.html`

#### 2. Video Generation Pipeline (HIGH PRIORITY)
- [ ] **Extend ShotSpec model** with video fields:
  - `duration_seconds`: Shot length
  - `camera_movement`: Pan, tilt, push-in, etc.
  - `transition_type`: Cut, fade, dissolve
  - `audio_cues`: SFX, dialogue, music
  - `motion_description`: Character/object movement
- [ ] **New generation step: SHOT_VIDEO_SPECS**
  - Generate video metadata for each shot
  - Context: shot description + stage setting
  - Output: Enhanced shot specs with video parameters
- [ ] **New generation step: SHOT_VIDEOS**
  - Model: Video generation API (Runway/Pika/etc.)
  - Input: Shot image + motion description + duration
  - Context: Previous/next shots for continuity
  - Storage: `data/{project_name}/videos/`
- [ ] **Video assembly**:
  - Stitch shots into scene videos
  - Apply transitions
  - Add audio layers
  - Export per-scene and full storyboard videos

#### 3. Enhanced HTML with Video Integration
- [ ] Embed video players for shots
- [ ] Video download links
- [ ] Show both static image and video versions

#### 4. Code Quality Improvements (MEDIUM PRIORITY)
- [ ] Add type hints to all functions
- [ ] Write comprehensive docstrings
- [ ] Unit tests for artifact_adapters.py
- [ ] Integration tests for full pipeline
- [ ] Example scripts for common workflows

### Phase 6: Production Features (FUTURE)

1. **User Choice Interface**
   - [ ] Interactive selection for logline/lore/narrative options
   - [ ] Web-based UI for pipeline management
   - [ ] Choice persistence and revision history

2. **Advanced Guides**
   - [x] `narrative_guide.md` for plot structure guidance (completed)
   - [ ] `scenes_guide.md` for scene breakdown guidance
   - [ ] Guide templates for different genres

3. **Performance & Quality**
   - [ ] Caching for expensive operations
   - [ ] Advanced retry strategies with fallback models
   - [ ] Quality validation for generated content
   - [ ] Shot editing and regeneration capabilities

4. **Export & Integration**
   - [ ] Export to professional formats (Final Draft, Adobe Premiere XML)
   - [ ] Audio generation and synchronization
   - [ ] Collaborative review workflows
   - [ ] Version control for storyboards

---

## Quick Start

### Running the Full Pipeline

```bash
# Install dependencies
uv sync

# Run full pipeline
uv run python script_v3.py
```

### Interactive Step-by-Step Testing

```bash
# Run with pause points between steps
uv run python test_pipeline_step_by_step.py
```

### Using as a Library

```python
from storyboard_core import StoryboardSpec, GenerationStep, run_generation_pipeline

# Initialize artifact
artifact = StoryboardSpec(
    name="my_story",
    title="My Story Title",
    moodboard_path="path/to/moodboard.png"
)

# Define user inputs
user_inputs = {
    GenerationStep.LOGLINE: "Generate story concepts based on the moodboard",
    GenerationStep.LORE: "Create fantasy world with ancient magic",
    # ... etc for all steps
}

# Run pipeline
completed_artifact = await run_generation_pipeline(artifact, user_inputs)
```

---

## Configuration

### Model Selection (in `storyboard_core/pipeline.py`)

```python
def _select_model_for_step(step: GenerationStep) -> Tuple[str, str]:
    # Lore: GPT-5 + medium reasoning
    if step == GenerationStep.LORE:
        return "openai/gpt-5", "medium"

    # Image generation: Gemini 2.5 Flash
    if step in [COMPONENT_IMAGES, STAGE_SHOT_IMAGES, SHOT_IMAGES]:
        return "google/gemini-2.5-flash-image-preview", "minimal"

    # Other text: Gemini 2.5 Flash
    return "google/gemini-2.5-flash", "minimal"
```

### Image Storage

- **Format:** PNG
- **Naming:** `{type}_{sanitized_name}_{YYYYMMDD_HHMMSS}.png`
- **Location:** `data/{project_name}/images/`

### Retry Logic

- **Component images:** 2 retries (3 total attempts)
- **Stage shot images:** 2 retries
- **Individual shot images:** 2 retries

---

## Testing Status

### ✅ Manually Tested
- Full pipeline execution from moodboard to final shots
- Checkpoint save/resume functionality
- Error handling and retry logic
- Batch processing for all image types

### ⚠️ Needs Testing
- HTML generation (not yet implemented)
- Video generation (not yet implemented)
- User choice interfaces (not yet implemented)
- Edge cases with unusual prompts/content

### ❌ Not Implemented
- Automated unit tests
- Integration test suite
- Performance benchmarks
- Load testing

---

## Dependencies

**Python:** 3.12+
**Package Manager:** uv

**Key Libraries:**
- `pydantic` - Data validation and models
- `openai` - LLM API (via openrouter)
- `python-dotenv` - Environment variables

**API Services:**
- OpenRouter (for GPT-5 and Gemini models)

---

## Environment Setup

```bash
# Required environment variables in .env
OPENROUTER_API_KEY=your_key_here

# Optional
DEFAULT_MODEL=google/gemini-2.5-flash
```

---

*For detailed implementation history, see `implementation.md`*