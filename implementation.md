# Storyboard Generation Pipeline Implementation

## Overview

This document outlines the implementation of a dynamic, dependency-driven pipeline for generating complete storyboards from user prompts. The system uses a core artifact as the single source of truth and generates structured outputs through LLM calls with automatic field description propagation.

## Architecture & Design Choices

### 1. Single Source of Truth
- **Core Artifact**: `StoryboardSpec` serves as the storage model with Optional fields
- **Dynamic DTOs**: Context and output DTOs are generated dynamically from core artifact field descriptions
- **Field Propagation**: All DTO descriptions are derived from the core artifact using Pydantic field introspection

### 2. Dual Model System
- **Storage Models**: Core artifact and nested models (PropSpec, SceneSpec, etc.) with Optional fields
- **Communication Models**: Strictly validated DTOs for LLM communication with no optional fields
- **Separation**: Clear distinction between storage (progressive generation) and communication (strict validation)

### 3. Dependency-Driven Execution
- **DAG Structure**: Generation steps have explicit dependencies (e.g., narrative depends on lore)
- **Automatic Ordering**: Pipeline automatically determines execution order based on completed dependencies
- **State Tracking**: Completion status checked before each step execution

### 4. Batch Processing Architecture
- **Parallel Generation**: Uses `batch_llm` from openrouter_wrapper for concurrent operations
- **Async Throughout**: Full async/await support for performance
- **Model Selection**: GPT-5 for structured outputs, Gemini 2.5 Flash for image generation

## Files Created/Modified

### Created Files
1. **`generation_pipeline.py`** (moved from prompts/)
   - Core pipeline implementation
   - Dynamic DTO generation functions
   - Dependency management
   - Pipeline orchestrator

2. **`script_v3.py`**
   - Usage example script
   - Demonstrates full pipeline execution
   - User input configuration

### Modified Files
1. **`prompts/storyboard_artifact.py`**
   - Added `image_url: Optional[str]` to `PropSpec`
   - Added `image_url: Optional[str]` to `ShotSpec`
   - Extended storage models for image tracking

## Current Implementation Status

### ✅ Completed Features

#### Dynamic DTO System
- `create_context_dto()`: Generates context DTOs from artifact field descriptions
- `create_output_dto()`: Generates structured output DTOs with strict validation
- `extract_context_data()`: Extracts relevant data from artifact for context building

#### Dependency Management
- `DEPENDENCIES` dict: Maps generation steps to their prerequisites
- `get_completion_status()`: Checks which steps have been completed
- `get_next_steps()`: Determines available steps based on dependencies

#### Basic Pipeline Operations
- `patch_artifact()`: Updates core artifact from LLM structured outputs
- `generate_step()`: Executes single generation step with proper context
- `generate_batch_components()`: Parallel component image generation
- `run_generation_pipeline()`: Main orchestrator function

#### Generation Steps (Partial)
- ✅ Lore generation (3 options)
- ✅ Narrative generation (3 options)
- ✅ Scene generation (names, descriptions, sketches)
- ✅ Component descriptions (props + scene mapping)
- ✅ Component images (batch parallel generation)
- ⚠️ Stage shot descriptions (structure exists, not integrated)
- ❌ Stage shot images (not implemented)
- ❌ Individual shot images (not implemented)

## Outstanding Implementation Tasks

### 1. Image Storage System
**Priority: HIGH**
- [ ] Create `data/` folder structure for local image storage
- [ ] Implement `save_image_to_data()` function to replace `save_image_and_get_url()`
- [ ] Store base64 images as PNG/JPG files with systematic naming
- [ ] Update artifact with local file paths instead of URLs
- [ ] #NEED_INPUT: Preferred image format (PNG/JPG) and naming convention

### 2. Complete Scene Processing Pipeline
**Priority: HIGH**

#### Stage Shot Descriptions
- [ ] Implement `generate_batch_stage_shots()` function
- [ ] Create context extraction for scene-level data
- [ ] Integrate with main pipeline orchestrator
- [ ] Update `patch_scene_stage_shot()` for batch processing

#### Stage Shot Images
- [ ] Implement `generate_stage_shot_images()` using Gemini 2.5 Flash
- [ ] Build context with scene description + component image references
- [ ] Store images in data folder with scene-based naming
- [ ] Update scene.stage_setting_shot.image_url with local paths

#### Individual Shot Generation
- [ ] Implement shot description generation per scene
- [ ] Create `generate_scene_shots()` for individual shots
- [ ] Generate shot images using stage setting as reference
- [ ] Update scene.shots with complete shot specifications

### 3. Enhanced Patching System
**Priority: MEDIUM**

#### Scene-Level Patchers
- [ ] Complete `patch_scene_stage_shot()` integration
- [ ] Implement `patch_scene_shots()` for individual shots
- [ ] Add validation for scene consistency

#### Image URL Patchers
- [ ] Extend `patch_prop_image_url()` for data folder paths
- [ ] Implement `patch_shot_image_url()` for shot images
- [ ] Add batch patching functions for multiple images

### 4. Pipeline Orchestration Enhancements
**Priority: MEDIUM**

#### Per-Scene Processing
- [ ] Add scene iteration logic to `run_generation_pipeline()`
- [ ] Implement per-scene batch processing
- [ ] Handle scene-specific context building
- [ ] Add progress tracking per scene

#### User Choice Integration
- [ ] Implement user selection interface for multi-option outputs
- [ ] Add choice persistence between pipeline runs
- [ ] Create review/approval workflow for generated content
- [ ] #NEED_INPUT: Preferred user interface (CLI prompts, web UI, config file)

### 5. Error Handling & Validation
**Priority: MEDIUM**

#### Generation Failure Handling
- [ ] Add retry logic for failed LLM calls
- [ ] Implement fallback strategies for image generation failures
- [ ] Add validation for generated content quality
- [ ] Create error recovery mechanisms

#### Dependency Validation
- [ ] Validate dependency completeness before step execution
- [ ] Add meaningful error messages for missing dependencies
- [ ] Implement dependency chain validation

### 6. Performance & Monitoring
**Priority: LOW**

#### Progress Tracking
- [ ] Add detailed progress reporting with percentages
- [ ] Implement step timing and performance metrics
- [ ] Create pipeline execution summaries

#### Caching & Optimization
- [ ] Add caching for expensive operations
- [ ] Implement resume capability for interrupted pipelines
- [ ] Optimize batch sizes for different generation types

## Configuration Requirements

### #NEED_INPUT: User Configuration Decisions

1. **Image Storage**
   - Preferred image format: PNG/JPG? : PNG
   - Naming convention: `{step}_{item_name}_{timestamp}.ext`? : it should be like component_componentname_timestamp or shot_shotname_timestamp
   - Folder structure: `data/images/{project_name}/` or `data/{project_name}/images/`? Do the latter

2. **User Choice Interface**
   - CLI prompts for multi-option selection? Yes leave room for this, but for now just pretend user selects option 1 every time. We will web interface later

3. **Model Configuration**
   - Reasoning effort level for structured outputs? : For lore, narrative, and scenes put medium and use gpt-5, rest of them dont think.
   - Image generation model preferences (Gemini 2.5 Flash vs others)? : always use gemini 2.5 flash
   - Fallback models for generation failures? : fallback to gemini 2.5 pro for everything else except lore, narrative, and scenes. 

4. **Pipeline Behavior**
   - Auto-proceed with default choices or always prompt user? : for now auto proceed, as above assume user always picks option 1
   - Save intermediate results at each step? : yes put it in data/
   - Enable pipeline resumption from any step? : yes that woukd be udeak

## Next Steps

### Phase 1: Core Image Pipeline 
1. Implement data folder image storage system
2. Complete stage shot descriptions integration
3. Add stage shot image generation
4. Test end-to-end pipeline for first 7 steps

### Phase 2: Individual Shot Generation 
1. Implement individual shot description generation
2. Add shot image generation with stage setting reference
3. Complete all 9 generation steps
4. Add comprehensive error handling

### Phase 3: User Experience 
1. Implement user choice interfaces
2. Add progress tracking and monitoring
3. Create pipeline resumption capability
4. Performance optimization and caching

## Testing Strategy

### Unit Tests Needed
- [ ] DTO generation functions
- [ ] Dependency validation logic
- [ ] Patching functions
- [ ] Image storage operations

### Integration Tests Needed
- [ ] End-to-end pipeline execution
- [ ] Error recovery scenarios
- [ ] Performance benchmarks
- [ ] User choice workflows

### Manual Testing
- [ ] Complete storyboard generation with real content
- [ ] Image quality validation
- [ ] Dependency chain correctness
- [ ] User experience flow

---

*Last Updated: 2025-09-28*
*Status: Phase 1 - Core Implementation Complete, Scene Processing In Progress*