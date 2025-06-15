# Technical Architecture Documentation

## Component Dependency Map

```
📁 Production Image Editor Architecture

┌─────────────────────────────────────────────────────────────────────┐
│                        MAIN ENTRY POINT                            │
│  main_production.py (ProductionImageEditor)                        │
│  ├── Application initialization                                    │
│  ├── File dialog management                                        │
│  ├── Image loading with CropRotateUI integration                   │
│  └── Cleanup and resource management                               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                      SERVICE LAYER                                 │
│  core/application.py (ApplicationService)                          │
│  ├── Business logic coordination                                   │
│  ├── Service management (ImageProcessor, FileManager, Segmenter)   │
│  ├── Cross-cutting concerns                                        │
│  └── Error handling and validation                                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                       UI LAYER                                     │
│  ui/main_window_production.py (ProductionMainWindow)              │
│  ├── Layout management and window setup                            │
│  ├── Event handling and mouse interactions                         │
│  ├── Component coordination                                        │
│  ├── Segmentation system integration                               │
│  ├── Mask management and overlays                                  │
│  └── Display updates and rendering                                 │
└─────┬───────────────────┬───────────────────┬───────────────────────┘
      │                   │                   │
┌─────▼─────┐    ┌────────▼────────┐    ┌─────▼─────────────────┐
│UI PANELS  │    │CROP & ROTATE    │    │SEGMENTATION SYSTEM    │
│           │    │                 │    │                       │
│Tool Panel│    │crop_rotate.py   │    │segmentation.py        │
│Modular    │    │├── BBox Renderer│    │├── SAM Integration   │
│├── Exposure    │├── Mouse Events │    │├── Memory Mgmt       │
│├── Color       │├── Visual Style │    │├── GPU/CPU Fallback  │
│├── Masks       │└── Texture Mgmt │    │└── Box/Auto Segment  │
│├── Curves      │                 │    │                       │
│└── Histogram   │                 │    │bounding_box_renderer  │
└───────────────┘    └─────────────────┘    │├── Visual Style     │
                                           │├── Mouse Interaction │
                                           │├── Coordinate Mapping│
                                           │└── Unified Rendering │
                                           └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                                 │
├─────────────────────┬───────────────────────┬───────────────────────┤
│IMAGE PROCESSOR      │FILE MANAGER           │MEMORY UTILS           │
│                     │                       │                       │
│image_processor.py   │file_manager.py        │memory_utils.py        │
│├── Effects Pipeline │├── Load/Save Ops      │├── GPU Monitoring     │
│├── Mask Integration │├── Format Support     │├── CUDA Cache Mgmt    │
│├── Parameter Apply  │├── Error Handling     │├── Memory Cleanup     │
│└── Selective Edit   │└── Path Validation    │└── Performance Opt    │
└─────────────────────┴───────────────────────┴───────────────────────┘
```

## Data Flow Architecture

```
USER INTERACTION FLOW
=====================

1. IMAGE LOADING
   User → File Dialog → ApplicationService → FileManager → ImageProcessor
   ↓
   CropRotateUI Creation → Texture Setup → Display Update
   ↓
   Segmentation System Init → BoundingBox Renderer Setup

2. PARAMETER CHANGES
   User → Tool Panel → Main Window → ApplicationService → ImageProcessor
   ↓
   Effect Application (Global or Mask-based) → Texture Update → Display

3. SEGMENTATION WORKFLOW
   User → Box Selection Toggle → Segmentation Mode Enable
   ↓
   Mouse Events → BoundingBox Renderer → Coordinate Translation
   ↓
   Confirm → SAM Processing → Mask Generation → Overlay Update

4. CROP/ROTATE OPERATIONS
   User → Crop Mode Toggle → BoundingBox Renderer (Crop Style)
   ↓
   Mouse Events → Rectangle Adjustment → Preview Update
   ↓
   Apply → Image Processing → Display Update
```

## Component Communication Patterns

```
EVENT DELEGATION PATTERN
========================

ProductionMainWindow
├── Mouse Events
│   ├── _on_mouse_down()
│   │   ├── Crop Mode → CropRotateUI.on_mouse_down()
│   │   ├── Segmentation Mode → segmentation_bbox_renderer.on_mouse_down()
│   │   └── Box Selection → _handle_box_selection_mouse_down()
│   │
│   ├── _on_mouse_drag()
│   │   ├── Crop Mode → CropRotateUI.on_mouse_drag()
│   │   ├── Segmentation Mode → segmentation_bbox_renderer.on_mouse_drag()
│   │   └── Box Selection → _handle_box_selection_drag()
│   │
│   └── _on_mouse_release()
│       ├── Crop Mode → CropRotateUI.on_mouse_release()
│       ├── Segmentation Mode → segmentation_bbox_renderer.on_mouse_release()
│       └── Box Selection → _handle_box_selection_release()
│
├── Parameter Updates
│   └── _on_parameter_change()
│       ├── ApplicationService.update_parameters()
│       ├── ImageProcessor.apply_effects()
│       └── Display Update
│
└── Segmentation Events
    ├── enable_segmentation_mode()
    ├── _update_segmentation_overlay()
    ├── confirm_segmentation_selection()
    └── segment_with_box()
```

## Memory Management Strategy

```
MEMORY OPTIMIZATION LAYERS
===========================

1. APPLICATION LEVEL
   ├── MemoryManager.cleanup_gpu_memory()
   ├── ApplicationService.cleanup()
   └── Graceful shutdown procedures

2. SEGMENTATION LEVEL
   ├── Automatic device selection (GPU/CPU)
   ├── Dynamic image resizing based on memory
   ├── CUDA cache clearing between operations
   └── Fallback processing for memory errors

3. PROCESSING LEVEL
   ├── Efficient texture management
   ├── Raw texture usage for performance
   ├── Memory pool optimization
   └── Garbage collection coordination

4. UI LEVEL
   ├── Update throttling (60 FPS limit)
   ├── Texture reuse where possible
   ├── Component cleanup on shutdown
   └── Event handler cleanup
```

## Visual Styling Consistency

```
BOUNDING BOX RENDERER UNIFIED STYLING
======================================

Default Style (Used by Both Modes):
├── Box Color: (64, 64, 64, 255)     # Dark Gray
├── Handle Color: (13, 115, 184, 255) # Blue
├── Box Thickness: 2 pixels
├── Handle Size: 20 pixels
└── Handle Threshold: 50 pixels

Application:
├── Crop Mode: Uses default styling
└── Segmentation Mode: Explicitly set to match default

Visual Consistency Achieved:
├── Identical appearance across modes
├── Same interaction patterns
├── Unified user experience
└── Professional look and feel
```

## Error Handling Strategy

```
LAYERED ERROR HANDLING
======================

1. APPLICATION LAYER
   ├── Try-catch in main entry points
   ├── Graceful degradation on failures
   ├── User-friendly error messages
   └── Automatic cleanup on errors

2. SERVICE LAYER
   ├── Validation of inputs
   ├── Resource availability checks
   ├── Error propagation with context
   └── Service state management

3. PROCESSING LAYER
   ├── Memory error detection
   ├── GPU/CPU fallback mechanisms
   ├── File I/O error handling
   └── Image format validation

4. UI LAYER
   ├── Component existence checks
   ├── Event handler safety
   ├── Display update protection
   └── User feedback on errors
```

This technical documentation provides a comprehensive view of the production image editor's architecture, showing how all components work together to deliver a robust, maintainable, and user-friendly application.
