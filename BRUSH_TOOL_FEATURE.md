# Brush Tool Feature

The Brush Tool allows users to manually draw masks directly on the image for precise mask creation and editing. **The brush tool is integrated directly into the Mask Panel** for seamless workflow integration.

## Features

### 1. Brush Size Control
- **Slider Range**: 1-100 pixels
- **Default**: 20 pixels
- Allows for precision work with small brushes or broader coverage with larger brushes

### 2. Opacity Control
- **Range**: 0.1-1.0 (10%-100%)
- **Default**: 1.0 (100%)
- Controls the transparency of the brush strokes

### 3. Hardness Control
- **Range**: 0.0-1.0 (0%-100%)
- **Default**: 0.8 (80%)
- Controls the edge softness of the brush
- 0.0 = Very soft edges
- 1.0 = Hard edges

### 4. Eraser Mode
- **Toggle**: On/Off
- **Default**: Off (Paint mode)
- When enabled, the brush removes mask content instead of adding it

## How to Use

### 1. Access the Brush Tool
1. Load an image in the editor
2. In the tool panel, check the "Masks" checkbox to open the mask panel
3. In the mask panel, check the "Brush Tool" checkbox
4. The brush controls will appear below

### 2. Adjust Brush Settings
- **Brush Size**: Adjust for the desired stroke width
- **Opacity**: Lower for subtle additions, higher for solid coverage
- **Hardness**: Adjust based on desired edge softness
- **Eraser Mode**: Toggle to switch between painting and erasing

### 3. Draw on the Image
1. Move your mouse over the image - you'll see a brush cursor
2. Click and drag to paint/erase
3. The mask overlay will show in real-time as you draw

### 4. Manage the Mask
- **Clear Brush**: Click "Clear Brush" to remove all brush strokes
- **Add to Masks**: Click "Add to Masks" to add the current brush mask to the mask table

## Integration with Mask Workflow

The brush tool is fully integrated with the existing mask workflow:

### Seamless Mask Management
- **Brush masks appear directly in the mask table** below the brush controls
- Can be selected, renamed, and managed like any other mask
- Supports all mask operations (grouping, deletion, etc.)

### Basic Editing
- Brush masks can be selected and edited like any other mask
- Apply exposure, color, and other adjustments to brush-defined areas

### Inpainting
- Use brush masks as inpainting targets
- Perfect for precise area selection for content generation

### Mask Operations
- Brush masks support all standard mask operations:
  - ✅ Selection and multi-selection
  - ✅ Renaming
  - ✅ Grouping/ungrouping
  - ✅ Deletion
  - ✅ Overlay visibility controls

## Technical Implementation

### Architecture
- **Integrated into MasksPanel**: Brush controls are part of the mask management panel
- **BrushRenderer**: Handles brush stroke rendering and mask generation
- **Event Integration**: Mouse events are handled through the centralized event system

### Performance Features
- Real-time brush cursor display
- Optimized stroke rendering with interpolation between points
- Efficient mask overlay updates

### Coordinate System
- Automatically handles coordinate transformations between screen space and texture space
- Accounts for image rotation, scaling, and positioning

## Location in Interface

The Brush Tool is located in the **Mask Panel**:
```
Tool Panel
├── Exposure Controls
├── Color Effects
├── Crop & Rotate
└── Masks ←── Brush Tool is here
    ├── Segmentation Controls
    ├── Brush Tool Controls ←── Here
    ├── Mask Management
    └── Mask Table ←── Brush masks appear here
```

## Tips for Best Results

1. **Start with larger brush sizes** for base coverage, then refine with smaller brushes
2. **Use lower opacity** for gradual mask building
3. **Toggle Eraser Mode** to refine mask edges
4. **Combine with auto-segmentation** - use brush tool to refine automatically generated masks
5. **Use hardness settings** based on the type of edge you want (soft for natural transitions, hard for precise selections)

## Keyboard Shortcuts

- **Escape**: Cancel current brush operation
- **Delete**: Clear current brush mask (when brush tool is active)

## Compatibility

The Brush Tool works seamlessly with all other editor features:
- ✅ Compatible with Crop & Rotate
- ✅ Compatible with Auto-segmentation
- ✅ Compatible with Mask Editing
- ✅ Compatible with Inpainting workflow
- ✅ Supports mask overlay visibility controls
- ✅ Integrated with mask table and management system
