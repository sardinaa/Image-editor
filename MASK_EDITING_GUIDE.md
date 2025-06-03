# Mask-Based Editing Guide

## Overview
The image editor now supports mask-based editing, allowing you to apply editing tools (exposure, contrast, saturation, curves, etc.) to specific regions of your image defined by segmentation masks.

## How to Use Mask Editing

### Step 1: Create Segmentation Masks
1. Load an image using **File → Load Image**
2. Use the **box selection tool** to draw rectangles around objects you want to segment
3. The AI segmentation will automatically create masks for the selected objects
4. Multiple masks can be created and will appear in the **Masks** list

### Step 2: Enable Mask Editing
1. In the **Masks** section of the tool panel, select a mask from the list
2. Check the **"Edit Selected Mask Only"** checkbox
3. The status will show "Editing: [Mask Name]" in yellow text
4. Now all editing tools will only affect the selected mask region

### Step 3: Apply Edits to Masked Region
With mask editing enabled, you can use any of the following tools:

#### Basic Editing Tools
- **Exposure**: Adjust brightness only in the masked area
- **Illumination**: Apply gamma correction to masked pixels
- **Contrast**: Modify contrast within the mask
- **Shadow/Whites/Blacks**: Tone adjustments for masked region
- **Saturation**: Color intensity changes for masked area
- **Texture**: Sharpening effects on masked pixels
- **Temperature**: Color temperature adjustments

#### Advanced Tools
- **RGB Curves**: Apply custom tone curves only to the masked region
- All curve adjustments (Red, Green, Blue channels) respect the mask

### Step 4: Switch Between Masks
- Select different masks from the list to edit different regions
- The "Edit Selected Mask Only" mode will automatically switch to the newly selected mask
- You can disable mask editing by unchecking the checkbox to return to global editing

## Features

### Selective Editing
- Only pixels within the selected mask are affected by editing tools
- Pixels outside the mask remain completely unchanged
- Smooth blending at mask boundaries for natural results

### Real-time Preview
- All mask-based edits update in real-time
- Histogram updates reflect changes in the masked region
- Visual feedback shows exactly which areas are being edited

### Mask Management
- **Delete Mask**: Remove unwanted masks
- **Rename Mask**: Give meaningful names to your masks
- **Mask Visibility**: Selected masks are highlighted in the preview

## Technical Details

### Mask Processing
- Masks are automatically converted to binary format (0 or 1)
- Smooth blending is applied at mask edges
- All editing operations preserve the original image quality

### Performance
- Mask editing adds minimal computational overhead
- Real-time updates maintain smooth performance
- Memory usage is optimized for large images

### Compatibility
- Works with all existing editing tools
- Fully compatible with crop and rotate functionality
- Supports all image formats (PNG, JPG, TIFF, RAW, etc.)

## Tips and Best Practices

### Creating Good Masks
1. Use precise box selections around objects for better segmentation
2. Multiple small selections often work better than one large selection
3. The AI segmentation works best with clear object boundaries

### Editing Workflow
1. Create all needed masks first before editing
2. Use descriptive names for masks (e.g., "Sky", "Building", "Person")
3. Test edits on individual masks before applying to multiple regions
4. Combine mask editing with global adjustments for best results

### Troubleshooting
- If mask editing doesn't work, ensure a mask is selected from the list
- The yellow status text shows the current editing mode
- Disable and re-enable mask editing if issues occur
- Check that the image processor is properly loaded

## Example Workflow

1. **Load a landscape photo**
2. **Create masks**: Draw boxes around sky, mountains, and foreground
3. **Edit sky**: Select sky mask, increase exposure and saturation
4. **Edit mountains**: Select mountain mask, adjust contrast and shadows
5. **Edit foreground**: Select foreground mask, enhance texture and warmth
6. **Final touches**: Disable mask editing and apply global adjustments

This selective editing approach allows for professional-level photo editing with precise control over different regions of your image.