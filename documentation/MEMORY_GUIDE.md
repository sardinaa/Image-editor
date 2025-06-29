# Image Editor - Memory Management Guide

This guide helps you use the image editor's segmentation feature efficiently, especially on systems with limited GPU memory.

## Quick Memory Check

Before using segmentation, run the memory checker:

```bash
python check_gpu_memory.py
```

This will analyze your system and provide specific recommendations.

## Memory Management Features

The image editor now includes several automatic memory management features:

### 1. **Automatic Device Selection**
- The editor automatically detects if your GPU has sufficient memory
- Falls back to CPU if GPU memory is insufficient
- You can see which device is being used in the console output

### 2. **Dynamic Image Resizing**
- Images are automatically resized based on available GPU memory
- Lower memory systems use smaller processing sizes (512px vs 1024px)
- Results are scaled back to original image dimensions

### 3. **Memory Cleanup**
- CUDA cache is cleared before each segmentation
- Automatic garbage collection
- Memory monitoring during processing

### 4. **Fallback Processing**
- If segmentation fails due to memory, tries with smaller image size
- Can automatically switch to CPU if GPU runs out of memory
- Graceful error handling with informative messages

## Troubleshooting Memory Issues

### If you get "CUDA out of memory" errors:

1. **Check available memory:**
   ```bash
   python check_gpu_memory.py
   ```

2. **Set environment variable:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python main.py
   ```

3. **Close other GPU applications:**
   - Web browsers with GPU acceleration
   - Other AI/ML applications
   - Games or graphics software

4. **Restart the application:**
   - Memory can accumulate over time
   - Fresh start often resolves issues

5. **Use smaller images:**
   - Resize images before loading
   - The editor will work with any size, but smaller is safer

### Memory Requirements by GPU Size:

| GPU Memory | Recommended | Max Image Size | Notes |
|------------|-------------|----------------|-------|
| < 3GB      | CPU Mode    | Any           | Slower but stable |
| 3-6GB      | GPU         | 768x768       | Monitor memory usage |
| 6-8GB      | GPU         | 1024x1024     | Should work well |
| > 8GB      | GPU         | 1024x1024+    | Full performance |

## Advanced Settings

### Force CPU Mode
If you want to always use CPU (slower but no memory limits):

Edit `ui/main_window.py` and change:
```python
self.segmenter = ImageSegmenter(device="cpu")
```

### Adjust Maximum Image Size
Edit `ui/segmentation.py` and modify the `max_image_size` values in `_adjust_max_size_for_memory()`.

## Performance Tips

1. **Load smaller images** when possible
2. **Close browser tabs** that use GPU
3. **Restart the application** if you notice slowdowns
4. **Use the memory checker** to verify optimal settings
5. **Process one image at a time** for best results

## Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "CUDA out of memory" | Insufficient GPU memory | Use CPU mode or smaller images |
| "Failed to create segmenter" | Model loading failed | Check model file exists, try CPU mode |
| "Memory-related error detected" | Various memory issues | Restart application, check available memory |

## Monitoring Memory Usage

The console output will show:
- Device being used (CUDA/CPU)
- Available memory before processing
- Image resize operations
- Memory cleanup operations
- Fallback operations when needed

Watch for these messages to understand what's happening with memory management.
