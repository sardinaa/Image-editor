# Advanced Image Editor

A desktop image editing application built with Python and DearPyGUI that combines traditional photo editing tools with modern AI capabilities. The editor provides standard adjustment controls (exposure, color grading, curves) alongside advanced features like automatic object segmentation using Meta's Segment Anything Model (SAM) and AI-powered content replacement through Stable Diffusion inpainting models. The application supports mask-based selective editing, allowing users to apply different adjustments to specific regions of an image or generate new content within selected areas using text prompts.

## Features

### Core Editing Tools

Standard image editing controls with real-time preview and histogram display.

![Core Editing Demo](docs/gifs/core-editing-demo.gif)

- **Exposure Controls**: Adjust exposure, highlights, shadows, whites, and blacks values
- **Color Adjustments**: Modify saturation and temperature settings for color balance
- **Enhancement Tools**: Apply contrast, texture, and grain effects
- **Curves Editor**: Edit tone curves with RGB channel control
- **Histogram Display**: Real-time RGB histogram that updates with adjustments

### Transform Tools

Basic geometric transformation operations.

![Transform Tools Demo](docs/gifs/transform-tools-demo.gif)

- **Crop & Rotate**: Interactive cropping with rotation controls
- **Flip Operations**: Horizontal and vertical image flipping
- **Smart Boundaries**: Automatic detection of maximum crop areas

### AI-Powered Features

Integration with machine learning models for automated image processing.

![AI Features Demo](docs/gifs/ai-features-demo.gif)

- **Auto-Segmentation**: Uses Meta's Segment Anything Model (SAM) for object detection
- **Mask Generation**: Automatic creation of selection masks from detected objects
- **Generative Inpainting**: Content replacement using Stable Diffusion models with text prompts
- **Hardware Optimization**: GPU acceleration with CPU fallback support

### Masking System

Tools for creating and managing image selections.

![Advanced Masking Demo](docs/gifs/advanced-masking-demo.gif)

- **Multiple Masks**: Support for multiple simultaneous selection masks
- **Selective Editing**: Apply different adjustments to individual masked regions
- **Mask Visualization**: Overlay display with configurable colors and opacity
- **Interactive Selection**: Click-based object selection using AI boundaries

### Interface

User interface built with DearPyGUI framework.

![User Interface Demo](docs/gifs/user-interface-demo.gif)

- **Modular Panels**: Component-based UI with organized tool sections
- **Real-time Updates**: Immediate visual feedback for parameter changes
- **Performance Monitoring**: Memory management and GPU resource optimization
- **Export Support**: Save functionality for PNG, JPG, TIFF, and BMP formats

## Getting Started

### Requirements

- Python 3.8+
- Optional: CUDA-compatible GPU for AI features
- 8GB RAM minimum (16GB recommended for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image-editor
   ```

2. **Install dependencies**
   ```bash
   pip install dearpygui opencv-python numpy pillow
   
   # For AI features (optional but recommended)
   pip install torch torchvision diffusers transformers
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

3. **Download AI Models** (Optional)
   
   For AI features, place these models in `assets/models/`:
   - **SAM Model**: `sam_vit_h_4b8939.pth` - Download from [Segment Anything](https://github.com/facebookresearch/segment-anything)
   - **Inpainting Model**: `512-inpainting-ema.ckpt` - Stable Diffusion inpainting checkpoint

### Running

```bash
python main_production.py
```

## Usage

### Basic Operations

1. **Load Image**: Use File menu or dialog to open supported formats (PNG, JPG, JPEG, BMP, TIFF)
2. **Edit**: Use exposure and color panels to adjust image parameters
3. **Transform**: Enable crop mode to rotate, flip, or crop the image  
4. **Export**: Save the edited image in desired format

### AI Features

1. **Segmentation**: Open Masks panel and click "Auto Segment" to detect objects
2. **Selective Editing**: Select masks and apply different adjustments to regions
3. **Inpainting**: Select a mask, enter text prompts, and click "Reimagine" to replace content

### Performance Notes

- AI features use GPU when available, fall back to CPU
- Large images are automatically resized for processing
- Segmentation runs in background to maintain UI responsiveness

## Architecture

### Service Layer Architecture
```
├── core/
│   ├── application.py          # Central application coordinator
│   └── services/               # Business logic services
│       ├── image_service.py    # Image operations and state
│       ├── mask_service.py     # Mask management
│       ├── segmentation_service.py  # AI segmentation
│       └── generative_service.py    # AI inpainting
├── processing/
│   ├── image_processor.py      # Core image processing
│   ├── segmentation.py         # SAM integration
│   └── file_manager.py         # I/O operations
├── ui/
│   ├── components/             # Modular UI panels
│   ├── interactions/           # User interaction handlers
│   ├── renderers/              # Visual rendering components
│   └── windows/                # Main window management
└── utils/
    ├── memory_utils.py         # Performance optimization
    ├── performance_config.py   # Configuration management
    └── ui_helpers.py           # UI utilities
```

### Key Design Patterns

- **Service Layer**: Separates business logic from UI
- **Component Architecture**: Modular, reusable UI components
- **Event-Driven**: Reactive updates with callback system
- **Memory Management**: Automatic GPU/CPU optimization
- **Error Handling**: Graceful degradation for missing dependencies

## Configuration

### Performance Settings

The application automatically configures optimal settings based on your hardware:

- **GPU Detection**: Automatically uses CUDA when available
- **Memory Limits**: Adaptive based on available RAM
- **Processing Threads**: Optimized for your CPU cores
- **Cache Management**: Intelligent cleanup and optimization

### Customization

Edit `utils/performance_config.py` to adjust:
- Maximum visible mask overlays
- Update throttling intervals
- Memory usage thresholds
- Rendering quality settings

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce image size or disable GPU acceleration
- Close other GPU-intensive applications
- Use CPU fallback mode

**"Model not found"**
- Ensure AI models are in `assets/models/` directory
- Check file permissions and paths
- Download models from official sources

**"Slow performance"**
- Enable GPU acceleration if available
- Reduce number of visible mask overlays
- Close unnecessary applications

### Error Handling

The application includes comprehensive error handling:
- Automatic fallback to CPU for GPU errors
- Graceful degradation when AI models are missing
- Memory cleanup on application exit
- Detailed error logging for debugging

## 🤝 Contributing

Contributions are welcome. Please read the guidelines below before submitting changes.

### Development Setup

1. Fork and clone the repository
2. Install dependencies as described in the installation section
3. Create a feature branch: `git checkout -b feature-name`

### Guidelines

- Follow PEP 8 style conventions
- Add type hints where appropriate
- Test changes with different image formats and sizes
- Ensure AI features work with both GPU and CPU
- Update documentation for user-facing changes

### Contribution Types

- **Bug Reports**: Include reproduction steps and system info
- **Features**: Discuss large changes in an issue first
- **Documentation**: Improve existing docs or add examples
- **Performance**: Profile and optimize bottlenecks

### Testing

- Test with various image formats and sizes
- Verify GPU/CPU fallback behavior
- Check memory usage with large images
- Ensure UI responsiveness during processing

## License

MIT License - see [LICENSE](LICENSE) file for details.

This license allows commercial and private use, modification, and distribution of the software.

## Acknowledgments

- Meta AI for Segment Anything Model
- Stability AI for Stable Diffusion  
- Hugging Face for Diffusers library
- OpenCV community for computer vision tools
- DearPyGUI team for the GUI framework

## 📝 TODO

### Bug Fixes
- [x] Reopen Image (texture is not updated properly)
- [ ] Fix pan interaction breaking after extended use
- [ ] Start image at position (0,0) instead of centered

### Feature Additions
- [ ] Invert mask functionality
- [ ] Before and after transform view
- [ ] Perspective editing tools
- [ ] Batch processing support
- [ ] HDR image format support

---

This is an open-source image editor that combines traditional editing tools with AI capabilities. The AI features require additional model downloads but are optional for basic functionality.
