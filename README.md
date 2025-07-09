# Advanced Image Editor

A desktop image editing application built with Python and DearPyGUI that combines traditional photo editing tools with modern AI capabilities. The editor provides standard adjustment controls (exposure, color grading, curves) alongside advanced features like automatic object segmentation using Meta's Segment Anything Model (SAM) and AI-powered content replacement through Stable Diffusion inpainting models. The application supports mask-based selective editing, allowing users to apply different adjustments to specific regions of an image or generate new content within selected areas using text prompts.

## Features

### Core Editing Tools

Standard image editing controls with real-time preview and histogram display.

![Core Editing Demo](docs/gifs/color.gif)

- **Exposure Controls**: Adjust exposure, highlights, shadows, whites, and blacks values
- **Color Adjustments**: Modify saturation and temperature settings for color balance
- **Enhancement Tools**: Apply contrast, texture, and grain effects
- **Curves Editor**: Edit tone curves with RGB channel control
- **Histogram Display**: Real-time RGB histogram that updates with adjustments

### Transform Tools

Basic geometric transformation operations.

![Transform Tools Demo](docs/gifs/crop_and_rotate.gif)

- **Crop & Rotate**: Interactive cropping with rotation controls
- **Flip Operations**: Horizontal and vertical image flipping
- **Smart Boundaries**: Automatic detection of maximum crop areas

### AI-Powered Features

Integration with machine learning models for automated image processing.

![AI Features Demo](docs/gifs/ai_features.gif)

- **Auto-Segmentation**: Uses Meta's Segment Anything Model (SAM) for object detection
- **Mask Generation**: Automatic creation of selection masks from detected objects
- **Generative Inpainting**: Content replacement using Stable Diffusion models with text prompts
- **Hardware Optimization**: GPU acceleration with CPU fallback support

### Masking System

Tools for creating and managing image selections.

- **Multiple Masks**: Support for multiple simultaneous selection masks
- **Selective Editing**: Apply different adjustments to individual masked regions
- **Mask Visualization**: Overlay display with configurable colors and opacity
- **Interactive Selection**: Click-based object selection using AI boundaries

### Interface

User interface built with DearPyGUI framework.

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

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.

## License

MIT License - see [LICENSE](LICENSE) file for details.

This license allows commercial and private use, modification, and distribution of the software.

## Acknowledgments

- Meta AI for Segment Anything Model
- Stability AI for Stable Diffusion  
- Hugging Face for Diffusers library
- OpenCV community for computer vision tools
- DearPyGUI team for the GUI framework