# Contributing to Advanced Image Editor

Contributions are welcome! Please read the guidelines below before submitting changes.

## Development Setup

1. Fork and clone the repository
2. Install dependencies as described in the installation section of the README
3. Create a feature branch: `git checkout -b feature-name`

## Guidelines

- Follow PEP 8 style conventions
- Add type hints where appropriate
- Test changes with different image formats and sizes
- Ensure AI features work with both GPU and CPU
- Update documentation for user-facing changes

## Contribution Types

- **Bug Reports**: Include reproduction steps and system info
- **Features**: Discuss large changes in an issue first
- **Documentation**: Improve existing docs or add examples
- **Performance**: Profile and optimize bottlenecks

## Testing

- Test with various image formats and sizes
- Verify GPU/CPU fallback behavior
- Check memory usage with large images
- Ensure UI responsiveness during processing

## Pull Request Process

1. Ensure your code follows the guidelines above
2. Update documentation if you're changing user-facing functionality
3. Test your changes thoroughly
4. Submit a pull request with a clear description of your changes

## Code of Conduct

Please be respectful and constructive in all interactions. We're building this together!

## Questions?

Feel free to open an issue if you have questions about contributing or need help getting started.

## 📝 TODO

### Bug Fixes
- [x] Reopen Image (texture is not updated properly)
- [ ] Fix pan interaction breaking after extended use
- [ ] Start image at position (0,0) instead of centered

### Feature Additions
- [ ] Performance configuration dialog 
- [ ] Invert mask functionality
- [ ] Before and after transform view
- [ ] Perspective editing tools
- [ ] Batch processing support
- [ ] HDR image format support

---

This is an open-source image editor that combines traditional editing tools with AI capabilities. The AI features require additional model downloads but are optional for basic functionality.

