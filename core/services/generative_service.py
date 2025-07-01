"""Generative service for mask-based image synthesis using an inpainting pipeline."""
from typing import Optional
import numpy as np
from PIL import Image
import os

from utils.memory_utils import MemoryManager, ErrorHandler, ResourceManager

try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionPipeline
    import torch
    import diffusers
    
    # Check diffusers version for compatibility
    diffusers_version = getattr(diffusers, '__version__', 'unknown')
    print(f"Using diffusers version: {diffusers_version}")
    
except Exception as e:  # pragma: no cover - diffusers may not be installed
    StableDiffusionInpaintPipeline = None
    StableDiffusionXLInpaintPipeline = None
    StableDiffusionPipeline = None
    torch = None
    print(f"Diffusers library not available: {e}")


class GenerativeService:
    """Service for inpainting masked regions with a generative model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.resource_manager = ResourceManager()
        
        # Fallback models to try if the primary model fails
        # Order them by compatibility - most reliable first
        self.fallback_models = [
            "runwayml/stable-diffusion-inpainting",       # Last resort inpainting model
            "runwayml/stable-diffusion-v1-5",            # SD1.5 base (works with img2img approach)
            "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL - more modern 
            "CompVis/stable-diffusion-v1-4"             # SD1.4 fallback
        ]

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        if StableDiffusionInpaintPipeline is None:
            raise ImportError("diffusers library is required for generative features")

        if self.model_path is None:
            raise ValueError("No model path specified for generative service. Please provide a valid model path or model name.")

        dev = self.device or MemoryManager.select_optimal_device(min_gpu_memory_gb=6.0)
        
        # Force CPU mode if CUDA has issues
        if dev == "cuda" and torch is not None:
            try:
                if not torch.cuda.is_available():
                    print("CUDA not available, forcing CPU mode")
                    dev = "cpu"
            except Exception:
                print("CUDA check failed, forcing CPU mode")
                dev = "cpu"
        
        dtype = torch.float16 if dev == "cuda" else torch.float32
        
        # Try the specified model first, then fallbacks
        models_to_try = [self.model_path] + [m for m in self.fallback_models if m != self.model_path]
        
        for i, model_path in enumerate(models_to_try):
            try:
                print(f"Attempting to load model: {model_path}")
                
                # Different loading strategies for different models
                load_kwargs = {
                    'torch_dtype': dtype,
                    'safety_checker': None,
                    'requires_safety_checker': False
                }
                
                # Determine pipeline type based on model
                pipeline_class = StableDiffusionInpaintPipeline  # Default
                
                if "stable-diffusion-xl" in model_path.lower():
                    # Try SDXL inpainting first, fallback to regular SDXL
                    try:
                        if StableDiffusionXLInpaintPipeline:
                            pipeline_class = StableDiffusionXLInpaintPipeline
                            print(f"Using SDXL Inpainting pipeline for {model_path}")
                    except:
                        pipeline_class = StableDiffusionPipeline
                        print(f"SDXL Inpainting not available, using regular SDXL for {model_path}")
                elif "inpainting" not in model_path.lower():
                    # For non-inpainting models, use regular SD pipeline
                    pipeline_class = StableDiffusionPipeline
                    print(f"Using regular SD pipeline for non-inpainting model: {model_path}")
                
                # Special handling for different model types
                if "stable-diffusion-2" in model_path:
                    # SD2 models sometimes need specific configurations
                    load_kwargs['use_safetensors'] = True
                elif "runwayml" in model_path:
                    # Original RunwayML model may need different settings
                    load_kwargs['revision'] = "fp16" if dtype == torch.float16 else None
                
                # Allow loading from legacy .ckpt or .safetensors files
                if (os.path.isfile(model_path) and
                        model_path.endswith((".ckpt", ".safetensors")) and
                        hasattr(pipeline_class, "from_single_file")):
                    self.pipeline = pipeline_class.from_single_file(
                        model_path,
                        **load_kwargs,
                    )
                else:
                    self.pipeline = pipeline_class.from_pretrained(
                        model_path,
                        **load_kwargs
                    )
                self.pipeline.to(dev)
                
                # Enable memory efficient attention if available
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    try:
                        self.pipeline.enable_attention_slicing()
                        print(f"Enabled attention slicing for model {model_path}")
                    except Exception as attention_error:
                        print(f"Could not enable attention slicing for {model_path}: {attention_error}")
                
                # Enable CPU offload only if accelerate is available and working
                if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    try:
                        # Check if accelerate is properly available
                        import accelerate
                        self.pipeline.enable_model_cpu_offload()
                        print(f"Enabled CPU offload for model {model_path}")
                    except ImportError:
                        print(f"Accelerate library not available, skipping CPU offload for {model_path}")
                    except Exception as offload_error:
                        print(f"Could not enable CPU offload for {model_path}: {offload_error}")
                        # Continue without CPU offload - this is not critical
                
                # Test the pipeline with a small dummy input to catch compatibility issues early
                try:
                    test_image = Image.new('RGB', (512, 512), color='white')
                    test_mask = Image.new('L', (512, 512), color='black')
                    
                    # Test based on pipeline type
                    if isinstance(self.pipeline, StableDiffusionPipeline):
                        # Regular SD pipeline - test with text2img
                        _ = self.pipeline(
                            prompt="test",
                            num_inference_steps=1,
                            output_type="pil"
                        )
                    else:
                        # Inpainting pipeline - test with img2img
                        _ = self.pipeline(
                            prompt="test",
                            image=test_image,
                            mask_image=test_mask,
                            num_inference_steps=1,
                            output_type="pil"
                        )
                    
                    # Test with a different size to catch size-specific issues
                    if not isinstance(self.pipeline, StableDiffusionPipeline):
                        test_image_768 = Image.new('RGB', (768, 512), color='white')
                        test_mask_768 = Image.new('L', (768, 512), color='black')
                        _ = self.pipeline(
                            prompt="test",
                            image=test_image_768,
                            mask_image=test_mask_768,
                            num_inference_steps=1,
                            output_type="pil"
                        )
                    
                    print(f"Model {model_path} passed compatibility test")
                except Exception as test_error:
                    print(f"Model {model_path} failed compatibility test: {test_error}")
                    # Clean up this pipeline and try next
                    self.pipeline = None
                    error_str = str(test_error).lower()
                    if ("sizes of tensors must match" in error_str or 
                        "size mismatch" in error_str or
                        "dimension" in error_str or
                        "expected size" in error_str or
                        "expected size 64 but got size 512" in str(test_error) or  # Specific SD2 issue
                        "pipeline" in error_str and "expected" in error_str):
                        print(f"Compatibility issue detected, trying next model...")
                        continue  # Try next model
                    # Re-raise if it's a different error that might be fixable
                    raise test_error
                
                self.resource_manager.register_model("inpaint_pipeline", self.pipeline, dev)
                print(f"Successfully loaded and tested model: {model_path}")
                return self.pipeline
                
            except Exception as e:
                print(f"Failed to load model {model_path}: {str(e)}")
                self.pipeline = None  # Ensure cleanup
                
                # Check for specific error types to provide better feedback
                error_str = str(e).lower()
                if "accelerator" in error_str and "not found" in error_str:
                    print(f"Model {model_path} requires accelerate library for CPU offload")
                elif "pipeline" in error_str and "expected" in error_str:
                    print(f"Model {model_path} is incompatible with StableDiffusionInpaintPipeline")
                elif "connection" in error_str or "timeout" in error_str:
                    print(f"Network issue loading {model_path}")
                elif "disk" in error_str or "space" in error_str:
                    print(f"Disk space issue loading {model_path}")
                elif "cuda" in error_str:
                    print(f"CUDA-related issue with {model_path}, trying CPU mode")
                elif "expected size 64 but got size 512" in str(e):
                    print(f"SD2 tensor dimension incompatibility detected with {model_path}")
                elif "sizes of tensors must match" in str(e):
                    print(f"Tensor size mismatch with {model_path} - trying different model")
                
                if i == len(models_to_try) - 1:  # Last model, re-raise the error
                    raise RuntimeError(
                        f"Failed to load any generative model. Last error with '{model_path}': {str(e)}. "
                        f"This may be due to:\n"
                        f"1. Model compatibility issues with current diffusers version ({diffusers_version})\n"
                        f"2. Insufficient GPU memory (requires ~6GB VRAM)\n"
                        f"3. Network connectivity issues during model download\n"
                        f"4. Corrupted model cache (try clearing ~/.cache/huggingface/)\n"
                        f"5. Insufficient disk space for model download (~10GB required)"
                    )
                else:
                    print(f"Trying next fallback model...")
                    continue
        
        raise RuntimeError("No models could be loaded")

    def reimagine(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> np.ndarray:
        """Generate new content for the masked region."""
        pipe = self._load_pipeline()
        
        # Validate inputs
        if image.shape[:2] != mask.shape[:2]:
            print(f"Warning: Image shape {image.shape[:2]} != mask shape {mask.shape[:2]}")
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST).astype(bool if mask.dtype == bool else mask.dtype)
        
        # Check mask validity
        if not np.any(mask):
            print("Error: Mask is empty, cannot perform inpainting")
            return None
            
        mask_area_ratio = np.sum(mask) / mask.size
        print(f"Mask covers {mask_area_ratio:.1%} of the image")
        
        if mask_area_ratio < 0.001:  # Less than 0.1%
            print("Warning: Mask area is very small, inpainting may not be effective")
        elif mask_area_ratio > 0.8:  # More than 80%
            print("Warning: Mask area is very large, may affect overall image quality")
        
        # Convert inputs to PIL Images with proper preprocessing
        if image.shape[2] == 4:  # RGBA
            img_pil = Image.fromarray(image[:, :, :3])  # Remove alpha for processing
        else:
            img_pil = Image.fromarray(image)
        
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
            
        # Process mask
        if mask.dtype == bool:
            mask_img = mask.astype(np.uint8) * 255
        else:
            mask_img = mask.copy()
            if mask_img.max() <= 1:
                mask_img = (mask_img * 255).astype(np.uint8)
                
        # Ensure mask is binary (0 or 255)
        mask_img = (mask_img > 127).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_img, mode='L')
        
        # Store original size for later restoration
        original_size = img_pil.size
        print(f"Input image size: {original_size}")
        print(f"Mask size: {mask_pil.size}")
        
        # Validate mask content
        mask_array_check = np.array(mask_pil)
        if np.sum(mask_array_check > 0) == 0:
            print("Error: Processed mask is empty")
            return None
        
        # For SD2 inpainting models, use standard sizes that work well
        # Common working sizes: 512x512, 768x768, 1024x1024
        def _get_optimal_size(size):
            """Get optimal size for SD2 inpainting models."""
            width, height = size
            
            # Standard sizes that work well with SD2
            standard_sizes = [
                (512, 512),
                (768, 768),
                (1024, 1024),
                (512, 768),
                (768, 512)
            ]
            
            # Find the closest standard size that's not too much larger
            aspect_ratio = width / height
            
            # For very different aspect ratios, use rectangular sizes
            if aspect_ratio > 1.5:  # Wide image
                return (768, 512)
            elif aspect_ratio < 0.67:  # Tall image
                return (512, 768)
            else:  # Square-ish image
                # Choose based on total pixel count
                total_pixels = width * height
                if total_pixels <= 512 * 512:
                    return (512, 512)
                elif total_pixels <= 768 * 768:
                    return (768, 768)
                else:
                    return (1024, 1024)
        
        # Get optimal processing size
        target_size = _get_optimal_size(original_size)
        print(f"Target processing size: {target_size}")
        
        # Always resize to target size for consistent processing, but preserve aspect ratio
        img_pil_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
        mask_pil_resized = mask_pil.resize(target_size, Image.Resampling.NEAREST)
        
        # Ensure mask is properly formatted after resize
        mask_array = np.array(mask_pil_resized)
        # Ensure mask remains binary after resize
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        mask_pil_resized = Image.fromarray(mask_array, mode='L')
        
        # Final validation
        final_mask_check = np.array(mask_pil_resized)
        if np.sum(final_mask_check > 0) == 0:
            print("Error: Mask became empty after resize")
            return None
            
        print(f"Final mask area after resize: {np.sum(final_mask_check > 0)} pixels")

        def _run():
            try:
                print(f"Running inference with size: {target_size}")
                print(f"Using prompt: '{prompt}'")
                
                # Check if this is an inpainting pipeline or regular pipeline
                if isinstance(pipe, StableDiffusionPipeline):
                    print("Using regular SD pipeline with img2img simulation")
                    # For regular SD models, we need to simulate inpainting
                    # We'll use img2img with a modified prompt
                    
                    # Create a blended image where the mask area is more neutral
                    img_array = np.array(img_pil_resized)
                    mask_array = np.array(mask_pil_resized)
                    
                    # Create a version where masked area is averaged/blurred
                    from PIL import ImageFilter
                    blurred_img = img_pil_resized.filter(ImageFilter.GaussianBlur(radius=15))
                    blurred_array = np.array(blurred_img)
                    
                    # Blend original and blurred in mask area
                    mask_norm = mask_array.astype(float) / 255.0
                    mask_3d = np.stack([mask_norm] * 3, axis=-1)
                    
                    prepared_array = img_array * (1 - mask_3d) + blurred_array * mask_3d
                    prepared_img = Image.fromarray(prepared_array.astype(np.uint8))
                    
                    # Enhanced prompt for better guidance
                    enhanced_prompt = f"high quality, detailed, {prompt}"
                    negative_prompt = "blurry, low quality, distorted, ugly, deformed"
                    
                    # Use img2img with high strength in mask area
                    result = pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=prepared_img,
                        strength=0.85,  # High strength for good changes
                        num_inference_steps=max(30, num_inference_steps),
                        guidance_scale=max(7.5, guidance_scale),
                    ).images[0]
                    
                else:
                    print("Using dedicated inpainting pipeline")
                    # Enhanced prompt for inpainting
                    enhanced_prompt = f"high quality, detailed, {prompt}"
                    negative_prompt = "blurry, low quality, distorted, ugly, deformed, artifacts"
                    
                    # Use regular inpainting pipeline with enhanced settings
                    generation_kwargs = {
                        'prompt': enhanced_prompt,
                        'negative_prompt': negative_prompt,
                        'image': img_pil_resized,
                        'mask_image': mask_pil_resized,
                        'num_inference_steps': max(30, num_inference_steps),
                        'guidance_scale': max(7.5, guidance_scale),
                        'strength': 0.95,  # High strength for inpainting
                    }
                    
                    result = pipe(**generation_kwargs).images[0]
                
                # Always resize back to original size
                result_resized = result.resize(original_size, Image.Resampling.LANCZOS)
                
                return np.array(result_resized)
                
            except Exception as e:
                error_str = str(e)
                print(f"Inference error: {error_str}")
                
                # Handle specific tensor size mismatch errors
                if ("Sizes of tensors must match" in error_str or 
                    "size mismatch" in error_str.lower() or
                    "Expected size" in error_str or
                    "expected size 64 but got size 512" in error_str):  # Specific SD2 issue
                    
                    # Try fallback with different approach
                    print("Attempting fallback with different size constraints...")
                    try:
                        # Fallback: use 512x512 and let the model handle it
                        fallback_size = (512, 512)
                        img_fallback = img_pil.resize(fallback_size, Image.Resampling.LANCZOS)
                        mask_fallback = mask_pil.resize(fallback_size, Image.Resampling.NEAREST)
                        
                        # Ensure mask is still binary after resize
                        mask_fallback_array = np.array(mask_fallback)
                        mask_fallback_array = (mask_fallback_array > 127).astype(np.uint8) * 255
                        mask_fallback = Image.fromarray(mask_fallback_array, mode='L')
                        
                        if isinstance(pipe, StableDiffusionPipeline):
                            # Img2img fallback
                            from PIL import ImageFilter
                            blurred_fallback = img_fallback.filter(ImageFilter.GaussianBlur(radius=15))
                            
                            result = pipe(
                                prompt=f"detailed, {prompt}",
                                image=blurred_fallback,
                                strength=0.8,
                                num_inference_steps=max(20, num_inference_steps // 2),
                                guidance_scale=guidance_scale
                            ).images[0]
                        else:
                            # Inpainting fallback
                            result = pipe(
                                prompt=f"detailed, {prompt}",
                                image=img_fallback,
                                mask_image=mask_fallback,
                                num_inference_steps=max(20, num_inference_steps // 2),
                                guidance_scale=guidance_scale
                            ).images[0]
                        
                        # Resize back to original
                        result_resized = result.resize(original_size, Image.Resampling.LANCZOS)
                        print("Fallback generation successful")
                        return np.array(result_resized)
                        
                    except Exception as fallback_error:
                        print(f"Fallback also failed: {fallback_error}")
                        
                        # If this is the SD2 tensor size issue, try forcing a model switch
                        if ("expected size 64 but got size 512" in str(fallback_error) or 
                            "expected size 64 but got size 512" in error_str):
                            print("Detected SD2 compatibility issue, forcing model switch...")
                            # Clear the current pipeline and force reload with different model
                            self.pipeline = None
                            self.resource_manager.move_to_cpu_if_needed("inpaint_pipeline")
                            
                            # Try to reload with the first fallback model
                            try:
                                print("Attempting to reload with runwayml/stable-diffusion-inpainting...")
                                self.model_path = "runwayml/stable-diffusion-inpainting"
                                new_pipe = self._load_pipeline()
                                
                                # Try the generation again with the new model
                                result = new_pipe(
                                    prompt=prompt,
                                    image=img_fallback,
                                    mask_image=mask_fallback,
                                    num_inference_steps=max(20, num_inference_steps // 2),
                                    guidance_scale=guidance_scale
                                ).images[0]
                                
                                result_resized = result.resize(original_size, Image.Resampling.LANCZOS)
                                print("Model switch successful")
                                return np.array(result_resized)
                                
                            except Exception as switch_error:
                                print(f"Model switch also failed: {switch_error}")
                        
                        raise RuntimeError(
                            f"Model failed with both standard and fallback approaches. "
                            f"Original error: {error_str}. "
                            f"This may indicate an incompatibility with the current image dimensions "
                            f"({original_size}) or model configuration."
                        )
                else:
                    raise e

        return ErrorHandler.safe_gpu_operation(_run, operation_name="reimagine")

    def cleanup(self) -> None:
        """Clean up resources used by the generative model."""
        self.resource_manager.move_to_cpu_if_needed("inpaint_pipeline")
        MemoryManager.clear_cuda_cache()
        self.pipeline = None