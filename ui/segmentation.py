import os
import cv2
import numpy as np
import torch
import gc
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class ImageSegmenter:
    def __init__(self, model_type="vit_h", checkpoint=None, device="auto"):
        if checkpoint is None:
            checkpoint = os.path.join(os.path.dirname(__file__), '../models/sam_vit_h_4b8939.pth')
        
        abs_checkpoint_path = os.path.abspath(checkpoint)
        print(f"Loading checkpoint from: {abs_checkpoint_path}")
        
        if not os.path.exists(abs_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {abs_checkpoint_path}")
        
        # Determine the best device to use
        self.device = self._select_device(device)
        print(f"Using device: {self.device}")
        
        # Set memory optimization environment variables
        if self.device == "cuda":
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.model = sam_model_registry[model_type](checkpoint=abs_checkpoint_path)
        self.model.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.predictor = SamPredictor(self.model)
        
        # Memory management settings
        self.max_image_size = 1024  # Default max size
        self._adjust_max_size_for_memory()
    
    def _select_device(self, device):
        """Select the best available device for computation"""
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    # Test CUDA availability and memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)
                    print(f"GPU memory available: {gpu_memory_gb:.2f} GB")
                    
                    # If GPU has less than 4GB, prefer CPU
                    if gpu_memory_gb < 4.0:
                        print("Low GPU memory detected, using CPU for better stability")
                        return "cpu"
                    return "cuda"
                except Exception as e:
                    print(f"CUDA test failed: {e}, falling back to CPU")
                    return "cpu"
            else:
                print("CUDA not available, using CPU")
                return "cpu"
        return device
    
    def _adjust_max_size_for_memory(self):
        """Adjust maximum image size based on available memory"""
        if self.device == "cuda":
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                # Adjust max size based on GPU memory
                if gpu_memory_gb < 4.0:
                    self.max_image_size = 512
                elif gpu_memory_gb < 6.0:
                    self.max_image_size = 768
                else:
                    self.max_image_size = 1024
                    
                print(f"Adjusted max image size to {self.max_image_size} based on GPU memory")
            except Exception as e:
                print(f"Failed to adjust image size: {e}")
                self.max_image_size = 512  # Conservative default
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache to free up memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _get_available_memory(self):
        """Get available GPU memory in MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**2)
        return float('inf')  # Assume unlimited for CPU
    
    def _resize_image_with_memory_constraint(self, image):
        """Resize image considering memory constraints"""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        # If image is already small enough, return as is
        if max_dim <= self.max_image_size:
            return image
        
        # Calculate new dimensions
        if height >= width:
            new_height = self.max_image_size
            new_width = int(round((self.max_image_size / height) * width))
        else:
            new_width = self.max_image_size
            new_height = int(round((self.max_image_size / width) * height))
        
        print(f"Resizing image from ({height}, {width}) to ({new_height}, {new_width}) for memory efficiency")
        return cv2.resize(image, (new_width, new_height))

    def segment(self, image):
        # Convert RGBA image to RGB if needed
        if image.shape[-1] == 4:
            print("Converting RGBA image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Store original dimensions for scaling masks back
        original_height, original_width = image.shape[:2]
        print(f"Original image dimensions: {original_width}x{original_height}")
        
        # Check memory before processing
        memory_info = self.get_memory_info()
        print(f"Memory before segmentation: {memory_info}")
        
        # Clear CUDA cache before processing
        self._clear_cuda_cache()
        
        # Ensure model is on the correct device
        self._ensure_model_on_device()
        
        try:
            # Resize image with memory constraints
            resized_image = self._resize_image_with_memory_constraint(image)
            
            # Pass the resized image in HWC (3-dimensional) format to the generator.
            masks = self.mask_generator.generate(resized_image)
            
            print(f"Generated {len(masks)} masks before scaling")
            
            # Calculate scaling factors
            resize_h, resize_w = resized_image.shape[:2]
            scale_w = original_width / resize_w
            scale_h = original_height / resize_h
            
            # Scale all masks back to original image dimensions
            for i, mask in enumerate(masks):
                if 'segmentation' in mask:
                    original_mask_shape = mask['segmentation'].shape
                    scaled_mask = cv2.resize(mask['segmentation'].astype(np.uint8), 
                                           (original_width, original_height), 
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask['segmentation'] = scaled_mask
                    print(f"Mask {i}: scaled from {original_mask_shape} to {scaled_mask.shape}")
                    
                    # Update area to reflect the scaled mask
                    mask['area'] = int(scaled_mask.sum())
                    
                    # Scale bbox coordinates back to original image
                    if 'bbox' in mask:
                        bbox = mask['bbox']
                        mask['bbox'] = [
                            int(bbox[0] * scale_w),
                            int(bbox[1] * scale_h),
                            int(bbox[2] * scale_w),
                            int(bbox[3] * scale_h)
                        ]
            
            print(f"Returning {len(masks)} masks scaled to original image dimensions")
            return masks
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error: {e}")
            return self._fallback_segment(image)
        except Exception as e:
            print(f"Segmentation error: {e}")
            return self._fallback_segment(image)
    
    def _fallback_segment(self, image):
        """Fallback segmentation with reduced memory usage"""
        print("Attempting fallback segmentation with reduced memory usage...")
        
        # Clear memory
        self._clear_cuda_cache()
        
        # Try with even smaller image size
        original_height, original_width = image.shape[:2]
        fallback_size = min(512, self.max_image_size // 2)
        
        height, width = image.shape[:2]
        if height >= width:
            new_height = fallback_size
            new_width = int(round((fallback_size / height) * width))
        else:
            new_width = fallback_size
            new_height = int(round((fallback_size / width) * height))
        
        print(f"Fallback: resizing image from ({height}, {width}) to ({new_height}, {new_width})")
        resized_image = cv2.resize(image, (new_width, new_height))
        
        try:
            # If still on CUDA and failing, try moving to CPU
            if self.device == "cuda":
                print("Moving model to CPU for fallback segmentation...")
                self.model.to("cpu")
                self.mask_generator = SamAutomaticMaskGenerator(self.model)
                
            masks = self.mask_generator.generate(resized_image)
            
            # Scale back to original dimensions
            scale_w = original_width / new_width
            scale_h = original_height / new_height
            
            for mask in masks:
                if 'segmentation' in mask:
                    scaled_mask = cv2.resize(mask['segmentation'].astype(np.uint8), 
                                           (original_width, original_height), 
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask['segmentation'] = scaled_mask
                    mask['area'] = int(scaled_mask.sum())
                    
                    if 'bbox' in mask:
                        bbox = mask['bbox']
                        mask['bbox'] = [
                            int(bbox[0] * scale_w),
                            int(bbox[1] * scale_h),
                            int(bbox[2] * scale_w),
                            int(bbox[3] * scale_h)
                        ]
            
            print(f"Fallback segmentation successful: {len(masks)} masks generated")
            return masks
            
        except Exception as e:
            print(f"Fallback segmentation failed: {e}")
            return []
        
    def segment_with_box(self, image, box):
        """
        Segment an image based on a bounding box input.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            box: Bounding box in format [x1, y1, x2, y2]
            
        Returns:
            List of masks in same format as automatic segmentation
        """
        # Convert RGBA image to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Store original dimensions for scaling masks back
        original_height, original_width = image.shape[:2]
        print(f"Segment with box - Original image dimensions: {original_width}x{original_height}")
        print(f"Input box: {box}")
        
        # Check memory before processing
        memory_info = self.get_memory_info()
        print(f"Memory before box segmentation: {memory_info}")
        
        # Clear CUDA cache before processing
        self._clear_cuda_cache()
        
        # Ensure model is on the correct device
        self._ensure_model_on_device()
        
        try:
            # Resize image with memory constraints
            resized_image = self._resize_image_with_memory_constraint(image)
            resize_h, resize_w = resized_image.shape[:2]
            
            # Scale the box coordinates to match the resized image
            scale_w = resize_w / original_width
            scale_h = resize_h / original_height
            scaled_box = [
                int(box[0] * scale_w),
                int(box[1] * scale_h),
                int(box[2] * scale_w),
                int(box[3] * scale_h)
            ]
            
            # Validate scaled box coordinates
            print(f"Scaled box coordinates: {scaled_box}")
            for i, coord in enumerate(scaled_box):
                if not isinstance(coord, (int, float)) or not np.isfinite(coord):
                    raise ValueError(f"Invalid scaled box coordinate {i}: {coord}")
            
            # Ensure box coordinates are within image bounds
            scaled_box[0] = max(0, min(scaled_box[0], resize_w - 1))  # x1
            scaled_box[1] = max(0, min(scaled_box[1], resize_h - 1))  # y1
            scaled_box[2] = max(0, min(scaled_box[2], resize_w))      # x2
            scaled_box[3] = max(0, min(scaled_box[3], resize_h))      # y2
            
            # Ensure valid box dimensions
            if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
                raise ValueError(f"Invalid box dimensions: {scaled_box}")
            
            print(f"Final validated scaled box: {scaled_box}")
            
            # Set the image embedding in the predictor
            self.predictor.set_image(resized_image)
            
            # Convert to numpy array with explicit dtype
            box_array = np.array(scaled_box, dtype=np.int32)
            print(f"Box array for prediction: {box_array}, dtype: {box_array.dtype}")
            
            # Get the mask prediction for the given box
            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=True
            )
            
            # Format results to match the automatic segmentation format
            result_masks = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                original_mask_shape = mask.shape
                # Scale mask back to original image dimensions
                scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                       (original_width, original_height), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                
                print(f"Box mask {i}: scaled from {original_mask_shape} to {scaled_mask.shape}, score: {score}")
                
                # Scale bbox back to original image dimensions
                original_scale_w = original_width / resize_w
                original_scale_h = original_height / resize_h
                original_bbox = [
                    int(scaled_box[0] * original_scale_w),
                    int(scaled_box[1] * original_scale_h),
                    int(scaled_box[2] * original_scale_w),
                    int(scaled_box[3] * original_scale_h)
                ]
                
                result_masks.append({
                    'segmentation': scaled_mask,
                    'area': int(scaled_mask.sum()),
                    'bbox': original_bbox,
                    'predicted_iou': float(score),
                    'point_coords': [],
                    'stability_score': float(score),
                    'crop_box': [0, 0, original_width, original_height]
                })
            
            print(f"Returning {len(result_masks)} box-generated masks scaled to original image dimensions")
            # Sort by score
            result_masks = sorted(result_masks, key=lambda x: x['predicted_iou'], reverse=True)
            return result_masks
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory during box segmentation: {e}")
            return self._fallback_segment_with_box(image, box)
        except Exception as e:
            print(f"Box segmentation error: {e}")
            return self._fallback_segment_with_box(image, box)
    
    def _fallback_segment_with_box(self, image, box):
        """Fallback box segmentation with reduced memory usage"""
        print("Attempting fallback box segmentation with reduced memory usage...")
        
        # Clear memory
        self._clear_cuda_cache()
        
        # Try with even smaller image size
        original_height, original_width = image.shape[:2]
        fallback_size = min(512, self.max_image_size // 2)
        
        height, width = image.shape[:2]
        if height >= width:
            new_height = fallback_size
            new_width = int(round((fallback_size / height) * width))
        else:
            new_width = fallback_size
            new_height = int(round((fallback_size / width) * height))
        
        print(f"Fallback box: resizing image from ({height}, {width}) to ({new_height}, {new_width})")
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Scale box coordinates
        scale_w = new_width / width
        scale_h = new_height / height
        scaled_box = [
            int(box[0] * scale_w),
            int(box[1] * scale_h),
            int(box[2] * scale_w),
            int(box[3] * scale_h)
        ]
        
        # Validate and clamp coordinates
        scaled_box[0] = max(0, min(scaled_box[0], new_width - 1))   # x1
        scaled_box[1] = max(0, min(scaled_box[1], new_height - 1))  # y1
        scaled_box[2] = max(0, min(scaled_box[2], new_width))       # x2
        scaled_box[3] = max(0, min(scaled_box[3], new_height))      # y2
        
        # Ensure valid box dimensions
        if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
            print(f"Invalid fallback box dimensions: {scaled_box}")
            return []
        
        print(f"Fallback scaled box: {scaled_box}")
        
        try:
            # If still on CUDA and failing, try moving to CPU
            if self.device == "cuda":
                print("Moving model to CPU for fallback box segmentation...")
                self.model.to("cpu")
                self.predictor = SamPredictor(self.model)
                
            self.predictor.set_image(resized_image)
            
            # Convert to numpy array with explicit dtype
            box_array = np.array(scaled_box, dtype=np.int32)
            
            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=True
            )
            
            # Scale back to original dimensions
            result_masks = []
            orig_scale_w = original_width / new_width
            orig_scale_h = original_height / new_height
            
            for i, (mask, score) in enumerate(zip(masks, scores)):
                scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                       (original_width, original_height), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                
                original_bbox = [
                    int(scaled_box[0] * orig_scale_w),
                    int(scaled_box[1] * orig_scale_h),
                    int(scaled_box[2] * orig_scale_w),
                    int(scaled_box[3] * orig_scale_h)
                ]
                
                result_masks.append({
                    'segmentation': scaled_mask,
                    'area': int(scaled_mask.sum()),
                    'bbox': original_bbox,
                    'predicted_iou': float(score),
                    'point_coords': [],
                    'stability_score': float(score),
                    'crop_box': [0, 0, original_width, original_height]
                })
            
            print(f"Fallback box segmentation successful: {len(result_masks)} masks generated")
            result_masks = sorted(result_masks, key=lambda x: x['predicted_iou'], reverse=True)
            return result_masks
            
        except Exception as e:
            print(f"Fallback box segmentation failed: {e}")
            return []
    
    def get_memory_info(self):
        """Get current memory usage information"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
            free = total - allocated
            return {
                'device': 'cuda',
                'allocated_mb': allocated,
                'cached_mb': cached,
                'free_mb': free,
                'total_mb': total
            }
        else:
            return {
                'device': 'cpu',
                'allocated_mb': 0,
                'cached_mb': 0,
                'free_mb': float('inf'),
                'total_mb': float('inf')
            }
    
    def cleanup_memory(self):
        """Explicitly clean up memory and reset model if needed"""
        if self.device == "cuda" and torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # If memory is still critically low, move model to CPU temporarily
            memory_info = self.get_memory_info()
            if memory_info['free_mb'] < 500:  # Less than 500MB free
                print(f"Critical memory situation: {memory_info['free_mb']:.1f}MB free")
                print("Moving model to CPU to free GPU memory...")
                self.model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
                # Update device reference but don't recreate predictor/generator yet
                self._temp_cpu_mode = True
    
    def _ensure_model_on_device(self):
        """Ensure model is on the correct device before use"""
        if hasattr(self, '_temp_cpu_mode') and self._temp_cpu_mode:
            if self.device == "cuda":
                # Try to move back to CUDA if we have enough memory
                memory_info = self.get_memory_info()
                if memory_info['free_mb'] > 1000:  # At least 1GB free
                    print("Moving model back to CUDA...")
                    self.model.to(self.device)
                    self.mask_generator = SamAutomaticMaskGenerator(self.model)
                    self.predictor = SamPredictor(self.model)
                    self._temp_cpu_mode = False
                else:
                    print("Insufficient GPU memory, keeping model on CPU")
                    self.mask_generator = SamAutomaticMaskGenerator(self.model)
                    self.predictor = SamPredictor(self.model)