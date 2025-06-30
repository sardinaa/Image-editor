import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from utils.memory_utils import MemoryManager, ResourceManager

class ImageSegmenter:
    def __init__(self, model_type="vit_h", checkpoint=None, device="auto"):
        if checkpoint is None:
            checkpoint = os.path.join(os.path.dirname(__file__), '../assets/models/sam_vit_h_4b8939.pth')
        
        abs_checkpoint_path = os.path.abspath(checkpoint)
        
        if not os.path.exists(abs_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {abs_checkpoint_path}")
        
        # Determine the best device to use
        self.device = self._select_device(device)
        
        # Set memory optimization environment variables
        if self.device == "cuda":
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Initialize resource manager
        self.resource_manager = ResourceManager()
        
        self.model = sam_model_registry[model_type](checkpoint=abs_checkpoint_path)
        self.model.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.predictor = SamPredictor(self.model)
        
        # Register model with resource manager
        self.resource_manager.register_model("sam_segmenter", self.model, self.device)
        
        # Memory management settings
        self.max_image_size = 1024  # Default max size
        self._adjust_max_size_for_memory()
    
    def _select_device(self, device):
        """Select the best available device for computation"""
        if device == "auto":
            selected_device = MemoryManager.select_optimal_device(min_gpu_memory_gb=4.0)
            return selected_device
        return device
    
    def _adjust_max_size_for_memory(self):
        """Adjust maximum image size based on available memory"""
        self.max_image_size = MemoryManager.get_recommended_image_size(self.device)
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache to free up memory"""
        MemoryManager.clear_cuda_cache()
    
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

        return cv2.resize(image, (new_width, new_height))

    def segment(self, image):
        # Convert RGBA image to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Store original dimensions for scaling masks back
        original_height, original_width = image.shape[:2]
        
        # Clear CUDA cache before processing
        self._clear_cuda_cache()
        
        # Ensure model is on the correct device
        self._ensure_model_on_device()
        
        try:
            # Resize image with memory constraints
            resized_image = self._resize_image_with_memory_constraint(image)
            
            # Pass the resized image in HWC (3-dimensional) format to the generator.
            masks = self.mask_generator.generate(resized_image)
            
            # Calculate scaling factors
            resize_h, resize_w = resized_image.shape[:2]
            scale_w = original_width / resize_w
            scale_h = original_height / resize_h
            
            # Scale all masks back to original image dimensions
            for i, mask in enumerate(masks):
                if 'segmentation' in mask:
                    scaled_mask = cv2.resize(mask['segmentation'].astype(np.uint8), 
                                           (original_width, original_height), 
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask['segmentation'] = scaled_mask
                    
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
        
            return masks
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error: {e}")
            return self._fallback_segment(image)
        except Exception as e:
            print(f"Segmentation error: {e}")
            return self._fallback_segment(image)
    
    def _fallback_segment(self, image):
        """Fallback segmentation with reduced memory usage"""
        
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

        resized_image = cv2.resize(image, (new_width, new_height))
        
        try:
            # If still on CUDA and failing, try moving to CPU
            if self.device == "cuda":
                self.resource_manager.move_to_cpu_if_needed("sam_segmenter")
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
            
            # Set the image embedding in the predictor
            self.predictor.set_image(resized_image)
            
            # Convert to numpy array with explicit dtype
            box_array = np.array(scaled_box, dtype=np.int32)
            
            # Get the mask prediction for the given box
            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=True
            )
            
            # Format results to match the automatic segmentation format
            result_masks = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Scale mask back to original image dimensions
                scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                       (original_width, original_height), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                
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
            return []
        
        try:
            # If still on CUDA and failing, try moving to CPU
            if self.device == "cuda":
                self.resource_manager.move_to_cpu_if_needed("sam_segmenter")
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
            
            result_masks = sorted(result_masks, key=lambda x: x['predicted_iou'], reverse=True)
            return result_masks
            
        except Exception as e:
            print(f"Fallback box segmentation failed: {e}")
            return []
    
    def cleanup_memory(self):
        """Explicitly clean up memory and reset model if needed"""
        # Use resource manager for intelligent memory management
        self.resource_manager.move_to_cpu_if_needed("sam_segmenter")
        MemoryManager.clear_cuda_cache()
    
    def _ensure_model_on_device(self):
        """Ensure model is on the correct device before use"""
        # Try to move model back to GPU if it was temporarily moved to CPU
        if self.device == "cuda":
            moved_back = self.resource_manager.move_back_to_gpu_if_possible("sam_segmenter")
            if moved_back:
                # Recreate predictor and generator with updated model
                self.mask_generator = SamAutomaticMaskGenerator(self.model)
                self.predictor = SamPredictor(self.model)