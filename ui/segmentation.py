import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class ImageSegmenter:
    def __init__(self, model_type="vit_h", checkpoint=None, device="cuda"):
        if checkpoint is None:
            checkpoint = os.path.join(os.path.dirname(__file__), '../models/sam_vit_h_4b8939.pth')
        
        abs_checkpoint_path = os.path.abspath(checkpoint)
        print(f"Loading checkpoint from: {abs_checkpoint_path}")
        
        if not os.path.exists(abs_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {abs_checkpoint_path}")
        
        self.model = sam_model_registry[model_type](checkpoint=abs_checkpoint_path)
        self.model.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.predictor = SamPredictor(self.model)

    def segment(self, image):
        # Convert RGBA image to RGB if needed
        if image.shape[-1] == 4:
            print("Converting RGBA image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Store original dimensions for scaling masks back
        original_height, original_width = image.shape[:2]
        print(f"Original image dimensions: {original_width}x{original_height}")
            
        # Resize image so that its long side is exactly 1024 pixels.
        height, width = image.shape[:2]
        if height >= width:
            new_height = 1024
            new_width = int(round((1024 / height) * width))
        else:
            new_width = 1024
            new_height = int(round((1024 / width) * height))
        
        # Debug: print new dimensions
        print(f"Resizing image from ({height}, {width}) to ({new_height}, {new_width})")
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Verify that the long side is exactly 1024 pixels
        assert max(new_height, new_width) == 1024, "Long side is not 1024 pixels."
        
        # Pass the resized image in HWC (3-dimensional) format to the generator.
        masks = self.mask_generator.generate(resized_image)
        
        print(f"Generated {len(masks)} masks before scaling")
        
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
                    scale_w = original_width / new_width
                    scale_h = original_height / new_height
                    bbox = mask['bbox']
                    mask['bbox'] = [
                        int(bbox[0] * scale_w),
                        int(bbox[1] * scale_h),
                        int(bbox[2] * scale_w),
                        int(bbox[3] * scale_h)
                    ]
        
        print(f"Returning {len(masks)} masks scaled to original image dimensions")
        return masks
        
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
        
        # Resize image as done in segment()
        height, width = image.shape[:2]
        if height >= width:
            new_height = 1024
            new_width = int(round((1024 / height) * width))
        else:
            new_width = 1024
            new_height = int(round((1024 / width) * height))
        
        # Resize the image
        scale_w = new_width / width
        scale_h = new_height / height
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Scale the box coordinates to match the resized image
        scaled_box = [
            int(box[0] * scale_w),
            int(box[1] * scale_h),
            int(box[2] * scale_w),
            int(box[3] * scale_h)
        ]
        
        # Set the image embedding in the predictor
        self.predictor.set_image(resized_image)
        
        # Get the mask prediction for the given box
        masks, scores, logits = self.predictor.predict(
            box=np.array(scaled_box),
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
            original_scale_w = original_width / new_width
            original_scale_h = original_height / new_height
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