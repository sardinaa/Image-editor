"""
Mask service for managing mask operations and state.
"""
from typing import List, Dict, Any, Optional


class MaskService:
    """Service for managing mask operations and state."""
    
    def __init__(self):
        self.layer_masks: List[Dict[str, Any]] = []
        self.mask_names: List[str] = []
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        self.mask_parameters: Dict[int, Dict[str, Any]] = {}
        self.global_parameters: Optional[Dict[str, Any]] = None
        self.mask_inpainting_files: Dict[int, List[str]] = {}  # Track inpainting files per mask
        self.mask_pre_inpainting_states: Dict[int, Dict[str, Any]] = {}  # Track pre-inpainting image states
    
    def add_masks(self, masks: List[Dict[str, Any]], names: Optional[List[str]] = None) -> None:
        """Add new masks to the collection."""
        start_index = len(self.layer_masks)
        self.layer_masks.extend(masks)
        
        # Add names for new masks
        if names and len(names) >= len(masks):
            self.mask_names.extend(names[:len(masks)])
        else:
            for idx in range(len(masks)):
                self.mask_names.append(f"Mask {start_index + idx + 1}")
    
    def replace_all_masks(self, masks: List[Dict[str, Any]], names: Optional[List[str]] = None) -> None:
        """Replace all existing masks with new ones."""
        self.layer_masks = masks
        
        if names and len(names) >= len(masks):
            self.mask_names = names[:len(masks)]
        else:
            self.mask_names = [f"Mask {idx + 1}" for idx in range(len(masks))]
    
    def clear_all_masks(self) -> None:
        """Clear all masks."""
        self.layer_masks.clear()
        self.mask_names.clear()
        self.mask_parameters.clear()
        self.mask_inpainting_files.clear()
        self.mask_pre_inpainting_states.clear()
        self.mask_editing_enabled = False
        self.current_mask_index = -1
    
    def delete_mask(self, mask_index: int) -> bool:
        """Delete a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            # Clean up inpainting files for this mask
            self._cleanup_inpainting_files(mask_index)
            
            self.layer_masks.pop(mask_index)
            if mask_index < len(self.mask_names):
                self.mask_names.pop(mask_index)
            
            # Remove parameters for this mask
            if mask_index in self.mask_parameters:
                del self.mask_parameters[mask_index]
            
            # Remove inpainting files tracking for this mask
            if mask_index in self.mask_inpainting_files:
                del self.mask_inpainting_files[mask_index]
            
            # Remove pre-inpainting state for this mask
            if mask_index in self.mask_pre_inpainting_states:
                del self.mask_pre_inpainting_states[mask_index]
            
            # Adjust parameter indices for masks after the deleted one
            new_params = {}
            for idx, params in self.mask_parameters.items():
                if idx > mask_index:
                    new_params[idx - 1] = params
                elif idx < mask_index:
                    new_params[idx] = params
            self.mask_parameters = new_params
            
            # Adjust inpainting files tracking for masks after the deleted one
            new_inpainting_files = {}
            for idx, files in self.mask_inpainting_files.items():
                if idx > mask_index:
                    new_inpainting_files[idx - 1] = files
                elif idx < mask_index:
                    new_inpainting_files[idx] = files
            self.mask_inpainting_files = new_inpainting_files
            
            # Adjust pre-inpainting states for masks after the deleted one
            new_pre_inpainting_states = {}
            for idx, state in self.mask_pre_inpainting_states.items():
                if idx > mask_index:
                    new_pre_inpainting_states[idx - 1] = state
                elif idx < mask_index:
                    new_pre_inpainting_states[idx] = state
            self.mask_pre_inpainting_states = new_pre_inpainting_states
            
            # Adjust current mask index
            if self.current_mask_index == mask_index:
                self.current_mask_index = -1
                self.mask_editing_enabled = False
            elif self.current_mask_index > mask_index:
                self.current_mask_index -= 1
            
            return True
        return False
    
    def rename_mask(self, mask_index: int, new_name: str) -> bool:
        """Rename a specific mask."""
        if 0 <= mask_index < len(self.mask_names):
            self.mask_names[mask_index] = new_name
            return True
        return False
    
    def save_mask_parameters(self, mask_index: int, parameters: Dict[str, Any]) -> None:
        """Save parameters for a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            self.mask_parameters[mask_index] = parameters.copy()
    
    def get_masks(self) -> List[Dict[str, Any]]:
        """Get all masks."""
        return self.layer_masks
    
    def get_mask_names(self) -> List[str]:
        """Get all mask names."""
        return self.mask_names
    
    def add_inpainting_files(self, mask_index: int, file_paths: List[str]) -> None:
        """Associate inpainting files with a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            if mask_index not in self.mask_inpainting_files:
                self.mask_inpainting_files[mask_index] = []
            self.mask_inpainting_files[mask_index].extend(file_paths)
    
    def get_inpainting_files(self, mask_index: int) -> List[str]:
        """Get inpainting files associated with a specific mask."""
        return self.mask_inpainting_files.get(mask_index, [])
    
    def _cleanup_inpainting_files(self, mask_index: int) -> None:
        """Clean up inpainting files associated with a specific mask."""
        import os
        
        files_to_delete = self.mask_inpainting_files.get(mask_index, [])
        deleted_count = 0
        
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted inpainting file: {file_path}")
            except Exception as e:
                print(f"Error deleting inpainting file {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} inpainting files for mask {mask_index}")
    
    def cleanup_orphaned_inpainting_files(self) -> int:
        """Clean up inpainting files that are no longer associated with any mask."""
        import os
        import glob
        
        if not os.path.exists("inpainting_results"):
            return 0
        
        # Get all current inpainting files
        all_tracked_files = set()
        for files in self.mask_inpainting_files.values():
            all_tracked_files.update(files)
        
        # Find all inpainting files in the directory
        inpainting_patterns = [
            "inpainting_results/inpainted_full_*.png",
            "inpainting_results/inpainted_mask_only_*.png"
        ]
        
        all_existing_files = set()
        for pattern in inpainting_patterns:
            all_existing_files.update(glob.glob(pattern))
        
        # Find orphaned files (exist but not tracked)
        orphaned_files = all_existing_files - all_tracked_files
        
        deleted_count = 0
        for file_path in orphaned_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Cleaned up orphaned inpainting file: {file_path}")
            except Exception as e:
                print(f"Error deleting orphaned file {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} orphaned inpainting files")
        
        return deleted_count
    
    def save_pre_inpainting_state(self, mask_index: int, image_processor_state: Dict[str, Any]) -> None:
        """Save the image processor state before inpainting for potential restoration."""
        if 0 <= mask_index < len(self.layer_masks):
            self.mask_pre_inpainting_states[mask_index] = image_processor_state.copy()
    
    def get_pre_inpainting_state(self, mask_index: int) -> Optional[Dict[str, Any]]:
        """Get the pre-inpainting state for a specific mask."""
        return self.mask_pre_inpainting_states.get(mask_index)
    
    def restore_pre_inpainting_state(self, mask_index: int, image_processor) -> bool:
        """Restore the image processor to its pre-inpainting state."""
        if mask_index not in self.mask_pre_inpainting_states:
            return False
        
        try:
            state = self.mask_pre_inpainting_states[mask_index]
            
            # Restore the original image
            if 'original_image' in state:
                image_processor.original = state['original_image'].copy()
            
            # Restore committed edits
            if 'committed_global_edits' in state:
                image_processor.committed_global_edits = state['committed_global_edits'].copy()
            
            if 'committed_mask_edits' in state:
                image_processor.committed_mask_edits = state['committed_mask_edits'].copy()
            
            # Reset current image
            image_processor.current = image_processor.original.copy()
            image_processor.clear_optimization_cache()
            
            print(f"Restored image processor to pre-inpainting state for mask {mask_index}")
            return True
            
        except Exception as e:
            print(f"Error restoring pre-inpainting state for mask {mask_index}: {e}")
            return False
