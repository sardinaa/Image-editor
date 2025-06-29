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
        self.mask_editing_enabled = False
        self.current_mask_index = -1
    
    def delete_mask(self, mask_index: int) -> bool:
        """Delete a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            self.layer_masks.pop(mask_index)
            if mask_index < len(self.mask_names):
                self.mask_names.pop(mask_index)
            
            # Remove parameters for this mask
            if mask_index in self.mask_parameters:
                del self.mask_parameters[mask_index]
            
            # Adjust parameter indices for masks after the deleted one
            new_params = {}
            for idx, params in self.mask_parameters.items():
                if idx > mask_index:
                    new_params[idx - 1] = params
                elif idx < mask_index:
                    new_params[idx] = params
            self.mask_parameters = new_params
            
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
