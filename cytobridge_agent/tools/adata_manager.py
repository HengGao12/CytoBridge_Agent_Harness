"""
AnnData Manager - Singleton manager for shared AnnData objects.

This module provides a centralized way to manage AnnData objects across
all agents and toolkits, eliminating redundant disk loads and reducing
memory usage.

Usage:
    from .adata_manager import AnnDataManager
    
    manager = AnnDataManager()
    adata = manager.load("path/to/data.h5ad")  # Loads or returns cached
    
    # After modifications
    manager.save("path/to/output.h5ad")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import scanpy as sc

logger = logging.getLogger(__name__)


class AnnDataManager:
    """
    Singleton manager for shared AnnData object.
    
    All agents and toolkits should use this instead of loading adata directly.
    This ensures only one copy of adata exists in memory at a time.
    
    Key behaviors:
    - load(path): Returns cached adata if same path, otherwise loads from disk
    - get(): Returns current adata (or None if not loaded)
    - update(adata, path): Updates the managed adata (e.g., after training)
    - save(path): Saves current adata to disk and updates path tracking
    - clear(): Clears the cached adata (for testing/cleanup)
    """
    
    _instance: Optional["AnnDataManager"] = None
    
    def __new__(cls) -> "AnnDataManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adata = None
            cls._instance._current_path = None
            cls._instance._is_dirty = False
            logger.debug("AnnDataManager singleton created")
        return cls._instance
    
    def load(self, path: str, force_reload: bool = False) -> AnnData:
        """
        Load adata from disk, or return cached if same path.
        
        Args:
            path: Path to the .h5ad file
            force_reload: If True, reload from disk even if path matches
            
        Returns:
            The loaded AnnData object
        """
        path_str = str(Path(path).resolve())
        
        # Return cached if same path and not forcing reload
        if (self._adata is not None 
            and self._current_path == path_str 
            and not force_reload
            and not self._is_dirty):
            logger.debug(f"Returning cached adata for {path_str}")
            return self._adata
        
        # Load from disk
        logger.info(f"Loading adata from {path_str}")
        self._adata = sc.read_h5ad(path_str)
        self._current_path = path_str
        self._is_dirty = False
        return self._adata
    
    def get_path(self) -> Optional[str]:
        """
        Get the path of the currently loaded adata.
        
        Returns:
            The path string, or None if not loaded
        """
        return self._current_path

    def bind_path(self, path: str) -> str:
        """
        Bind the active AnnData path without loading the object in this process.

        Large or corrupt .h5ad files should be read by isolated worker
        processes, not by the planner process. Tools that need data can use
        this path and load inside their own subprocess boundary.
        """
        path_str = str(Path(path).resolve())
        self._adata = None
        self._current_path = path_str
        self._is_dirty = False
        logger.info("Bound adata path without loading object: %s", path_str)
        return path_str
    
    def update(self, adata: AnnData, path: Optional[str] = None) -> None:
        """
        Update the managed adata (e.g., after training returns modified adata).
        
        Args:
            adata: The new AnnData object
            path: Optional new path (if adata was saved to a new location)
        """
        logger.info(f"Updating managed adata (path: {path or 'unchanged'})")
        self._adata = adata
        if path is not None:
            self._current_path = str(Path(path).resolve())
        self._is_dirty = False
    
    def save(self, path: str) -> str:
        """
        Save current adata to disk and update path tracking.
        
        Args:
            path: Path to save the .h5ad file
            
        Returns:
            The absolute path where file was saved
            
        Raises:
            ValueError: If no adata is loaded
        """
        if self._adata is None:
            raise ValueError("No adata loaded to save")
        
        path_str = str(Path(path).resolve())
        logger.info(f"Saving adata to {path_str}")
        self._adata.write_h5ad(path_str)
        self._current_path = path_str
        self._is_dirty = False
        return path_str
    
    def mark_dirty(self) -> None:
        """
        Mark adata as modified externally (needs reload on next access).
        
        Use this when adata was saved to disk by another process and 
        you want the next load() to re-read from disk.
        """
        self._is_dirty = True
    
    def clear(self) -> None:
        """
        Clear the cached adata.
        
        Useful for testing or forcing a complete reload.
        """
        logger.info("Clearing cached adata")
        self._adata = None
        self._current_path = None
        self._is_dirty = False
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.
        
        Only use this for testing purposes.
        """
        if cls._instance is not None:
            cls._instance._adata = None
            cls._instance._current_path = None
            cls._instance._is_dirty = False
        cls._instance = None
