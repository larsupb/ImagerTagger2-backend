"""
ImageDataSet - Manages an image/video dataset with captions, masks, and thumbnails.

This module provides a fully encapsulated dataset class for managing media collections
with associated metadata. It is designed to be instantiated per-session and stored
in gr.State for multi-user Gradio deployments.
"""

import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple, Callable, Any

from PIL import Image

from lib.media_item import MediaItem, is_image, is_video, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from lib.media_cache import generate_thumbnail

logger = logging.getLogger(__name__)


def _not_filtered(file: str, ignore_list: List[str]) -> bool:
    """Check if a file should not be filtered out based on ignore patterns."""
    return not any(ignore_pattern in file for ignore_pattern in ignore_list)


class ImageDataSet:
    """
    Manages an image/video dataset with captions, masks, and thumbnails.

    This class provides a fully encapsulated interface for managing media collections.
    All access to internal data goes through properties and methods.

    Example usage:
        dataset = ImageDataSet()
        dataset.load("/path/to/images", masks_dir="/path/to/masks")

        for item in dataset:
            print(item.filename)

        caption = dataset.read_caption(0)
        dataset.save_caption(0, "new caption text")
    """

    def __init__(self):
        """Initialize an empty dataset."""
        self._items: List[MediaItem] = []
        self._caption_cache: Dict[str, str] = {}
        self._bookmarks: Dict[str, bool] = {}
        self._base_dir: str = ""
        self._masks_dir: Optional[str] = None
        self._initialized: bool = False

    # =========================================================================
    # Properties (read-only access)
    # =========================================================================

    @property
    def is_initialized(self) -> bool:
        """Check if the dataset has been successfully loaded."""
        return self._initialized

    @property
    def is_empty(self) -> bool:
        """Check if the dataset contains no items."""
        return len(self._items) == 0

    @property
    def count(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._items)

    @property
    def base_dir(self) -> str:
        """Return the base directory of the dataset."""
        return self._base_dir

    @property
    def masks_dir(self) -> Optional[str]:
        """Return the masks directory, if configured."""
        return self._masks_dir

    @property
    def has_mask_support(self) -> bool:
        """Check if mask support is enabled for this dataset."""
        return self._masks_dir is not None

    # =========================================================================
    # Container protocol methods
    # =========================================================================

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._items)

    def __getitem__(self, index: int) -> MediaItem:
        """
        Get a MediaItem by index.

        Args:
            index: The index of the item to retrieve

        Returns:
            The MediaItem at the given index

        Raises:
            IndexError: If the index is out of bounds
        """
        return self._items[index]

    def __iter__(self) -> Iterator[MediaItem]:
        """Iterate over all MediaItems in the dataset."""
        return iter(self._items)

    def __bool__(self) -> bool:
        """Return True if the dataset is initialized and non-empty."""
        return self._initialized and len(self._items) > 0

    # =========================================================================
    # Item access methods
    # =========================================================================

    def get_item(self, index: int) -> Optional[MediaItem]:
        """
        Safely get a MediaItem by index.

        Args:
            index: The index of the item to retrieve

        Returns:
            The MediaItem at the given index, or None if index is invalid
        """
        if not self._initialized or index < 0 or index >= len(self._items):
            return None
        return self._items[index]

    def get_media_path(self, index: int) -> Optional[str]:
        """
        Get the media file path for an item by index.

        Args:
            index: The index of the item

        Returns:
            The media file path, or None if index is invalid
        """
        item = self.get_item(index)
        return item.media_path if item else None

    def get_caption_path(self, index: int) -> Optional[str]:
        """
        Get the caption file path for an item by index.

        Args:
            index: The index of the item

        Returns:
            The caption file path, or None if index is invalid
        """
        item = self.get_item(index)
        return item.caption_path if item else None

    def get_mask_path(self, index: int) -> Optional[str]:
        """
        Get the mask file path for an item by index.

        Args:
            index: The index of the item

        Returns:
            The mask file path, or None if index is invalid or no mask support
        """
        item = self.get_item(index)
        return item.mask_path if item else None

    def get_thumbnail(self, index: int) -> Optional[str]:
        """
        Get the thumbnail path for an item by index.

        Args:
            index: The index of the item

        Returns:
            The thumbnail path, or None if index is invalid or no thumbnail
        """
        item = self.get_item(index)
        return item.thumbnail_path if item else None

    def get_all_thumbnails(self) -> List[Optional[str]]:
        """
        Get all thumbnail paths in the dataset.

        Returns:
            List of thumbnail paths (some may be None if generation failed)
        """
        return [item.thumbnail_path for item in self._items]

    def find_index(self, path: str) -> int:
        """
        Find the index of an item by its media path.

        Args:
            path: The media file path to search for

        Returns:
            The index of the item

        Raises:
            ValueError: If the path is not found
        """
        for i, item in enumerate(self._items):
            if item.media_path == path:
                return i
        raise ValueError(f"Path not found: {path}")

    # =========================================================================
    # Dataset loading and management
    # =========================================================================

    def load(
        self,
        path: str,
        masks_dir: Optional[str] = None,
        only_missing_captions: bool = False,
        ignore_list: Optional[List[str]] = None,
        subdirectories: bool = False,
        load_thumbnails: bool = True
    ) -> None:
        """
        Load a dataset from a directory.

        Args:
            path: Path to the directory containing media files
            masks_dir: Optional path to directory containing mask files
            only_missing_captions: If True, only load images without captions
            ignore_list: List of patterns to ignore in filenames
            subdirectories: If True, search subdirectories recursively
            load_thumbnails: If True, generate thumbnails for gallery display
        """
        if ignore_list is None:
            ignore_list = []

        # Reset state
        self._items = []
        self._caption_cache = {}
        self._bookmarks = {}
        self._initialized = path and os.path.exists(path)

        if not self._initialized:
            return

        self._base_dir = path
        self._masks_dir = masks_dir

        # Create masks directory if needed
        if self._masks_dir and not os.path.exists(self._masks_dir):
            os.makedirs(self._masks_dir)

        def should_include(file_path: str, file_name: str) -> bool:
            if not (is_image(file_name) or is_video(file_name)):
                return False
            if not _not_filtered(file_name, ignore_list):
                return False
            if only_missing_captions:
                item = MediaItem.from_media_path(file_path, self._masks_dir)
                return not item.caption_exists()
            return True

        # Collect media files
        media_paths = []
        for root, dirs, files in os.walk(path):
            if not subdirectories and root != path:
                continue
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if should_include(full_path, file_name):
                    media_paths.append(full_path)

        media_paths.sort()

        # Create MediaItem instances
        for media_path in media_paths:
            item = MediaItem.from_media_path(media_path, self._masks_dir)
            if load_thumbnails:
                item.thumbnail_path = generate_thumbnail(media_path)
            self._items.append(item)

        # Load bookmarks
        bookmarks_path = os.path.join(self._base_dir, "bookmarks.json")
        if os.path.exists(bookmarks_path):
            with open(bookmarks_path, "r", encoding="utf8") as f:
                self._bookmarks = json.load(f)

    def prune_orphaned_captions(self, path: Optional[str] = None, subdirectories: bool = False) -> int:
        """
        Remove caption files that don't have a corresponding media file.

        Args:
            path: Directory to scan (defaults to base_dir)
            subdirectories: If True, scan subdirectories recursively

        Returns:
            Number of orphaned captions removed
        """
        if path is None:
            path = self._base_dir

        removed_count = 0
        for root, dirs, files in os.walk(path):
            if not subdirectories and root != path:
                continue
            for f in files:
                if not f.lower().endswith('.txt'):
                    continue
                full_path = os.path.join(root, f)
                media_base = os.path.splitext(full_path)[0]
                potential_media = [media_base + ext for ext in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS]
                # Append case for uppercase extensions
                potential_media += [media_base + ext.upper() for ext in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS]
                if not any(os.path.exists(p) for p in potential_media):
                    os.remove(full_path)
                    removed_count += 1

        return removed_count

    # =========================================================================
    # Caption operations
    # =========================================================================

    def read_caption(self, index: int) -> str:
        """
        Read the caption text for an item.

        Args:
            index: The index of the item

        Returns:
            The caption text, or empty string if not found
        """
        item = self.get_item(index)
        if not item:
            return ""

        caption_text = ""
        if os.path.exists(item.caption_path):
            with open(item.caption_path, 'r', encoding='utf-8') as f:
                caption_text = f.read()

        self._caption_cache[item.caption_path] = caption_text
        return caption_text

    def read_tags(self, index: int) -> List[str]:
        """
        Read and parse the caption into individual tags.

        Args:
            index: The index of the item

        Returns:
            List of tags (comma-separated values from caption)
        """
        caption = self.read_caption(index)
        tags = caption.split(",")
        tags = [tag.strip() for tag in tags]
        tags = [tag for tag in tags if tag]
        return tags

    def save_caption(self, index: int, new_text: str) -> bool:
        """
        Save caption text for an item.

        Args:
            index: The index of the item
            new_text: The new caption text to save

        Returns:
            True if saved successfully, False otherwise
        """
        item = self.get_item(index)
        if not item:
            logger.warning(f"save_caption: Invalid index {index}")
            return False

        # Only save if text has changed
        cached = self._caption_cache.get(item.caption_path, None)
        if cached is not None and cached == new_text:
            logger.debug(f"save_caption: No changes for {item.caption_path}")
            return True

        logger.info(f"save_caption: Saving caption for {item.filename} (index={index})")
        try:
            with open(item.caption_path, 'w', encoding='utf-8') as f:
                f.write(new_text)

            self._caption_cache[item.caption_path] = new_text
            logger.debug(f"save_caption: Successfully saved {item.caption_path}")
            return True
        except Exception as e:
            logger.error(f"save_caption: Failed to save {item.caption_path}: {e}")
            return False

    # =========================================================================
    # File operations
    # =========================================================================

    def delete_item(self, index: int) -> bool:
        """
        Delete an item and all associated files.

        Args:
            index: The index of the item to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        item = self.get_item(index)
        if not item:
            logger.warning(f"delete_item: Invalid index {index}")
            return False

        logger.info(f"delete_item: Deleting {item.filename} (index={index})")

        try:
            # Delete media file
            if os.path.exists(item.media_path):
                logger.debug(f"delete_item: Removing media file {item.media_path}")
                os.remove(item.media_path)
            else:
                logger.warning(f"delete_item: Media file not found: {item.media_path}")

            # Delete mask file if exists
            if item.mask_exists():
                logger.debug(f"delete_item: Removing mask file {item.mask_path}")
                os.remove(item.mask_path)

            # Delete caption file if exists
            if os.path.exists(item.caption_path):
                logger.debug(f"delete_item: Removing caption file {item.caption_path}")
                os.remove(item.caption_path)

            # Remove from caption cache
            self._caption_cache.pop(item.caption_path, None)

            # Remove from items list
            del self._items[index]

            logger.info(f"delete_item: Successfully deleted {item.filename}")
            return True

        except Exception as e:
            logger.error(f"delete_item: Failed to delete {item.filename}: {e}")
            return False

    def rename_item(self, index: int, new_name: str) -> Tuple[bool, str, Optional[str]]:
        """
        Rename an item to a user-specified name.

        Args:
            index: The index of the item to rename
            new_name: New filename without extension

        Returns:
            Tuple of (success, message, new_path or None)
        """
        item = self.get_item(index)
        if not item:
            logger.warning(f"rename_item: Invalid index {index}")
            return False, "Invalid index or dataset not initialized", None

        new_name = new_name.strip()
        if not new_name:
            logger.warning(f"rename_item: Empty filename provided for {item.filename}")
            return False, "Filename cannot be empty", None

        invalid_chars = '/\\:*?"<>|'
        if any(c in new_name for c in invalid_chars):
            logger.warning(f"rename_item: Invalid characters in new name '{new_name}' for {item.filename}")
            return False, f"Filename contains invalid characters: {invalid_chars}", None

        new_media_path = os.path.join(item.directory, new_name + item.extension)
        new_caption_path = os.path.splitext(new_media_path)[0] + ".txt"

        # Check for existing file (unless it's the same file)
        if os.path.exists(new_media_path) and new_media_path != item.media_path:
            logger.warning(f"rename_item: File already exists: {new_media_path}")
            return False, "A file with this name already exists", None

        # No changes needed
        if new_media_path == item.media_path:
            logger.debug(f"rename_item: No changes needed for {item.filename}")
            return True, "No changes made", new_media_path

        logger.info(f"rename_item: Renaming {item.filename} to {new_name}{item.extension} (index={index})")

        try:
            # Rename media file
            logger.debug(f"rename_item: Renaming media file {item.media_path} -> {new_media_path}")
            os.rename(item.media_path, new_media_path)

            # Rename caption file if exists
            if os.path.exists(item.caption_path):
                logger.debug(f"rename_item: Renaming caption file {item.caption_path} -> {new_caption_path}")
                os.rename(item.caption_path, new_caption_path)

            # Rename mask file if exists
            if item.mask_exists():
                new_mask_path = os.path.join(
                    self._masks_dir,
                    os.path.splitext(new_name)[0] + ".png"
                ) if self._masks_dir else None
                if new_mask_path:
                    logger.debug(f"rename_item: Renaming mask file {item.mask_path} -> {new_mask_path}")
                    os.rename(item.mask_path, new_mask_path)

            # Update caption cache
            if item.caption_path in self._caption_cache:
                self._caption_cache[new_caption_path] = self._caption_cache.pop(item.caption_path)
                logger.debug(f"rename_item: Updated caption cache")

            # Update item paths
            item.update_paths_after_rename(new_media_path, self._masks_dir)

            logger.info(f"rename_item: Successfully renamed to {new_name}{item.extension}")
            return True, "File renamed successfully", new_media_path

        except OSError as e:
            logger.error(f"rename_item: Failed to rename {item.filename} to {new_name}: {e}")
            return False, f"Error renaming file: {str(e)}", None

    def rename_item_numbered(self, index: int, offset: int = 0) -> Optional[str]:
        """
        Rename an item to a 5-digit numbered format.

        Args:
            index: The index of the item to rename
            offset: Offset to add to the index for numbering

        Returns:
            The new name if successful, None otherwise
        """
        item = self.get_item(index)
        if not item:
            return None

        new_name = str(offset + index).zfill(5)
        success, message, new_path = self.rename_item(index, new_name)

        if success and new_path:
            return new_name
        else:
            print(message)
            return None

    def copy_item(self, index: int, target_dir: str) -> bool:
        """
        Copy an item and its caption to a target directory.

        Args:
            index: The index of the item to copy
            target_dir: Target directory path

        Returns:
            True if copied successfully, False otherwise
        """
        item = self.get_item(index)
        if not item:
            logger.warning(f"copy_item: Invalid index {index}")
            return False

        logger.info(f"copy_item: Copying {item.filename} to {target_dir} (index={index})")

        try:
            if not os.path.exists(target_dir):
                logger.debug(f"copy_item: Creating target directory {target_dir}")
                os.makedirs(target_dir)

            logger.debug(f"copy_item: Copying media file {item.media_path} -> {target_dir}")
            shutil.copy2(item.media_path, target_dir)

            if os.path.exists(item.caption_path):
                logger.debug(f"copy_item: Copying caption file {item.caption_path} -> {target_dir}")
                shutil.copy2(item.caption_path, target_dir)

            logger.info(f"copy_item: Successfully copied {item.filename}")
            return True

        except Exception as e:
            logger.error(f"copy_item: Failed to copy {item.filename}: {e}")
            return False

    def add_item(self, media_path: str) -> int:
        """
        Add a new media file to the dataset.

        Args:
            media_path: Absolute path to the media file to add

        Returns:
            The index of the newly added item
        """
        item = MediaItem.from_media_path(media_path, self._masks_dir)
        item.thumbnail_path = generate_thumbnail(media_path)
        self._items.append(item)
        return len(self._items) - 1

    def update_image(self, index: int, new_image: Image.Image) -> bool:
        """
        Update an image with a new PIL Image.

        Args:
            index: The index of the item to update
            new_image: The new PIL Image

        Returns:
            True if updated successfully, False otherwise
        """
        item = self.get_item(index)
        if not item:
            logger.warning(f"update_image: Invalid index {index}")
            return False

        logger.info(f"update_image: Updating image {item.filename} (index={index})")

        try:
            logger.debug(f"update_image: Saving new image to {item.media_path}")
            new_image.save(item.media_path)

            # Update thumbnail
            logger.debug(f"update_image: Regenerating thumbnail for {item.filename}")
            item.thumbnail_path = generate_thumbnail(item.media_path)

            logger.info(f"update_image: Successfully updated {item.filename}")
            return True

        except Exception as e:
            logger.error(f"update_image: Failed to update {item.filename}: {e}")
            return False

    # =========================================================================
    # Bookmark operations
    # =========================================================================

    def is_bookmarked(self, index: int) -> bool:
        """
        Check if an item is bookmarked.

        Args:
            index: The index of the item

        Returns:
            True if bookmarked, False otherwise
        """
        item = self.get_item(index)
        if not item:
            return False

        return self._bookmarks.get(item.filename, False)

    def set_bookmark(self, index: int, value: bool) -> None:
        """
        Set the bookmark status for an item.

        Args:
            index: The index of the item
            value: True to bookmark, False to unbookmark
        """
        item = self.get_item(index)
        if not item:
            return

        self._bookmarks[item.filename] = value
        self._save_bookmarks()

    def toggle_bookmark(self, index: int) -> bool:
        """
        Toggle the bookmark status for an item.

        Args:
            index: The index of the item

        Returns:
            The new bookmark status
        """
        current = self.is_bookmarked(index)
        new_status = not current
        self.set_bookmark(index, new_status)
        return new_status

    def _save_bookmarks(self) -> None:
        """Save bookmarks to the bookmarks.json file."""
        if not self._base_dir:
            return
        bookmarks_path = os.path.join(self._base_dir, "bookmarks.json")
        with open(bookmarks_path, "w", encoding="utf8") as f:
            json.dump(self._bookmarks, f, indent=4)

    # =========================================================================
    # Dataset operations
    # =========================================================================

    def backup(self) -> Optional[str]:
        """
        Create a compressed backup of all media and caption files.

        Returns:
            Path to the backup file, or None if failed
        """
        if not self._initialized or self.is_empty:
            return None

        to_compress = []
        for item in self._items:
            to_compress.append(item.media_path)
            if os.path.exists(item.caption_path):
                to_compress.append(item.caption_path)

        target_dir = self._base_dir
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(target_dir, f'dataset_backup_{date}.zip')

        with zipfile.ZipFile(backup_path, 'w') as zipf:
            for f in to_compress:
                if os.path.exists(f):
                    zipf.write(f, os.path.basename(f), compress_type=zipfile.ZIP_DEFLATED)

        return backup_path

    def scan(self, func: Callable[[int, str, Optional[str], Optional[str]], Any]) -> List[Any]:
        """
        Scan through all items and apply a function to each.

        Args:
            func: Function taking (index, media_path, mask_path, caption_path)

        Returns:
            List of results from applying func to each item
        """
        if not self._initialized:
            raise Exception("Dataset not initialized!")

        results = []
        for i, item in enumerate(self._items):
            mask_path = item.mask_path if item.mask_exists() else None
            caption_path = item.caption_path if os.path.exists(item.caption_path) else None
            results.append(func(i, item.media_path, mask_path, caption_path))

        return results

    # =========================================================================
    # Legacy compatibility (deprecated - will be removed in future versions)
    # =========================================================================

    @property
    def initialized(self) -> bool:
        """Deprecated: Use is_initialized instead."""
        return self._initialized

    @property
    def mask_support(self) -> bool:
        """Deprecated: Use has_mask_support instead."""
        return self.has_mask_support

    def size(self) -> int:
        """Deprecated: Use len(dataset) or dataset.count instead."""
        return len(self._items)

    def empty(self) -> bool:
        """Deprecated: Use dataset.is_empty instead."""
        return self.is_empty

    def mask_paths(self, index: int) -> Optional[str]:
        """Deprecated: Use get_mask_path(index) instead."""
        return self.get_mask_path(index)

    def read_caption_at(self, index: int) -> str:
        """Deprecated: Use read_caption(index) instead."""
        return self.read_caption(index)

    def read_tags_at(self, index: int) -> List[str]:
        """Deprecated: Use read_tags(index) instead."""
        return self.read_tags(index)

    def delete_image(self, index: int) -> bool:
        """Deprecated: Use delete_item(index) instead."""
        return self.delete_item(index)

    def rename_image(self, index: int, offset: int) -> Optional[str]:
        """Deprecated: Use rename_item_numbered(index, offset) instead."""
        return self.rename_item_numbered(index, offset)

    def rename_image_to(self, index: int, new_name: str) -> Tuple[bool, str, Optional[str]]:
        """Deprecated: Use rename_item(index, new_name) instead."""
        return self.rename_item(index, new_name)

    def copy_media(self, index: int, target_dir: str) -> bool:
        """Deprecated: Use copy_item(index, target_dir) instead."""
        return self.copy_item(index, target_dir)

    @property
    def media_paths(self) -> List[str]:
        """Deprecated: Direct access to media_paths. Use get_media_path(index) or iterate over items."""
        return [item.media_path for item in self._items]

    @property
    def caption_paths(self) -> List[str]:
        """Deprecated: Direct access to caption_paths. Use get_caption_path(index) instead."""
        return [item.caption_path for item in self._items]

    @property
    def thumbnail_images(self) -> List[Optional[str]]:
        """Deprecated: Direct access to thumbnails. Use get_all_thumbnails() instead."""
        return self.get_all_thumbnails()

    @property
    def masks_path(self) -> Optional[str]:
        """Deprecated: Use masks_dir instead."""
        return self._masks_dir

    def prune(self, path: str, subdirectories: bool = False) -> int:
        """Deprecated: Use prune_orphaned_captions(path, subdirectories) instead."""
        return self.prune_orphaned_captions(path, subdirectories)
