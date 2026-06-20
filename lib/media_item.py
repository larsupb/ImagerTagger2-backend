"""
MediaItem dataclass representing a single media file with its associated files.

This module provides a clean abstraction for handling media files (images/videos)
along with their caption files, mask files, and cached thumbnails.
"""

from dataclasses import dataclass, field
import os
from typing import Optional


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')


def is_image(path: str) -> bool:
    """Check if a file path points to an image file."""
    return path.lower().endswith(IMAGE_EXTENSIONS)


def is_video(path: str) -> bool:
    """Check if a file path points to a video file."""
    return path.lower().endswith(VIDEO_EXTENSIONS)


def derive_caption_path(media_path: str) -> str:
    """Derive the caption file path from a media file path."""
    return os.path.splitext(media_path)[0] + ".txt"


def derive_mask_path(media_path: str, masks_dir: str) -> str:
    """Derive the mask file path from a media file path and masks directory."""
    base_name = os.path.basename(media_path)
    mask_file_name = os.path.splitext(base_name)[0] + ".png"
    return os.path.join(masks_dir, mask_file_name)


@dataclass
class MediaItem:
    """
    Represents a single media item (image or video) with all associated files.

    Attributes:
        media_path: Full absolute path to the image/video file
        caption_path: Full absolute path to the .txt caption file
        mask_path: Full absolute path to the mask file (or None if no mask support)
        thumbnail_path: Path to the cached thumbnail/gif (or None if not cached)
    """
    media_path: str
    caption_path: str
    mask_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    @classmethod
    def from_media_path(cls, media_path: str, masks_dir: Optional[str] = None) -> 'MediaItem':
        """
        Create a MediaItem from a media file path.

        Args:
            media_path: Full absolute path to the media file
            masks_dir: Optional directory where mask files are stored

        Returns:
            A new MediaItem instance with derived paths
        """
        caption_path = derive_caption_path(media_path)
        mask_path = derive_mask_path(media_path, masks_dir) if masks_dir else None
        return cls(
            media_path=media_path,
            caption_path=caption_path,
            mask_path=mask_path,
            thumbnail_path=None
        )

    @property
    def filename(self) -> str:
        """Return the filename with extension (e.g., 'image.jpg')."""
        return os.path.basename(self.media_path)

    @property
    def basename(self) -> str:
        """Return the filename without extension (e.g., 'image')."""
        return os.path.splitext(self.filename)[0]

    @property
    def extension(self) -> str:
        """Return the file extension including the dot (e.g., '.jpg')."""
        ext = os.path.splitext(self.media_path)[1]
        # Normalize jpg/jpeg to .jpg
        if ext.lower() in ['.jpg', '.jpeg']:
            return '.jpg'
        return ext

    @property
    def directory(self) -> str:
        """Return the parent directory of the media file."""
        return os.path.dirname(self.media_path)

    @property
    def is_video(self) -> bool:
        """Check if this item is a video file."""
        return is_video(self.media_path)

    @property
    def is_image(self) -> bool:
        """Check if this item is an image file."""
        return is_image(self.media_path)

    def media_exists(self) -> bool:
        """Check if the media file exists on disk."""
        return os.path.exists(self.media_path)

    def caption_exists(self) -> bool:
        """Check if the caption file exists and is non-empty."""
        return os.path.exists(self.caption_path) and os.path.getsize(self.caption_path) > 0

    def mask_exists(self) -> bool:
        """Check if the mask file exists on disk."""
        return self.mask_path is not None and os.path.exists(self.mask_path)

    def thumbnail_exists(self) -> bool:
        """Check if the thumbnail/cache file exists on disk."""
        return self.thumbnail_path is not None and os.path.exists(self.thumbnail_path)

    def update_paths_after_rename(self, new_media_path: str, masks_dir: Optional[str] = None) -> None:
        """
        Update all paths after a rename operation.

        Args:
            new_media_path: The new media file path
            masks_dir: Optional directory where mask files are stored
        """
        self.media_path = new_media_path
        self.caption_path = derive_caption_path(new_media_path)
        if masks_dir:
            self.mask_path = derive_mask_path(new_media_path, masks_dir)
        # Note: thumbnail_path is not updated as it's based on content hash

    def __repr__(self) -> str:
        return f"MediaItem(filename={self.filename!r}, has_caption={self.caption_exists()}, has_mask={self.mask_exists()})"
