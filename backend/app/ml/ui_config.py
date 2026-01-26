"""
Professional UI Configuration for SwishVision.

Provides consistent styling, fonts, colors, and rendering utilities
for all video outputs to ensure professional, high-quality visuals.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import os


# ============================================================================
# COLOR PALETTE - Professional, consistent colors
# ============================================================================

class Colors:
    """Professional color palette (BGR format for OpenCV)."""

    # Team Colors - Vibrant and distinguishable
    TEAM_A = (34, 139, 34)      # Forest Green (BGR)
    TEAM_B = (19, 69, 139)      # Saddle Brown / Dark Orange-Red (BGR)
    REFEREE = (0, 215, 255)     # Gold (BGR)

    # UI Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DARK_GRAY = (40, 40, 40)
    MEDIUM_GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)

    # Overlay backgrounds (with alpha support)
    OVERLAY_DARK = (20, 20, 20)
    OVERLAY_LIGHT = (240, 240, 240)

    # Accent colors for different stages
    DETECTION_ACCENT = (0, 191, 255)    # Deep Sky Blue
    TRACKING_ACCENT = (147, 20, 255)     # Deep Pink
    TEAM_ACCENT = (0, 215, 255)         # Gold
    TACTICAL_ACCENT = (34, 139, 34)     # Forest Green


# Default team color mapping
DEFAULT_TEAM_COLORS = {
    0: Colors.TEAM_A,
    1: Colors.TEAM_B,
    -1: Colors.REFEREE,
}


# ============================================================================
# FONT CONFIGURATION
# ============================================================================

class FontConfig:
    """
    Font configuration with Arial support.

    OpenCV doesn't support TTF fonts directly, so we use PIL for text rendering
    when high-quality output is needed, with fallback to OpenCV fonts.
    """

    # Try to find Arial font on the system
    FONT_PATHS = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/Library/Fonts/Arial.ttf",                      # macOS alternative
        "C:\\Windows\\Fonts\\arial.ttf",                 # Windows
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux alternative
    ]

    @staticmethod
    def get_arial_font(size: int = 32) -> Optional[ImageFont.FreeTypeFont]:
        """
        Get Arial font for PIL rendering.

        Args:
            size: Font size in points

        Returns:
            ImageFont object or None if not found
        """
        for path in FontConfig.FONT_PATHS:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue

        # Fallback to default PIL font
        try:
            return ImageFont.truetype("Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    # OpenCV font fallback
    CV_FONT = cv2.FONT_HERSHEY_SIMPLEX
    CV_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ============================================================================
# TEXT SIZE CONFIGURATION
# ============================================================================

class TextSize:
    """Standard text sizes for consistency."""

    TITLE = 48          # Stage titles
    HEADING = 36        # Section headings
    LABEL = 28          # Player labels, main text
    BODY = 24           # Body text, stats
    SMALL = 20          # Small annotations
    TINY = 16           # Minimal text


# ============================================================================
# RENDERING UTILITIES
# ============================================================================

def put_text_pil(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = TextSize.BODY,
    color: Tuple[int, int, int] = Colors.WHITE,
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 8,
    align: str = 'left'
) -> np.ndarray:
    """
    Render high-quality text using PIL with Arial font.

    Args:
        img: Image array (BGR)
        text: Text to render
        position: (x, y) position (top-left)
        font_size: Font size in points
        color: Text color (BGR)
        bg_color: Optional background color (BGR)
        padding: Padding around text if bg_color is set
        align: Text alignment ('left', 'center', 'right')

    Returns:
        Image with text rendered
    """
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Get font
    font = FontConfig.get_arial_font(font_size)

    # Convert BGR to RGB for PIL
    text_color_rgb = (color[2], color[1], color[0])

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Adjust position based on alignment
    x, y = position
    if align == 'center':
        x -= text_width // 2
    elif align == 'right':
        x -= text_width

    # Draw background if specified
    if bg_color is not None:
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        bg_bbox = (
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        )
        draw.rectangle(bg_bbox, fill=bg_rgb)

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color_rgb)

    # Convert back to BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


def put_text_cv(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = Colors.WHITE,
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 8
) -> np.ndarray:
    """
    Render text using OpenCV (faster but lower quality).

    Use this for real-time applications where speed matters more than quality.

    Args:
        img: Image array (BGR)
        text: Text to render
        position: (x, y) position (bottom-left for OpenCV)
        font_scale: Font scale factor
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Optional background color (BGR)
        padding: Padding around text if bg_color is set

    Returns:
        Image with text rendered
    """
    x, y = position

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, FontConfig.CV_FONT, font_scale, thickness
    )

    # Draw background if specified
    if bg_color is not None:
        cv2.rectangle(
            img,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )

    # Draw text
    cv2.putText(img, text, (x, y), FontConfig.CV_FONT, font_scale, color, thickness, cv2.LINE_AA)

    return img


def add_title_bar(
    img: np.ndarray,
    title: str,
    height: int = 60,
    bg_color: Tuple[int, int, int] = Colors.OVERLAY_DARK,
    text_color: Tuple[int, int, int] = Colors.WHITE,
    use_pil: bool = True
) -> np.ndarray:
    """
    Add a professional title bar to the top of an image.

    Args:
        img: Image array (BGR)
        title: Title text
        height: Height of title bar in pixels
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
        use_pil: Use PIL for high-quality text (default: True)

    Returns:
        Image with title bar added
    """
    # Draw background
    cv2.rectangle(img, (0, 0), (img.shape[1], height), bg_color, -1)

    # Add title text
    if use_pil:
        img = put_text_pil(
            img,
            title,
            position=(20, 10),
            font_size=TextSize.TITLE,
            color=text_color
        )
    else:
        put_text_cv(
            img,
            title,
            position=(20, height - 15),
            font_scale=1.2,
            color=text_color,
            thickness=2
        )

    return img


def add_stats_overlay(
    img: np.ndarray,
    stats: dict,
    position: str = 'bottom-left',
    bg_color: Tuple[int, int, int] = Colors.OVERLAY_DARK,
    text_color: Tuple[int, int, int] = Colors.WHITE,
    use_pil: bool = True
) -> np.ndarray:
    """
    Add a stats overlay to the image.

    Args:
        img: Image array (BGR)
        stats: Dictionary of stat_name -> value
        position: Position of overlay ('bottom-left', 'bottom-right', 'top-left', 'top-right')
        bg_color: Background color with transparency (BGR)
        text_color: Text color (BGR)
        use_pil: Use PIL for high-quality text

    Returns:
        Image with stats overlay
    """
    if not stats:
        return img

    height, width = img.shape[:2]
    margin = 20
    padding = 15
    line_height = 35

    # Calculate overlay dimensions
    num_lines = len(stats)
    overlay_height = padding * 2 + line_height * num_lines
    overlay_width = 300

    # Calculate position
    if 'right' in position:
        x = width - overlay_width - margin
    else:
        x = margin

    if 'bottom' in position:
        y = height - overlay_height - margin
    else:
        y = margin

    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + overlay_width, y + overlay_height), bg_color, -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Add border
    cv2.rectangle(img, (x, y), (x + overlay_width, y + overlay_height), Colors.LIGHT_GRAY, 2)

    # Add stats text
    text_y = y + padding + 25
    for stat_name, value in stats.items():
        text = f"{stat_name}: {value}"
        if use_pil:
            img = put_text_pil(
                img,
                text,
                position=(x + padding, text_y - 20),
                font_size=TextSize.BODY,
                color=text_color
            )
        else:
            put_text_cv(img, text, (x + padding, text_y), 0.7, text_color, 2)
        text_y += line_height

    return img


def create_player_label(
    img: np.ndarray,
    text: str,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    use_pil: bool = True
) -> np.ndarray:
    """
    Create a professional player label above bounding box.

    Args:
        img: Image array (BGR)
        text: Label text (e.g., "#23 LeBron James")
        bbox: Bounding box (x1, y1, x2, y2)
        color: Team color (BGR)
        use_pil: Use PIL for high-quality text

    Returns:
        Image with label added
    """
    x1, y1, x2, y2 = map(int, bbox)

    if use_pil:
        # Calculate label size
        font = FontConfig.get_arial_font(TextSize.LABEL)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw background rectangle
        label_y = max(10, y1 - text_height - 15)
        cv2.rectangle(img, (x1, label_y), (x1 + text_width + 16, label_y + text_height + 12), color, -1)

        # Add white border
        cv2.rectangle(img, (x1, label_y), (x1 + text_width + 16, label_y + text_height + 12), Colors.WHITE, 2)

        # Render text
        img = put_text_pil(img, text, (x1 + 8, label_y + 4), TextSize.LABEL, Colors.WHITE)
    else:
        # OpenCV fallback
        (text_width, text_height), baseline = cv2.getTextSize(text, FontConfig.CV_FONT, 0.6, 2)
        label_y = max(10, y1 - text_height - 10)

        cv2.rectangle(img, (x1, label_y), (x1 + text_width + 12, y1), color, -1)
        cv2.rectangle(img, (x1, label_y), (x1 + text_width + 12, y1), Colors.WHITE, 2)
        cv2.putText(img, text, (x1 + 6, y1 - 6), FontConfig.CV_FONT, 0.6, Colors.WHITE, 2, cv2.LINE_AA)

    return img


# ============================================================================
# VIDEO OUTPUT CONFIGURATION
# ============================================================================

class VideoConfig:
    """Video output settings using mp4v codec (as per reference notebook)."""

    # mp4v codec
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

    # Quality settings
    DEFAULT_FPS = 30.0
    DEFAULT_BITRATE = 8000  # kbps for high quality

    @staticmethod
    def create_writer(
        output_path: str,
        fps: float,
        width: int,
        height: int,
        use_h264: bool = False  # Deprecated parameter, ignored
    ) -> cv2.VideoWriter:
        """
        Create a video writer using mp4v codec.

        Args:
            output_path: Output video path
            fps: Frames per second
            width: Video width
            height: Video height
            use_h264: Ignored (kept for API compatibility)

        Returns:
            VideoWriter object
        """
        writer = cv2.VideoWriter(output_path, VideoConfig.FOURCC, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")

        return writer
