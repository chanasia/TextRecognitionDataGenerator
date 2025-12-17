"""Text image generation with Thai language support."""

import random as rnd
from typing import Tuple, List, Dict
from PIL import Image, ImageColor, ImageDraw, ImageFont
import math
import numpy as np

from trdg.utils import get_text_width, get_text_height, get_text_bbox
from trdg.thai_utils import (
    decompose_thai_grapheme,
    normalize_grapheme,
    split_grapheme_clusters,
    has_upper_vowel,
    has_lower_vowel,
    contains_thai
)
from trdg.thai_bbox import measure_grapheme_bboxes


def _calculate_horizontal_bounds(
        image_font: ImageFont.FreeTypeFont,
        graphemes: List[str],
        word_split: bool,
        text_parts: List[str],
        space_width: float,
        character_spacing: int,
        is_thai: bool
) -> Tuple[float, float]:
    """
    Simulate text layout to find exact pixel bounds (min_x, max_x).
    This fixes the clipping issue for italic/bold fonts with overhangs.
    """
    min_x = float('inf')
    max_x = float('-inf')
    cursor_x = 0.0

    # Create an iterator that mimics the rendering loops exactly
    iterator = []

    if word_split and text_parts:
        # Word split mode logic
        for part in text_parts:
            if part == " ":
                iterator.append((" ", True)) # (text, is_space)
            else:
                # For Thai, we split into graphemes; for others, chars
                sub_graphemes = split_grapheme_clusters(part) if is_thai else list(part)
                for g in sub_graphemes:
                    iterator.append((g, False))
    else:
        # Standard character spacing mode
        for g in graphemes:
            iterator.append((g, False))

    # Simulation Loop
    for i, (char, is_space) in enumerate(iterator):
        if is_space:
            # Logic matches _render_simple_text and _calculate_char_positions
            space_w = get_text_width(image_font, " ")
            advance = int(space_w * space_width)

            # Spaces don't have ink, but they advance the cursor
            # We track cursor position for max_x to ensure trailing spaces aren't cut
            min_x = min(min_x, cursor_x)
            max_x = max(max_x, cursor_x + advance)

            cursor_x += advance
        else:
            advance = get_text_width(image_font, char)

            # Get actual pixel bounds relative to the pen (0,0)
            if char.strip():
                try:
                    left, _, right, _ = image_font.getbbox(char)
                    # Convert to global coordinates
                    global_left = cursor_x + left
                    global_right = cursor_x + right

                    min_x = min(min_x, global_left)
                    max_x = max(max_x, global_right)
                except Exception:
                    # Fallback if getbbox fails
                    min_x = min(min_x, cursor_x)
                    max_x = max(max_x, cursor_x + advance)
            else:
                # Invisible chars (like zero-width)
                min_x = min(min_x, cursor_x)
                max_x = max(max_x, cursor_x)

            cursor_x += advance

            # Apply character spacing (if not the very last element)
            if not word_split and character_spacing > 0 and i < len(iterator) - 1:
                cursor_x += character_spacing

    if min_x == float('inf'):
        return 0, 0

    return min_x, max_x


def _render_simple_text(
        txt_img_draw: ImageDraw.ImageDraw,
        txt_mask_draw: ImageDraw.ImageDraw,
        image_font: ImageFont.FreeTypeFont,
        text: str,
        fill: Tuple[int, int, int],
        stroke_width: int,
        stroke_fill_color: Tuple[int, int, int],
        word_split: bool,
        space_width: float,
        character_spacing: int,
        y_offset: int,
        start_x: float = 0  # Added: Start position offset
) -> List[Dict]:
    """Render non-Thai text using simple character-based approach."""
    if word_split:
        words = text.split(" ")
        splitted_text = []
        for w in words:
            splitted_text.append(w)
            splitted_text.append(" ")
        if splitted_text:
            splitted_text.pop()
    else:
        splitted_text = list(text)

    piece_widths = []
    for p in splitted_text:
        if p == " ":
            piece_widths.append(int(get_text_width(image_font, " ") * space_width))
        else:
            piece_widths.append(get_text_width(image_font, p))

    char_positions = []

    # Calculate cumulative widths for positioning
    current_x = start_x # Start with the calculated offset

    for i, p in enumerate(splitted_text):
        # Determine spacing for this step
        spacing = character_spacing if not word_split and i > 0 else 0
        current_x += spacing

        # Draw text at current_x
        txt_img_draw.text(
            (current_x, y_offset),
            p,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        txt_mask_draw.text(
            (current_x, y_offset),
            p,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

        # [แก้ไขจุดที่ 3] เช็คว่าไม่ใช่ช่องว่าง ก่อนเก็บ BBox
        if p.strip():
            left, top, right, bottom = get_text_bbox(image_font, p)
            char_positions.append({
                "grapheme": p,
                "bbox": (current_x, y_offset + top, current_x + (right - left), y_offset + bottom)
            })

        # Advance cursor for next character
        current_x += piece_widths[i]

    return char_positions


def _render_thai_mask_components(
        txt_mask_draw: ImageDraw.ImageDraw,
        image_font: ImageFont.FreeTypeFont,
        components: Dict,
        x_pos: int,
        y_offset: int,
        base_idx: int,
        stroke_width: int,
        stroke_fill_color: Tuple[int, int, int]
) -> None:
    """Render Thai grapheme components to mask image."""
    def get_mask_color(idx: int) -> Tuple[int, int, int]:
        return ((idx + 1) // (255 * 255), (idx + 1) // 255, (idx + 1) % 255)

    mask_idx = base_idx

    if components['base'] or components['leading']:
        base_text = (components['leading'] if components['leading'] else '') + \
                   (components['base'] if components['base'] else '')
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_text,
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['upper_vowel']:
        txt_mask_draw.text(
            (x_pos, y_offset),
            components['base'] + components['upper_vowel'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['upper_diacritic']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_for_render + components['upper_diacritic'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['upper_tone']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_for_render + components['upper_tone'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['lower']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_for_render + components['lower'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['trailing']:
        txt_mask_draw.text(
            (x_pos, y_offset),
            (components['base'] if components['base'] else '') + components['trailing'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )


def _render_thai_text(
        txt_img_draw: ImageDraw.ImageDraw,
        txt_mask_draw: ImageDraw.ImageDraw,
        image_font: ImageFont.FreeTypeFont,
        graphemes: List[str],
        fill: Tuple[int, int, int],
        stroke_width: int,
        stroke_fill_color: Tuple[int, int, int],
        word_split: bool,
        text: str,
        space_width: float,
        character_spacing: int,
        y_offset: int,
        start_x: float = 0 # Added: Start position offset
) -> List[Dict]:
    """Render Thai text with detailed component-level bboxes."""
    if word_split:
        words = text.split(" ")
        text_parts = []
        for i, word in enumerate(words):
            text_parts.append(word)
            if i < len(words) - 1:
                text_parts.append(" ")

        # Pass start_x to calculation
        char_positions, _ = _calculate_char_positions(
            image_font, graphemes, y_offset, word_split=True,
            text_parts=text_parts, space_width=space_width, start_x=start_x
        )

        char_index = 0
        for pos in char_positions:
            g_normalized = normalize_grapheme(pos["grapheme"])

            txt_img_draw.text(
                (pos["x_offset"], y_offset),
                g_normalized,
                fill=fill,
                font=image_font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill_color,
            )

            char_index = _render_grapheme_to_mask(
                txt_mask_draw, image_font, pos["components"],
                pos["x_offset"], y_offset, char_index,
                stroke_width, stroke_fill_color
            )
    else:
        # Pass start_x to calculation
        char_positions, _ = _calculate_char_positions(
            image_font, graphemes, y_offset, start_x=start_x
        )

        for i, pos in enumerate(char_positions):
            g_normalized = normalize_grapheme(pos["grapheme"])

            # Recalculate x_pos including spacing (since _calculate_char_positions returns base width offsets)
            # The logic here is tricky: _calculate_char_positions accumulates widths, but here we add spacing.
            # We need to ensure we use the 'start_x' + cumulative width + cumulative spacing.

            # However, _calculate_char_positions already returns 'x_offset' which is cumulative width + start_x.
            # We just need to add the character spacing component.

            x_pos = pos["x_offset"] + (i * character_spacing if character_spacing > 0 else 0)

            # Update the stored offset in char_positions because we are shifting it by spacing
            pos["x_offset"] = x_pos

            # Also shift the detailed component bboxes
            shift_amount = (i * character_spacing if character_spacing > 0 else 0)
            if shift_amount > 0:
                # Helper to shift bbox
                def shift(b): return (b[0]+shift_amount, b[1], b[2]+shift_amount, b[3])
                if pos['base_bbox']: pos['base_bbox'] = shift(pos['base_bbox'])
                if pos['leading_bbox']: pos['leading_bbox'] = shift(pos['leading_bbox'])
                if pos['upper_vowel_bbox']: pos['upper_vowel_bbox'] = shift(pos['upper_vowel_bbox'])
                if pos['upper_tone_bbox']: pos['upper_tone_bbox'] = shift(pos['upper_tone_bbox'])
                if pos['upper_diacritic_bbox']: pos['upper_diacritic_bbox'] = shift(pos['upper_diacritic_bbox'])
                if pos['lower_bbox']: pos['lower_bbox'] = shift(pos['lower_bbox'])
                if pos['trailing_bbox']: pos['trailing_bbox'] = shift(pos['trailing_bbox'])

            txt_img_draw.text(
                (x_pos, y_offset),
                g_normalized,
                fill=fill,
                font=image_font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill_color,
            )

            _render_thai_mask_components(
                txt_mask_draw, image_font, pos["components"],
                x_pos, y_offset, i, stroke_width, stroke_fill_color
            )

    return char_positions


def _calculate_char_positions(
        image_font: ImageFont.FreeTypeFont,
        graphemes: List[str],
        y_offset: int,
        word_split: bool = False,
        text_parts: List[str] = None,
        space_width: float = 1.0,
        start_x: float = 0 # Added: Start position
) -> Tuple[List[Dict], int]:
    """Calculate character positions and bboxes for Thai graphemes."""
    char_positions = []
    x_offset = start_x # Initialize with start_x

    if word_split and text_parts:
        for part in text_parts:
            part_graphemes = split_grapheme_clusters(part)

            for g in part_graphemes:
                g_width = get_text_width(image_font, g)

                # [แก้ไขจุดที่ 1] เช็คว่าไม่ใช่ช่องว่าง ถึงจะคำนวณ BBox
                if g.strip():
                    components = decompose_thai_grapheme(g)
                    bboxes = measure_grapheme_bboxes(image_font, g, components, x_offset, y_offset)

                    char_positions.append({
                        "grapheme": g,
                        "x_offset": x_offset,
                        "components": components,
                        **bboxes,
                        "is_sara_am": components['is_sara_am']
                    })

                x_offset += g_width

            if part == " ":
                space_w = get_text_width(image_font, " ")
                x_offset = x_offset - space_w + int(space_w * space_width)
    else:
        for g in graphemes:
            g_width = get_text_width(image_font, g)

            if g.strip():
                components = decompose_thai_grapheme(g)
                bboxes = measure_grapheme_bboxes(image_font, g, components, x_offset, y_offset)

                char_positions.append({
                    "grapheme": g,
                    "x_offset": x_offset,
                    "components": components,
                    **bboxes,
                    "is_sara_am": components['is_sara_am']
                })

            x_offset += g_width

    return char_positions, x_offset


def _render_grapheme_to_mask(
        txt_mask_draw: ImageDraw.ImageDraw,
        image_font: ImageFont.FreeTypeFont,
        components: Dict,
        x_offset: int,
        y_offset: int,
        char_index: int,
        stroke_width: int,
        stroke_fill_color: Tuple[int, int, int]
) -> int:
    """Render Thai grapheme components to mask for word_split mode. Returns updated char_index."""
    def get_mask_color(idx):
        return ((idx + 1) // (255 * 255), (idx + 1) // 255, (idx + 1) % 255)

    if components['base'] or components['leading']:
        base_text = (components['leading'] if components['leading'] else '') + \
                   (components['base'] if components['base'] else '')
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_text,
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['upper_vowel']:
        txt_mask_draw.text(
            (x_offset, y_offset),
            components['base'] + components['upper_vowel'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['upper_diacritic']:
        upper_d = components['upper_diacritic']
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_for_render + upper_d,
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['upper_tone']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_for_render + components['upper_tone'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['lower']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_for_render + components['lower'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['trailing']:
        txt_mask_draw.text(
            (x_offset, y_offset),
            (components['base'] if components['base'] else '') + components['trailing'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    return char_index


def generate(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        orientation: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        word_split: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """
    Generate text image with optional Thai language support.

    Returns (image, mask, char_positions) tuple.
    """
    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            stroke_width,
            stroke_fill,
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        word_split: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """
    Generate horizontal text image.

    Automatically detects Thai text and uses appropriate rendering method.
    Returns (image, mask, char_positions) tuple.
    """
    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)

    is_thai = contains_thai(text)

    if is_thai:
        graphemes = split_grapheme_clusters(text)
    else:
        graphemes = list(text)

    # 1. Prepare text parts for word split mode logic (if needed)
    text_parts = []
    if word_split:
        words = text.split(" ")
        for i, word in enumerate(words):
            text_parts.append(word)
            if i < len(words) - 1:
                text_parts.append(" ")

    # 2. Calculate Exact Layout Bounds (Fix for clipping bugs)
    # We simulate layout to find actual ink bounds (min_x, max_x)
    min_x, max_x = _calculate_horizontal_bounds(
        image_font, graphemes, word_split, text_parts, space_width, character_spacing, is_thai
    )

    # 3. Calculate Height Bounds
    # Scan height of all characters to ensure we capture ascenders/descenders
    left, top, right, bottom = get_text_bbox(image_font, text)
    min_y = top
    max_y = bottom

    if is_thai:
        for g in graphemes:
            if g.strip():
                try:
                    _, g_t, _, g_b = image_font.getbbox(g)
                    min_y = min(min_y, g_t)
                    max_y = max(max_y, g_b)
                except Exception:
                    pass

    # 4. Determine Canvas Dimensions
    # Height
    y_offset = -min_y
    text_height = max_y - min_y
    if text_height <= 0: text_height = font_size # fallback

    # Width: Use the calculated bounds
    # max_x - min_x gives the tight width.
    # If min_x is negative (left overhang), we need to shift drawing by -min_x
    text_width = math.ceil(max_x - min_x)
    start_x = -min_x

    # Safety checks
    if text_width <= 0: text_width = 1

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill_color = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    if is_thai:
        char_positions = _render_thai_text(
            txt_img_draw, txt_mask_draw, image_font, graphemes,
            fill, stroke_width, stroke_fill_color,
            word_split, text, space_width, character_spacing, y_offset,
            start_x=start_x # Pass the calculated start offset
        )
    else:
        char_positions = _render_simple_text(
            txt_img_draw, txt_mask_draw, image_font, text,
            fill, stroke_width, stroke_fill_color,
            word_split, space_width, character_spacing, y_offset,
            start_x=start_x # Pass the calculated start offset
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), char_positions
    else:
        return txt_img, txt_mask, char_positions


def _generate_vertical_text(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """
    Generate vertical text image.

    Returns (image, mask, char_positions) tuple.
    """
    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)

    graphemes = split_grapheme_clusters(text)

    left, top, right, bottom = get_text_bbox(image_font, text)
    x_offset_base = -left

    space_height = int(get_text_height(image_font, " ") * space_width)

    grapheme_heights = [
        get_text_height(image_font, g) if g != " " else space_height for g in graphemes
    ]
    text_width = right - left
    text_height = sum(grapheme_heights) + character_spacing * len(graphemes)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill_color = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    char_positions = []

    for i, g in enumerate(graphemes):
        g_left, g_top, g_right, g_bottom = get_text_bbox(image_font, g)
        y_offset = -g_top
        y_pos = sum(grapheme_heights[0:i]) + i * character_spacing + y_offset

        char_positions.append({
            "grapheme": g,
            "bbox": (x_offset_base, y_pos + g_top, x_offset_base + (g_right - g_left), y_pos + g_bottom),
            "is_upper_vowel": has_upper_vowel(g),
            "is_lower_vowel": has_lower_vowel(g)
        })

        txt_img_draw.text(
            (x_offset_base, y_pos),
            g,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        txt_mask_draw.text(
            (x_offset_base, y_pos),
            g,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255, 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), char_positions
    else:
        return txt_img, txt_mask, char_positions