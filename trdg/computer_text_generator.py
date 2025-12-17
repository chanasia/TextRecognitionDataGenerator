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
import uharfbuzz as hb

from trdg.vector_engine import FontVectorEngineHB
from trdg.thai_utils import contains_thai

_vector_engines = {}

def _get_vector_engine(font_path: str, size: int) -> FontVectorEngineHB:
    """Get or create a cached vector engine."""
    key = (font_path, size)
    if key not in _vector_engines:
        _vector_engines[key] = FontVectorEngineHB(font_path, size)
    return _vector_engines[key]

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
    """
    min_x = float('inf')
    max_x = float('-inf')
    cursor_x = 0.0

    iterator = []
    if word_split and text_parts:
        for part in text_parts:
            if part == " ":
                iterator.append((" ", True))
            else:
                sub_graphemes = split_grapheme_clusters(part) if is_thai else list(part)
                for g in sub_graphemes:
                    iterator.append((g, False))
    else:
        for g in graphemes:
            iterator.append((g, False))

    for i, (char, is_space) in enumerate(iterator):
        if is_space:
            space_w = get_text_width(image_font, " ")
            advance = int(space_w * space_width)
            min_x = min(min_x, cursor_x)
            max_x = max(max_x, cursor_x + advance)
            cursor_x += advance
        else:
            advance = get_text_width(image_font, char)
            if char.strip():
                try:
                    left, _, right, _ = image_font.getbbox(char)
                    global_left = cursor_x + left
                    global_right = cursor_x + right
                    min_x = min(min_x, global_left)
                    max_x = max(max_x, global_right)
                except Exception:
                    min_x = min(min_x, cursor_x)
                    max_x = max(max_x, cursor_x + advance)
            else:
                min_x = min(min_x, cursor_x)
                max_x = max(max_x, cursor_x)

            cursor_x += advance
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
        start_x: float = 0
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
    current_x = start_x

    for i, p in enumerate(splitted_text):
        spacing = character_spacing if not word_split and i > 0 else 0
        current_x += spacing

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

        if p.strip():
            left, top, right, bottom = get_text_bbox(image_font, p)
            char_positions.append({
                "grapheme": p,
                "bbox": (current_x, y_offset + top, current_x + (right - left), y_offset + bottom)
            })

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
    """Render Thai grapheme components to mask image. (Legacy)"""
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
    """Render Thai grapheme components to mask for word_split mode. Returns updated char_index. (Legacy)"""
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
    Generate horizontal text image using Vector Engine for exact layout.
    """

    try:
        engine = _get_vector_engine(font, font_size)
    except Exception as e:
        print(f"[Error] Could not load Vector Engine for {font}: {e}")
        return None, None, []

    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(engine.hb_font, buf)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    cursor_x, cursor_y = 0, 0
    glyph_layout_data = []

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        glyph_name = engine.ttfont.getGlyphName(info.codepoint)

        # HarfBuzz Units -> Pixels
        x_offset = pos.x_offset / 64
        y_offset = pos.y_offset / 64
        x_advance = pos.x_advance / 64
        y_advance = pos.y_advance / 64

        # Apply spacing adjustment (ถ้าไม่ใช่ตัวสุดท้าย)
        if character_spacing > 0:
            x_advance += character_spacing

        # Base Position of this glyph
        current_draw_x = cursor_x + x_offset
        current_draw_y = cursor_y + y_offset

        # Decompose to find True BBox
        components = engine.decompose_glyph(glyph_name)

        has_ink = False
        for comp in components:
            bbox = comp.get('bbox')
            if bbox:
                has_ink = True
                bx1, by1, bx2, by2 = bbox

                # Global Coord Calculation (Font Coordinate System: Y-Up)
                global_x1 = current_draw_x + bx1
                global_x2 = current_draw_x + bx2
                global_y1 = current_draw_y + by1
                global_y2 = current_draw_y + by2

                min_x = min(min_x, global_x1)
                max_x = max(max_x, global_x2)
                min_y = min(min_y, global_y1)
                max_y = max(max_y, global_y2)

        glyph_layout_data.append({
            "glyph_name": glyph_name,
            "components": components,
            "draw_x": current_draw_x,
            "draw_y": current_draw_y,
            "x_advance": x_advance
        })

        cursor_x += x_advance
        cursor_y += y_advance

    # Safety Fallback
    if min_x == float('inf'): min_x, max_x = 0, 0
    if min_y == float('inf'): min_y, max_y = 0, font_size

    #Tight Fit
    text_height = int(math.ceil(max_y - min_y))
    text_width = int(math.ceil(max_x - min_x))

    if text_width <= 0: text_width = 1
    if text_height <= 0: text_height = 1

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")

    # Colors
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

    start_offset_x = -min_x
    font_top_y = max_y

    txt_img_draw.text(
        (start_offset_x, font_top_y),
        text,
        fill=fill,
        font=image_font,
        anchor="ls",  # Left-Baseline Alignment
        stroke_width=stroke_width,
        stroke_fill=stroke_fill_color,
        language="th"  # Hint for PIL
    )

    char_positions = []
    mask_idx = 0

    for g_data in glyph_layout_data:
        base_draw_x = g_data['draw_x'] + start_offset_x
        base_draw_y = g_data['draw_y']

        components = g_data['components']

        # Data Structure
        char_info = {
            "glyph_name": g_data['glyph_name'],
            "bbox": None,
            "base_bbox": None,
            "leading_bbox": None,
            "upper_vowel_bbox": None,
            "upper_tone_bbox": None,
            "upper_diacritic_bbox": None,
            "lower_bbox": None,
            "trailing_bbox": None
        }

        all_comp_bboxes = []

        # Color for this character/glyph in the mask
        mask_color = ((mask_idx + 1) // (255 * 255), (mask_idx + 1) // 255, (mask_idx + 1) % 255)

        for comp in components:
            bbox = comp.get('bbox')
            role = comp.get('role')

            if bbox:
                bx1, by1, bx2, by2 = bbox

                # Convert to Image Coordinate System
                img_x1 = base_draw_x + bx1
                img_x2 = base_draw_x + bx2
                img_y1 = font_top_y - (base_draw_y + by2)  # Y-Flip (Top)
                img_y2 = font_top_y - (base_draw_y + by1)  # Y-Flip (Bottom)

                final_bbox = (int(img_x1), int(img_y1), int(img_x2), int(img_y2))
                all_comp_bboxes.append(final_bbox)

                # Assign to Role
                if role == "BASE":
                    char_info["base_bbox"] = final_bbox
                elif role == "LEADING_VOWEL":
                    char_info["leading_bbox"] = final_bbox
                elif role == "UPPER_VOWEL":
                    char_info["upper_vowel_bbox"] = final_bbox
                elif role == "TONE":
                    char_info["upper_tone_bbox"] = final_bbox
                elif role == "UPPER_DIACRITIC":
                    char_info["upper_diacritic_bbox"] = final_bbox
                elif role == "LOWER_VOWEL":
                    char_info["lower_bbox"] = final_bbox
                elif role == "TRAILING_VOWEL":
                    char_info["trailing_bbox"] = final_bbox
                elif role == "SARA_AA":
                    char_info["trailing_bbox"] = final_bbox
                elif role == "NIKHAHIT":
                    char_info["upper_diacritic_bbox"] = final_bbox

                txt_mask_draw.rectangle(final_bbox, fill=mask_color)

        if all_comp_bboxes:
            # Calculate total bbox for the glyph
            min_bx = min(b[0] for b in all_comp_bboxes)
            min_by = min(b[1] for b in all_comp_bboxes)
            max_bx = max(b[2] for b in all_comp_bboxes)
            max_by = max(b[3] for b in all_comp_bboxes)
            char_info['bbox'] = (min_bx, min_by, max_bx, max_by)

            char_positions.append(char_info)
            mask_idx += 1

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