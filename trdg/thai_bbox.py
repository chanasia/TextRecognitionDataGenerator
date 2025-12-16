"""Thai text bbox measurement using pixel-level analysis."""

from typing import Tuple, List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from trdg.utils import get_text_width, get_text_bbox
from trdg.thai_utils import THAI_TONE_MARKS, SARA_AA, NIKHAHIT, normalize_grapheme


def _measure_pixels(
        image_font: ImageFont,
        base_char: str,
        component_char: str,
        x_offset: int,
        y_offset: int,
        is_tone: bool = False
) -> Optional[Tuple[int, int, int, int]]:
    """Core pixel-level bbox measurement (With Noise Thresholding)."""
    if not component_char or not base_char:
        return None

    try:
        composite = base_char + component_char
        comp_left, comp_top, comp_right, comp_bottom = image_font.getbbox(composite)
        base_left, base_top, base_right, base_bottom = image_font.getbbox(base_char)
    except Exception:
        return None

    # Mask Matching
    base_mask = image_font.getmask(base_char)
    base_pixels = np.array(base_mask).reshape(base_mask.size[1], base_mask.size[0])

    comp_mask = image_font.getmask(composite)
    comp_pixels = np.array(comp_mask).reshape(comp_mask.size[1], comp_mask.size[0])

    max_h = max(base_pixels.shape[0], comp_pixels.shape[0])
    max_w = max(base_pixels.shape[1], comp_pixels.shape[1])

    base_offset_y = base_top - comp_top

    base_padded = np.zeros((max_h, max_w), dtype=base_pixels.dtype)
    comp_padded = np.zeros((max_h, max_w), dtype=comp_pixels.dtype)

    if base_offset_y >= 0:
        end_y = min(base_offset_y + base_pixels.shape[0], max_h)
        base_padded[base_offset_y:end_y, :base_pixels.shape[1]] = base_pixels[:end_y - base_offset_y, :]
    else:
        start_y = -base_offset_y
        base_padded[:base_pixels.shape[0] - start_y, :base_pixels.shape[1]] = base_pixels[start_y:, :]

    comp_padded[:comp_pixels.shape[0], :comp_pixels.shape[1]] = comp_pixels

    diff_pixels = (comp_padded.astype(int) - base_padded.astype(int))

    max_diff = diff_pixels.max() if diff_pixels.size > 0 else 0
    noise_threshold = max(20, int(max_diff * 0.2))

    tone_pixels = diff_pixels > noise_threshold

    tone_rows, tone_cols = np.where(tone_pixels)

    if len(tone_rows) == 0:
        return None

    font_size = image_font.size
    gap_threshold = max(1, int(font_size * 0.025))

    comp_rows_all = np.where(comp_padded > 0)[0]
    comp_rows_unique = sorted(set(comp_rows_all.tolist()))
    gap_end = comp_rows_unique[-1]

    for i in range(len(comp_rows_unique) - 1):
        gap = comp_rows_unique[i + 1] - comp_rows_unique[i]
        if gap > gap_threshold and comp_rows_unique[i] < base_offset_y:
            gap_end = comp_rows_unique[i]
            break

    tone_mask = tone_rows <= gap_end
    tone_rows = tone_rows[tone_mask]
    tone_cols = tone_cols[tone_mask]

    if len(tone_rows) == 0:
        return None

    # Calculate Raw BBox
    x1 = x_offset + tone_cols.min() + comp_left
    y1 = y_offset + tone_rows.min() + comp_top
    x2 = x_offset + tone_cols.max() + 1 + comp_left
    y2 = y_offset + tone_rows.max() + 1 + comp_top

    if is_tone:
        padding = max(1, int(font_size * 0.02))  # Dynamic Padding

        x1 = x1 - padding
        x2 = x2 + padding
        y2 = y2 + padding

    return (x1, y1, x2, y2)


def _measure_leading(
        image_font: ImageFont,
        grapheme: str,
        leading_char: str,
        x_offset: int,
        y_offset: int
) -> Optional[Tuple[int, int, int, int]]:
    """Measure leading vowel bbox using font masks."""
    if not leading_char or not grapheme:
        return None

    full_left, full_top, full_right, full_bottom = image_font.getbbox(grapheme)

    full_mask = image_font.getmask(grapheme)
    full_pixels = np.array(full_mask).reshape(full_mask.size[1], full_mask.size[0])

    grapheme_without_leading = grapheme.replace(leading_char, '')

    if not grapheme_without_leading:
        rows, cols = np.where(full_pixels > 0)
        if len(rows) == 0:
            return None

        x1 = x_offset + cols.min() + full_left
        y1 = y_offset + rows.min() + full_top
        x2 = x_offset + cols.max() + 1 + full_left
        y2 = y_offset + rows.max() + 1 + full_top
        return (x1, y1, x2, y2)

    no_lead_mask = image_font.getmask(grapheme_without_leading)
    no_lead_pixels = np.array(no_lead_mask).reshape(no_lead_mask.size[1], no_lead_mask.size[0])

    leading_pixels = np.logical_and(full_pixels > 0, no_lead_pixels == 0)
    rows, cols = np.where(leading_pixels)

    if len(rows) == 0:
        return None

    x1 = x_offset + cols.min() + full_left
    y1 = y_offset + rows.min() + full_top
    x2 = x_offset + cols.max() + 1 + full_left
    y2 = y_offset + rows.max() + 1 + full_top
    return (x1, y1, x2, y2)


def _measure_sara_am(
        image_font: ImageFont,
        full_char: str,
        x_offset: int,
        y_offset: int,
        base_char_to_subtract: str = "",
        cut_out_bboxes: List[Optional[Tuple[int, int, int, int]]] = None
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    Measure sara am by subtracting Base and masking out known Tone/Vowel bboxes.
    """
    if not full_char:
        return None, None

    # 1. Render Full Image (น้ำ)
    try:
        full_left, full_top, full_right, full_bottom = image_font.getbbox(full_char)
        full_mask = image_font.getmask(full_char)
        full_pixels = np.array(full_mask).reshape(full_mask.size[1], full_mask.size[0])
    except:
        return None, None

    # 2. Render Base Image (น) เพื่อลบออก
    sub_pixels = None
    sub_top, sub_left = full_top, full_left

    if base_char_to_subtract:
        try:
            sub_mask = image_font.getmask(base_char_to_subtract)
            sub_pixels = np.array(sub_mask).reshape(sub_mask.size[1], sub_mask.size[0])
            sub_left, sub_top, _, _ = image_font.getbbox(base_char_to_subtract)
        except:
            sub_pixels = None

    # 3. เตรียม Canvas
    if sub_pixels is not None:
        max_h = max(full_pixels.shape[0], sub_pixels.shape[0])
        max_w = max(full_pixels.shape[1], sub_pixels.shape[1])
    else:
        max_h, max_w = full_pixels.shape

    full_padded = np.zeros((max_h, max_w), dtype=full_pixels.dtype)
    sub_padded = np.zeros((max_h, max_w), dtype=full_pixels.dtype)

    full_padded[:full_pixels.shape[0], :full_pixels.shape[1]] = full_pixels

    if sub_pixels is not None:
        dy = sub_top - full_top
        dx = sub_left - full_left
        y_start, x_start = max(0, dy), max(0, dx)
        y_end = min(y_start + sub_pixels.shape[0], max_h)
        x_end = min(x_start + sub_pixels.shape[1], max_w)

        sub_h, sub_w = y_end - y_start, x_end - x_start
        if sub_h > 0 and sub_w > 0:
            sub_padded[y_start:y_end, x_start:x_end] = sub_pixels[:sub_h, :sub_w]

    # 4. Subtraction: ลบ Base ออก -> เหลือ Tone + Nikhahit + Sara Aa
    am_pixels = (full_padded.astype(int) - sub_padded.astype(int)) > 0

    # 5. Masking: ลบ Tone/Vowel ออก (ตาม BBox ที่ส่งมา)
    # เราจะระบายสีดำทับลงไปใน am_pixels เลย
    if cut_out_bboxes:
        for bbox in cut_out_bboxes:
            if bbox:
                bx1, by1, bx2, by2 = bbox

                # แปลง Global Coords (bx, by) เป็น Local Coords ของ Full Image
                # Global X = x_offset + full_left + local_c
                # local_c = Global X - x_offset - full_left
                c1 = bx1 - x_offset - full_left
                r1 = by1 - y_offset - full_top
                c2 = bx2 - x_offset - full_left
                r2 = by2 - y_offset - full_top

                # Clip ให้อยู่ในขอบเขตภาพ
                c1, r1 = max(0, c1), max(0, r1)
                c2, r2 = min(max_w, c2), min(max_h, r2)

                if r2 > r1 and c2 > c1:
                    am_pixels[r1:r2, c1:c2] = 0  # ลบวรรณยุกต์ทิ้ง!

    # 6. หา Gap (ตอนนี้เหลือแค่ Nikhahit กับ Sara Aa แล้ว)
    rows, cols = np.where(am_pixels)
    if len(rows) == 0:
        return None, None

    min_row, max_row = rows.min(), rows.max()
    row_sums = np.sum(am_pixels, axis=1)

    gap_row = -1
    found_upper = False

    for r in range(min_row, max_row):
        if row_sums[r] > 0:
            found_upper = True
        if found_upper and row_sums[r] == 0:
            # ต้องมีเนื้อหาข้างล่างด้วย (สระอา)
            if np.sum(row_sums[r + 1:]) > 0:
                gap_row = r
                break

    if gap_row == -1:
        # Fallback logic (ถ้ามันชิดกันมากจนไม่มี gap หรือมีแค่ส่วนเดียว)
        # พยายามตัดครึ่งทางแนวตั้งถ้ามันสูงเกินไป
        gap_row = (min_row + max_row) // 2

    upper_mask = (rows <= gap_row)
    lower_mask = (rows > gap_row)

    upper_rows, upper_cols = rows[upper_mask], cols[upper_mask]
    lower_rows, lower_cols = rows[lower_mask], cols[lower_mask]

    # 7. สร้าง Result BBox
    upper_bbox = None
    if len(upper_rows) > 0:
        upper_bbox = (
            x_offset + upper_cols.min() + full_left,
            y_offset + upper_rows.min() + full_top,
            x_offset + upper_cols.max() + 1 + full_left,
            y_offset + upper_rows.max() + 1 + full_top
        )

    trailing_bbox = None
    if len(lower_rows) > 0:
        trailing_bbox = (
            x_offset + lower_cols.min() + full_left,
            y_offset + lower_rows.min() + full_top,
            x_offset + lower_cols.max() + 1 + full_left,
            y_offset + lower_rows.max() + 1 + full_top
        )

    return upper_bbox, trailing_bbox

def measure_grapheme_bboxes(
        image_font: ImageFont,
        grapheme: str,
        components: Dict,
        x_offset: int,
        y_offset: int
) -> Dict:
    """Measure all bboxes for a Thai grapheme in one call."""
    result = {
        "leading_bbox": None,
        "base_bbox": None,
        "upper_vowel_bbox": None,
        "upper_tone_bbox": None,
        "upper_diacritic_bbox": None,
        "lower_bbox": None,
        "trailing_bbox": None,
    }

    g_normalized = normalize_grapheme(grapheme)

    if components['leading']:
        result["leading_bbox"] = _measure_leading(
            image_font, g_normalized, components['leading'], x_offset, y_offset
        )

    if components['base']:
        base_width = get_text_width(image_font, components['base'])
        left, top, right, bottom = get_text_bbox(image_font, components['base'])
        result["base_bbox"] = (x_offset, y_offset + top, x_offset + base_width, y_offset + bottom)

        # --- [จุดที่ต้องแก้: แทนที่ Block นี้ทั้งหมด] ---
    if components['is_sara_am']:
        # 1. วัดวรรณยุกต์/สระบน ตามปกติ (Logic เดิม)
        if components['upper_tone']:
            # เคล็ดลับ: ส่ง base_with_nikhahit (Base+วงกลม) ไปวัด เพื่อให้ได้ตำแหน่ง Tone ที่สูงถูกต้อง
            base_with_nikhahit = (components['base'] or '') + NIKHAHIT
            result["upper_tone_bbox"] = _measure_pixels(
                image_font, base_with_nikhahit, components['upper_tone'],
                x_offset, y_offset, is_tone=True
            )

        if components['upper_vowel']:
            # ปกติสระอำไม่ค่อยมี upper vowel ซ้อน แต่เผื่อไว้
            base_with_nikhahit = (components['base'] or '') + NIKHAHIT
            result["upper_vowel_bbox"] = _measure_pixels(
                image_font, base_with_nikhahit, components['upper_vowel'],
                x_offset, y_offset
            )

        # 2. แยกสระอำ (นิคหิต + สระอา)
        # วิธีใหม่: ส่งแค่ Base ไปลบ และส่ง BBox ของ Tone ไป Mask ทิ้ง
        nikhahit_bbox, sara_aa_bbox = _measure_sara_am(
            image_font, g_normalized, x_offset, y_offset,
            base_char_to_subtract=components['base'],
            cut_out_bboxes=[result["upper_tone_bbox"], result["upper_vowel_bbox"]]
        )

        result["upper_diacritic_bbox"] = nikhahit_bbox
        result["trailing_bbox"] = sara_aa_bbox
    else:
        has_vowel = bool(components['upper_vowel'])
        has_tone = bool(components['upper_tone'])
        has_diacritic = bool(components['upper_diacritic'])
        num_upper = sum([has_vowel, has_tone, has_diacritic])

        if num_upper > 1 and components['base']:
            component_list = [
                components['upper_vowel'],
                components['upper_tone'],
                components['upper_diacritic']
            ]
            accumulated_base = components['base']

            for i, comp in enumerate(component_list):
                if comp:
                    is_tone = comp in THAI_TONE_MARKS
                    bbox = _measure_pixels(image_font, accumulated_base, comp, x_offset, 0, is_tone)
                    if bbox:
                        bbox = (bbox[0], bbox[1] + y_offset, bbox[2], bbox[3] + y_offset)

                    if i == 0:
                        result["upper_vowel_bbox"] = bbox
                    elif i == 1:
                        result["upper_tone_bbox"] = bbox
                    else:
                        result["upper_diacritic_bbox"] = bbox

                    accumulated_base = accumulated_base + comp

        elif num_upper == 1:
            if has_vowel:
                if components['base']:
                    result["upper_vowel_bbox"] = _measure_pixels(
                        image_font, components['base'], components['upper_vowel'],
                        x_offset, y_offset
                    )
                else:
                    left, top, right, bottom = get_text_bbox(image_font, components['upper_vowel'])
                    result["upper_vowel_bbox"] = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

            if has_tone:
                if components['base']:
                    result["upper_tone_bbox"] = _measure_pixels(
                        image_font, components['base'], components['upper_tone'],
                        x_offset, y_offset, is_tone=True
                    )
                else:
                    left, top, right, bottom = get_text_bbox(image_font, components['upper_tone'])
                    result["upper_tone_bbox"] = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

            if has_diacritic:
                if components['base']:
                    result["upper_diacritic_bbox"] = _measure_pixels(
                        image_font, components['base'], components['upper_diacritic'],
                        x_offset, y_offset
                    )
                else:
                    left, top, right, bottom = get_text_bbox(image_font, components['upper_diacritic'])
                    result["upper_diacritic_bbox"] = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

    if components['lower']:
        if components['base']:
            result["lower_bbox"] = _measure_pixels(
                image_font, components['base'], components['lower'],
                x_offset, y_offset
            )
        else:
            left, top, right, bottom = get_text_bbox(image_font, components['lower'])
            result["lower_bbox"] = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

    if components['trailing'] and not components['is_sara_am']:
        trailing_left, trailing_top, trailing_right, trailing_bottom = get_text_bbox(
            image_font, components['trailing']
        )
        base_width = get_text_width(image_font, components['base']) if components['base'] else 0
        result["trailing_bbox"] = (
            x_offset + base_width,
            y_offset + trailing_top,
            x_offset + base_width + (trailing_right - trailing_left),
            y_offset + trailing_bottom
        )

    return result