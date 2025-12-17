"""
Vector Engine for TrueType/OpenType fonts.
Handles complex Thai glyph composition, ligature splitting, and exact bounding box extraction
using HarfBuzz and FontTools.
"""

import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.misc.transform import Transform
from typing import Tuple, Dict, Optional, List

from trdg.thai_utils import (
    NAME_NIKHAHIT, NAME_SARA_AA, NAME_SARA_AM, NAME_TONES,
    THAI_LEADING_VOWELS, THAI_UPPER_VOWELS, THAI_LOWER_CHARS,
    THAI_TONE_MARKS, THAI_UPPER_DIACRITICS, THAI_TRAILING_VOWELS
)


class FontVectorEngineHB:
    def __init__(self, font_path: str, size: int):
        self.font_path = font_path
        self.size = size
        self.ttfont = TTFont(font_path)
        self.glyph_set = self.ttfont.getGlyphSet()
        self.units_per_em = self.ttfont['head'].unitsPerEm
        self.scale = self.size / self.units_per_em

        # HarfBuzz Setup
        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        self.hb_font = hb.Font(face)
        # HarfBuzz uses 26.6 fixed point format
        self.hb_font.scale = (int(self.size * 64), int(self.size * 64))

    def _to_pixel_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """Convert Design Units to Pixels."""
        if not bbox: return None
        return (bbox[0] * self.scale, bbox[1] * self.scale,
                bbox[2] * self.scale, bbox[3] * self.scale)

    def _merge_bboxes(self, bboxes):
        """Merge multiple bboxes into one enclosing bbox."""
        if not bboxes: return None
        return (min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                max(b[2] for b in bboxes), max(b[3] for b in bboxes))

    def classify_by_name(self, name: str) -> str:
        """Classify glyph role based on its name or Unicode value."""
        n_lower = name.lower()

        # Check Specific Names
        if any(x in n_lower for x in NAME_NIKHAHIT): return "NIKHAHIT"
        if any(x in n_lower for x in NAME_SARA_AA): return "SARA_AA"

        # Try to decode 'uniXXXX'
        unicode_char = None
        if n_lower.startswith("uni") and len(n_lower) >= 7:
            try:
                hex_str = n_lower[3:7]
                unicode_char = chr(int(hex_str, 16))
            except:
                pass

        # Classify by Unicode Category
        if unicode_char:
            if unicode_char in THAI_LEADING_VOWELS: return "LEADING_VOWEL"
            if unicode_char in THAI_UPPER_VOWELS: return "UPPER_VOWEL"
            if unicode_char in THAI_LOWER_CHARS: return "LOWER_VOWEL"
            if unicode_char in THAI_TONE_MARKS: return "TONE"
            if unicode_char == '\u0E4D': return "NIKHAHIT"
            if unicode_char in THAI_UPPER_DIACRITICS: return "UPPER_DIACRITIC"
            if unicode_char in THAI_TRAILING_VOWELS:
                if unicode_char == '\u0E32': return "SARA_AA"
                return "TRAILING_VOWEL"
            if '\u0E01' <= unicode_char <= '\u0E2E': return "BASE"

        if 'tone' in n_lower or 'mai' in n_lower: return "TONE"

        return "BASE"

    def _split_contours_vertical(self, glyph_name: str, threshold_ratio: float = 0.4) -> Tuple[List, List]:
        """Split glyph contours into high and low groups based on Y-coordinate."""
        rec_pen = RecordingPen()
        self.glyph_set[glyph_name].draw(rec_pen)

        contours = []
        current = []
        for cmd, args in rec_pen.value:
            if cmd == 'moveTo':
                if current: contours.append(current)
                current = []
            current.append((cmd, args))
        if current: contours.append(current)

        high_group, low_group = [], []
        threshold = self.units_per_em * threshold_ratio

        for cmds in contours:
            pen = BoundsPen(None)
            for cmd, args in cmds: getattr(pen, cmd)(*args)
            if pen.bounds:
                y_center = (pen.bounds[1] + pen.bounds[3]) / 2
                if y_center > threshold:
                    high_group.append(pen.bounds)
                else:
                    low_group.append(pen.bounds)
        return high_group, low_group

    def _get_component_bbox(self, comp) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        """Get bbox and Y-center of a composite component."""
        if hasattr(comp, 'transform'):
            matrix = comp.transform
        else:
            matrix = [1, 0, 0, 1, comp.x, comp.y]

        t = Transform(matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5])
        pen = BoundsPen(self.glyph_set)
        try:
            self.glyph_set[comp.glyphName].draw(TransformPen(pen, t))
            if pen.bounds:
                return self._to_pixel_bbox(pen.bounds), (pen.bounds[1] + pen.bounds[3]) / 2
        except:
            pass
        return None, 0

    def _handle_sara_am_split(self, glyph_name: str) -> List[Dict]:
        """Handle Explicit Sara Am (uni0E33) decomposition."""
        glyph = self.ttfont['glyf'][glyph_name]
        results = []
        if glyph.isComposite():
            for comp in glyph.components:
                bbox, y_center = self._get_component_bbox(comp)
                if bbox:
                    role = self.classify_by_name(comp.glyphName)
                    if role == "BASE":
                        if y_center > self.units_per_em * 0.4:
                            role = "NIKHAHIT"
                        else:
                            role = "SARA_AA"
                    results.append({"role": role, "bbox": bbox, "name": comp.glyphName})
        else:
            nikhahit_bboxes, sara_aa_bboxes = self._split_contours_vertical(glyph_name, 0.4)
            if nikhahit_bboxes:
                results.append({"role": "NIKHAHIT", "bbox": self._to_pixel_bbox(self._merge_bboxes(nikhahit_bboxes)),
                                "name": glyph_name + "_ni"})
            if sara_aa_bboxes:
                results.append({"role": "SARA_AA", "bbox": self._to_pixel_bbox(self._merge_bboxes(sara_aa_bboxes)),
                                "name": glyph_name + "_aa"})
        return results

    def _handle_ligature_split(self, glyph_name: str, components: List[Dict]) -> List[Dict]:
        """Handle Composite Ligature (e.g. Tone + Nikhahit)."""
        components.sort(key=lambda x: x['y_center_units'])
        is_stacked_mark = (
                    len(components) >= 2 and all(c['y_center_units'] > self.units_per_em * 0.3 for c in components))
        results = []
        if is_stacked_mark:
            c_low = components[0];
            c_low['role'] = 'NIKHAHIT';
            results.append(c_low)
            for c_high in components[1:]: c_high['role'] = 'TONE'; results.append(c_high)
        else:
            for c in components:
                if c['role'] in ['BASE', 'UNKNOWN']:
                    if c['y_center_units'] > self.units_per_em * 0.5:
                        c['role'] = 'TONE'
                    elif c['y_center_units'] < 0:
                        c['role'] = 'LOWER_VOWEL'
                results.append(c)
        return results

    def _handle_simple_ligature_split(self, glyph_name: str) -> List[Dict]:
        """Handle Simple Ligature (merged contours)."""
        high_group, low_group = self._split_contours_vertical(glyph_name, 0.3)
        all_contours = high_group + low_group
        if not all_contours: return []

        contours_with_y = []
        for b in all_contours: contours_with_y.append(((b[1] + b[3]) / 2, b))
        contours_with_y.sort(key=lambda x: x[0])

        results = []
        if len(contours_with_y) >= 2:
            max_gap = 0;
            split_idx = 1
            for i in range(len(contours_with_y) - 1):
                gap = contours_with_y[i + 1][0] - contours_with_y[i][0]
                if gap > max_gap: max_gap = gap; split_idx = i + 1

            nikhahit_parts = [x[1] for x in contours_with_y[:split_idx]]
            tone_parts = [x[1] for x in contours_with_y[split_idx:]]

            if nikhahit_parts: results.append(
                {"role": "NIKHAHIT", "bbox": self._to_pixel_bbox(self._merge_bboxes(nikhahit_parts)),
                 "name": glyph_name + "_ni"})
            if tone_parts: results.append({"role": "TONE", "bbox": self._to_pixel_bbox(self._merge_bboxes(tone_parts)),
                                           "name": glyph_name + "_tn"})
        else:
            results.append(
                {"role": "TONE", "bbox": self._to_pixel_bbox(self._merge_bboxes([x[1] for x in contours_with_y])),
                 "name": glyph_name})
        return results

    # --- Main Interface ---

    def decompose_glyph(self, glyph_name: str) -> List[Dict]:
        """Decompose a glyph into vector components with roles and bounding boxes."""
        if glyph_name not in self.ttfont['glyf']: return []

        # Explicit Sara Am check
        if any(x in glyph_name.lower() for x in NAME_SARA_AM):
            return self._handle_sara_am_split(glyph_name)

        glyph = self.ttfont['glyf'][glyph_name]

        # Composite Glyph
        if glyph.isComposite():
            raw_comps = []
            for comp in glyph.components:
                bbox, y_center = self._get_component_bbox(comp)
                if bbox:
                    raw_comps.append({
                        "name": comp.glyphName,
                        "bbox": bbox,
                        "y_center_units": y_center,
                        "role": self.classify_by_name(comp.glyphName)
                    })
            return self._handle_ligature_split(glyph_name, raw_comps)

        # Simple Glyph
        else:
            # Check for Simple Ligature (Nikhahit+Tone merged)
            is_ligature_name = ("0e4d" in glyph_name.lower() and any(
                t in glyph_name.lower() for t in ["0e48", "0e49", "0e4a", "0e4b"]))
            if is_ligature_name:
                return self._handle_simple_ligature_split(glyph_name)

            # Standard Simple
            pen = BoundsPen(self.glyph_set)
            self.glyph_set[glyph_name].draw(pen)
            if pen.bounds:
                y_center = (pen.bounds[1] + pen.bounds[3]) / 2
                role = self.classify_by_name(glyph_name)
                # Heuristic fallback
                if role == 'BASE':
                    if y_center > self.units_per_em * 0.6:
                        role = 'TONE'
                    elif y_center > self.units_per_em * 0.4:
                        role = 'UPPER_VOWEL'
                    elif y_center < 0:
                        role = 'LOWER_VOWEL'

                return [{"role": role, "bbox": self._to_pixel_bbox(pen.bounds), "name": glyph_name}]
            return []