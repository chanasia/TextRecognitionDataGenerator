import math
import numpy as np
from PIL import Image
import random


def rotate_point(point, angle, center):
    """Rotate a point (x, y) around a center (cx, cy) by a given angle (degrees)."""
    angle_rad = math.radians(angle)
    x, y = point
    cx, cy = center

    # Translate to origin
    x -= cx
    y -= cy

    # Rotate
    new_x = x * math.cos(angle_rad) + y * math.sin(angle_rad)
    new_y = -x * math.sin(angle_rad) + y * math.cos(angle_rad)

    # Translate back
    new_x += cx
    new_y += cy

    return new_x, new_y


def rotate_bbox(bbox, angle, old_center, new_center):
    """Rotate a bbox (x1, y1, x2, y2) and return the new bounding box enclosing the rotated rectangle."""
    x1, y1, x2, y2 = bbox
    points = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    ]

    # หมุนจุดทั้ง 4 ของ bbox
    rotated_points = [rotate_point(p, angle, old_center) for p in points]

    # ปรับ offset ตาม center ใหม่ (เพราะภาพขนาดเปลี่ยนหลังหมุน)
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]

    final_points = [(x + dx, y + dy) for x, y in rotated_points]

    # หา min/max ใหม่เพื่อสร้าง bbox สี่เหลี่ยมผืนผ้า
    xs = [p[0] for p in final_points]
    ys = [p[1] for p in final_points]

    return (min(xs), min(ys), max(xs), max(ys))


def apply_rotation(img, mask, char_positions, angle):
    """Rotate image, mask, and all bboxes inside char_positions."""
    if angle == 0:
        return img, mask, char_positions

    # 1. Rotate Image & Mask (Expand=True เพื่อไม่ให้ภาพขาด)
    # PIL rotate เป็น Counter-Clockwise แต่ TRDG ปกติรับค่าเป็น Clockwise (หรือกลับกัน)
    # เราใช้ angle ตรงๆ ถ้าหมุนผิดทางให้ใส่ -angle
    old_w, old_h = img.size
    old_center = (old_w / 2, old_h / 2)

    rotated_img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    rotated_mask = mask.rotate(angle, expand=True, resample=Image.NEAREST)

    new_w, new_h = rotated_img.size
    new_center = (new_w / 2, new_h / 2)

    # 2. Update BBoxes
    new_char_positions = []

    for char_pos in char_positions:
        new_pos = char_pos.copy()

        # หมุน BBox หลัก (Grapheme)
        if 'bbox' in char_pos:
            new_pos['bbox'] = rotate_bbox(char_pos['bbox'], angle, old_center, new_center)

        # หมุน BBox ย่อย (Components: Base, Tone, Vowel...)
        component_keys = [
            'base_bbox', 'leading_bbox', 'upper_vowel_bbox',
            'upper_tone_bbox', 'upper_diacritic_bbox',
            'lower_bbox', 'trailing_bbox'
        ]

        for key in component_keys:
            if char_pos.get(key):
                new_pos[key] = rotate_bbox(char_pos[key], angle, old_center, new_center)

        new_char_positions.append(new_pos)

    return rotated_img, rotated_mask, new_char_positions


def apply_curve(img, mask, char_positions, amplitude=20):
    """ทำให้ภาพโค้งเป็นรูปภูเขา (Arch) พร้อมแก้ BBox"""
    if amplitude == 0: return img, mask, char_positions

    w, h = img.size
    # คำนวณความสูงใหม่ (เผื่อที่ให้ส่วนโค้ง)
    new_h = h + abs(amplitude)

    # สร้างภาพเปล่า
    new_img = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
    new_mask = Image.new("L", (w, new_h), 0)

    img_arr = np.array(img)
    mask_arr = np.array(mask)

    # สูตร Sine Wave (0 -> 1 -> 0)
    x_coords = np.arange(w)
    y_offsets = (amplitude * np.sin(x_coords * np.pi / w)).astype(int)

    # ย้าย Pixel ขึ้น/ลง
    for x in range(w):
        off = y_offsets[x]
        if off < 0: off = 0
        src_len = min(h, new_h - off)

        # Shift pixels down/up
        new_img.paste(img.crop((x, 0, x + 1, src_len)), (x, off))
        new_mask.paste(mask.crop((x, 0, x + 1, src_len)), (x, off))

    # แก้ไข BBox
    new_chars = []
    for char in char_positions:
        new_c = char.copy()

        # ฟังก์ชันย่อยคำนวณ BBox ใหม่
        def shift_bbox(bbox):
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cx = max(0, min(w - 1, cx))
            off = y_offsets[cx]  # ใช้ offset ตรงกึ่งกลางกล่อง
            return (x1, y1 + off, x2, y2 + off)

        if 'bbox' in new_c: new_c['bbox'] = shift_bbox(new_c['bbox'])

        for k in ['base_bbox', 'leading_bbox', 'upper_vowel_bbox', 'upper_tone_bbox', 'lower_bbox', 'trailing_bbox',
                  'upper_diacritic_bbox']:
            if new_c.get(k): new_c[k] = shift_bbox(new_c[k])

        new_chars.append(new_c)

    return new_img, new_mask, new_chars