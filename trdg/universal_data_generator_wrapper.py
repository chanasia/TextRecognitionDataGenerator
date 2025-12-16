import os
import random
import json
import numpy as np
from PIL import Image, ImageFilter

from trdg.computer_text_generator import generate
from trdg.transform_utils import apply_rotation, apply_curve


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class UniversalDataGeneratorWrapper:
    @classmethod
    def generate_from_tuple(cls, t):
        """
        Wrapper function to handle generation + transformation (rotation) properly for Thai BBoxes.
        Matches the tuple structure from run.py
        """
        (
            index, text, font, out_dir, size, extension, skew_angle, random_skew,
            blur, random_blur, background_type, distorsion, distorsion_orientation,
            handwritten, name_format, width, alignment, text_color, orientation,
            space_width, character_spacing, margins, fit, output_mask, word_split,
            image_dir, stroke_width, stroke_fill, image_mode, output_bboxes, output_coco
        ) = t

        # 1. Generate Plain Image
        image, mask, char_positions = generate(
            text, font, text_color, size, orientation, space_width,
            character_spacing, fit, word_split, stroke_width, stroke_fill
        )

        # 2. Apply Rotation (Skew)
        angle = 0
        if skew_angle != 0 or random_skew:
            if random_skew:
                angle = random.randint(0 - skew_angle, skew_angle)
            else:
                angle = skew_angle

            image, mask, char_positions = apply_rotation(image, mask, char_positions, angle)

        if distorsion == 3:
            # ใช้ distorsion_orientation (-do) เป็นตัวกำหนดความสูงของโค้ง (เช่น 20, 30 px)
            curve_amount = distorsion_orientation if distorsion_orientation > 0 else 20
            image, mask, char_positions = apply_curve(image, mask, char_positions, amplitude=curve_amount)

        # 3. Apply Blur
        final_blur = 0
        if blur > 0:
            final_blur = blur
            if random_blur:
                final_blur = random.randint(0, blur)

        if final_blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=final_blur))

        # 4. Apply Background
        if background_type == 0:
            # Gaussian Noise
            noise = Image.effect_noise(image.size, random.randint(30, 60))
            noise = noise.convert(image.mode)
            image = Image.blend(image, noise, 0.15)
        elif background_type == 1:
            # Plain White
            pass
        elif background_type == 2:
            # Quasicrystal (เลียนแบบลายกระดาษ/Texture)
            # ในที่นี้ขอใช้ Noise เบาๆ แทนเพื่อความง่าย หรือข้ามไป
            pass

        # Create final image with background
        final_image = Image.new("RGB", image.size, (255, 255, 255))
        final_image.paste(image, (0, 0), mask=image)

        # 5. Save Output
        if name_format == 0:
            name = f"{text}_{index}.{extension}"
        elif name_format == 1:
            name = f"{index}_{text}.{extension}"
        elif name_format == 2:
            name = f"{index}.{extension}"
        else:
            name = f"{text}_{index}.{extension}"

        valid_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in ('_', '-', '.', ' ')])
        image_path = os.path.join(out_dir, valid_name)

        final_image.save(image_path)

        # Save Metadata (COCO Generator will pick this up)
        metadata_filename = image_path.rsplit('.', 1)[0] + "_metadata.json"

        metadata = {
            "file_name": valid_name,
            "text": text,
            "width": final_image.width,
            "height": final_image.height,
            "image_id": index,
            "char_bboxes": [cp['bbox'] for cp in char_positions if 'bbox' in cp],
            "char_positions": char_positions
        }

        with open(metadata_filename, "w", encoding="utf8") as f:
            json.dump(metadata, f, ensure_ascii=False, cls=NumpyEncoder)

        return