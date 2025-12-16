import os
import random as rnd
import json

from PIL import Image, ImageFilter, ImageStat

from trdg import computer_text_generator, background_generator, distorsion_generator
from trdg.utils import mask_to_bboxes, make_filename_valid

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        cls.generate(*t)

    @classmethod
    def generate(
            cls,
            index: int,
            text: str,
            font: str,
            out_dir: str,
            size: int,
            extension: str,
            skewing_angle: int,
            random_skew: bool,
            blur: int,
            random_blur: bool,
            background_type: int,
            distorsion_type: int,
            distorsion_orientation: int,
            is_handwritten: bool,
            name_format: int,
            width: int,
            alignment: int,
            text_color: str,
            orientation: int,
            space_width: int,
            character_spacing: int,
            margins: int,
            fit: bool,
            output_mask: bool,
            word_split: bool,
            image_dir: str,
            stroke_width: int = 0,
            stroke_fill: str = "#282828",
            image_mode: str = "RGB",
            output_bboxes: int = 0,
            output_coco: bool = False,
    ) -> Image:
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        char_positions = []
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
            image, mask, char_positions = computer_text_generator.generate(
                text,
                font,
                text_color,
                size,
                orientation,
                space_width,
                character_spacing,
                fit,
                word_split,
                stroke_width,
                stroke_fill,
            )
        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distortion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            if distorted_img.size[1] == 0:
                print(f"Skipping text '{text}' - zero height image")
                return

            resize_ratio = (float(size - vertical_margin) / float(distorted_img.size[1]))
            new_width = int(distorted_img.size[0] * resize_ratio)

            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (new_width, size - vertical_margin), Image.Resampling.NEAREST
            )
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            if distorted_img.size[0] == 0:
                print(f"Skipping text '{text}' - zero width image")
                return
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.Resampling.NEAREST
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background_img = background_generator.image(
                background_height, background_width, image_dir
            )
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        try:
            resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
            background_img_st = ImageStat.Stat(background_img)

            resized_img_px_mean = sum(resized_img_st.mean[:2]) / 3
            background_img_px_mean = sum(background_img_st.mean) / 3

            if abs(resized_img_px_mean - background_img_px_mean) < 15:
                print("value of mean pixel is too similar. Ignore this image")
                print("resized_img_st \n {}".format(resized_img_st.mean))
                print("background_img_st \n {}".format(background_img_st.mean))
                return
        except Exception as err:
            return

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        text_offset_x = margin_left
        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
            text_offset_x = margin_left
        elif alignment == 1:
            text_offset_x = int(background_width / 2 - new_text_width / 2)
            background_img.paste(
                resized_img,
                (text_offset_x, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (text_offset_x, margin_top),
            )
        else:
            text_offset_x = background_width - new_text_width - margin_right
            background_img.paste(
                resized_img,
                (text_offset_x, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (text_offset_x, margin_top),
            )

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        background_img = background_img.convert(image_mode)
        background_mask = background_mask.convert(image_mode)

        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.random() * blur
        )
        final_image = background_img.filter(gaussian_filter)
        final_mask = background_mask.filter(gaussian_filter)

        #####################################
        # Generate name for resulting image #
        #####################################
        # We remove spaces if space_width == 0
        if space_width == 0:
            text = text.replace(" ", "")
        if name_format == 0:
            name = "{}_{}".format(text, str(index))
        elif name_format == 1:
            name = "{}_{}".format(str(index), text)
        elif name_format == 2:
            name = str(index)
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            name = "{}_{}".format(text, str(index))

        name = make_filename_valid(name, allow_unicode=True)
        image_name = "{}.{}".format(name, extension)
        mask_name = "{}_mask.png".format(name)
        box_name = "{}_boxes.txt".format(name)
        tess_box_name = "{}.box".format(name)

        if out_dir is not None:
            final_image.save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                final_mask.save(os.path.join(out_dir, mask_name))
            if output_bboxes == 1:
                bboxes = mask_to_bboxes(final_mask)
                with open(os.path.join(out_dir, box_name), "w") as f:
                    for bbox in bboxes:
                        f.write(" ".join([str(v) for v in bbox]) + "\n")
            if output_bboxes == 2:
                bboxes = mask_to_bboxes(final_mask, tess=True)
                with open(os.path.join(out_dir, tess_box_name), "w") as f:
                    for bbox, char in zip(bboxes, text):
                        f.write(
                            " ".join([char] + [str(v) for v in bbox] + ["0"]) + "\n"
                        )

            if output_coco:
                # Transform char_positions to match final image
                if orientation == 0:
                    resize_ratio = (float(size - vertical_margin) / float(distorted_img.size[1]))
                else:
                    resize_ratio = (float(size - horizontal_margin) / float(distorted_img.size[0]))

                def transform_bbox(bbox, ratio, offset_x, offset_y):
                    """Apply resize and offset to bbox"""
                    if bbox is None:
                        return None
                    return [
                        int(bbox[0] * ratio) + offset_x,
                        int(bbox[1] * ratio) + offset_y,
                        int(bbox[2] * ratio) + offset_x,
                        int(bbox[3] * ratio) + offset_y
                    ]

                char_positions_json = []
                char_bboxes = []

                for pos in char_positions:
                    transformed_pos = {
                        "grapheme": pos["grapheme"],
                        "leading_bbox": transform_bbox(pos.get("leading_bbox"), resize_ratio, text_offset_x,
                                                       margin_top),
                        "base_bbox": transform_bbox(pos.get("base_bbox"), resize_ratio, text_offset_x, margin_top),
                        "upper_vowel_bbox": transform_bbox(pos.get("upper_vowel_bbox"), resize_ratio, text_offset_x,
                                                           margin_top),
                        "upper_tone_bbox": transform_bbox(pos.get("upper_tone_bbox"), resize_ratio, text_offset_x,
                                                          margin_top),
                        "upper_diacritic_bbox": transform_bbox(pos.get("upper_diacritic_bbox"), resize_ratio,
                                                               text_offset_x, margin_top),
                        "lower_bbox": transform_bbox(pos.get("lower_bbox"), resize_ratio, text_offset_x, margin_top),
                        "trailing_bbox": transform_bbox(pos.get("trailing_bbox"), resize_ratio, text_offset_x,
                                                        margin_top),
                        "is_sara_am": pos.get("is_sara_am", False)
                    }
                    char_positions_json.append(transformed_pos)

                    # Extract all non-None bboxes as separate characters
                    if transformed_pos["leading_bbox"]:
                        char_bboxes.append(transformed_pos["leading_bbox"])
                    if transformed_pos["base_bbox"]:
                        char_bboxes.append(transformed_pos["base_bbox"])
                    if transformed_pos["upper_vowel_bbox"]:
                        char_bboxes.append(transformed_pos["upper_vowel_bbox"])
                    if transformed_pos["upper_diacritic_bbox"]:
                        char_bboxes.append(transformed_pos["upper_diacritic_bbox"])
                    if transformed_pos["upper_tone_bbox"]:
                        char_bboxes.append(transformed_pos["upper_tone_bbox"])
                    if transformed_pos["lower_bbox"]:
                        char_bboxes.append(transformed_pos["lower_bbox"])
                    if transformed_pos["trailing_bbox"]:
                        char_bboxes.append(transformed_pos["trailing_bbox"])

                metadata = {
                    "image_id": index,
                    "file_name": image_name,
                    "width": background_width,
                    "height": background_height,
                    "text": text,
                    "char_bboxes": char_bboxes,
                    "char_positions": char_positions_json,
                    "text_offset": (text_offset_x, margin_top)
                }

                metadata_name = "{}_metadata.json".format(name)
                with open(os.path.join(out_dir, metadata_name), "w", encoding="utf8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
        else:
            if output_mask == 1:
                return final_image, final_mask
            return final_image