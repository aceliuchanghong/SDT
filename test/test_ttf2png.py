# -*- coding: UTF-8 -*-
import os
import fontforge
import json
import re
import concurrent.futures


def convert_ttf_to_png(json_path, font_paths, output_dir, sample_count=6):
    """
    Convert TTF files to PNG images based on characters specified in a JSON file.

    :param json_path: Path to the JSON file containing character sets.
    :param font_paths: List of paths to TTF files.
    :param output_dir: Directory to save the generated PNG images.
    :param sample_count: Number of characters to generate per font file.
    """

    def process_font(font_path):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        print(f"Processing file: {font_path}")

        font = fontforge.open(font_path)  # Open the font file
        font.em = 256
        output_subdir = os.path.join(output_dir, font_name)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        with open(json_path, 'r', encoding='utf-8') as f:
            cjk = json.load(f)
        cn_charset = cjk["gbk"]

        count = 0
        for c in cn_charset:
            if count < sample_count:
                print("pic_ing on:", c)
                try:
                    glyph = font[ord(c)]  # Get the glyph for the character
                    glyph.export(os.path.join(output_subdir, f"{c}.png"), 255)
                    count += 1
                except Exception as e:
                    print(f"Glyph not found for character {c}: {e}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_font, font_paths)


if __name__ == '__main__':
    json_path = r'D:\aProject\py\SDT\test\txt9169.json'
    root_dir = r'D:\aProject\py\SDT\test\font_dir'
    output_dir = './LYJ300-1223'
    sample_count = 6

    font_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.ttf')]
    convert_ttf_to_png(json_path, font_paths, output_dir, sample_count)
