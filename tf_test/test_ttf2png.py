# -*- coding: UTF-8 -*-
import os
import fontforge
import json
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_ttf_to_png(json_path, font_paths, output_dir, sample_count=6):
    def process_font(font_path):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        logger.info(f"Processing file: {font_path}")
        try:
            try:
                font = fontforge.open(font_path)  # Open the font file
            except Exception as e:
                logger.error(f"Error open font {font_path}: {e}")
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
                    logger.info(f"the {count + 1} pic_ing on:{c}")
                    try:
                        glyph = font[ord(c)]  # Get the glyph for the character
                        glyph.export(os.path.join(output_subdir, f"{c}.png"), 255)
                        count += 1
                    except Exception as e:
                        logger.error(f"Glyph not found for character {c}: {e}")
        except Exception as e:
            logger.error(f"Error processing font {font_path}: {e}")

    # Ensure all paths in font_paths are absolute
    absolute_font_paths = [os.path.abspath(path) for path in font_paths]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_font, absolute_font_paths)
    # for font_path in absolute_font_paths:
    #     process_font(font_path)


def get_ttf_files(directory):
    """
    递归地遍历指定目录及其子目录，找到所有 .ttf 文件，并返回这些文件的绝对路径列表。

    :param directory: 要遍历的目录路径
    :return: 所有 .ttf 文件的绝对路径列表
    """
    ttf_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ttf'):
                ttf_files.append(os.path.abspath(os.path.join(root, file)))

    return ttf_files


if __name__ == '__main__':
    """
    ffpython D:\\aProject\\py\\SDT\\tf_test\\test_ttf2png.py
    """
    json_path = r'D:\aProject\py\SDT\tf_test\txt9169.json'
    output_dir = './LCH_pics'

    ttf_dir = r'D:\download\Chinese-Fonts-Dataset-main\Chinese-Fonts-Dataset-main\ttf格式\衬线体\仿宋'
    sample_count = 10000

    ttf_list = [
        r'D:\aProject\py\SDT\z_new_start\generate_utils\LXGWWenKaiGB-Light.ttf'
    ]
    # ttf_list = get_ttf_files(ttf_dir)

    convert_ttf_to_png(json_path, ttf_list, output_dir, sample_count)
