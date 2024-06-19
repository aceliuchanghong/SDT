# -*- coding: UTF-8 -*-
import os
import fontforge
import json
import re

jsonPath = r'D:\aProject\py\SDT\test\txt9169.json'

rootdir = r'D:\aProject\py\SDT\test\font_dir'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
for i in range(0, len(list)):

    alphas = re.findall(r"\D+", list[i])[0].split(".")[0]

    path = os.path.join(rootdir, list[i])
    print(path)
    if os.path.isfile(path):
        font = fontforge.open(path)  # Open a font
        font.em = 256
        filePath = './LYJ300-1223/' + alphas
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        cjk = json.load(open(jsonPath))
        CN_CHARSET = cjk["gbk"]
        # print(CN_CHARSET)

        count = 0
        sample_count = 6  # 生成几个字

        for c in CN_CHARSET:
            if count <= sample_count:
                count += 1
                try:
                    pen = font[ord(c)]  # 获取字形 unicode 编码 包含此字形的字体
                    pen.export('./LYJ300-1223/' + alphas + '/' + c + '.png', 255)
                except Exception as e:
                    print("None", e)
