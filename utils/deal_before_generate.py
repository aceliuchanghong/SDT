import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.judge_font import get_files
import os

# 设置基础目录，文件后缀名，保存路径，集数量和展示图片数量
base_dir = '../style_samples'
suffix = ".jpg"
save_pics_path = 'suit_pics2'
set_nums = 10
show_pics_num = 2

# 确保保存图片的目录存在
if not os.path.exists(save_pics_path):
    os.makedirs(save_pics_path)


def resize_thin_character(pics):
    """
    将输入图片进行处理，提取出字符骨架并显示部分结果。

    参数:
    pics (list): 图片文件路径列表
    """
    length = len(pics)
    index = 0
    for pic in tqdm(pics, total=length):  # 使用tqdm显示处理进度
        # 读取图片为灰度图
        style_img = cv2.imdecode(np.fromfile(pic, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # 调整图片大小为64x64
        fix = cv2.resize(style_img, (64, 64))

        # 反转颜色，使前景为白色，背景为黑色
        fix = cv2.bitwise_not(fix)

        # 使用形态学操作获取骨架
        size = np.size(fix)
        skel = np.zeros(fix.shape, np.uint8)

        # 二值化处理
        ret, fix = cv2.threshold(fix, 127, 255, cv2.THRESH_BINARY)
        # 获取形态学操作的结构元素
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # 使用循环进行骨架提取
        while True:
            open = cv2.morphologyEx(fix, cv2.MORPH_OPEN, element)  # 形态学开操作
            temp = cv2.subtract(fix, open)  # 获取开操作的差异
            eroded = cv2.erode(fix, element)  # 腐蚀操作
            skel = cv2.bitwise_or(skel, temp)  # 合并结果
            fix = eroded.copy()  # 更新fix

            if cv2.countNonZero(fix) == 0:  # 如果图片中所有像素都为零，跳出循环
                break

        # 反转骨架图像颜色，使骨架为黑色，背景为白色
        skel = cv2.bitwise_not(skel)

        # 如果需要展示部分结果图片
        if show_pics_num > index:
            plt.imshow(skel, cmap='gray')
            plt.show()
        # 保存处理后的图片
        save_path = os.path.join(save_pics_path, f'skel_{index}.jpg')
        cv2.imwrite(save_path, skel)
        index += 1


if __name__ == '__main__':
    # 获取文件列表
    files_list = get_files(base_dir, suffix)
    # 调用函数处理图片
    resize_thin_character(files_list)
