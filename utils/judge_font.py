import os
import pickle
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import numpy as np
from utils.check_db import excute_sqlite_sql
from utils.config import create_table_sql, table_select_nums_sql

path_test = '../data/CASIA_CHINESE/test_style_samples'
path_train = '../data/CASIA_CHINESE/train_style_samples'
show_num_img = 2


def num2label(nums):
    label = {
        '0': '草书',
        '1': '楷书',
    }
    if nums in ['0', '1']:
        return label[nums]
    else:
        return 'DK'


def get_user_input():
    while True:
        user_input = input("是否楷书? (1==>是,0==>不是):")
        if user_input in ['0', '1']:
            return user_input
        else:
            print("输入错误,仅支持0或者1,请重新输入评价。")


def get_files(path, suffix):
    files_with_suffix = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                files_with_suffix.append(os.path.join(root, file).replace("\\", '/'))
    return files_with_suffix


def update_file_pkl(file_type):
    the_files = get_files(file_type, 'pkl')
    new_down = gr.Dropdown(choices=the_files)
    return new_down


def select_file_txt(pkl_file):
    file_name = pkl_file.split('/')[-1]
    pkl_file_type = pkl_file.split('/')[-2].split('/')[-1]
    pkl_file = pkl_file_type + "/" + file_name

    ans = excute_sqlite_sql(table_select_nums_sql, (pkl_file,), False)
    print(ans)
    if not ans:
        ans = 'xx'
    ans = num2label(ans)
    return ans


def select_file_pics(pkl_file):
    i = 0
    pics = []
    samples = pickle.load(open(pkl_file, 'rb'))
    for item in samples:
        img_array = item['img'].astype(np.uint8)
        pil_image = Image.fromarray(img_array, mode='L')
        pil_image = pil_image.resize((300, 300))
        pics.append(pil_image)
        i += 1
        if i >= show_num_img:
            break
    return pics[0], pics[1]


def select_file_old(file_type, pkl_file):
    the_files = get_files(file_type, 'pkl')
    index = the_files.index(pkl_file)
    if index > 0:
        return the_files[index - 1]
    else:
        return the_files[-1]


def select_file_new(file_type, pkl_file):
    the_files = get_files(file_type, 'pkl')
    index = the_files.index(pkl_file)
    if index < len(the_files) - 1:
        return the_files[index + 1]
    else:
        return the_files[0]


def create_app():
    with gr.Blocks(title="pick fonts") as demo:
        with gr.Row():
            file_type = gr.Dropdown(label='选择路径', choices=[path_test, path_train],
                                    scale=5, info='pkl file path you choose')
            file_pkl = gr.Dropdown(label='选择pkl文件', choices=[], scale=5, info='pkl file list you choose',
                                   interactive=True)
            ans = gr.Textbox(label='标记结果', scale=2, info='pkl file label', interactive=True)
        with gr.Row():
            old_one = gr.Button(value='上一个', variant='secondary', scale=5)
            next_one = gr.Button(value='下一个', variant='secondary', scale=5)
        with gr.Row():
            write_one = gr.Textbox(label='标记', scale=5, placeholder='0')
            submit_button = gr.Button(value='提交', variant='primary', scale=5)
        with gr.Row():
            pic1 = gr.Image(label='文字图片预览1', scale=5, image_mode='L', height=800, width=300)
            pic2 = gr.Image(label='文字图片预览2', scale=5, image_mode='L', height=800, width=300)

        # 添加事件处理器
        file_type.change(fn=update_file_pkl, inputs=file_type, outputs=file_pkl)
        file_pkl.change(fn=select_file_txt, inputs=file_pkl, outputs=ans)
        file_pkl.change(fn=select_file_pics, inputs=file_pkl, outputs=[pic1, pic2])

        old_one.click(fn=select_file_old, inputs=[file_type, file_pkl], outputs=file_pkl)
        next_one.click(fn=select_file_new, inputs=[file_type, file_pkl], outputs=file_pkl)
    return demo


if __name__ == '__main__':
    # test_file = get_files(path_test, 'pkl')
    # for _ in test_file:
    #     i = 0
    #     samples = pickle.load(open(_, 'rb'))
    #     for item in samples:
    #         plt.imshow(item['img'], cmap='gray')
    #         plt.show()
    #         i += 1
    #         if i >= show_num_img:
    #             break
    #     user_choice = get_user_input()
    #     print(user_choice)
    excute_sqlite_sql(create_table_sql)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=12345, share=False)
