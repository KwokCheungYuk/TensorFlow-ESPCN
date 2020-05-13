import glob
import os
import h5py
import numpy as np

from PIL import Image

# 原始训练图片的文件夹
FOLDER_PATH = "images/BSDS500_train1"
# 训练图片大小
INPUT_IMG_SIZE = 17
# 放大倍数
RATIO = 2
# LR裁剪步长
STRIDE = 5
# 图片通道数
IMAGE_CHANNEL = 3


""" 
 RGB与YCbCr互转
"""
mat = np.array(
    [[ 65.481, 128.553, 24.966 ],
     [-37.797, -74.203, 112.0  ],
     [  112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])
 
def rgb2ycbcr(rgb_img_data):
    ycbcr_img_data = np.zeros(rgb_img_data.shape)
    for x in range(rgb_img_data.shape[0]):
        for y in range(rgb_img_data.shape[1]):
            ycbcr_img_data[x, y, :] = np.round(np.dot(mat, rgb_img_data[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img_data

def ycbcr2rgb(ycbcr_img_data):
    rgb_img_data = np.zeros(ycbcr_img_data.shape, dtype=np.uint8)
    for x in range(ycbcr_img_data.shape[0]):
        for y in range(ycbcr_img_data.shape[1]):
            [r, g, b] = ycbcr_img_data[x,y,:]
            rgb_img_data[x, y, :] = np.maximum(0, np.minimum(255, np.round(np.dot(mat_inv, ycbcr_img_data[x, y, :] - offset) * 255.0)))
    return rgb_img_data

# 对原始图片进行处理
def process_original(img_path):
    original_img = Image.open(img_path)
    lr_img = original_img.resize((int(original_img.size[0] / RATIO), int(original_img.size[1] / RATIO)),
                                     Image.ANTIALIAS)
    hr_img = original_img.resize(
        ((original_img.size[0] // RATIO) * RATIO, (original_img.size[1] // RATIO) * RATIO),
        Image.ANTIALIAS)
    # 论文提到只考虑YCbCr空间
    lr_img_ycbcr_data = rgb2ycbcr(np.asarray(lr_img))
    hr_img_ycbcr_data = rgb2ycbcr(np.asarray(hr_img))
    return np.asarray(lr_img_ycbcr_data), np.asarray(hr_img_ycbcr_data)


# 裁剪图片
def cut_img(img_path_array):
    sub_input_array = []
    sub_label_array = []
    for img_path in img_path_array:
        lr_img, hr_img = process_original(img_path)
        height, width, channel = lr_img.shape
        if channel != IMAGE_CHANNEL:
            continue
        for x in range(0, height - INPUT_IMG_SIZE + 1, STRIDE):
            for y in range(0, width - INPUT_IMG_SIZE + 1, STRIDE):
                sub_lr_img = lr_img[x: x + INPUT_IMG_SIZE, y: y + INPUT_IMG_SIZE]
                sub_lr_img = sub_lr_img.reshape([INPUT_IMG_SIZE, INPUT_IMG_SIZE, IMAGE_CHANNEL])
                # 归一化
                sub_lr_img = sub_lr_img / 255.0
                sub_input_array.append(sub_lr_img)

                hr_x = x * RATIO
                hr_y = y * RATIO
                sub_hr_img = hr_img[hr_x: hr_x + INPUT_IMG_SIZE * RATIO, hr_y: hr_y + INPUT_IMG_SIZE * RATIO]
                sub_hr_img = sub_hr_img.reshape([INPUT_IMG_SIZE * RATIO, INPUT_IMG_SIZE * RATIO, IMAGE_CHANNEL])
                # 归一化
                sub_hr_img = sub_hr_img / 255.0
                sub_label_array.append(sub_hr_img)
    return sub_input_array, sub_label_array


# 创建训练数据
def create_hdf5(input_hdf5, label_hdf5):
    save_folder_path = os.path.join(os.getcwd(), "train_data")
    if not os.path.isdir(save_folder_path):
        os.makedirs(os.path.join(save_folder_path))
    file_path = save_folder_path + '/train_data_' + str(RATIO) + '.h5'
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset("input", data=input_hdf5)
        hf.create_dataset("label", data=label_hdf5)


# 创建输入数据和标签
def create_input_label(data_folder=FOLDER_PATH):
    save_folder_path = os.path.join(os.getcwd(), "train_data")
    file_path = save_folder_path + '/train_data_' + str(RATIO) + '.h5'
    if not os.path.exists(file_path):
        print("---------------------------\nThere is no h5 data file. Now create it.\n---------------------------")
        data_path = os.path.join(os.getcwd(), data_folder)
        img_path_array = glob.glob(data_path + "/*.jpg")
        sub_input_array, sub_label_array = cut_img(img_path_array)

        # 创建hdf5文件
        input_hdf5 = np.asarray(sub_input_array)
        label_hdf5 = np.asarray(sub_label_array)
        create_hdf5(input_hdf5, label_hdf5)
    else:
        print("---------------------------\nThere is h5 data file. Directly use it.\n---------------------------")


if __name__ == "__main__":
    create_input_label()
