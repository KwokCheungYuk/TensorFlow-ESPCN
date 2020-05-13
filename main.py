import numpy as np
import tensorflow as tf
import preprocessing
import os
import glob
import espcn
from PIL import Image
import time
import PSNR

TEST_IMAGE_FOLDER = "images/BSDS500_test"

# 不使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.compat.v1.disable_eager_execution()

def train():
    print("---------------------------\nTrain model starts.\n---------------------------")
    preprocessing.create_input_label()
    with tf.compat.v1.Session() as session:
        network = espcn.ESPCN(session=session, img_height=preprocessing.INPUT_IMG_SIZE,
                        img_width=preprocessing.INPUT_IMG_SIZE,
                        img_channel=preprocessing.IMAGE_CHANNEL,
                        ratio=preprocessing.RATIO, is_train=True)
        network.train()


def rebuild(lr_image_data):
    h, w, c = lr_image_data.shape
    with tf.compat.v1.Session() as session:
        network = espcn.ESPCN(session=session, img_height=h,
                        img_width=w,
                        img_channel=preprocessing.IMAGE_CHANNEL,
                        ratio=preprocessing.RATIO, is_train=False)
        sr_image_data = network.apply_model(lr_image_data / 255.0)
        return sr_image_data

def test():
    test_path = os.path.join(os.getcwd(), TEST_IMAGE_FOLDER)
    imgs_path_array = glob.glob(test_path + "/*.jpg")
    psnr_arr = []
    for img_path in imgs_path_array:
        lr_img_data, hr_img_data = preprocessing.process_original(img_path)
        sr_img_data = rebuild(lr_img_data)
        psnr_arr.append(PSNR.cal_psnr(hr_img_data, sr_img_data))
        """
        lr_path = img_path.split('.')[0] + "_LR." + img_path.split('.')[-1]
        sr_path = img_path.split('.')[0] + "_SR." + img_path.split('.')[-1]
        lr_img_rgb_data = preprocessing.ycbcr2rgb(lr_img_data)
        sr_img_rgb_data = preprocessing.ycbcr2rgb(sr_img_data)
        lr_img_rgb = Image.fromarray(lr_img_rgb_data.astype('uint8'))
        sr_img_rgb = Image.fromarray(sr_img_rgb_data.astype('uint8'))
        lr_img_rgb.save(lr_path)
        sr_img_rgb.save(sr_path)
        """
    psnr_mean = np.sum(psnr_arr) / len(psnr_arr)
    #print(psnr_arr)
    print("This model's PSNR: [%.4f]" % (psnr_mean))
    with open("PSNR.txt", 'a') as file:
        file.write("---------------------------------------------------\n")
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        file.write("Ratio: " +  str(preprocessing.RATIO) + "\n")
        file.write("Activation Function: relu\n")
        file.write("Epoch: " + espcn.get_epoch() + "\tBatch_size: " + espcn.get_batch_size() + "\n")
        file.write("Test images set: " + TEST_IMAGE_FOLDER .split('/')[-1] + "\n")
        file.write("Average PSNR: [%.4f]\n" % (psnr_mean))
        file.write("---------------------------------------------------\n\n\n")


def apply():
    print("Please enter image's path: ", end="")
    img_path = input()
    lr_image = Image.open(img_path)
    lr_ycbcr_data = preprocessing.rgb2ycbcr(np.asarray(lr_image))
    sr_image_data = rebuild(lr_ycbcr_data)
    sr_rgb_data = preprocessing.ycbcr2rgb(sr_image_data)
    sr_path = img_path.split('.')[0] + "_x" + str(preprocessing.RATIO) + "." + img_path.split('.')[-1]
    sr_img_rgb = Image.fromarray(sr_rgb_data.astype('uint8'))
    sr_img_rgb.save(sr_path)


if __name__ == '__main__':
    print("1.Train model.\n2.Test model.\n3.Apply model")
    print("Please enter op: ", end="")
    a = int(input())
    if a == 1:
        train()
        print("---------------------------\nTrain completed.\n---------------------------")
    elif a == 2:
        print("---------------------------\nTest model starts.\n---------------------------")
        test()
    elif a == 3:
        print("1. 2 x magnification.\n2. 3 x magnification.\n3. 5 x magnification.\n4. 7 x magnification.")
        print("Please enter op: ", end="")
        b = int(input())
        if b == 1:
            preprocessing.RATIO = 2
            apply()
        elif b == 2:
            preprocessing.RATIO = 3
            apply()
        elif b == 3:
            preprocessing.RATIO = 5
            apply()
        elif b == 4:
            preprocessing.RATIO = 7
            apply()
        else:
            print("Error op code!")
    else:
        print("Error op code!")
