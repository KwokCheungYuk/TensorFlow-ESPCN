import time

import h5py
import os.path
import numpy as np
import tensorflow as tf
import PSNR
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

BACTH_SIZE = 50

EPOCH = 1000

def get_epoch():
    return str(EPOCH)

def get_batch_size():
    return str(BACTH_SIZE)


"""
 定义权值函数
 shape=[filter_size, filter_size, channel_number, kernel_number]
"""
def weight_variable(name2, shape2):
    initializer2 = tf.compat.v1.truncated_normal_initializer(stddev=0.1)
    return tf.compat.v1.get_variable(name=name2, shape=shape2, initializer=initializer2)


"""
 定义偏置函数
 shape=[kernel_number]
"""
def bias_variable(shape2, name2):
    initializer2 = tf.constant_initializer(0.0)
    return tf.compat.v1.get_variable(name=name2, shape=shape2, initializer=initializer2)


"""
 定义卷积操作函数
 x为卷积层的输入数据
 weight为卷积参数
 x_move、y_move分别是在x方向、y方向上的步长
 bias 为偏置函数
"""
def conv2d(x, weight, x_move, y_move, bias):
    return tf.nn.bias_add(tf.nn.conv2d(x, weight, strides=[1, x_move, y_move, 1], padding='SAME'), bias)


class ESPCN:
    def __init__(self, session, img_height, img_width, img_channel, ratio, is_train):
        self.session = session
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.ratio = ratio
        # 用于判断用户是想训练模型还是运用模型
        self.is_train = is_train

        self.batch_size = BACTH_SIZE
        # 根据论文给出的实现细节
        self.epoch = EPOCH
        self.learning_rate = 0.00001
        # 训练数据所在路径
        self.train_folder = "train_data"
        self.train_data_path = self.train_folder + '/train_data_' + str(self.ratio) + '.h5'
        
        # checkpoint路径
        self.check_folder = "checkpoint"

        # 创建神经网络
        # shape: [图片数量，高度，宽度，通道]
        self.images = tf.compat.v1.placeholder(
            tf.float32, [None, self.img_height, self.img_width,
                         self.img_channel],
            name="input_data")
        self.labels = tf.compat.v1.placeholder(
            tf.float32, [None, self.img_height * self.ratio,
                         self.img_width * self.ratio, self.img_channel],
            name="labels")
        self.prediction = self.inference()
        # 根据论文loss使用均方误差
        self.loss = tf.reduce_mean(tf.square(self.labels - self.prediction))
        self.saver = tf.compat.v1.train.Saver()


    """
     定义卷积神经网络
    """
    def inference(self):
        """
        第一层卷积
        根据论文，filter为5 * 5 * 64
        输入：img_height * img_width * img_channe
        输出: img_height * img_width * 64
        """
        with tf.compat.v1.variable_scope("conv1", reuse=tf.compat.v1.AUTO_REUSE):
            conv1_weights = weight_variable(name2="weights1", shape2=[5, 5, self.img_channel, 64])
            conv1_bias = bias_variable(name2="bias1", shape2=[64])
            conv1_temp = conv2d(self.images, conv1_weights, 1, 1, conv1_bias)
            # 根据论文，使用tanh激活函数
            conv1 = tf.nn.relu(conv1_temp)

        """
        第二层卷积
        根据论文，filter为3 * 3 * 32
        输入：img_height * img_width * 64
        输出: img_height * img_width * 32
        """
        with tf.compat.v1.variable_scope("conv2", reuse=tf.compat.v1.AUTO_REUSE):
            conv2_weights = weight_variable(name2="weights2", shape2=[3, 3, 64, 32])
            conv2_bias = bias_variable(name2="bias2", shape2=[32])
            conv2_temp = conv2d(conv1, conv2_weights, 1, 1, conv2_bias)
            # 根据论文，使用tanh激活函数
            conv2 = tf.nn.relu(conv2_temp)

        """
        第三层卷积
        根据论文，filter为3 * 3 * (ratio * ratio * img_channe)
        输入：img_height * img_width * 32
        输出: img_height * img_width * (ratio * ratio * img_channe)
        """
        with tf.compat.v1.variable_scope("conv3", reuse=tf.compat.v1.AUTO_REUSE):
            conv3_weights = weight_variable(name2="weights3",
                                            shape2=[3, 3, 32, self.ratio * self.ratio * self.img_channel])
            conv3_bias = bias_variable(name2="bias3", shape2=[self.ratio * self.ratio * self.img_channel])
            conv3_temp = conv2d(conv2, conv3_weights, 1, 1, conv3_bias)

        output = tf.nn.relu(self.rearrange_img(conv3_temp, self.ratio))
        return output


    """
     应用时候的移动
    """
    def apply_shuffle(self, sub_tensor, ratio):
        # 图片数量，高度，宽度，通道
        batch_size, row_size, col_size, channel_size = sub_tensor.get_shape().as_list()
        temp = tf.reshape(sub_tensor, (1, row_size, col_size, ratio, ratio))
        """
        按第二维度（高度）分割张量,得到row_size个子张量
        子张量shape = [1, 1, col_size, ratio, ratio]     
        """
        temp_sub_tensors = tf.split(temp, row_size, 1)
        """
        row_size个张量在第四维度拼接张量
        shape = [1, 1, col_size, row_size * ratio, ratio]
        """
        temp = tf.concat([t for t in temp_sub_tensors], 3)
        """
        按第三维度（宽度）分割张量,得到col_size个子张量
        子张量shape = [1, 1, 1, row_size * ratio, ratio]     
        """
        temp_sub_tensors = tf.split(temp, col_size, 2)
        """ 
        col_size个张量在第五维度拼接张量
        shape =  [1, 1, 1, row_size * ratio, ratio * col_size]   
        """
        temp = tf.concat([t for t in temp_sub_tensors], 4)
        return tf.reshape(temp, (1, row_size * ratio, col_size * ratio, 1))


    """
     训练时候的移动
    """
    def train_shuffle(self, sub_tensor, ratio):
        # 图片数量，高度，宽度，通道数
        batch_size, row_size, col_size, channel_size = sub_tensor.get_shape().as_list()
        temp = tf.reshape(sub_tensor, (self.batch_size, row_size, col_size, ratio, ratio))
        """
         [self.batch_size, row_size, col_size, ratio, ratio]
         => [batch_size, 1, col_size, ratio, ratio]
        """
        temp_sub_tensors = tf.split(temp, row_size, 1)
        """
         [batch_size, 1, col_size, ratio, ratio]
         => [bsize, 1, col_size, row_size * ratio, ratio]
        """
        temp = tf.concat([t for t in temp_sub_tensors], 3)
        """
         [bsize, 1, col_size, row_size * ratio, ratio]
         => [bsize, 1, 1, row_size * ratio, ratio]
        """
        temp_sub_tensors = tf.split(temp, col_size, 2)
        """
        [bsize, 1, 1, row_size * ratio, ratio]
         => [bsize, 1, 1, row_size * ratio, col_size * ratio]
        """
        temp = tf.concat([t for t in temp_sub_tensors], 4)
        return tf.reshape(temp, (self.batch_size, row_size * ratio, col_size * ratio, 1))


    """ 
     重排列
     x为经过三层卷积后的输出
     ration为放大比例
    """
    def rearrange_img(self, input_tensor, ratio):
        """
         张量切分处理
         第一维：图片张数；第二维：高度；第三维：宽度；第四位：深度
         按第四维度（深度）分割张量，切分成3个子张量R,G,B
         子张量shape = [batch_size, row_size, col_size, ratio * ratio]
        """
        input_sub_tensors = tf.split(input_tensor, 3, 3)
        # 训练的时候批处理
        if self.is_train:
            result = tf.concat([self.train_shuffle(s, ratio) for s in input_sub_tensors], 3)
        else:
            result = tf.concat([self.apply_shuffle(s, ratio) for s in input_sub_tensors], 3)
        return result


    """
    加载训练模型
    """
    def load_checkpoint(self):
        model_folder = "ratio_" + str(self.ratio) + "_" + str(self.epoch) + "_" + str(self.batch_size)+ "_relu"
        checkpoint_folder = os.path.join(self.check_folder, model_folder)
        check_state = tf.train.get_checkpoint_state(checkpoint_folder)
        if check_state and check_state.model_checkpoint_path:
            checkpoint_path = str(check_state.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(os.getcwd(), checkpoint_path))
            print("Checkpoint [from: %s] load succeeded.\n" % checkpoint_path)
        else:
            print("There is no Checkpoint!")


    """
     保存训练模型
    """
    def save_checkpoint(self, count):
        model_name = "ESPCN.model"
        model_folder = "ratio_" + str(self.ratio) + "_" + str(self.epoch) + "_" + str(self.batch_size) + "_relu"
        checkpoint_folder = os.path.join(self.check_folder, model_folder)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.saver.save(self.session, os.path.join(checkpoint_folder, model_name), global_step=count)

    """
     加载训练数据
    """
    def load_train_data(self):
        with h5py.File(self.train_data_path, 'r') as hf:
            input_data = np.array(hf.get("input"))
            label = np.array(hf.get("label"))
        return input_data, label


    """
     训练模型
    """
    def train(self):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_vars = tf.compat.v1.trainable_variables()
        self.train_op = optimizer.minimize(loss=self.loss, var_list=train_vars)
        # 初始化变量
        tf.compat.v1.global_variables_initializer().run()
        self.load_checkpoint()
        input_data, label = self.load_train_data()
        input_size = len(input_data)
        count = 0
        current_time = time.time()
        loss_mean_arr = []
        for e in range(self.epoch):
            loss_value_arr = []
            for batch_num in range(input_size // self.batch_size):
                batch_images = input_data[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                batch_labels = label[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                if len(batch_images) != self.batch_size:
                    continue
                _, loss_value = self.session.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                loss_value_arr.append(loss_value)
                count += 1
                if count % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time consumed: [%4.4f], loss value: [%.8f]" % (
                        (e + 1), count, time.time() - current_time, loss_value))
                # 定期保存模型
                if count % 500 == 0:
                    self.save_checkpoint(count)

            loss_mean_arr.append(np.sum(loss_value_arr) / len(loss_value_arr))

        # loss绘图
        plt.figure()
        plt.plot(loss_mean_arr, 'b-.')
        plt.ylabel('Average loss')
        plt.xlabel('Epoch')
        plt.title("Loss per Epoch")
        if not os.path.exists("loss_images"):
            os.makedirs("loss_images")
        save_path = 'loss_images/loss_' + str(self.ratio) + '_' + str(self.epoch) + '_' +str(self.batch_size) + '_relu.png'
        plt.savefig(save_path) 


    """
     应用模型
    """
    def apply_model(self, lr_image):
        self.load_checkpoint()
        output_arr = self.prediction.eval({self.images: lr_image.reshape(1, self.img_height, self.img_width, self.img_channel)})
        output_arr = np.where(output_arr < 0, 0, output_arr)
        sr_image = np.squeeze(output_arr) * 255.
        return np.uint8(sr_image)
