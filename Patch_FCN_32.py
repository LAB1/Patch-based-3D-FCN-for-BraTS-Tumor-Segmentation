import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import argparse
import sys
import tempfile
import os
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import tempfile
import math
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d_transposed_2X(x, W, output_shape):
    """conv3d_transposed returns a 3d transposed and a 2*2*2 uppooling layer with full stride"""
    return tf.nn.conv3d_transpose(x, W, output_shape, [1, 2, 2, 2, 1], padding='SAME')


def conv3d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    conv = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    return conv


def max_pooling_2X(x):
    """max_pool_2*2*2 downsamples a feature map by 2X."""
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')


"""
def batch_normalization(product, kernel_number,epsilon):

    batch_mean, batch_var = tf.nn.moments(product, [0, 1, 2, 3])
    scale = tf.Variable(tf.ones([kernel_number]))
    beta = tf.Variable(tf.zeros([kernel_number]))
    BN = tf.nn.batch_normalization(product, batch_mean, batch_var, beta, scale, epsilon)
    h_conv = tf.nn.relu(BN)


    return scale, beta, h_conv
"""


def batch_normalization(product, phase):
    bn_layer = tf.contrib.layers.batch_norm(product, center=True, scale=True, is_training=phase)
    return bn_layer


def deepnn_3d(patch, phase):
    """
    build the architecture of the CNN
    input - patch: a 32 * 32 * 32 * 4 image
    output -  y_conv: feature map 32 * 32 * 32 * 5, keep_prob: dropout setting
    """
    keep_prob = tf.placeholder(tf.float32)

    ################################################################################
    # convolutional component -1
    # Conv-1 3*3*3*32
    with tf.name_scope('conv-1'):
        W_conv1_32 = weight_variable([3, 3, 3, 4, 64])
        b_conv1_32 = bias_variable([64])
        product1_32 = conv3d(patch, W_conv1_32) + b_conv1_32
        bn_layer1_32 = batch_normalization(product1_32, phase)
        h_conv1_32 = tf.nn.relu(bn_layer1_32)
        dropout1_32 = tf.nn.dropout(h_conv1_32, keep_prob)
    # Conv-2 5*5*5*32
    with tf.name_scope('conv-2'):
        W_conv2_32 = weight_variable([5, 5, 5, 64, 64])
        b_conv2_32 = bias_variable([64])
        product2_32 = conv3d(dropout1_32, W_conv2_32) + b_conv2_32
        bn_layer2_32 = batch_normalization(product2_32, phase)
        h_conv2_32 = tf.nn.relu(bn_layer2_32)
        dropout2_32 = tf.nn.dropout(h_conv2_32, keep_prob)
    # 2*2 pooling CPU Support 16*16*32
    with tf.name_scope('pool-1'):
        h_pool1_32 = max_pooling_2X(dropout2_32)
    ################################################################################
    # convolutional component 1
    # Conv1 3*3*3*32
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 3, 64, 64])
        b_conv1 = bias_variable([64])
        product1 = conv3d(h_pool1_32, W_conv1) + b_conv1
        bn_layer1 = batch_normalization(product1, phase)
        h_conv1 = tf.nn.relu(bn_layer1)
        dropout1 = tf.nn.dropout(h_conv1, keep_prob)
    # Conv2 5*5*5*32
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 5, 64, 64])
        b_conv2 = bias_variable([64])
        product2 = conv3d(dropout1, W_conv2) + b_conv2
        bn_layer2 = batch_normalization(product2, phase)
        h_conv2 = tf.nn.relu(bn_layer2)
        dropout2 = tf.nn.dropout(h_conv2, keep_prob)
    # 2*2 pooling CPU Support 8*8*32
    with tf.name_scope('pool1'):
        h_pool1 = max_pooling_2X(dropout2)

    ################################################################################
    # convolutional component 2
    # Conv3 3*3*3*32
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        product3 = conv3d(h_pool1, W_conv3) + b_conv3
        bn_layer3 = batch_normalization(product3, phase)
        h_conv3 = tf.nn.relu(bn_layer3)
        dropout3 = tf.nn.dropout(h_conv3, keep_prob)
    # Conv4 3*3*3*32
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 3, 64, 64])
        b_conv4 = bias_variable([64])
        product4 = conv3d(dropout3, W_conv4) + b_conv4
        bn_layer4 = batch_normalization(product4, phase)
        h_conv4 = tf.nn.relu(bn_layer4)
        dropout4 = tf.nn.dropout(h_conv4, keep_prob)
    # 2*2*2 pooling CPU Support 4*4*4*32
    with tf.name_scope('pool2'):
        h_pool2 = max_pooling_2X(dropout4)

    ###############################################################################
    # convolutional component 3
    # Conv5 3*3*3*32
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 3, 64, 64])
        b_conv5 = bias_variable([64])
        product5 = conv3d(h_pool2, W_conv5) + b_conv5
        bn_layer5 = batch_normalization(product5, phase)
        h_conv5 = tf.nn.relu(bn_layer5)
        dropout5 = tf.nn.dropout(h_conv5, keep_prob)
    ###############################################################################
    # transposed convolutional component 1
    # dconv1 kernal 2*2*2*32  map 8*8*8*32
    with  tf.name_scope('trans_conv1'):
        W_trans_conv1 = weight_variable([2, 2, 2, 64, 64])
        b_trans_conv1 = bias_variable([64])
        h_conv5_shape = tf.shape(dropout5)
        output_shape_1 = tf.stack([h_conv5_shape[0], 8, 8, 8, 64])
        trans_product1 = conv3d_transposed_2X(dropout5, W_trans_conv1, output_shape_1) + b_trans_conv1
        h_trans_bn_layer1 = batch_normalization(trans_product1, phase)
        h_trans_conv1 = tf.nn.relu(h_trans_bn_layer1)
        dropout_trans_1 = tf.nn.dropout(h_trans_conv1, keep_prob)
        concat_map1 = tf.concat([h_conv4, dropout_trans_1], 4)  # feature map 8*8*8*64

    # Conv6 3*3*3*32
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 3, 128, 64])
        b_conv6 = bias_variable([64])
        product6 = conv3d(concat_map1, W_conv6) + b_conv6
        bn_layer6 = batch_normalization(product6, phase)
        h_conv6 = tf.nn.relu(bn_layer6)
        dropout6 = tf.nn.dropout(h_conv6, keep_prob)
    # Conv7 3*3*3*32
    with tf.name_scope('conv7'):
        W_conv7 = weight_variable([3, 3, 3, 64, 64])
        b_conv7 = bias_variable([64])
        product7 = conv3d(dropout6, W_conv7) + b_conv7
        bn_layer7 = batch_normalization(product7, phase)
        h_conv7 = tf.nn.relu(bn_layer7)
        dropout7 = tf.nn.dropout(h_conv7, keep_prob)
    ###############################################################################
    # transposed convolutional component 2
    # dconv2 kernal 2*2*2*32  map 16*16*16*32
    with  tf.name_scope('trans_conv2'):
        W_trans_conv2 = weight_variable([2, 2, 2, 64, 64])
        b_trans_conv2 = bias_variable([64])
        h_conv7_shape = tf.shape(dropout7)
        output_shape_2 = tf.stack([h_conv7_shape[0], 16, 16, 16, 64])
        trans_product2 = conv3d_transposed_2X(dropout7, W_trans_conv2, output_shape_2) + b_trans_conv2
        h_trans_bn_layer2 = batch_normalization(trans_product2, phase)
        h_trans_conv2 = tf.nn.relu(h_trans_bn_layer2)
        dropout_trans_2 = tf.nn.dropout(h_trans_conv2, keep_prob)
        concat_map2 = tf.concat([h_conv2, dropout_trans_2], 4)  # feature map 16*16*16*64

    # Conv8 5*5*5*32
    with tf.name_scope('conv8'):
        W_conv8 = weight_variable([5, 5, 5, 128, 64])
        b_conv8 = bias_variable([64])
        product8 = conv3d(concat_map2, W_conv8) + b_conv8
        bn_layer8 = batch_normalization(product8, phase)
        h_conv8 = tf.nn.relu(bn_layer8)
        dropout8 = tf.nn.dropout(h_conv8, keep_prob)
    # Conv9 3*3*3*32
    with tf.name_scope('conv9'):
        W_conv9 = weight_variable([3, 3, 3, 64, 64])
        b_conv9 = bias_variable([64])
        product9 = conv3d(dropout8, W_conv9) + b_conv9
        bn_layer9 = batch_normalization(product9, phase)
        h_conv9 = tf.nn.relu(bn_layer9)
        dropout9 = tf.nn.dropout(h_conv9, keep_prob)

    ############################################################################################
    # transposed convolutional component 3
    # dconv2 kernal 2*2*2*32  map 32*32*32*32
    with  tf.name_scope('trans_conv3'):
        W_trans_conv3 = weight_variable([2, 2, 2, 64, 64])
        b_trans_conv3 = bias_variable([64])
        h_conv9_shape = tf.shape(dropout9)
        output_shape_3 = tf.stack([h_conv9_shape[0], 32, 32, 32, 64])
        trans_product3 = conv3d_transposed_2X(dropout9, W_trans_conv3, output_shape_3) + b_trans_conv3
        h_trans_bn_layer3 = batch_normalization(trans_product3, phase)
        h_trans_conv3 = tf.nn.relu(h_trans_bn_layer3)
        dropout_trans_3 = tf.nn.dropout(h_trans_conv3, keep_prob)
        concat_map3 = tf.concat([h_conv2_32, dropout_trans_3], 4)  # feature map 32*32*32*64

    # Conv10 5*5*5*32
    with tf.name_scope('conv10'):
        W_conv10 = weight_variable([5, 5, 5, 128, 64])
        b_conv10 = bias_variable([64])
        product10 = conv3d(concat_map3, W_conv10) + b_conv10
        bn_layer10 = batch_normalization(product10, phase)
        h_conv10 = tf.nn.relu(bn_layer10)
        dropout10 = tf.nn.dropout(h_conv10, keep_prob)

    # Conv11 3*3*3*32
    with tf.name_scope('conv11'):
        W_conv11 = weight_variable([3, 3, 3, 64, 64])
        b_conv11 = bias_variable([64])
        product11 = conv3d(dropout10, W_conv11) + b_conv11
        bn_layer11 = batch_normalization(product11, phase)
        h_conv11 = tf.nn.relu(bn_layer11)
        dropout11 = tf.nn.dropout(h_conv11, keep_prob)

    ############################################################################################
    # output map 32*32*32*5
    # Conv12 1*1*1*5
    with tf.name_scope('conv12'):
        W_conv12 = weight_variable([1, 1, 1, 64, 5])
        b_conv12 = bias_variable([5])
        h_conv12 = tf.nn.conv3d(dropout11, W_conv12, strides=[1, 1, 1, 1, 1], padding='SAME') + b_conv12

    regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(
        W_conv4) + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_conv6) + tf.nn.l2_loss(W_conv7) + tf.nn.l2_loss(
        W_conv8) + tf.nn.l2_loss(W_conv9) + tf.nn.l2_loss(W_conv10) + tf.nn.l2_loss(W_trans_conv1) + tf.nn.l2_loss(
        W_trans_conv2)+ tf.nn.l2_loss(W_conv1_32)+ tf.nn.l2_loss(W_conv2_32)+ tf.nn.l2_loss(W_conv11)+ tf.nn.l2_loss(W_conv12)+ tf.nn.l2_loss(W_trans_conv3)

    return h_conv12, keep_prob, regularizers


def batch_whole_dice(truth, predict):
    true_positive = truth > 0
    predict_positive = predict > 0
    match = np.equal(true_positive, predict_positive)
    match_count = np.count_nonzero(match)
    print("match: ", match_count)
    P1 = np.count_nonzero(predict)
    print("P1: ", P1)
    T1 = np.count_nonzero(truth)
    print("T1: ", T1)
    full_back = np.zeros((32, 32, 32))
    non_back = np.invert(np.equal(truth, full_back))
    TP = np.logical_and(match, non_back)
    TP_count = np.count_nonzero(TP)
    print("TP_count: ", TP_count)
    if (P1 + T1) == 0:
        return 0
    else:
        return 2 * TP_count / (P1 + T1)


if __name__ == '__main__':

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.40
    #config.gpu_options.allow_growth = True

    batchsize = 15
    phase = tf.placeholder(tf.bool, name='phase')
    lamda = 1e-4
    x = tf.placeholder(tf.float32, [None, 32, 32, 32, 4])
    y_ = tf.placeholder(tf.float32, [None, 32, 32, 32, 5])
    y_fc, keep_prob, regularizers = deepnn_3d(x, phase)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_fc)
    cross_entropy = tf.reduce_mean(cross_entropy)
    L2_cross_entropy = tf.reduce_mean(cross_entropy + lamda * regularizers)

    beta = 0.7
    with tf.name_scope('weighted_loss'):
        class_weights = tf.constant([[[[beta, (1 - beta), (1 - beta), (1 - beta), (1 - beta)]]]])
        weights = tf.reduce_sum(class_weights * y_, axis=4)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=y_fc)
        weighted_losses = unweighted_losses * weights
    weighted_cross_entropy = tf.reduce_mean(weighted_losses)
    weighted_L2_cross_entropy = tf.reduce_mean(weighted_cross_entropy + lamda * regularizers)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(weighted_L2_cross_entropy)

    # evaluation

    # accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_fc, 4), tf.argmax(y_, 4))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # result
    predict = tf.argmax(y_fc, 4)
    true_label = tf.argmax(y_, 4)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model_32_1209_final/Patch_3dFCN_32.ckpt")
        print("Model restored.")
        datablock = np.arange(40) + np.ones((40))

        maxepoch = 1000
        epoch = 0
        #acc_list = np.ndarray()
        #dice_list = np.ndarray()
        #acc_list = np.load("./acc_list_norm.npy")
        #dice_list = np.load("./dice_list_norm.npy")
        no_dice_list = []
        while (epoch < maxepoch):
            print("epoch:", epoch)
            np.random.shuffle(datablock)
            block_index = 0
            for block_id in datablock:
                print("block: ", block_index)
                block_index += 1
                # file name bug
                train_src = './patch_train_32_imbalance/train_groundtruth_' + str(int(block_id)) + ".npy"
                train_groundtruth_src = './patch_train_32_imbalance/train_data_' + str(int(block_id)) + ".npy"
                #
                train_data = np.load(train_src)
                train_groundtruth = np.load(train_groundtruth_src)
                train_index = np.arange(len(train_groundtruth))
                np.random.shuffle(train_index)
                batch_begin = 0
                batch_end = batchsize
                print("training begin")
                for i in range(0, 20000):
                    if batch_end <= len(train_index):
                        batch_index = train_index[batch_begin: batch_end]
                        if (i + 1) % 1 == 0:
                            # np.random.shuffle(patientIDs)
                            train_accuracy = accuracy.eval(feed_dict={
                                x: train_data[batch_index], y_: train_groundtruth[batch_index], keep_prob: 1.0,
                                phase: 0})
                            train_label = predict.eval(feed_dict={
                                x: train_data[batch_index], y_: train_groundtruth[batch_index], keep_prob: 1.0,
                                phase: 0})
                            train_label_norm = predict.eval(feed_dict={
                                x: train_data[batch_index], y_: train_groundtruth[batch_index], keep_prob: 1.0,
                                phase: 1})
                            train_true_label = true_label.eval(feed_dict={y_: train_groundtruth[batch_index]})

                            print('step', i, 'training accuracy ', train_accuracy)
                            # print("result:", train_label)
                            print('whole_dice: ', batch_whole_dice(train_true_label, train_label))
                            print('norm_whole_dice: ', batch_whole_dice(train_true_label, train_label_norm))
                            #acc_list = np.insert(acc_list, len(acc_list), train_accuracy)
                            #dice_list = np.insert(dice_list, len(dice_list), batch_whole_dice(train_true_label, train_label))
                            no_dice_list.append(batch_whole_dice(train_true_label, train_label))
                        train_step.run(
                            feed_dict={x: train_data[batch_index], y_: train_groundtruth[batch_index], keep_prob: 0.6,
                                       phase: 0})
                        batch_begin += batchsize
                        batch_end += batchsize
                    else:
                        save_path = saver.save(sess, "./model_32_1209_final_new/Patch_3dFCN_32.ckpt")
                        print("Model saved in file: %s" % save_path)

                        np.save("./no_dice_list.npy",no_dice_list)

                        del train_data
                        del train_groundtruth
                        break
            save_path = saver.save(sess, "./model_32_1209_final_new/Patch_3dFCN_32.ckpt")
            print("Model saved in file: %s" % save_path)
            epoch += 1
