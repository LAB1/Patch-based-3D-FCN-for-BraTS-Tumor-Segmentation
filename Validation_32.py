import numpy as np
import tensorflow as tf
import tempfile
import sys
import Patch_FCN_32
from numba import jit
import Patch_Preprocessing_16
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
import tkinter
from scipy import stats


def whole_dice(truth, predict):
    print(truth.shape)
    print(predict.shape)
    true_positive = truth > 0
    predict_positive = predict > 0
    match = np.equal(true_positive,predict_positive)
    match_count = np.count_nonzero(match)
    print("match: ", match_count)
    P1 = np.count_nonzero(predict)
    print("P1: ", P1)
    T1 = np.count_nonzero(truth)
    print("T1: ", T1)
    full_back = np.zeros((155,240,240))
    non_back = np.invert(np.equal(truth,full_back))
    TP = np.logical_and(match, non_back)
    TP_count = np.count_nonzero(TP)
    print("TP_count: ", TP_count)
    plt.imshow(predict[60], cmap='gray')

    if (P1+T1) == 0:
        return 0
    else:
        return 2*TP_count/(P1+T1)

def padding_16(data):
    # data 4*155*240*240
    pad_data = np.lib.pad(data, ((0,0),(31,31),(31,31),(31,31)), 'minimum')
    return pad_data

def predict_patch_list_16(init,data, window_size, model_src):
    #data: x*y*z*size*size*size*4
    #output: n*(x,y,z) n*size*size*size

    batchsize = 15
    ######################
    """
    train_src = './patch_train_32_bal/train_groundtruth_' + str(int(1)) + ".npy"
    train_groundtruth_src = './patch_train_32_bal/train_data_' + str(int(1)) + ".npy"
    train_data = np.load(train_src)
    train_groundtruth = np.load(train_groundtruth_src)
    train_index = np.arange(len(train_groundtruth))
    np.random.shuffle(train_index)
    batch_index = train_index[0: batchsize-1]
    train_patch = train_data[batch_index]
    """
    #######################


    batch_catch = []
    index_list = []
    patch_predict_list = []
    catch_num = 0
    phase = tf.placeholder(tf.bool, name='phase')
    x_ = tf.placeholder(tf.float32, [None, window_size, window_size, window_size, 4])
    y_fc, keep_prob, regularizers = Patch_FCN_32.deepnn_3d(x_,phase)
    predict = tf.argmax(y_fc, 4)
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_src)
        for x in range(0, len(data)):
            for y in range(0, len(data[0])):
                for z in range(0, len(data[0][0])):
                    print(x,y,z)
                    index_list.append((x,y,z))
                    feed_data = np.array(data[x,y,z,:,:,:,:])
                    batch_catch.append(feed_data)
                    catch_num += 1
                    if catch_num == batchsize or (x+y+z) == (len(data) + len(data[0])+len(data[0][0])-3):

                        train_label = predict.eval(feed_dict={x_: batch_catch, keep_prob: 1.0, phase: 0})
                        print(np.sum(train_label))
                        catch_num = 0
                        del batch_catch
                        batch_catch = []
                        for patchlabel in train_label:
                            patch_predict_list.append(patchlabel)
    tf.reset_default_graph()


    patch_predict_list = np.array(patch_predict_list)
    print("patch_predict_list ", patch_predict_list.shape)
    return index_list, patch_predict_list

def window_predicts(patch_index_list, data, window_size, stride):
    #patch_index_list: n*(x,y,z)
    # data: n*size*size*size
    # output: dic{} (pi,pj,pk):[features]
    pixel_feature_dic = {} #(pi,pj,pk):[features]
    print("data_label ", data.shape)
    for counter in range(0,len(data)):
        for i in range(0,window_size):
            pi = stride*patch_index_list[counter][0] + i
            for j in range(0, window_size):
                pj = stride*patch_index_list[counter][1] + j
                for k in range(0, window_size):
                    pk = stride*patch_index_list[counter][2] + k
                    if pixel_feature_dic.__contains__((pi,pj,pk)):
                        pixel_feature_dic[(pi,pj,pk)].append(data[counter][i][j][k])
                    else:
                        pixel_feature_dic[(pi, pj, pk)]= [data[counter][i][j][k]]
    imagesize_feature_dic = {}
    print(len(pixel_feature_dic))
    for key in pixel_feature_dic.keys():
        nkey = (key[0] - window_size + 1, key[1] - window_size + 1, key[2] - window_size + 1)
        if  nkey[0] >= 0 and nkey[0]<155 and nkey[1] >= 0 and nkey[1]<240 and nkey[2] >= 0 and nkey[2]<240:
            nkey = (key[0]-window_size+1, key[1]-window_size+1, key[2]-window_size+1)
            imagesize_feature_dic[nkey] = pixel_feature_dic[key]
    print(len(imagesize_feature_dic))
    return imagesize_feature_dic



def predict_16(init,data, window_size, stride, model_src):
    #standardized data 4*155*240*240
    data = padding_16(data)
    patchs_multi_model = []
    for model in range(0,len(data)):
        patch_x = []
        for x in range(0,len(data[0])-window_size, stride):
            patch_y = []
            for y in range(0,len(data[0][0])-window_size, stride):
                patch_z = []
                for z in range(0,len(data[0][0][0])-window_size, stride):
                    patch = data[model,x:x+window_size,y:y+window_size,z:z+window_size]
                    patch_z.append(patch)
                patch_y.append(patch_z)
            patch_x.append(patch_y)
        patchs_multi_model.append(patch_x)
    patchs_multi_model = np.array(patchs_multi_model)
    patchs_multi_model = patchs_multi_model.transpose(1, 2, 3, 4, 5, 6, 0)
    patches_index_list, predict_patches = predict_patch_list_16(init,patchs_multi_model, window_size, model_src)
    predict_features_dic = window_predicts(patches_index_list, predict_patches, window_size, stride)
    predict_matrix = transform_dic(predict_features_dic)
    return predict_features_dic, predict_matrix

def transform_dic(features_dic):
    predict_matrix = np.zeros((155,240,240))
    for key in features_dic.keys():
        #predict_matrix[key[0],key[1],key[2]] = stats.mode(features_dic[key], axis=None)[0]
        predict_matrix[key[0], key[1], key[2]] = features_dic[key][0]
    return predict_matrix

def getdata(dic, patientID):
    #return 4*155*240*240, 155*240*240
    print("patientID: ", patientID)
    featuremaps = [patientID * 5, patientID * 5 + 1, patientID * 5 + 2, patientID * 5 + 3, patientID * 5 + 4]
    dir_list = []
    for mapindex in featuremaps:
        dir_list.append(dic[mapindex])
    image_dic = {}
    print(dir_list)
    for dir in dir_list:
        if dir.find("Flair") != -1:
            Flair, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            Flair = (Flair - np.mean(Flair)) / np.std(Flair)
            image_dic[1] = Flair

        elif dir.find("MR_T1c.") != -1:
            T1c, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            T1c = (T1c - np.mean(T1c)) / np.std(T1c)
            image_dic[2] = T1c

        elif (dir.find("XX.O.OT") != -1) | (dir.find("XX.XX.OT.") != -1):
            truth, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            image_dic[0] = truth
        elif dir.find("MR_T1.") != -1:
            T1, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            T1 = (T1 - np.mean(T1)) / np.std(T1)
            image_dic[3] = T1


        elif dir.find("MR_T2") != -1:
            T2, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            T2 = (T2 - np.mean(T2)) / np.std(T2)
            image_dic[4] = T2

    data = [image_dic[1],image_dic[2],image_dic[3],image_dic[4]]
    return data, image_dic[0]


if __name__ == '__main__':
    train_dice = []
    train_dir = Patch_Preprocessing_16.readimages("BRATS2015_Training")

    result = []
    for i in range(200,274):
        x_path = "train_predict_init20/predict_" + str(i) +".npy"
        x_truth_path = "train_predict_init20/truth_" + str(i) +".npy"
        y_path = "train_predict/predict_" + str(i) +".npy"
        #y_truth_path = "train_predict/truth_" + str(i) +".npy"
        z_path = "train_predict_updated/predict_" + str(i) +".npy"
        #z_truth_path = "train_predict_updated/truth_" + str(i) +".npy"

        x = np.load(x_path)
        x_truth = np.load(x_truth_path)
        y = np.load(y_path)
        #y_truth = np.load(y_truth_path)
        z = np.load(z_path)
        #z_truth = np.load(z_truth_path)

        temp = []
        for i in range(155):
            temp_i = []
            for j in range(240):
                temp_j = []
                for l in range(240):
                    temp_l = []
                    temp_l.append(x[i][j][l])
                    temp_l.append(y[i][j][l])
                    temp_l.append(z[i][j][l])
                    temp_l = np.asarray(temp_l)
                    temp_j.append(temp_l)
                temp_j = np.asarray(temp_j)
                temp_j,count = stats.mode(temp_j,axis=1)
                temp_i.append(temp_j)
            temp_i = np.asarray(temp_i)
            temp.append(temp_i)

        temp = np.asarray(temp)
        temp.resize(155,240,240)

        xx = whole_dice(x_truth,temp)
        result.append(xx)
        print (xx)
    print (sum(result)/74)










'''
# bagging

    x = np.load("train_predict/predict_235.npy")
    y = np.load("train_predict_init20/predict_235.npy")
    z = np.load("train_predict_init20_new/predict_235.npy")
    w = np.load("train_predict_updated/predict_235.npy")

    temp_all = []
    print (x.shape)
    for i in range(155):
        temp_i = []
        for j in range(240):
            temp_j = []
            for k in range(240):
                temp = []
                temp.append(x[i][j][k])
                temp.append(y[i][j][k])
                temp.append(z[i][j][k])
                temp.append(w[i][j][k])
                temp_j.append(temp)
            temp_j = np.asarray(temp_j)
            #print (temp_j.shape)
            temp_j,count = stats.mode(temp_j,axis=1)
            #print (temp_j.shape)
            temp_i.append(temp_j)
        temp_i = np.asarray(temp_i)
        temp_all.append(temp_i)
    temp_all = np.asarray(temp_all)
    np.save("235",temp_all)
    np.squeeze(temp_all,axis=3)

    print (temp_all.shape)
    print (temp_all[0][0][0][0])
    #predict,count = stats.mode(temp_all,axis=1)
    #print (predict.shape)

   # first_dice = whole_dice(truth, predict)
    #print(first_dice)

    for i in range(200,274):
        data, truth = getdata(train_dir, i)
        predict_features_dic_16,predict_matrix_16 = predict_16(0,data, window_size=32, stride = 32, model_src = "./model_32_init20/Patch_3dFCN_32.ckpt")

        predict_str = './train_predict_init20/predict_' + str(i)
        np.save(predict_str, predict_matrix_16)
        truth_str = './train_predict_init20/truth_' + str(i)
        np.save(truth_str, truth)

        first = predict_matrix_16
        first_dice = whole_dice(truth, first)
        print(first_dice)
        train_dice.append(first_dice)
        if i <200:
            saving_str = './Figure_1209/train_' + str(i)
        else:
            saving_str = './Figure_1209/test_' + str(i)
        #plt.savefig(saving_str)
        #second = predict_matrix_16[:, :, :, 1]
        #second_dice = whole_dice(truth, second)
        #print(second_dice)
        #plt.show()
    with open('output_1209.txt', 'w+') as f:
        sys.stdout = f
        print(train_dice)
'''


