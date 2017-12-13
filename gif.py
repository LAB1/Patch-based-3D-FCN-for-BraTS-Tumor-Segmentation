import numpy as np
import Patch_Preprocessing_16
import Validation_16
import scipy.misc


if __name__ == '__main__':
    patientID = 234
    predict_str = './train_predict/predict_' + str(patientID)+".npy"
    truth_str = './train_predict/truth_' + str(patientID)+".npy"
    truth = np.load(truth_str)
    predict = np.load(predict_str)
    train_dir = Patch_Preprocessing_16.readimages("BRATS2015_Training")
    featuremaps = [patientID * 5, patientID * 5 + 1, patientID * 5 + 2, patientID * 5 + 3, patientID * 5 + 4]
    dir_list = []
    for mapindex in featuremaps:
        dir_list.append(train_dir[mapindex])
    image_dic = {}
    print(dir_list)

    for dir in dir_list:
        if dir.find("Flair") != -1:
            Flair, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            image_dic[1] = Flair

        elif dir.find("MR_T1c.") != -1:
            T1c, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            image_dic[2] = T1c

        elif dir.find("MR_T1.") != -1:
            T1, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            image_dic[3] = T1


        elif dir.find("MR_T2") != -1:
            T2, origin, spacing = Patch_Preprocessing_16.load_itk(dir)
            image_dic[4] = T2

    for i in range(0,155):
        predict_str = './gif/predict_' + str(patientID)+"_"+str(i)+".jpg"
        rescaled = (255.0 / predict[i].max() * (predict[i] - predict[i].min())).astype(np.uint8)
        print(np.shape(rescaled))
        scipy.misc.imsave(predict_str, rescaled)


        truth_str = './gif/truth_' + str(patientID) + "_" + str(i)+".jpg"
        rescaled = (255.0 / truth[i].max() * (truth[i] - truth[i].min())).astype(np.uint8)
        scipy.misc.imsave(truth_str, rescaled)

        T1_str = './gif/flair_' + str(patientID) + "_" + str(i)+".jpg"
        rescaled = (255.0 / image_dic[1][i].max() * (image_dic[1][i] - image_dic[1][i].min())).astype(np.uint8)
        scipy.misc.imsave(T1_str, rescaled)
