import SimpleITK as sitk
import numpy as np
import os
import random



#learn bounding box
def learn_boundingbox(dir_scr):
    box = [[75,75], [120,120], [120,120]]
    for dir in dir_scr:
        if (dir.find("XX.O.OT") != -1) | (dir.find("XX.XX.OT.") != -1):
            truth, origin, spacing = load_itk(dir)
            for i in range(0,154):
                if np.sum(truth[i])>0:
                    if i < box[0][0]:
                        box[0][0] = i
                    break
            for i in range(0,154):
                if np.sum(truth[154-i])>0:
                    if (154-i) > box[0][1]:
                        box[0][1] = (154-i)
                    break
            for i in range(0,239):
                if np.sum(truth[:,i,:])>0:
                    if i < box[1][0]:
                        box[1][0] = i
                    break
            for i in range(0,239):
                if np.sum(truth[:,239-i,:])>0:
                    if (239-i) > box[1][1]:
                        box[1][1] = (239-i)
                    break
            for i in range(0,239):
                if np.sum(truth[:,:,i])>0:
                    if i < box[2][0]:
                        box[2][0] = i
                    break
            for i in range(0,239):
                if np.sum(truth[:,:,239-i])>0:
                    if (239-i) > box[2][1]:
                        box[2][1] = (239-i)
                    break
            print(box)
    return box


def tumor_box(truth):
    box = [[75, 75], [120, 120], [120, 120]]
    for i in range(0, 154):
        if np.sum(truth[i]) > 0:
            if i < box[0][0]:
                box[0][0] = i
            break
    for i in range(0, 154):
        if np.sum(truth[154 - i]) > 0:
            if (154 - i) > box[0][1]:
                box[0][1] = (154 - i)
            break
    for i in range(0, 239):
        if np.sum(truth[:, i, :]) > 0:
            if i < box[1][0]:
                box[1][0] = i
            break
    for i in range(0, 239):
        if np.sum(truth[:, 239 - i, :]) > 0:
            if (239 - i) > box[1][1]:
                box[1][1] = (239 - i)
            break
    for i in range(0, 239):
        if np.sum(truth[:, :, i]) > 0:
            if i < box[2][0]:
                box[2][0] = i
            break
    for i in range(0, 239):
        if np.sum(truth[:, :, 239 - i]) > 0:
            if (239 - i) > box[2][1]:
                box[2][1] = (239 - i)
            break
    print(box)
    return box

# cut regions
def cut_region(scan, box=[[5,149],[39,215],[35,211]]): #144*176*176
    return scan[box[0][0]:box[0][1],box[1][0]:box[1][1],box[2][0]:box[2][1]]

def get_patches(scan, box=[16,16,16]):
    # input 144*176*176 output 9*11*11 16*16*16
    patch_list = []
    for i in range(0,9):
        for j in range(0,11):
            for k in range(0,11):
                patch = scan[(i*16):(i*16+16),(j*16):(j*16+16),(k*16):(k*16+16)]
                patch_list.append(patch)
    return patch_list


# Load Brats Data
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


#get data path
def readimages(scr):
    files = os.listdir(scr)
    imagedic_list = []
    for HLdic in files:
        HLfiles = os.listdir(scr + "/" + HLdic)
        for patientdic in HLfiles:
            scandic = os.listdir(scr + "/" + HLdic + "/" + patientdic)
            # patient_scans = []
            for datadic in scandic:
                datafile = os.listdir(scr + "/" + HLdic + "/" + patientdic + "/" + datadic)
                for file in datafile:
                    if file[0] == 'V':
                        path = scr + "/" + HLdic + "/" + patientdic + "/" + datadic + "/" + file
                        imagedic_list.append(path)
    return imagedic_list

def onehot_truth(truth, windowsize):
    background = np.zeros((windowsize, windowsize, windowsize))
    tumor1 = np.zeros((windowsize, windowsize, windowsize))
    tumor2 = np.zeros((windowsize, windowsize, windowsize))
    tumor3 = np.zeros((windowsize, windowsize, windowsize))
    tumor4 = np.zeros((windowsize, windowsize, windowsize))
    for d in range(0,windowsize):
        for x in range(0, windowsize):
            for y in range(0, windowsize):
                if truth[d][x][y] == 0:
                    background[d][x][y] = 1
                elif truth[d][x][y] == 1:
                    tumor1[d][x][y] = 1
                elif truth[d][x][y] == 2:
                    tumor2[d][x][y] = 1
                elif truth[d][x][y] == 3:
                    tumor3[d][x][y] = 1
                elif truth[d][x][y] == 4:
                    tumor4[d][x][y] = 1
    truth3D_list = [background, tumor1, tumor2, tumor3, tumor4]
    return truth3D_list


#read patch
def get_next_batch(train_dir, patientID, windowsize, stride, prob):
    print("patientID: ",patientID)
    featuremaps = [patientID * 5, patientID * 5 + 1, patientID * 5 + 2, patientID * 5 + 3, patientID * 5 + 4]
    dir_list = []
    for mapindex in featuremaps:
        dir_list.append(train_dir[mapindex])
    image_dic = {}
    print(dir_list)
    for dir in dir_list:
        if dir.find("Flair") != -1:
            Flair, origin, spacing = load_itk(dir)
            Flair = (Flair - np.mean(Flair)) / np.std(Flair)
            image_dic[1] = Flair

        elif dir.find("MR_T1c.") != -1:
            T1c, origin, spacing = load_itk(dir)
            T1c = (T1c - np.mean(T1c)) / np.std(T1c)
            image_dic[2] = T1c

        elif (dir.find("XX.O.OT") != -1) | (dir.find("XX.XX.OT.") != -1):
            truth, origin, spacing = load_itk(dir)
            image_dic[0] = truth

        elif dir.find("MR_T1.") != -1:
            T1, origin, spacing = load_itk(dir)
            T1 = (T1 - np.mean(T1)) / np.std(T1)
            image_dic[3] = T1


        elif dir.find("MR_T2") != -1:
            T2, origin, spacing = load_itk(dir)
            T2 = (T2 - np.mean(T2)) / np.std(T2)
            image_dic[4] = T2

    samples_truth, samples_data = box_sampling(image_dic, windowsize, stride, prob)
    samples_truth = samples_truth.transpose(0, 2, 3, 4, 1)
    samples_data = samples_data.transpose(0, 2, 3, 4, 1)
    print(samples_truth.shape)
    return samples_truth, samples_data

def box_sampling(image_dic, windowsize,stride, prob):
    truth_patch_list = []
    data_patch_list = []
    T0_array = (image_dic[0] == 0)
    T1_array = (image_dic[0] > 0)
    T2_array = (image_dic[0] == 2)
    T3_array = (image_dic[0] == 3)
    T4_array = (image_dic[0] == 4)
    T0 = np.count_nonzero(T0_array)
    T1 = np.count_nonzero(T1_array)
    T2 = np.count_nonzero(T2_array)
    T3 = np.count_nonzero(T3_array)
    T4 = np.count_nonzero(T4_array)

    #ratio = T1/T0
    ratio = 1
    Pcount = 0
    Ncount = 0
    for x in range(0+int(windowsize/2),155-int(windowsize/2),stride):
        for y in range(0+int(windowsize/2),240-int(windowsize/2),stride):
            for z in range(0+int(windowsize/2),240-int(windowsize/2),stride):
                randomprob = np.random.random_sample()
                if (image_dic[0][x][y][z]>0 and randomprob < prob) or (image_dic[0][x][y][z]==0 and randomprob < (prob*ratio)):
                    if image_dic[0][x][y][z] > 0:
                        Pcount += 1
                    else:
                        Ncount += 1
                    patch_0 = image_dic[0][int(x-windowsize/2):int(x + windowsize/2), int(y-windowsize/2):int(y + windowsize/2), int(z-windowsize/2):z + int(windowsize/2)]
                    patch_1 = image_dic[1][int(x-windowsize/2):int(x + windowsize/2), int(y-windowsize/2):int(y + windowsize/2), int(z-windowsize/2):z + int(windowsize/2)]
                    patch_2 = image_dic[2][int(x-windowsize/2):int(x + windowsize/2), int(y-windowsize/2):int(y + windowsize/2), int(z-windowsize/2):z + int(windowsize/2)]
                    patch_3 = image_dic[3][int(x-windowsize/2):int(x + windowsize/2), int(y-windowsize/2):int(y + windowsize/2), int(z-windowsize/2):z + int(windowsize/2)]
                    patch_4 = image_dic[4][int(x-windowsize/2):int(x + windowsize/2), int(y-windowsize/2):int(y + windowsize/2), int(z-windowsize/2):z + int(windowsize/2)]
                    truth_patch = patch_0
                    onehot_truth_patch = onehot_truth(truth_patch, windowsize)
                    data_patch = [patch_1, patch_2, patch_3, patch_4]
                    truth_patch_list.append(onehot_truth_patch)
                    data_patch_list.append(data_patch)
    truth_patch_list = np.array(truth_patch_list)
    data_patch_list = np.array(data_patch_list)
    print("P: ",Pcount)
    print("N: ",Ncount)
    return truth_patch_list,data_patch_list

if __name__ == '__main__':

    train_dir = readimages("BRATS2015_Training")
    train_data, train_groundtruth = get_next_batch(train_dir, 0, 32, 5, 0.01)
    block_id = 1
    for i in range(1, 201):
        print(i)
        if i < 200:
            batch_data, batch_truth = get_next_batch(train_dir, i, 32, 5, 0.01)
        if (i == 200) or (i%5 == 0):
            print(i)
            data_str = './patch_train_32_imbalance/train_data_' + str(block_id)
            truth_str = './patch_train_32_imbalance/train_groundtruth_' + str(block_id)
            np.save(data_str, train_data)
            np.save(truth_str, train_groundtruth)
            print("save")
            del train_data
            del train_groundtruth
            if i == 274:
                break
            train_data = batch_data
            train_groundtruth = batch_truth
            block_id+=1
        else:
            train_data = np.concatenate((train_data, batch_data.copy()))
            train_groundtruth = np.concatenate((train_groundtruth, batch_truth.copy()))

