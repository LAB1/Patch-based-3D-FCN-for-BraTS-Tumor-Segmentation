from sklearn import svm
import numpy as np
import Validation_16
import Patch_FCN_16
import Patch_Preprocessing_16

def train_sampling(p_data, t_data, p_prob, n_prob):
    sample_list = []
    labels = []
    for i in range(0,len(t_data),2):
        print(i)
        for j in range(0,len(t_data[0]),2):
            for k in range(0,len(t_data[0][0]),2):
                rand = np.random.random_sample()
                if t_data[i][j][k] > 0:
                    if rand < p_prob:
                        sample_list.append(p_data[i,j,k,:])
                        labels.append(t_data[i][j][k])
                else:
                    if rand < n_prob:
                        sample_list.append(p_data[i,j,k,:])
                        labels.append(t_data[i][j][k])
    sample_list = np.array(sample_list)
    labels = np.array(labels)
    return sample_list, labels


def stacking_train(p_data, t_data, p_prob, n_prob):
    #p_data 155(n)*240*240*features
    #t_data 155(n)*240*240
    sample_list, labels = train_sampling(p_data, t_data, p_prob, n_prob)
    print(sample_list.shape)
    print("train")
    svm_model = svm.SVC()
    svm_model.fit(sample_list, labels)
    return svm_model

def stacking_predict(model, p_data):
    s_predict = np.zeros((155,240,240))
    for i in range(0,len(p_data)):
        print(i)
        for j in range(0,len(p_data[0])):
            for k in range(0,len(p_data[0][0])):
                sample = p_data[i, j, k, :]
                if np.sum(sample) < 8:
                    s_predict[i,j,k] = 0
                else:
                    s_predict[i, j, k] = 1
    return s_predict

if __name__ == '__main__':
    predict_str1 = './train_predict/predict_' + str(5) + '.npy'
    truth_str1 = './train_predict/truth_' + str(5) + '.npy'
    p5 = np.load(predict_str1)
    t5 = np.load(truth_str1)
    """
    predict_str1 = './train_predict/predict_' + str(1)
    truth_str1 = './train_predict/truth_' + str(1)
    p1 = np.load(predict_str1)
    t1 = np.load(truth_str1)

    p = np.concatenate(p5,p1)
    t = np.concatenate(t5,t1)
    """
    p = p5
    t = t5
    model = stacking_train(p, t, 1, 1e-3)
    s_predict = stacking_predict(model, p5)
    print("dice")
    first_dice = Validation_16.whole_dice(t5, s_predict)
    print(first_dice)
