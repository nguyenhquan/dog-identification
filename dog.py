import numpy as np
from numpy import random
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd


# declaring variables
TRAINING_DATA_IMG_PATH = "E:/Desktop/dog_breed/data/train/"
TEST_DATA_IMG_PATH = "E:/Desktop/dog_breed/data/test/img/"
TEST_DATA_LABEL_PATH = "E:/Desktop/dog_breed/data/test/labels_6_breed.csv"

def load_data_set(feat_detect):

    test_data_label = pd.read_csv(TEST_DATA_LABEL_PATH)
    training_data = []
    test_data = []


    print("Loading Test Data .....")
    img = cv2.imread('E:/Desktop/4.jpg')
    (kp, desc) = get_features(img, feat_detect)
    test_data.append((desc, 'dog'))


    return  np.array(test_data)

def get_features(image, feature_detector):

    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gs_image = cv2.resize(gs_image, (256, 256))
    kp, descriptors = feature_detector.detectAndCompute(gs_image, mask=None)
    if descriptors is None:
        return kp, None
    return kp, np.array(descriptors)




def predict_image(knn_classifier, svm_classifier, k_means, test_set, k_cluster_no):
    test_feature = np.zeros((np.shape(test_set)[0], k_cluster_no))
    test_label = test_set[:, -1]
    for i in range(np.shape(test_set)[0]):
        desc, label = test_set[i][0], test_set[i][1]
        r = k_means.predict(desc)
        r_unique = np.unique(r, return_counts=True)
        for j in range(np.shape(r_unique)[1]):
            test_feature[i][r_unique[0][j]] = r_unique[1][j]


    knn_result = knn_classifier.predict(test_feature)
    print('KNN PREDICT:')
    print(knn_result)

    svm_result2 = svm_classifier.predict(test_feature)
    print('SVM PREDICT:')
    print(svm_result2)


if __name__ == "__main__":

    k_cluster = 10
    fd = cv2.xfeatures2d.SIFT_create()
    test_set = load_data_set(fd)


    from joblib import dump,load
    clf=load('knnmodel.joblib')
    svm_clf=load('svmmodel.joblib')
    k_mean_clr=load('kmeansmodel.joblib')

    predict_image(clf, svm_clf, k_mean_clr, test_set, k_cluster)
