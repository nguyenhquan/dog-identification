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

    print("Loading Training Data .....")
    folder_list = os.listdir(TRAINING_DATA_IMG_PATH)
    for folder in folder_list:
        file_list = os.listdir(TRAINING_DATA_IMG_PATH + folder + "/")
        for image_name in file_list:
            img = cv2.imread(TRAINING_DATA_IMG_PATH + folder + "/" + image_name)
            (kp, desc) = get_features(img, feat_detect)
            training_data.append((desc, folder))

    print("Loading Test Data .....")
    for val in test_data_label.values:
        img = cv2.imread(TEST_DATA_IMG_PATH + val[0] + ".jpg")
        (kp, desc) = get_features(img, feat_detect)
        test_data.append((desc, val[1]))

    random.shuffle(training_data)
    return np.array(training_data), np.array(test_data)


def get_features(image, feature_detector):

    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gs_image = cv2.resize(gs_image, (256, 256))
    kp, descriptors = feature_detector.detectAndCompute(gs_image, mask=None)
    if descriptors is None:
        return kp, None
    return kp, np.array(descriptors)


def initializing_classifier(clust_cnt):

    knn_classifier = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='brute')
    svm_classifier = SVC(probability=True, kernel='linear', C=3.67, gamma=5.383)

    kmeans_classifier = KMeans(clust_cnt)
    feature_detector = cv2.xfeatures2d.SIFT_create()
    return knn_classifier, svm_classifier, kmeans_classifier, feature_detector


def k_mean_clustering(descriptor_list, k_means):

    descriptors = descriptor_list[0][0]
    for descriptor, label in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    k_means.fit(descriptors)
    return k_means


def train_classifier(knn_classifier, svm_classifier, train_data, train_label):

    print('Training KNN Classifier')
    knn_classifier.fit(train_data, train_label)
    print('Training SVM Classifier')
    svm_classifier.fit(train_data, train_label)
    return knn_classifier, svm_classifier


def bag_of_features(descriptor_list, k_mean_cluster, k_clusters):

    no_of_data = np.shape(descriptor_list)[0]

    x_lab = np.zeros((no_of_data, k_clusters))
    y_lab = descriptor_list[:, -1]
    t = 0
    for i in range(no_of_data):
        d = descriptor_list[i][0]
        for j in range(np.shape(d)[0]):
            cluster_index = k_mean_cluster[t]
            x_lab[i][cluster_index] = x_lab[i][cluster_index] + 1
            t = t + 1

    return x_lab, y_lab


def predict_accuracy(knn_classifier, svm_classifier, k_means, test_set, k_cluster_no):

    test_feature = np.zeros((np.shape(test_set)[0], k_cluster_no))
    test_label = test_set[:, -1]
    for i in range(np.shape(test_set)[0]):
        desc, label = test_set[i][0], test_set[i][1]
        r = k_means.predict(desc)
        r_unique = np.unique(r, return_counts=True)
        for j in range(np.shape(r_unique)[1]):
            test_feature[i][r_unique[0][j]] = r_unique[1][j]

    knn_result = knn_classifier.predict(test_feature)
    svm_result2 = svm_classifier.predict(test_feature)


    knn_acc = svm_acc = ada_acc = 0
    for l in range(np.shape(test_feature)[0]):
        if test_label[l] == knn_result[l]:
            knn_acc = knn_acc + 1
        if test_label[l] == svm_result2[l]:
           svm_acc = svm_acc + 1


    knn_acc = (knn_acc / np.shape(test_feature)[0]) * 100
    svm_acc = (svm_acc / np.shape(test_feature)[0]) * 100

    print('KNN: ', knn_acc, '%; SVM: ', svm_acc, '%')


if __name__ == "__main__":

    k_cluster = 10
    print("Initializing Classifiers .....")
    knn_clr, svm_clr, k_means, fd = initializing_classifier(k_cluster)
    training_set, test_set = load_data_set(fd)

    print('Clustering features into', k_cluster, 'clusters .....')
    k_mean_clr = k_mean_clustering(training_set, k_means)

    print('Creating Bag of Features .....')
    x_label, y_label = bag_of_features(training_set, k_mean_clr.labels_, k_cluster)

    clf, svm_clf = train_classifier(knn_clr, svm_clr, x_label, y_label)
    from joblib import dump,load
    dump(svm_clf, 'svmmodel.joblib')
    dump(clf, 'knnmodel.joblib')
    dump(k_mean_clr, 'kmeansmodel.joblib')

    #predict_accuracy(clf, svm_clf, k_mean_clr, test_set, k_cluster)
