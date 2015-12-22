import numpy as np
import sys
import time

from skimage import color
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Split input line from the training text file
def line_split( line ):
    split_line = line.split(' ')
    return split_line[0], int(split_line[1])

# Load the training set of images into memory, convert them to gray scale
def load_training_set( file ):
    images = []
    classify = []
    img_size = [250, 250]
    img_classify = 0

    try:
        for line in file:
            img_url, img_classify = line_split( line )
            classify = np.hstack((classify, img_classify))
            img = imread(img_url, as_grey=True)
            img = resize(img, img_size)
            images.append(img)
    finally:
        file.close()

    return images, classify

def load_unknown_set( file ):
    images = []
    img_size = [250, 250]

    try:
        for line in file:
            img = imread(line.rstrip('\n'), as_grey=True)
            img = resize(img, img_size)
            images.append(img)
    finally:
        file.close()

    return images

def hog_svm_classifier( training_images, classify, unknown_images ):
    #print( "Histogram of Oriented Gradients Classifier" )

    hog_set = []

    for img in training_images:
        fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        hog_set.append(fd)

    # Testing with known data set
    X_train, X_test, y_train, y_test = train_test_split(
        hog_set, classify, test_size=0.25)

    clf = SVC(kernel='linear', class_weight='balanced')
    clf = clf.fit(X_train, y_train)

    # Testing how well the data was fit
    y_pred = clf.predict(X_test)
    y_pred = clf.predict(X_test)
    y_f1 = f1_score(y_test, y_pred, average=None)
    y_precision = precision_score(y_test, y_pred, average=None)
    y_recall = recall_score(y_test, y_pred, average=None)

    y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]

    #print( "Histogram of Oriented Gradients Classifier Fitted" )

    return y_scores

def pca_svm_classifier( training_images, classify, unknown_images ):
    #print( "Principal Component Analysis Classifier" )

    flat_training_images = []
    for img in training_images:
        flat_training_images.append(img.flatten())

    # Testing with known data set
    X_train, X_test, y_train, y_test = train_test_split(
        flat_training_images, classify, test_size=0.25)

    pca = RandomizedPCA(n_components=len(X_train), whiten=True).fit(X_train)
    eigen_images = pca.components_.reshape((len(X_train), 250, 250))
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = SVC(kernel='linear', class_weight='balanced')
    clf = clf.fit(X_train_pca, y_train)

    # Testing how well the data was fit
    y_pred = clf.predict(X_test_pca)
    y_f1 = f1_score(y_test, y_pred, average=None)
    y_precision = precision_score(y_test, y_pred, average=None)
    y_recall = recall_score(y_test, y_pred, average=None)

    y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]

    #print( "Principal Component Analysis Classifier Fitted" )

    return y_scores

def k_near_svm_classifier( training_images, classify, unknown_images ):
    #print( "K-Nearest Neighbor Classifier" )

    flat_training_images = []
    for img in training_images:
        flat_training_images.append(img.flatten())

    # Testing with known data set
    X_train, X_test, y_train, y_test = train_test_split(
        flat_training_images, classify, test_size=0.25)

    clf = KNeighborsClassifier(10).fit(X_train, y_train)

    # Testing how well the data was fit
    y_pred = clf.predict(X_test)
    y_f1 = f1_score(y_test, y_pred, average=None)
    y_precision = precision_score(y_test, y_pred, average=None)
    y_recall = recall_score(y_test, y_pred, average=None)

    y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]

    #print( "K-Nearest Neighbor Classifier Fitted" )

    return y_scores

def svm_classifier( training_images, classify, unknown_images ):
    #print( "SVM Classifier" )

    flat_training_images = []
    for img in training_images:
        flat_training_images.append(img.flatten())

    # Testing with the known dataset
    X_train, X_test, y_train, y_test = train_test_split(
        flat_training_images, classify, test_size=0.25)

    clf = SVC(kernel='linear', class_weight='balanced')
    clf = clf.fit(X_train, y_train)

    # Testing how well the known data set was Fitted
    y_pred = clf.predict(X_test)
    y_f1 = f1_score(y_test, y_pred, average=None)
    y_precision = precision_score(y_test, y_pred, average=None)
    y_recall = recall_score(y_test, y_pred, average=None)

    y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]

    #print( "SVM Classifier Fitted" )

    return y_scores

def save_classified_set( classified_images, classified_set ):
    return 0


if len(sys.argv) != 5:
    print( 'There was ' + str(len(sys.argv)-1) + ' arguements' )
    print( 'Four arguements must be entered, 2 input text files, 1 output text file and an algorithm identifier' )

else:
    # Open the entered files
    trainer_set = open(sys.argv[1], 'r')
    unknown_set = open(sys.argv[2], 'r')
    classified_set = open(sys.argv[3], 'w')

    # Load the training data
    training_images, classify = load_training_set( trainer_set )
    print( "Training Images Loaded" )

    # Load the unknown data
    unknown_images = load_unknown_set( unknown_set )
    print( "Unknown Images Loaded" )

    if int(sys.argv[4]) > 4 or int(sys.argv[4]) < 1:
        print( "Algorithm identifier must be between 1-4" )
    else:
        '''
        if int(sys.argv[4]) == 1:
            classified_images = hog_svm_classifier(training_images, classify, unknown_images)
        elif int(sys.argv[4]) == 2:
            classified_images = pca_svm_classifier(training_images, classify, unknown_images)
        elif int(sys.argv[4]) == 3:
            classified_images = k_near_svm_classifier(training_images, classify, unknown_images)
        elif int(sys.argv[4]) == 4:
            classified_images = svm_classifier(training_images, classify, unknown_images)
        '''

        for i in [1,2,3,4]:
            y_pred_array = [[],[],[],[],[],[]]
            start_time = time.time()
            for j in range(100):
                if i == 1:
                    y_pred = hog_svm_classifier(training_images, classify, unknown_images)
                elif i == 2:
                    y_pred = pca_svm_classifier(training_images, classify, unknown_images)
                elif i == 3:
                    y_pred = k_near_svm_classifier(training_images, classify, unknown_images)
                elif i == 4:
                    y_pred = svm_classifier(training_images, classify, unknown_images)

                y_pred_array[0].append(y_pred[0])
                y_pred_array[1].append(y_pred[1])
                y_pred_array[2].append(y_pred[2])
                y_pred_array[3].append(y_pred[3])
                y_pred_array[4].append(y_pred[4])
                y_pred_array[5].append(y_pred[5])

            finish_time = time.time()
            mean = []
            std = []
            mean.append(np.mean(np.array(y_pred_array[0])))
            mean.append(np.mean(np.array(y_pred_array[1])))
            mean.append(np.mean(np.array(y_pred_array[2])))
            mean.append(np.mean(np.array(y_pred_array[3])))
            mean.append(np.mean(np.array(y_pred_array[4])))
            mean.append(np.mean(np.array(y_pred_array[5])))

            std.append(np.std(np.array(y_pred_array[0])))
            std.append(np.std(np.array(y_pred_array[1])))
            std.append(np.std(np.array(y_pred_array[2])))
            std.append(np.std(np.array(y_pred_array[3])))
            std.append(np.std(np.array(y_pred_array[4])))
            std.append(np.std(np.array(y_pred_array[5])))
            print( "It took %f to complete testing" % (finish_time-start_time))
            print( "\t\tPrecision\tRecall\t\tF1" )
            print( "Mean 0\t\t%f\t%f\t%f" % (mean[0], mean[1], mean[2]))
            print( "Mean 0\t\t%f\t%f\t%f" % (mean[3], mean[4], mean[5]))
            print( "St.Dev 0\t%f\t%f\t%f" % (std[0], std[1], std[2]))
            print( "St.Dev 0\t%f\t%f\t%f" % (std[3], std[4], std[5]))

        save_classified_set( classified_images, classified_set )
