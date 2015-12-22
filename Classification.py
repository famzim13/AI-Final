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
    print( "Histogram of Oriented Gradients Classifier" )

    clf = SVC(kernel='linear', class_weight='balanced')
    classified = [0]*len(unknown_images)

    start_time = time.time()

    for i in range(15):
        training_hog_set = []
        for img in training_images:
            fd = hog(img, orientations=8, pixels_per_cell=(16+i, 16+i),
                        cells_per_block=(1+(i//3), 1+(i//3)))
            training_hog_set.append(fd)

        unknown_hog_set = []
        for img in unknown_images:
            fd = hog(img, orientations=8, pixels_per_cell=(16+i, 16+i),
                        cells_per_block=(1+(i//3), 1+(i//3)))
            unknown_hog_set.append(fd)

        '''
        Data splitting with a known dataset, held-out
        X_train, X_test, y_train, y_test = train_test_split(
            flat_training_images, classify, test_size=0.25)
        '''

        clf = clf.fit(training_hog_set, classify)

        class_pred = clf.predict(unknown_hog_set)

        for j in range(len(classified)):
            classified[j] += class_pred[j]

        '''
        Testing how well the data was fit
        y_pred = clf.predict(X_test)
        y_pred = clf.predict(X_test)
        y_f1 = f1_score(y_test, y_pred, average=None)
        y_precision = precision_score(y_test, y_pred, average=None)
        y_recall = recall_score(y_test, y_pred, average=None)
        y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]
        '''

    finish_time = time.time()

    for i in range(len(classified)):
        classified[i] = float("{0:.2f}".format(classified[i]/15))

    print( "Histogram of Oriented Gradients Classifier Fitted" )

    return classified

def pca_svm_classifier( training_images, classify, unknown_images ):
    print( "Principal Component Analysis Classifier" )

    flat_training_images = []
    for img in training_images:
        flat_training_images.append(img.flatten())

    flat_unknown_images = []
    for img in unknown_images:
        flat_unknown_images.append(img.flatten())

    classified = [0]*len(unknown_images)

    start_time = time.time()

    for i in range(5):

        '''
        Data splitting with a known dataset, held-out
        X_train, X_test, y_train, y_test = train_test_split(
            flat_training_images, classify, test_size=0.25)
        '''

        pca = RandomizedPCA(n_components=len(flat_training_images)//(i+1), whiten=True).fit(flat_training_images)
        eigen_images = pca.components_.reshape((len(X_train)//(i+1), 250, 250))
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = SVC(kernel='linear', class_weight='balanced')
        clf = clf.fit(X_train_pca, y_train)

        class_pred = clf.predict(flat_unknown_images)

        for j in range(len(classified)):
            classified[j] += class_pred[j]

        '''
        Testing how well the data was fit
        y_pred = clf.predict(X_test_pca)
        y_f1 = f1_score(y_test, y_pred, average=None)
        y_precision = precision_score(y_test, y_pred, average=None)
        y_recall = recall_score(y_test, y_pred, average=None)
        y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]
        '''

    finish_time = time.time()
    print( "It took %f seconds" % (finish_time-start_time))

    for i in range(len(classified)):
        classified[i] = float("{0:.2f}".format(classified[i]/5))

    print( "Principal Component Analysis Classifier Fitted" )

    return classified

def k_nearest_classifier( training_images, classify, unknown_images ):
    print( "K-Nearest Neighbor Classifier" )

    flat_training_images = []
    for img in training_images:
        flat_training_images.append(img.flatten())

    flat_unknown_images = []
    for img in unknown_images:
        flat_unknown_images.append(img.flatten())

    classified = [0]*len(unknown_images)

    for i in range(50):

        '''
        Data splitting with a known dataset, held-out
        X_train, X_test, y_train, y_test = train_test_split(
            flat_training_images, classify, test_size=0.25)
        '''

        clf = KNeighborsClassifier(5*(i+1)).fit(flat_training_images, classify)

        class_pred = clf.predict(flat_unknown_images)

        for j in range(len(classified)):
            classified[j] += class_pred[j]

        '''
        Testing how well the data was fit
        y_pred = clf.predict(X_test)
        y_f1 = f1_score(y_test, y_pred, average=None)
        y_precision = precision_score(y_test, y_pred, average=None)
        y_recall = recall_score(y_test, y_pred, average=None)
        y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]
        '''

    for i in range(len(classified)):
        classified[i] = float("{0:.2f}".format(classified[i]/50))

    print( "K-Nearest Neighbor Classifier Fitted" )

    return classified

def svm_classifier( training_images, classify, unknown_images ):
    print( "SVM Classifier" )

    flat_training_images = []
    for img in training_images:
        flat_training_images.append(img.flatten())

    flat_unknown_images = []
    for img in unknown_images:
        flat_unknown_images.append(img.flatten())

    classified = [0]*len(unknown_images)

    start_time = time.time()

    for i in range(15):

        '''
        Data splitting with a known dataset, held-out
        X_train, X_test, y_train, y_test = train_test_split(
            flat_training_images, classify, test_size=0.25)
        '''

        clf = SVC(kernel='linear', class_weight='balanced', tol=1e3*(i+1))
        clf = clf.fit(flat_training_images, classify)

        class_pred = clf.predict(flat_unknown_images)

        for j in range(len(classified)):
            classified[j] += class_pred[j]

        '''
        Testing how well the known data set was Fitted
        y_pred = clf.predict(X_test)
        y_f1 = f1_score(y_test, y_pred, average=None)
        y_precision = precision_score(y_test, y_pred, average=None)
        y_recall = recall_score(y_test, y_pred, average=None)
        y_scores = [y_precision[0], y_recall[0], y_f1[0], y_precision[1], y_recall[1], y_f1[1]]
        '''

    finish_time = time.time()
    for i in range(len(classified)):
        classified[i] = float("{0:.2f}".format(classified[i]/15))
    print( classified )

    print( "It took %f seconds" % (finish_time-start_time))

    print( "SVM Classifier Fitted" )

    return

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
        if int(sys.argv[4]) == 1:
            classified_images = hog_svm_classifier(training_images, classify, unknown_images)
        elif int(sys.argv[4]) == 2:
            classified_images = pca_svm_classifier(training_images, classify, unknown_images)
        elif int(sys.argv[4]) == 3:
            classified_images = k_nearest_classifier(training_images, classify, unknown_images)
        elif int(sys.argv[4]) == 4:
            classified_images = svm_classifier(training_images, classify, unknown_images)
        '''
        This was used to test which method had the best, recall, precision and
        F1 scores and what the deviation was over 100 iterations
        Time measurements were also taken since those are important for determining
        how well an algorithm will work in a production environment
        for i in [1,2,3,4]:
            y_pred_array = [[],[],[],[],[],[]]
            start_time = time.time()
            for j in range(100):
                if i == 1:
                    y_pred = hog_svm_classifier(training_images, classify, unknown_images)
                elif i == 2:
                    y_pred = pca_svm_classifier(training_images, classify, unknown_images)
                elif i == 3:
                    y_pred = k_nearest_classifier(training_images, classify, unknown_images)
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
            '''

        save_classified_set( classified_images, classified_set )
