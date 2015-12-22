import sys
import numpy as np

from skimage import color
from skimage.io import imread
from skimage.transform import resize

from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# Split input line from the training text file
def line_split( line ):
    split_line = line.split(' ')
    return split_line[0], int(split_line[1])

# Extract features from the set of training images
def load_training_set( file ):
    images = []
    classify = []
    img_size = [200, 200]
    img_classify = 0

    try:
        for line in file:
            img_url, img_classify = line_split( line )
            classify = np.hstack((classify, img_classify))
            img = imread(img_url, as_grey=True)
            img = resize(img, img_size)
            images.append(img.flatten())
    finally:
        file.close()

    return images, classify

def hog_svm_classifier( images, classify ):
    return

def pca_svm_classifier( images, classify ):
    sum_score = 0
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(
            images, classify, test_size=0.25, random_state=42)

        pca = RandomizedPCA(n_components=len(X_train), whiten=True).fit(X_train)
        eigen_images = pca.components_.reshape((len(X_train), 200, 200))
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = SVC(kernel='linear', class_weight='balanced')
        clf = clf.fit(X_train_pca, y_train)

        sum_score += clf.score(X_test_pca, y_test)

        y_pred = clf.predict(X_test_pca)

    print( sum_score/5 )

def k_near_svm_classifier( images, classify ):
    return

def svm_classifier( images, classify ):
    sum_score = 0
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(
            images, classify, test_size=0.25, random_state=42)

        clf = SVC(kernel='linear', class_weight='balanced')
        clf = clf.fit(X_train, y_train)
        sum_score += clf.score(X_test, y_test)

        y_pred = clf.predict(X_test)
        print( y_test )
        print(classification_report(y_test, y_pred, target_names=['0', '1']))

    print( sum_score/5 )

if len(sys.argv) != 5:
    print( 'There was ' + str(len(sys.argv)-1) + ' arguements' )
    print( 'Four arguements must be entered, 2 input text files, 1 output text file and an algorithm identifier' )

else:
    trainer_set = open(sys.argv[1], 'r')
    images, classify = load_training_set( trainer_set )
    print( "Images Loaded" )
    if int(sys.argv[4]) > 4 or int(sys.argv[4]) < 1:
        print( "Algorithm identifier must be between 1-4" )
    else:
        if int(sys.argv[4]) == 1:
            hog_svm_classifier(images, classify)
        elif int(sys.argv[4]) == 2:
            pca_svm_classifier(images, classify)
        elif int(sys.argv[4]) == 3:
            k_near_svm_classifier(images, classify)
        elif int(sys.argv[4]) == 4:
            svm_classifier(images, classify)
