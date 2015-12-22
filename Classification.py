import sys
import numpy as np

from skimage import color
from skimage.io import imread
from skimage.transform import resize

from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.grid_search import GridSearchCV
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

def hog_feature_extraction( images, classify ):
    return

def pca_feature_extraction( images, classify ):
    X_train, X_test, y_train, y_test = train_test_split(
        images, classify, test_size=0.25, random_state=42)
    print( "Train and test set" )

    pca = RandomizedPCA(n_components=len(X_train), whiten=True).fit(X_train)
    print( "PCA fitted" )
    eigen_images = pca.components_.reshape((len(X_train), 200, 200))
    print( "Eigen images created" )
    X_train_pca = pca.transform(X_train)
    print( "Train pca created" )
    X_test_pca = pca.transform(X_test)
    print( "Test pca created" )
    print(X_test_pca)

    clf = SVC(kernel='linear', class_weight='balanced')
    print( "SVC created" )
    clf = clf.fit(X_train_pca, y_train)
    print( "SVC fitted" )

    y_pred = clf.predict(X_test_pca)
    print(y_pred)
    print(classification_report(y_test, y_pred, target_names=['0','1']))


def classified_images( classified_set, output_loaction ):
    return


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
            print(1)
        elif int(sys.argv[4]) == 2:
            pca_feature_extraction(images, classify)
        elif int(sys.argv[4]) == 3:
            print(3)
        elif int(sys.argv[4]) == 4:
            print(4)
