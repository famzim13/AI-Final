import sys
import numpy as np

from skimage import color
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import RandomizedPCA

# Split input line from the training text file
def line_split( line ):
    split_line = line.split(' ')
    return split_line[0], int(split_line[1])

# Extract features from the set of training images
def load_training_set( file ):
    images = []
    classify = []
    img_size = [250, 250]
    img_classify = 0

    try:
        for line in file:
            img_url, img_classify = line_split( line )
            classify = np.hstack((classify, img_classify))
            img = imread(img_url)
            img = color.rgb2gray(img)
            img = resize(img, img_size)
            images.append(img)
    finally:
        file.close()

    return images, classify

def hog_feature_extraction( images, classify ):
    return

def load_files( file_set ):
    default_slice = (slice(0, 250), slice(0, 250))

    h_slice, w_slice = default_slice
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    n_faces = len( file_set )
    faces = np.zeros((n_faces, h, w), dtype=np.float32)

    for i, file_set in enumerate( file_set ):
        img = imread( file_set )
        face = np.asarray(img[default_slice], dtype=np.float32)
        face /= 255.0
        face = imresize(face, [250, 250])
        face = face.mean(axis=2)
        faces[i, ...] = face

    return faces

def classified_images( classified_set, output_loaction ):
    return


if len(sys.argv) != 5:
    print( 'There was ' + str(len(sys.argv)-1) + ' arguements' )
    print( 'Four arguements must be entered, 2 input text files, 1 output text file and an algorithm identifier' )
else:
    trainer_set = open(sys.argv[1], 'r')
    images, classify = load_training_set( trainer_set )
#    print( imread(zero_trainer_set[0]))
#    zero_faces = load_files( zero_trainer_set )
#    one_faces = load_files( one_trainer_set )
#    zero_X = zero_faces.data
#    zero_n_featues = zero_X.shape[1]
#    one_X = one_faces.data
#    one_n_features = one_X.shape[1]
#    pca.fit(zero_faces.data)
    if int(sys.argv[4]) > 4 or int(sys.argv[4]) < 1:
        print( "Algorithm identifier must be between 1-4" )
    else:
        if int(sys.argv[4]) == 1:
            print(1)
        elif int(sys.argv[4]) == 2:
            print(2)
        elif int(sys.argv[4]) == 3:
            print(3)
        elif int(sys.argv[4]) == 4:
            print(4)
#        classified_images( classified_set, sys.argv[4] )
