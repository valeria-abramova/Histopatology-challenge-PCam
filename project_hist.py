# import the necessary packages
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
import skimage.color
import color_transfer
from sklearn.ensemble import RandomForestClassifier
from imutils import paths
import numpy as np
from skimage.feature import hog
from skimage.feature import greycomatrix, greycoprops
import imutils
import cv2
import os
import random
import mahotas as mt
import h5py
import pickle
from Random_forest import Forest_classifier_ROC


def hog_feature(image):
    """

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    flattened ndarray
        HOG descriptor

    """

    hogf = hog(image, feature_vector=True)
    return hogf

def haralick(roi):
    """

    Parameters
    ----------
    roi : numpy array
        Grayscale ROI

    Returns
    -------
    ndarray of np.double
        The array of 13 Haralick features

    """
    textures = mt.features.haralick(roi)


    return np.asarray(textures.mean(axis=0))

def tas(image):

    """

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    1D ndarray of feature values

    """

    return mt.features.pftas(image)

def extract_color_histogram(image, bins=(8, 8, 8)):
    """

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    1D ndarray
        flattened color histogram

    """

    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()

def lbp_feature(image):
    """

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    1D ndarray
        flattened histogram of lbp

    """

    b = [p for p in range(0, 100)]

    lbp = local_binary_pattern(image[:,:,2], 8, 1, 'uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=b)

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist.flatten()

def glcm_feature(image):
    """

    Parameters
    ----------
    image : ndarray
            grayscale image

    Returns
    -------
    list of calculated glcm features

    """


    glcm = greycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
    glcm_dissimilarity = greycoprops(glcm, 'contrast')[0, 0]
    glcm_corr = greycoprops(glcm, 'correlation')[0, 0]
    glcm_unif = greycoprops(glcm, 'ASM')[0, 0]


    return glcm_dissimilarity, glcm_corr, glcm_unif

def color_statistics(image):
    """
    gurus.pyimagesearch.com;

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    list of color statistics values

    """

    (means, stds) = cv2.meanStdDev(image)

    return means, stds

def mser_feature(image):

    """

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    1D np.array with a number of MSER regions

    """
    mser = cv2.MSER_create()
    msers, bboxes = mser.detectRegions(image)


    return np.reshape(np.asarray(msers).shape[0],1)

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist data\\train\\m0"))
imagePaths.extend(list(paths.list_images("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist data\\train\\b0")))
random.shuffle(imagePaths)

# grab validation paths
val_paths = list(paths.list_images("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist data\\val\\m0"))
val_paths.extend(list(paths.list_images("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist data\\val\\b0")))
random.shuffle(val_paths)


# initialize the features matrix and labels list

features = []
labels = []
features_val=[]
labels_val=[]
#
#path to the extracted bovw and extracting image indexes in a list

bovwDB = h5py.File("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist\\bovw.hdf5")
featuresDB = h5py.File("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist\\sumVal_features.hdf5", mode="r")
list_ids = [x for x in featuresDB["image_ids"]]

bovwDB_val = h5py.File("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist\\bovw_val.hdf5")
featuresDB_val = h5py.File("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist\\sumVal_features_valid_set.hdf5", mode="r")
list_ids_val = [x for x in featuresDB_val["image_ids"]]

# source image for color normalization
source = cv2.imread("C:\\Users\\z\\Desktop\\UdG\\CAD\\hist data\\train\\m_try\\m_trainImage98.png", cv2.IMREAD_UNCHANGED)

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # read the image
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)

    # extract image id and image label
    imageID = imagePath.split(os.path.sep)[-1].split(".")[0]
    label = imagePath.split(os.path.sep)[-1].split("_")[0]

    if label == "b":
        label = 0
    elif label == "m":
        label = 1

    # find index of the current image in the features file
    index = list_ids.index(imageID)

    # color normalization
    image = color_transfer.color_transfer(source,image)

    #extract features
    hist = extract_color_histogram(image)
    hist_rgb = extract_color_histogram(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))

    glcm1 = np.reshape(glcm_feature(image[:, :, 2])[0], 1)
    glcm2 = np.reshape(glcm_feature(image[:, :, 2])[1], 1)
    glcm3 = np.reshape(glcm_feature(image[:, :, 2])[2], 1)

    ts = tas(image)

    col_stat1 = np.asarray(color_statistics(image)[0]).flatten()
    col_stat2 = np.asarray(color_statistics(image)[1]).flatten()

    col_stat_rgb1 = np.asarray(color_statistics(skimage.color.hsv2rgb(image))[0]).flatten()
    col_stat_rgb2 = np.asarray(color_statistics(skimage.color.hsv2rgb(image))[1]).flatten()

    hist = np.concatenate((hist, hist_rgb,  col_stat1, col_stat2, col_stat_rgb1, col_stat_rgb2,glcm1, glcm2, glcm3, ts, bovwDB["bovw"][index]))
    features.append(hist)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

features = np.array(features)
labels = np.array(labels)

trainFeat = features
trainLabels = labels

for (i, valPath) in enumerate(val_paths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(valPath, cv2.IMREAD_UNCHANGED)
    imageID = valPath.split(os.path.sep)[-1].split(".")[0]

    index = list_ids_val.index(imageID)

    label = valPath.split(os.path.sep)[-1].split("_")[0]

    if label == "b":
        label = 0
    elif label == "m":
        label = 1

    image = color_transfer.color_transfer(source, image)

    hist = extract_color_histogram(image)

    glcm1 = np.reshape(glcm_feature(image[:, :, 2])[0], 1)
    glcm2 = np.reshape(glcm_feature(image[:, :, 2])[1], 1)
    glcm3 = np.reshape(glcm_feature(image[:, :, 2])[2], 1)

    ts = tas(image)
    hist_rgb = extract_color_histogram(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))

    col_stat1 = np.asarray(color_statistics(image)[0]).flatten()
    col_stat2 = np.asarray(color_statistics(image)[1]).flatten()

    col_stat_rgb1 = np.asarray(color_statistics(skimage.color.hsv2rgb(image))[0]).flatten()
    col_stat_rgb2 = np.asarray(color_statistics(skimage.color.hsv2rgb(image))[1]).flatten()


    hist = np.concatenate((hist, hist_rgb, col_stat1, col_stat2, col_stat_rgb1, col_stat_rgb2, glcm1, glcm2, glcm3,
                            ts, bovwDB_val["bovw"][index]))

    features_val.append(hist)
    labels_val.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(val_paths)))

features_val = np.array(features_val)
labels_val = np.array(labels_val)

testFeat = features_val
testLabels = labels_val

Forest_classifier_ROC(trainFeat, trainLabels, testFeat, testLabels)
