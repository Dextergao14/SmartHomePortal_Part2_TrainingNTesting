# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021
@author: Dexter
"""
import cv2
import numpy as np
import os
import csv
import re
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

import tensorflow as tf

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


# =============================================================================
# Helper Classes and Functions
# =============================================================================


class GestureDtls:
    """
     A class to handle Gesture Details
     Ex: GestureDetails("FanUp", "Increase Fan Speed", "13")
    """

    def __init__(self, gesture_Id, gesture_name, output_label):
        self.gesture_Id = gesture_Id
        self.gesture_name = gesture_name
        self.output_label = output_label


class GestureFeature:


    def __init__(self, gesture_detail: GestureDtls, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature


def extractFeature(folder_path, input_file, mid_frame_ctrer):

    middle_image = cv2.imread(frameExtractor(folder_path + input_file, folder_path + "frames/", mid_frame_ctrer),
                              cv2.IMREAD_GRAYSCALE)
    feature_extracted = HandShapeFeatureExtractor.extract_feature(HandShapeFeatureExtractor.get_instance(),
                                                                  middle_image)
    return feature_extracted


def getGesByName(gesture_file_name):
    """
        A Function to get gesture given its file name
    """
    for ele in gesture_data:
        if ele.gesture_Id == gesture_file_name.split('_')[0]:
            return ele
    return None


# a list to containg all gestures and thier details (Id, name, label)
gesture_data = [GestureDtls("Num0", "0", "0"), GestureDtls("Num1", "1", "1"),
                GestureDtls("Num2", "2", "2"), GestureDtls("Num3", "3", "3"),
                GestureDtls("Num4", "4", "4"), GestureDtls("Num5", "5", "5"),
                GestureDtls("Num6", "6", "6"), GestureDtls("Num7", "7", "7"),
                GestureDtls("Num8", "8", "8"), GestureDtls("Num9", "9", "9"),
                GestureDtls("FanDown", "Decrease Fan Speed", "10"),
                GestureDtls("FanOn", "FanOn", "11"), GestureDtls("FanOff", "FanOff", "12"),
                GestureDtls("FanUp", "Increase Fan Speed", "13"),
                GestureDtls("LightOff", "LightOff", "14"), GestureDtls("LightOn", "LightOn", "15"),
                GestureDtls("SetThermo", "SetThermo", "16")
                ]

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================

fVectors = []
train_data_path = "traindata/"
ctr = 0
for file in os.listdir(train_data_path):
    if not file.startswith('frames'):
        fVectors.append(GestureFeature(getGesByName(file),
                                       extractFeature(train_data_path, file, ctr)))
        ctr = ctr + 1


# =============================================================================
# Recognize the gesture (use cosine simlrty for comparing the vectors)
# =============================================================================

def gesDetection(gesture_folder_path, gesture_file_name, mid_frame_ctrer):

    video_feature = extractFeature(gesture_folder_path, gesture_file_name, mid_frame_ctrer)

    flag = True
    modis = 0
    gesture_detail: GestureDtls = GestureDtls("", "", "")
    while flag and modis < 5:
        simlrty = 1
        pos = 0
        idx = 0
        for featureVector in fVectors:
            cos = tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature, axis=-1)
            if cos < simlrty:
                simlrty = cos
                pos = idx
            idx = idx + 1
        gesture_detail = fVectors[pos].gesture_detail
        flag = False
        if flag:
            modis = modis + 1
    return gesture_detail


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================

test_data_path = "test/"
test_ctr = 0
print(fVectors)
with open('Results.csv', 'w', newline='') as results_file:
    fields_names = ['Output_Label']
    data_writer = csv.DictWriter(results_file, fieldnames=fields_names)
    data_writer.writeheader()

    for test_file in os.listdir(test_data_path):
        if not test_file.startswith('frames') and test_file.endswith('.mp4'):
            recognized_g_dtl = gesDetection(test_data_path, test_file, test_ctr)
            test_ctr = test_ctr + 1

            if recognized_g_dtl is None:
                print("Object is None")
            else:
                # access object property
                print(recognized_g_dtl.gesture_name)
            if recognized_g_dtl is not None:
                data_writer.writerow({'Output_Label': recognized_g_dtl.output_label})
            else:
                data_writer.writerow({'Output_Label': "4"})

        print(test_ctr)
