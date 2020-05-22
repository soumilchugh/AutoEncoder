import os, argparse
from pathlib import Path
import tensorflow as tf
import cv2
import imutils
import glob
import numpy as np
from numpy import array, exp
from collections import defaultdict
from tensorflow.python.client import timeline
import json
import numpy as np
from collections import defaultdict
import json
from scipy.spatial import distance
import matplotlib
import time
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import random
import copy
from PIL import Image
from scipy.stats import truncnorm
from scipy import spatial

import math

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

filepath = Path("C:/Users/soumi/Documents/test-Autoencoder").glob('*.jpg')

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def randomCircle(imagegray, spurious):
    height = 240
    width = 320
    #num_reflections = np.random.randint(1, 10)
    counter = 0
    for i in np.arange(0, 100):
        if counter >= spurious:
            break
        mask = np.zeros([height,width]).astype(np.uint8)
        x1 = random.randint(0,width)
        y1 = random.randint(0,height)
        gauss_dist = get_truncated_normal(mean=200, sd=55, low=128, upp=255)
        size = np.random.randint(1, 5)
        cv2.circle(mask,(x1,y1),size,(255,255,255),-1) # Red
        mask_ind = np.argwhere(mask.astype(np.float)== 255)
        g = gauss_dist.rvs(len(mask_ind))
        imagegray[mask_ind[:,0],mask_ind[:,1]] = g
        counter = counter + 1


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        tf.import_graph_def(graph_def, name = "load")
        flops = tf.profiler.profile(graph,options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP = ', flops.total_float_ops)

    with tf.Session(graph = graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #output = tf.get_default_graph().get_tensor_by_name("load/fcrelu:0")
        output = tf.get_default_graph().get_tensor_by_name("load/Inference/Output:0")
        inputPlaceholder = tf.get_default_graph().get_tensor_by_name("load/Input:0")
        isTrain = tf.get_default_graph().get_tensor_by_name("load/isTrain:0")
        global falseCount
        global trueCount
        finalAccuracyList = []
        closestList = []
        filenames = [file for file in filepath]
        previous = []
        current = []
        for file in filenames:
            filename, file_extension = os.path.splitext(str(file))
            name = str(os.path.basename(filename));
            print (name)
            test_data = []
            image = cv2.imread(str(file))
            imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            imagegray = clahe.apply(imagegray)
            randomCircle(imagegray,5)
            normalised_image = imagegray
            normalised_image1 = normalised_image.astype(float)/255.0
            normalised_image1 = np.expand_dims(normalised_image1, axis=2)
            test_data.append(normalised_image1)
            start = time.perf_counter()
            test_mask2 = sess.run(output,feed_dict={inputPlaceholder:test_data, isTrain:False})
            elapsed = time.perf_counter() - start
            print ("elapsed",elapsed)
            print (test_mask2.shape)
            test_mask = test_mask2[0]
            '''
            current = test_mask
            previous.append(current)
            for p in previous:
                result = 1 - spatial.distance.cosine(p, current)
                print (result)
            '''
            output_mask = (((test_mask[:,:,0]).astype(np.float)))
            cv2.imshow('Output',output_mask)
            cv2.imshow('Input',imagegray)
            cv2.imwrite('image.jpg',output_mask)
            k = cv2.waitKey(0)
            if k == 27:
                break


load_graph("C:/Users/soumi/Documents/Autoencoder-Model/model/model_learningRate_0.0001epochs_151_channelsize_16adam_mse/130/frozen_model.pb")
