# 알고리즘에 필요한 library

import base64
import datetime
import time
import io
import os
import keras
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as plt
import pandas as pd
import glob
import math
import cv2
import pickle
import umap as umap
import warnings

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cityblock
from math import cos, pi

from matplotlib import cm
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestCentroid

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import kshape
# from tslearn.clustering import KShape
from math import sqrt
from scipy import stats
from scipy.cluster import vq
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

from tensorflow.python.keras.backend import eager_learning_phase_scope
from keras.engine.topology import Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, Layer,LeakyReLU, \
    Conv1D, UpSampling1D, MaxPooling1D, Conv1DTranspose, ELU, Dropout, MaxPooling2D, UpSampling2D, concatenate, \
    Activation
from keras.layers import Input
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from keras import optimizers

# Dash에 필요한 library

from dash.dependencies import Input, Output, State
import dash_table
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc

from plotly.subplots import make_subplots

import plotly.graph_objects as go
import pickle
import copy
import pathlib
import urllib.request
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
# Multi-dropdown options
# from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS

# Download
from flask import Flask, send_file
from dash_extensions import Download

# 전역 변수


np.set_printoptions(threshold=np.inf)  # ...없이 출력하기

PATH = None
df = None
prev_path = None
pure_csv0 = None
pure_csv = None
change_cutting_method = 0

dataset = None
embedding_data = None
predict = None
dataset_pure = None
preprocessing_csv = None
process_label = None
csv_file_name = None

autoencoder_hist = None
dataset_pure_list = None
cutting_dataset = None
cutting_dataset_pure = None
embedding_csv = None

centroid_idx = None
centroid_value = None

len_min = None
len_max = None
len_mean = None

Kmean = KMeans(n_clusters=5)
pphist = None
emhist = None
# Function

"""

Data Proprocessing(0) : CSV로부터 DATA 입력받기
"""

# 데이터 입력 함수

"""
path : csv 파일의 경로
column : 데이터를 나타내는 칼럼명
"""


def align_timeseries_dataset(path, value_col, process_col=None):
    global prev_path
    global pure_csv
    global PATH
    global change_cutting_method

    global len_min
    global len_max
    global len_mean

    if path == None or value_col == None:
        return

    print("prev path : ", prev_path)
    print("global path : ", PATH)

    try:
        if pure_csv is None:
            print("pure_csv is None")
            input_csv = pd.read_csv(path, engine='python', encoding='euc-kr')
            input_csv = input_csv.astype({value_col: 'float32'})
            pure_csv = input_csv

        elif prev_path is None:
            print("if 문 2")
            input_csv = pure_csv

        elif change_cutting_method == 1 and PATH != prev_path:
            print("PATH != prev_path")
            input_csv = pd.read_csv(path, engine='python', encoding='euc-kr')
            input_csv = input_csv.astype({value_col: 'float32'})
            pure_csv = input_csv

        else:
            print("Else")
            input_csv = pure_csv

    except Exception as e:
        print(e)

    # input_csv = pd.read_csv(path, engine='python', encoding='euc-kr')
    # input_csv = input_csv.astype({value_col: 'float32'})

    # 결측치 제거
    input_csv = input_csv.dropna(subset=[value_col])

    # z_score
    zscore_dataset = z_score_normalize(input_csv[value_col])
    input_csv['z_score'] = zscore_dataset

    # min-max
    minmax_dataset = min_max_normalize(input_csv[value_col])
    input_csv['min_max'] = minmax_dataset

    # 전처리 csv 저장용 DataFrame
    preprocessing_csv = pd.DataFrame()

    # Process가 존재하지 않는 경우
    if process_col == None:
        process_set = None
        dataset_preprocessing = input_csv['min_max']
        dataset = input_csv[value_col]

    # Process가 존재하는 경우 process 별로 데이터 분리
    else:
        dataset_preprocessing = []
        dataset = []

        process_list = input_csv[process_col]
        process_set = list(set(process_list))

        for process in process_set:
            data = input_csv[(input_csv[process_col] == process)]
            data_preprocessing = data['min_max']
            data_pure = data[value_col]

            dataset_preprocessing.append(data_preprocessing)
            dataset.append(data_pure)

        preprocessing_csv['Process'] = process_list

        process_array = np.array(dataset_preprocessing)
        process_len = []

        for i in range(len(process_array)):
            process_len.append(process_array[i].shape[0])

        print("len 최대 : ", max(process_len))
        len_max = str(round(max(process_len), 3))
        print("len 최소 : ", min(process_len))
        len_min = str(round(min(process_len), 3))
        print("len 평균 : ", sum(process_len, 0.0) / len(process_len))
        len_mean = str(round(sum(process_len, 0.0) / len(process_len), 3))

    preprocessing_csv['Value'] = input_csv[value_col]
    preprocessing_csv['z_score'] = zscore_dataset
    preprocessing_csv['min_max'] = minmax_dataset

    return np.array(dataset_preprocessing), preprocessing_csv, process_set, np.array(dataset)


"""Data preprocessing function(1) - 시계열 데이터 길이 조절(truncation, padding, sliding window, DTW 등)"""


# 데이터 자르기 함수

def data_truncation(dataset):
    return_dataset = []
    max_size = 0
    min_size = 999999

    for i in range(len(dataset)):
        if dataset[i].size < min_size:
            min_size = dataset[i].size

    for i in range(len(dataset)):
        if dataset[i].size > min_size:
            return_dataset.append(dataset[i][:min_size])
        else:
            return_dataset.append(dataset[i])

    return np.array(return_dataset)


# 데이터 패딩 함수

def data_padding(dataset):
    return_dataset = []
    max_size = 0

    for i in range(len(dataset)):
        if dataset[i].size > max_size:
            max_size = dataset[i].size

    for i in range(len(dataset)):
        if dataset[i].size < max_size:
            return_dataset.append(np.pad(dataset[i], (0, max_size - dataset[i].size), 'constant', constant_values=0))
        else:
            return_dataset.append(dataset[i])

    return np.array(return_dataset)


# Sliding window 함수

def sliding_window(dataset, window_size=10, shift_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(window_size, shift=shift_size, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    return_dataset = list()

    for window in dataset:
        return_dataset.append(window.numpy())
    return_dataset = np.array(return_dataset)

    return return_dataset


# DTW 유사도를 통한 시계열 데이터 확장 함수 (작은 길이를 큰 길이로 맞춤)

def data_dtw(dataset):
    return_dataset = []
    max_size = 0
    max_index = 0

    for i in range(len(dataset)):
        if dataset[i].size > max_size:
            max_size = dataset[i].size
            max_index = i

    long_ts_data = dataset[max_index]

    for i in range(len(dataset)):
        if dataset[i].size < max_size:
            return_dataset.append(DTW_resize_algorithm(long_ts_data, dataset[i])[0])
        else:
            return_dataset.append(dataset[i])

    return np.array(return_dataset)


def DTW_resize_algorithm(long_ts_data, short_ts_data):
    if len(long_ts_data) == len(short_ts_data):
        return np.array(short_ts_data), np.array([0] * len(short_ts_data))

    step = 0
    similarity_degree_path = [0] * len(long_ts_data)
    long_ts_data = np.array(long_ts_data)
    short_ts_data = np.array(short_ts_data)

    path_coordinates = fastdtw(short_ts_data, long_ts_data)[1]

    for i in range(len(similarity_degree_path)):

        similarity_degree_path[i] = (long_ts_data[path_coordinates[step][1]] - short_ts_data[path_coordinates[step][0]])

        for j in range(step + 1, len(path_coordinates)):

            if path_coordinates[step][1] == path_coordinates[j][1]:
                similarity_degree_path[i] = similarity_degree_path[i] + (
                        long_ts_data[path_coordinates[j][1]] - short_ts_data[path_coordinates[j][0]])
                step = j
                continue

            else:
                step += 1
                break

    resize_ts_data = (long_ts_data - similarity_degree_path)

    return np.array(resize_ts_data), np.array(similarity_degree_path)


"""Data preprocessing function(2) - 데이터 정규화 or 일반화"""


# Min-Max Normalization: 모든 feature들의 스케일이 동일하지만, 이상치(outlier)를 잘 처리하지 못한다.

def min_max_normalize(lst):
    normalized = []

    min_value = min(lst)
    max_value = max(lst)

    for value in lst:
        normalized_num = (value - min_value) / (max_value - min_value)
        normalized.append(normalized_num)

    return np.array(normalized)


# Z-Score Normalization : 이상치(outlier)를 잘 처리하지만, 정확히 동일한 척도로 정규화 된 데이터를 생성하지는 않는다.

def z_score_normalize(lst):
    normalized = []

    mean_value = np.mean(lst)
    std_value = np.std(lst)

    for value in lst:
        normalized_num = (value - mean_value) / std_value
        normalized.append(normalized_num)
    return np.array(normalized)


"""Data preprocessing function(3) - 잠재 벡터 추출(UMAP, 이미지화 등)"""

"""
RP 알고리즘

serialize_vector : 시계열 데이터 value vector
"""


def RP_algorithm(serialize_vector):
    N = serialize_vector.size
    S = np.repeat(serialize_vector[None, :], N, axis=0)
    Z = np.abs(S - S.T)
    Z /= Z.max()
    Z *= 255
    Z = Z.astype('uint8')
    Z = np.array(Z)
    return Z


"""

GAF 알고리즘 

"""


def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    """To work with tabulate."""
    return (math.cos(a + b))


class GAF_algorithm:

    def __init__(self):
        pass

    def __call__(self, serie):
        """Compute the Gramian Angular Field of an image"""
        # Min-Max scaling
        min_ = np.amin(serie)
        max_ = np.amax(serie)
        scaled_serie = (2 * serie - max_ - min_) / (max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Polar encoding
        phi = np.arccos(scaled_serie)
        # Note! The computation of r is not necessary
        r = np.linspace(0, 1, len(scaled_serie))

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)

        gaf = (1 + gaf) * 255 / 2.0

        return gaf


"""
image_vector : 변경하고자 하는 이미지 리스트
img_size : 변경하고자 하는 이미지 크기
"""


def resize_img(image_vector_list, img_size):
    image_list = []

    for i in range(len(image_vector_list)):

        if len(image_vector_list[i]) > img_size:
            img = cv2.resize(image_vector_list[i], (img_size, img_size), interpolation=cv2.INTER_AREA)

        else:
            img = cv2.resize(image_vector_list[i], (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        img = img.astype('uint8')
        image_list.append(img)

    image_list = np.array(image_list)

    return image_list


# Auto_Encoder 함수

"""

dataset : (n,) shape의 데이터
LEARNING_LATE : 러닝레이트
BATCH_SIZE : 배치사이즈
EPOCHS : 에폭
TEST_SIZE : traing에서 뽑아낼 test 사이즈(default = 100)
IMG_SIZE : 이미지사이즈(default = 64)

"""


def embedding_AE(dataset, LEARNING_LATE, BATCH_SIZE, EPOCHS, IMAGING_FLAG='1', IMG_SIZE_FLAG='1', TEST_SIZE=500):
    global autoencoder_hist

    np.random.seed(1)
    tf.random.set_seed(1)

    latent_dim = 2
    dataset_img = []
    autoencoder_hist = []

    if IMAGING_FLAG == '1':
        for i in range(len(dataset)):
            dataset_img.append(RP_algorithm(dataset[i]))
    else:
        gaf = GAF_algorithm()
        for i in range(len(dataset)):
            dataset_img.append(gaf(dataset[i]))

    if IMG_SIZE_FLAG == '1':
        IMG_SIZE = 64
    elif IMG_SIZE_FLAG == '2':
        IMG_SIZE = 256
    else:
        IMG_SIZE = 512

    dataset_class_img = resize_img(dataset_img, IMG_SIZE)
    dataset_class_img = np.array(dataset_class_img)

    dataset_image = dataset_class_img.reshape(len(dataset), IMG_SIZE, IMG_SIZE)

    # Image -> train / test로 나누기

    train = dataset_image
    np.random.shuffle(train)

    train = train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    test = train[len(train) - TEST_SIZE:]
    train = train[:len(train) - TEST_SIZE]

    # 데이터 정규화

    train = train / dataset_image.max()
    test = test / dataset_image.max()

    # 체크포인트 설정

    ae_checkpoint_path = 'AE.ckpt'
    ae_checkpoint_dir = os.path.dirname(ae_checkpoint_path)

    ae_callback_early = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=50,
        verbose=0,
        mode='auto'
    )

    ae_callback_best = keras.callbacks.ModelCheckpoint(
        filepath=ae_checkpoint_path,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq=1
    )

    count = factor(IMG_SIZE)
    # AE Layer 설정

    encoder_input = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input')

    x = Conv2D(16, 3, strides=2, padding='same', activation='relu')(encoder_input)
    x = BatchNormalization()(x)

    lenl = 32

    for i in range(count - 4):
        x = Conv2D(lenl, 3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        lenl *= 2
    print(x.shape)
    x = Flatten()(x)
    print(x.shape)
    # x.reshape(,IMAG_SIZE)
    units = x.shape[1]

    # 2D 좌표로 표기하기 위하여 2를 출력값으로 지정
    embed = Dense(latent_dim, name='embedded')(x)

    x = Dense(units)(embed)
    x = Reshape((8, 8, IMG_SIZE))(x)

    lenl = IMG_SIZE / ((count % 2) + 1)

    for i in range(count - 4):
        x = Conv2DTranspose(lenl, 3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        lenl /= 2

    decoder_outputs = Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='sigmoid', name='output')(x)

    # 오토인코더 실행

    autoencoder = Model(encoder_input, decoder_outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_LATE), loss=tf.keras.losses.MeanSquaredError())

    # hist 저장
    for i in range(1, EPOCHS + 1):
        start = time.perf_counter()
        hist = autoencoder.fit(train, train, batch_size=BATCH_SIZE, epochs=1, validation_data=(test, test),
                               shuffle=True, callbacks=[ae_callback_early, ae_callback_best])
        end = time.perf_counter()

        execution_time = round(end - start, 3)

        result = str(i) + "  /  " + str(EPOCHS) + "  [  " + str(execution_time) + "  ms  ] " + "  -  loss  :  " + str(
            np.round(hist.history["loss"], 4))
                 # + " / val_loss : " + str(np.round(hist.history["val_loss"], 4))

        print(result)
        autoencoder_hist.append(result)
        autoencoder_hist.append(html.Br())

    autoencoder.summary()
    # hist = autoencoder.fit(train, train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test, test),
    #                       shuffle=True, callbacks=[ae_callback_early, ae_callback_best])

    # 이미지 그리기

    # decoded_images = autoencoder.predict(test)

    # draw_image_data(test,"Original Image",IMG_SIZE)
    # draw_image_data(decoded_images,"Reproduction Image",IMG_SIZE)

    autoencoder_hist.append("DONE!")

    # 인코딩된 잠재 벡터
    get_embedded = K.function([autoencoder.get_layer('input').input],
                              [autoencoder.get_layer('embedded').output])

    dataset_dimension = np.vstack(get_embedded([dataset_image]))
    dataset_dimension_data = np.vstack([dataset_dimension])
    dataset_dimension_data = dataset_dimension_data.reshape(-1, latent_dim)

    # history 정보 출력
    print(" history 정보 출력 ")
    for i in range(0, EPOCHS):
        print(i + 1, ' / ', EPOCHS, autoencoder_hist[i])

    return dataset_dimension_data


def factor(n):
    count = 0
    while n % 2 == 0:
        count += 1
        n = int(n / 2)
    return count


# UMAP 함수

"""
n_components : 축소하고자 하는 차원수
_n_neighbors : 작을수록 locality를 잘 나타내고, 커질수록 global structure를 잘 나타냄
_min_dist : 얼마나 점들을 조밀하게 묶을 것인지 (낮을 수록 조밀해짐)
"""


def embedding_UMAP(dataset, _n_neighbors=50, _min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=_n_neighbors, init='random', random_state=0, min_dist=_min_dist)
    embedding_2d = reducer.fit_transform(dataset)

    reducer = umap.UMAP(n_components=3, n_neighbors=_n_neighbors, init='random', random_state=0, min_dist=_min_dist)
    embedding_3d = reducer.fit_transform(dataset)
    return embedding_2d, embedding_3d


# PCA 함수
"""
n_component : 주성분 분석 개수
반환값 : PCA 실행 결과 
"""


def embedding_PCA(dataset, n_component=10):
    pca = PCA(n_components=n_component)
    pca.fit(dataset)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # print(pca.explained_variance_ratio_)

    pca = PCA(n_components=2)
    pca.fit(dataset)
    dataset_pca_2d = pca.transform(dataset)

    pca = PCA(n_components=3)
    pca.fit(dataset)
    dataset_pca_3d = pca.transform(dataset)

    return dataset_pca_2d, dataset_pca_3d, per_var


"""Clustering"""

# K-means 함수

"""
dimension_data=잠재벡터
MAX_CLUSTER_SIZE=최대 군집 개수
"""


def clustering_KMEANS(dimension_data, n_cluster, MAX_CLUSTER_SIZE=10):
    # class_dimension_data=잠재벡터,num_cluster=최대 군집 개수

    # n_cluster_list = cal_Silhouette(dimension_data, MAX_CLUSTER_SIZE, 5)
    # best_cluster = n_cluster_list[0]

    dimension_data = dimension_data.reshape(-1, 2)

    Kmean = KMeans(n_clusters=n_cluster)
    Kmean.fit(dimension_data)

    predict = Kmean.predict(dimension_data)

    # for center in n_cluster_list:
    #    draw_cluster_and_center(dimension_data, center)

    return predict


# K-shape 함수

"""
timeseries_data=
n_cluster=군집 수
"""


def clustering_KSHAPE(timeseries_data, n_cluster=2):
    ks = kshape(timeseries_data, n_clusters=n_cluster)
    # ks = KShape(n_clusters = n_cluster)
    cluster_found_kshape = ks.fit_predict(timeseries_data)
    print("cluster_found_kshape : ", cluster_found_kshape)

    return cluster_found_kshape


# DBSCAN 함수
"""
eps : 기준점으로부터의 거리
min_samples : 반경 내의 점의 개수
반환값 : dbscan 군집 결과
"""


def clsutering_DBSCAN(dataset, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    predict = dbscan.fit_predict(dataset)

    return predict


"""기타 함수"""


# 실루엣 다이어그램 그리는 함수

def plotSilhouette(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)

        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhoutte_avg = np.mean(silhouette_vals)
    plt.axvline(silhoutte_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('K')
    plt.xlabel('Silhouette value')
    plt.show()


#  실루엣 계수 계산 함수
#  max_cluster_num : 최대 클러스터 개수(k)
#  cluster_num : 추출할 실루엣 계수 높은 클러스터 개수

def cal_Silhouette(data, max_cluster_num, cluster_num):
    max = []

    for i in range(2, max_cluster_num + 1):
        km = KMeans(n_clusters=i, random_state=10)
        km_labels = km.fit_predict(data)
        max.append([i, silhouette_score(data, km_labels)])

        max.sort(key=lambda x: x[1], reverse=True)
    print("max : ", max)
    # 실루엣 계수 높은 상위 (clust_num)개만 추출해서 클러스터 개수 저장
    n_cluster_list = []
    n_cluster_value = []

    for i in max[0:cluster_num]:
        n_cluster_list.append(i[0])
        n_cluster_value.append(round(i[1] * 100, 2))

    return n_cluster_list, n_cluster_value


def draw_inertia_kshape(dimension_data, MAX_CLUSTER_SIZE=10):
    distortions_kshape = []

    for i in range(2, MAX_CLUSTER_SIZE + 1):
        #kshape = kshape(n_clusters=i, max_iter=100)
        kshape = KShape(n_clusters=i, max_iter=100)

        cluster_found = kshape.fit_predict(dimension_data)
        distortions_kshape.append(kshape.inertia_)

    plt.plot(range(2, MAX_CLUSTER_SIZE + 1), distortions_kshape, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


"""
cluster_list 의 k 값으로 실루엣 다이어그램 작성 함수

"""


def draw_Silhouette_Diagram(data, n_cluster_list):
    for i in range(len(n_cluster_list)):
        print("Cluster 개수 : ", n_cluster_list[i])

        km = KMeans(n_clusters=n_cluster_list[i], random_state=10)
        km_labels = km.fit_predict(data)
        plotSilhouette(data, km_labels)


"""
center : cluster center 수
data : cluster data
"""


def draw_cluster_and_center(data, center):
    Kmean = KMeans(n_clusters=center)
    Kmean.fit(data)

    plt.scatter(data[:, 0], data[:, 1], s=0.05, c=Kmean.labels_.astype(float))

    print("Center 개수 : ", center)
    for i in range(center):
        plt.scatter(Kmean.cluster_centers_[i, 0], Kmean.cluster_centers_[i, 1], s=50, c='red', marker='s')

    plt.show()


# 시계열 데이터 그리는 함수

"""
dimenstion_data : 2차원의 데이터
predict : 라벨 번호 list (정답)
label : 라벨 이름 list

"""


def draw_vector_data(dimenstion_data, predict=None, label=None, center=None):
    plt.figure(figsize=(15, 15))
    plt.rc('legend', fontsize='20')

    if predict is not None:
        if label is None:
            if center is None:
                plt.scatter(dimenstion_data[:, 0], dimenstion_data[:, 1], s=30, c=predict)
            else:
                plt.scatter(dimenstion_data[:, 0], dimenstion_data[:, 1], s=30, c=predict)
                plt.scatter(center[:, 0], center[:, 1], s=50, c='red', marker='s')
        else:
            plt.scatter(dimenstion_data[0], dimenstion_data[1], s=30, label=label, c=predict)

    else:
        plt.scatter(dimenstion_data[:, 0], dimenstion_data[:, 1], s=30, color='blue')


# 이미지 데이터 그리는 함수

"""
image_data : 이미지 데이터
title : 제목

n_images : 그릴 이미지 수
IMG_SIZE : reshape할 이미지 사이즈(image_data의 크기와 반드시 같아야함)

"""


def draw_image_data(image_data, title, n_images=5, IMG_SIZE=64):
    plt.figure(figsize=(100, 10))

    for i in range(n_images):
        ## display original
        ax = plt.subplot(1, n_images, i + 1)
        ax.set_title("Original Image")
        plt.imshow(image_data[i].reshape(IMG_SIZE, IMG_SIZE))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# 전처리데이터 다운로드 함수

def download_csv(path, dataframe, file_name):
    split_path = path.split('/')
    del split_path[-1]

    higher_path = "/".join(split_path)
    dataframe.to_csv(higher_path + '/' + file_name + '.csv')


# Outlier 보여주는 함수

def show_outlier(cutting_dataset, process_label, predict):
    outlier_list = []

    for index, value in enumerate(predict):
        if value == -1:
            outlier_list.append(index)

    if len(outlier_list) == 0:
        return

    outlier_list = np.array(outlier_list)

    draw_cnt = int(input("출력할 이상치 개수(현재 {}개의 이상치 탐색)".format(len(outlier_list))))

    if draw_cnt == 0:
        return

    for i in range(draw_cnt):
        idx = outlier_list[i]
        col_name = "Process " + str(process_label[idx])
        index_df = pd.DataFrame(cutting_dataset[idx], columns=[col_name])
        ax = index_df.plot()

    return ax


"""
show_scatter_to_plot : x,y 값으로 plot 그리는 함수
show_index_to_plot : index로 plot 그리는 함수
find_centroid_index : centroid와 가장 가까운 데이터(index,value) 반환하는 함수
"""


def whereclose(a, b, rtol=1e-05, atol=1e-08):
    return np.where(isclose(a, b, rtol, atol))


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


def find_scatter_index(embedding, x_value, y_value):
    finded_idx = whereclose(embedding, x_value)[0]

    for idx in finded_idx:
        if isclose(embedding[idx][1], y_value):
            return idx

    return None


def show_scatter_to_plot(embedding_dataset, cutting_dataset, process_label, x_value, y_value):
    idx = find_scatter_index(embedding_dataset, x_value, y_value)

    if idx is None:
        return None

    finded_process = process_label[idx]
    col_name = "Process " + str(finded_process)
    # index_df = pd.DataFrame(cutting_dataset[idx], columns=[col_name])
    # ax = index_df.plot()
    ax = px.line(cutting_dataset[idx])

    return ax


def show_index_to_plot(cutting_dataset, process_label, index):
    if index is None:
        return

    finded_process = process_label[index]
    col_name = "Process " + str(finded_process)
    index_df = pd.DataFrame(cutting_dataset[index], columns=[col_name])
    ax = index_df.plot()

    return ax


def find_centroid_index(embedding, predict):
    X = np.array(embedding)
    y = np.array(predict)
    clf = NearestCentroid()
    clf.fit(X, y)

    centroid = clf.centroids_
    centroid_idx = []

    predict_set = list(sorted(set(predict)))

    if predict_set[0] == -1:
        del predict_set[0]
        centroid = centroid[1:]

    for i in predict_set:
        closet_idx = None
        closet_distance = None

        for j in range(len(embedding)):
            if predict[j] == i:
                if closet_idx is None and closet_distance is None:
                    closet_idx = j
                    closet_distance = euclidean(centroid[i], embedding[j])
                else:
                    tmp_distance = euclidean(centroid[i], embedding[j])
                    if closet_distance > tmp_distance:
                        closet_idx = j
                        closet_distance = tmp_distance

        centroid_idx.append(closet_idx)
    return centroid_idx, centroid


def cal_rms(arr):
    square = 0
    mean = 0.0
    root = 0.0

    for i in range(len(arr)):
        square += (arr[i] ** 2)

    mean = (square / (float)(len(arr)))
    root = math.sqrt(mean)

    return root

#############################################################


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.css.config.serve_locally = True

# app.css.append_css({"external_url":"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.csshttps://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"})
server = app.server

# Create controls
# county_options = [
#     {"label": str(COUNTIES[county]), "value": str(county)} for county in COUNTIES
# ]
#
# well_status_options = [
#     {"label": str(WELL_STATUSES[well_status]), "value": str(well_status)}
#     for well_status in WELL_STATUSES
# ]
#
# well_type_options = [
#     {"label": str(WELL_TYPES[well_type]), "value": str(well_type)}
#     for well_type in WELL_TYPES
# ]

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "CLUSTERING FRAMEWORK",
                                )
                            ]
                        )
                    ],
                    className="title",
                    id="title",
                ),
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("kw.jpg"),
                            id="plotly-image",
                            width='90px',
                            height='30px',
                            className="img"
                        ),
                        html.Img(
                            src=app.get_asset_url("bistel.png"),
                            id="plotly-image2",
                            # width='90px',
                            # height='30px',
                            className="img2"

                        )
                    ],
                    className="logo",
                ),

            ],
            id="header",
            className="row flex-display",
        ),
        dcc.Tabs(
            id="app-tabs",
            value="tab1",
            className="custom-tabs",
            children=[
                dcc.Tab(
                    id="Preprocess-tab",
                    label="Data Preprocessing",
                    value="tab1",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[
                        html.Div(
                            [
                                html.Div(children=[
                                html.Div(
                                    [
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files', className='file')
                                            ]),
                                            style={
                                                "height": "150px",
                                                'lineHeight': '130px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=True, className="control_label"
                                        ),

                                        html.Div(
                                            children=[

                                                html.Div(
                                                    children=[
                                                        html.Div(html.Label('Is it splited?', className="dcc_control_l2"),className='inline'),
                                                        html.Div(

                                                            dcc.RadioItems(id='chk-process',
                                                                           options=[{'label': ' O ', 'value': 'o'},
                                                                                    {'label': ' X ',
                                                                                     'value': 'x'}], value='o',
                                                                           labelClassName='radio'),
                                                            className='inline', ), ]
                                                    , className='label_bm'
                                                ),


                                                html.Div(children=[
                                                    html.Div(children=[html.Label('Value Column', className="dcc_control_label"),
                                                                   dcc.Input(id='VC', type='text', value='Value',
                                                          className="input_val"),],className='pro_val'),
                                                    html.Div(children=[html.Label([
                                                "Process Column",

                                            ], id='PCN-label', className='dcc_control_label'),dcc.Input(id='PCN', className="input_val")],className='pro_val'),

                                                                   ],className='flex'),


                                                html.Button('Show Graph', id='graph-btn', n_clicks=0,
                                                            className="dcc_btn3")], className='label_tm'),
                                        html.Div(
                                            className='empty'
                                        ),

                                        html.Div(children=[
                                            html.Div(html.H3('Value Info'),
                                                     className='info_label'),
                                            html.Div(html.H3('Length Info'), className='info_label')
                                        ], style={'display': 'flex'}),

                                        html.Div(
                                            children=[
                                                html.Div(children=[html.Div(children=[
                                                    html.Label('Min : ', id='value_min0'),
                                                    html.Label('Max : ', id='value_max0'),
                                                    html.Label('Mean : ', id='value_mean0'),
                                                    html.Label('Variance : ', id='value_variance0'),
                                                    html.Label('RMS : ', id='value_rms0'), ],
                                                    className='info'
                                                ),

                                                html.Div(children=[

                                                    html.Label('111 ', id='value_min'),
                                                    html.Label('222', id='value_max'),
                                                    html.Label('333', id='value_mean'),
                                                    html.Label('444', id='value_variance'),
                                                    html.Label('555', id='value_rms'),
                                                ], className='info2',
                                                    id='value_info'),],className='info_half'),

                                                html.Div(children=[html.Div(children=[
                                                    html.Label('Min : ', id='len_min0'),
                                                    html.Label('Max : ', id='len_max0'),
                                                    html.Label('Mean : ', id='len_mean0'),
                                                ], className='info'
                                                ),

                                                html.Div(children=[
                                                    html.Label('111 ', id='len_min'),
                                                    html.Label('222', id='len_max'),
                                                    html.Label('333', id='len_mean'),
                                                ], className='info2')],className='info_half'),



                                            ], className='info3'),



                                        html.Div(
                                            children=[html.Div(children=[
                                                html.Label([
                                                    "Cutting method",html.Div([
                                                "?",html.Span([html.P("Truncation : 가장 짧은 데이터에 맞춰 나머지 \n        데이터를 자름"),
                                                               html.P("Padding : 가장 긴 데이터에 맞춰 나머지 데이터에 0을 추가"),
                                                               html.P("DTW : 유사성 측정 후, 유사성 보존하면서 길이를 늘림"),
                                                               html.A("DTW",href="https://en.wikipedia.org/wiki/Dynamic_time_warping", target='_blank')],className='tooltiptext')],className='q_pre'),
                                                    dcc.Dropdown(
                                                        id='cut_radio',
                                                        options=[{'label': i, 'value': i} for i in
                                                                 ['Truncation', 'Padding', 'DTW']],
                                                        value='', className="dcc_control3"
                                                    )
                                                ], id='CTM-label', className='dcc_input_p'), ],
                                                className='inline_flex'),
html.Div(children=[html.Div(html.Label(
                                                    ["Sliding Window",html.Div([
                                                "?",html.Span([html.P("Sliding Window : 데이터크기(TIME SLICE)와 이동간격(SHIFT SIZE)에 맞춰 데이터 셋 구성. 중첩가능")],className='tooltiptext')],className='q_pre2')],
                                                ),id='sw',className='cc_100'),html.Div(children=[html.Div(children=[html.Div(html.Label(
                                                    "TIME SLICE"
                                                ),className='cc_45',id='ts'),dcc.Input(id='TS', type='number',
                                                              className="ccc_45")], id='TS-label',className='c_45'),
html.Div(children=[html.Div(html.Label(
                                                    "SHIFT SIZE"
                                                ),className='cc_45',id='ss'),dcc.Input(id='SS', type='number',
                                                              className="ccc_45")], id='SS-label',className='c_45'),
],className='flex-display'),],className='c_75',id='sliding'),


                                                html.Button('Slice', id='slice-btn', n_clicks=0, className="dcc_btn2"),
],
                                            className='flex1'),
                                        html.Div(className='empty2'),
                                        html.Div(
                                            children=[
                                                html.Div(className='empty_inline'),
                                                html.Button("Download", id="prep_download-btn", n_clicks=0,
                                                            className="dcc_btn3"),
                                            ]
                                        ),

                                        Download(id="preprocessing_download"),
                                        html.Div(id='output-data-upload'),
                                        html.Div(id='graph'),
                                        html.Div([
                                            dbc.Modal(
                                                [
                                                    dbc.ModalHeader(),
                                                    dbc.ModalBody(html.H2("Cutting Success")),
                                                    dbc.ModalFooter(
                                                        dbc.Button("Close", id="close-md", className="ml-auto")
                                                    ),
                                                ],
                                                id="modal",
                                                size="sm",
                                                is_open=False,
                                                backdrop=True,
                                                # True, False or Static for modal to not be closed by clicking on backdrop
                                                scrollable=True,  # False or True if modal has a lot of text
                                                centered=True,  # True, False
                                                fade=True
                                            )
                                        ]),
                                    ],
                                    className="pretty_container_side1 "
                                ),
                            ],className="four columns"),

                                html.Div([
                                    html.Div(
                                        id="dataTableContainer",
                                        className="pretty_container_preprocessing_df"
                                    ),
                                    html.Div(
                                        id="GraphContainer",
                                        className="pretty_container_preprocessing_graph",
                                    )
                                ], id="preprocessing_right-column",
                                    className="eight columns",
                                ),
                            ],
                            className="column flex-display"
                        ),
                    ]),
                dcc.Tab(
                    id="Embedding-tab",
                    label="Embedding",
                    value="tab2",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[
                        html.Div([
                            html.Div([
                                html.Div(children=[html.Div(className="section-banner2", children="Previous Step"),
                                                   html.Div(id='pphistory', className='hist')], className='history'),
                                html.Div([
                                    html.Label([
                                        "Embedding method",html.Div([
                                                "?",html.Span([html.P("Embedding : 고차원의데이터를 저차원으로 축소")],className='tooltiptext')],className='q_em'),
                                        dcc.Dropdown(
                                            id='embedding_radio',
                                            options=[{'label': i, 'value': i} for i in ['Autoencoder', 'PCA', 'UMAP']],
                                            value='Autoencoder', className="dcc_control"
                                        )
                                    ], className='section-banner'),
                                    html.Div(children=[html.Div(html.H4('Autoencoder'), id='Autoencoder_label',
                                             className="dcc_control_h4"),
                                            # html.Div(html.Abbr('?',spellCheck='false', title="Hello, I am hover-enabled helpful information."),className='q'),
                                            # html.Div('?',className='q'),
                                            # html.Div(id='w',className='tooltip',children=[html.Label('!',className='q'),html.Div("고차원에서의 유사성을 저차원에서도",className='tooltiptext')]),

                                            # dbc.Label('?',id='ae',className='q'),
                                            # dbc.Tooltip("Tooltip on",id='tooltip',target='ae',placement="right",style={'font-size':19}),
# dbc.Tooltip("Tooltip on2",id='tooltip_pca',target='ae',placement="right",style={'font-size':19}),
# dbc.Tooltip(children=[html.P("고차원에서의 유사성을 저차원에서도"),html.P("유사하게 보존하기 위한 방법 ")],id='tooltip_umap',target='ae',placement="right",
#             style={'font-size':19,'padding':'5px','maxWidth': 600,})
html.Div([
                                                "?",html.Span(
[html.P("AE:입력층, 은닉층, 출력층으로 구성"),html.P("    은닉층에서 차원 축소 후"),html.P("    출력층이 입력층의 값을 복원하도록 학습")
    ,html.P("    데이터를 분할해 여러번 학습하는 과정 필요"),html.P("Learning_Rate : 학습률. 일반적으로 0.001이 많이 쓰임"),html.P("batch_size : 한번의 학습에 쓰일 데이터 크기")
    ,html.P("Epoch : 모든 데이터 학습 반복 횟수"),html.A("Wikipedia",href="https://en.wikipedia.org/wiki/Autoencoder#:~:text=An%20autoencoder%20is%20a%20type,to%20ignore%20signal%20%E2%80%9Cnoise%E2%80%9D.", target='_blank')],className='tooltiptext'
                                                ,id='tooltip'),html.Span(
[html.P("PCA :원데이터의 분산을 최대한 유지하려함"),html.P("       선형방식의 정사영을 하기 때문에 데이터들이 뭉개질 수 있음")
 ,html.A("Wikipedia",href="https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.", target='_blank')],className='tooltiptext2'
                                                ,id='tooltip_pca'),
                                                html.Span(
[html.P("UMAP : 고차원에서의 유사성을 저차원에서도"),html.P("           유사하게 보존하기 위한 방법 "),html.P("n_neighbors: 값이 클수록 고차원의 전체적인 구조를")
 ,html.P("                  작을수록 지역적인 구조를 파악"),html.P("min_dist: 저차원에서 두 벡터 사이의 최소 거리"),
 html.A("Docs of UMAP",href="https://umap-learn.readthedocs.io/en/latest/",target='_blank')],className='tooltiptext3'
                                                ,id='tooltip_umap')
                                            ],id='ae',className='q'),

                                    ],className='flex-display'),
                                    html.Div(
                                        children=[html.Div(children=[
                                            html.Div(html.Label('Learning_Rate'), id='Learning_Rate_label',
                                                     className="dcc_auto"),
                                            dcc.Input(id='Learning_Rate_input', type='number',
                                                      className="dcc_auto2"), ], className='auto_con'),
                                            html.Div(children=[
                                                html.Div(html.Label('Batch_Size'), id='Batch_Size_label',
                                                         className="dcc_auto"),
                                                dcc.Input(id='Batch_Size_input', type='number',
                                                          className="dcc_auto2"), ], className='auto_con'),
                                            html.Div(children=[html.Div(html.Label('Epoch'), id='Epoch_label',
                                                                        className="dcc_auto"),
                                                               dcc.Input(id='Epoch_input', type='number',
                                                                         className="dcc_auto2"), ],
                                                     className='auto_con'), ], className='flex-display'),

                                    # html.Div(html.Label('Test Data Size'), id='Test_Data_Size_label',
                                    #          className="dcc_control_label"),
                                    # dcc.Input(id='Test_Data_Size_input', type='number', className="dcc_control_label"),
                                    html.Div(id='autoencoder_options',
                                             children=[
                                                 html.Div(children=[html.Div(html.Label(" Imaging\nAlgorithm"),
                                                                             id='Imaging_Algorithm_label',
                                                                             className="dcc_control_image"),
                                                                    html.Div(children=[
                                                                        dcc.RadioItems(id='IMAGING_FLAG',
                                                                                       options=[{'label': 'RP',
                                                                                                 'value': '1'},
                                                                                                {'label': 'GAF',
                                                                                                 'value': '2'}],
                                                                                       className='radio_image'), ]
                                                                        , className='inline_image'), ],
                                                          className='inline_flex_image'),
                                                 html.Div(children=[
                                                     html.Div(html.Label("IMAGE\n  SIZE"), id='Imaging_Size_label',
                                                              className="dcc_control_image"),

                                                     html.Div(
                                                         children=[

                                                             dcc.RadioItems(id='IMAGING_SIZE_FLAG',
                                                                            options=[{'label': 'small', 'value': '1'},
                                                                                     {'label': 'middle', 'value': '2'},
                                                                                     {'label': 'large', 'value': '3'}],
                                                                            className='radio_image2'), ]
                                                         , className='inline_image2'), ],
                                                     className='inline_flex_image'),

                                             ]
                                             , className='flex-display-image'),

                                    html.Div(
                                        id='umap_op',
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div(html.Label('n_neighbors'), id='n_neighbors_label',
                                                             className="dcc_control_umap1"),
                                                    dcc.Input(id='n_neighbors_input', type='number',
                                                              className="dcc_control_umap2"),
                                                ], className='inline_flex_umap'),
                                            html.Div(
                                                children=[
                                                    html.Div(html.Label('min_dist'), id='min_dist_label',
                                                             className="dcc_control_umap1"),
                                                    dcc.Input(id='min_dist_input', type='number',
                                                              className="dcc_control_umap2"),
                                                ], className='inline_flex_umap'),

                                        ], className='umap_options'),

                                    html.Div(
                                        children=[
                                            html.Button('Embedding', id='embedding-btn', n_clicks=0,
                                                        className="dcc_btn4"),
                                            html.Button("Download", id="embedding_download-btn", n_clicks=0,
                                                        className="dcc_btn4"),
                                        ], className='flex-display'),

                                    Download(id="embedding_download"),
                                    html.Div(id='ae_history_div', children=[
                                        html.Label('Autoencoder is Updating', id='ae_hist'),
                                        dcc.Interval(id='interval_component', interval=1000, n_intervals=0)
                                    ], className="scroll_container_hist"),

                                ], className="pretty_container_side1 ",
                                )], className='four columns'),

                            html.Div([
                                html.Div(id='learn', children=[], className="pretty_container_embedding_graph"),
                                html.Div(id='loading_plot', children=[], className="scroll_container_graph")

                            ],
                                id="embedding_right-column",
                                className="eight columns",
                            ),
                        ], className="column flex-display"
                        )]),
                dcc.Tab(
                    id="Clustering-tab",
                    label="Clustering",
                    value="tab3",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[
                        html.Div([
                            html.Div([
                                html.Div(
                                    children=[html.Div(className="section-banner2", children="Previous Steps"),
                                              html.Div(id='emhistory1', className='hist'),
                                              html.Div(id='emhistory', className='hist2'), ], className='history'),
                                html.Div([
                                    html.Label([
                                        "Clustering method",
                                        dcc.Dropdown(
                                            id='clustering_radio',
                                            options=[{'label': i, 'value': i} for i in
                                                     ['K-Means', 'DBSCAN', 'K-shape']],
                                            value='K-Means', className="dcc_control"
                                        )
                                    ], className='section-banner'),
                                    html.Div(children=[
html.Div(html.H4('K-Means'), id='K-Means_label', className="dcc_control_label"),
html.Div([
                                                "?",html.Span([html.P("K-means : 군집끼리의 거리 차이의 분산을 최소화하는 방식으로 동작")
                                                               ,html.P("Silhouette Coefficient : 클러스터 내의 응집도와 클러스터 간의 분리도를 고려한 지표"),
                                                               html.A("Wikipedia",href="https://en.wikipedia.org/wiki/K-means_clustering", target='_blank')],className='tooltiptext',id="cl_k")
,html.Span([html.P("DBSCAN : 밀도 기반의 군집화 알고리즘. EPS(지정거리) 내에 MIN_SAMPLES(최소 샘플 개수)만큼 데이터가 존재한다면 하나의 군집으로 인식"),
            html.A("Wikipedia",href="https://en.wikipedia.org/wiki/DBSCAN",target='_blank')],className='tooltiptext2',id="cl_db")
,html.Span([html.P("K-shape : 데이터 간의 유사성을 직접 비교"),html.P("             계산량이 많음"),html.A("Efficient and Accurate Clustering of Time Series",href="http://www1.cs.columbia.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf", target='_blank')],className='tooltiptext3',id="cl_ks")],className='q_cl'),
                                    ],className='flex-display'),

                                    html.Div(
                                        children=[
                                            html.Div(id='k-means',
                                                     children=[
                                                         html.Div(html.Label('MAX_CLUSTER_SIZE'),
                                                                  id='MAX_CLUSTER_SIZE_label',
                                                                  className='cluster_label', ),
                                                         dcc.Input(id='MAX_CLUSTER_SIZE_input', type='number',
                                                                   className='cluster_max'),
                                                     ], className='cluster_flex_inline',
                                                     ),

                                            html.Div(id='dbscan',
                                                     children=[
                                                         html.Div(
                                                             children=[
                                                                 html.Div(html.Label('EPS'), id='EPS_label',
                                                                          className="dcc_control_dbscan"),
                                                                 dcc.Input(id='EPS_input', type='number',
                                                                           className="dcc_control_dbscan2"),

                                                             ], className='dcc_control_cluster'),
                                                         html.Div(
                                                             children=[
                                                                 html.Div(html.Label('MIN_SAMPLES'),
                                                                          id='MIN_SAMPLES_label',
                                                                          className="dcc_control_dbscan"),
                                                                 dcc.Input(id='MIN_SAMPLES_input', type='number',
                                                                           className="dcc_control_dbscan2"),
                                                             ], className='dcc_control_cluster'),

                                                     ], className='cluster_flex_inline2'),

                                            html.Button('Clustering', id='Clustering-btn', n_clicks=0,
                                                        className="cluster_btn"),
                                        ], className='cluster_flex'),
                                    html.Div(children=[
                                        html.Div(html.Label('Silhouette Coefficient'),
                                                 id='Silhouette_Coefficient_label', className='silhouette_label'),
                                        html.Div(dcc.Dropdown(
                                            id='k-means_clustering_radio',
                                            options=[],
                                            value=''
                                        ), className='silhouette_radio'), ], className='silhouette'),

                                ], className="pretty_container_side1 ",

                                )], className='four columns'),
                            html.Div([
                                html.Div(id='Cluster_Plot', children=[], className="scroll_container_plot", ),
                                html.Div(id='Cluster_Graph&Hover', style={'display': 'flex'},
                                         children=[html.Div(className='half_container', id='Cluster_Graph', children=[
                                             dcc.Graph(
                                                 id='cluster-result', )
                                         ]),
                                                   html.Div(className='half_container', id='Cluster_Hover',
                                                            children=[])], className="scroll_container_graph"),

                                html.Div(id='K_means_Cluster_Plot', children=[], className="scroll_container_plot", ),
                                html.Div(id='K_means_Cluster_Graph&Hover', style={'display': 'flex'}, children=[
                                    html.Div(className='half_container', id='K_means_Cluster_Graph', children=[
                                        dcc.Graph(
                                            id='K-Means-result')
                                    ]),
                                    html.Div(className='half_container', id='K_means_Cluster_Hover', children=[])],
                                         className="scroll_container_graph"
                                         ),

                                html.Div(id="Outlier_Plot", children=[], className="scroll_container_outlier")
                            ],
                                id="clustering_right-column",
                                className="eight columns",
                            ),
                        ],
                            className="column flex-display")
                    ])]
        )
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# # Create callbacks
# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("count_graph", "figure")],
# )

def parse_contents(contents, filename, date):
    global PATH
    global df
    global prev_path
    global change_cutting_method

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            global PATH
            global prev_path
            global df
            global change_cutting_method

            # Assume that the user uploaded a CSV file
            prev_path = PATH
            change_cutting_method = 0
            PATH = io.StringIO(decoded.decode('utf-8'))
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            prev_path = PATH
            change_cutting_method = 0
            PATH = io.StringIO(decoded.decode('utf-8'))
            df = pd.read_excel(io.BytesIO(decoded), engine='python', encoding='euc-kr')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df[:5].to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_data={
                'backgroundColor': '#161a28',
                'color': 'white',
                'font-family': "Open Sans",
                'text-align': 'center'
            },
            style_header={
                'backgroundColor': '#161a28',
                'color': 'white',
                'font-family': "Open Sans",
                'text-align': 'center'
            }
        )
    ])


@app.callback(Output('dataTableContainer', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    global csv_file_name

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        csv_file_name = list_of_names
        print("csv file name : ", csv_file_name)
        return children


# Show Graph 버튼 누르면 그래프 보여주는 함수

@app.callback([Output('GraphContainer', 'children'),
               Output('value_min', 'children'),
               Output('value_max', 'children'),
               Output('value_mean', 'children'),
               Output('value_variance', 'children'),
               Output('value_rms', 'children'),
               Output('value_info', 'style'),
               Output('len_min', 'children'),
               Output('len_max', 'children'),
               Output('len_mean', 'children'), ],

              Input('graph-btn', 'n_clicks'),
              Input('VC', 'value'),
              Input('chk-process', 'value'),
              Input('PCN', 'value'))
def show_graph(n_clicks, VC, chk, pcn):
    global PATH
    global df
    global dataset
    global dataset_pure
    global preprocessing_csv
    global process_label

    global len_min
    global len_max
    global len_mean

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'graph-btn' in changed_id:
        df = df.astype({VC: 'float32'})
        dataset = df[VC].to_numpy()
        dataset_pure = dataset

        value_min = str(round(dataset_pure.min(),3))
        value_max = str(round(dataset_pure.max(),3))
        value_mean = str(round(dataset_pure.mean(),3))
        value_variance = str(round(dataset_pure.var(),3))
        value_rms = str(round(cal_rms(dataset_pure),3))

        if chk == 'x':
            len_min = ''
            len_max = ''
            len_mean = ''

        else:
            # read_and_separate(PATH,VC, pcn)
            try:
                dataset, preprocessing_csv, process_label, dataset_pure = align_timeseries_dataset(PATH, VC, pcn)
            except:
                print("시계열 데이터 cutting ERROR")

        children = [
            dcc.Graph(
                id='example-graph',
                style={"width": "100%", "height": "100%", "color": "white"},
                figure=px.line(df[VC], color_discrete_sequence=["#f4d44d"], template='plotly_dark').update_layout({
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "xaxis": dict(
                        showline=False, showgrid=False, zeroline=False
                    ),
                    'xaxis_title': "time",
                    "yaxis": dict(
                        showgrid=False, showline=False, zeroline=False
                    ),
                    "autosize": True,
                    "showlegend": False
                })

            )
        ]
        return [children, value_min, value_max, value_mean, value_variance, value_rms,
                {'display': 'inline-block', 'margin-right': '25px'}, len_min, len_max, len_mean]
    else:
        return ['', 'ㅤ', 'ㅤ', 'ㅤ', 'ㅤ', 'ㅤ', {'display': 'inline-block', 'margin-right': '95px'}, 'ㅤ', 'ㅤ', 'ㅤ']