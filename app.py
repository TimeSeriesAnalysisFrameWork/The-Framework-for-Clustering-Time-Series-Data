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

