import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
import keras
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torch.optim as optim
# from torchvision.models import resnet18
# from PIL import Image
import pandas as pd
import seaborn as sns
from datetime import datetime

file_directory_all = ['C:/Users/hou/Desktop/radar_competition/r1/',
                    'C:/Users/hou/Desktop/radar_competition/f1/',
                    'C:/Users/hou/Desktop/radar_competition/f2/',
                    'C:/Users/hou/Desktop/radar_competition/f3/',
                    'C:/Users/hou/Desktop/radar_competition/fff1/',
                    'C:/Users/hou/Desktop/radar_competition/fff2/',
                    'C:/Users/hou/Desktop/radar_competition/fff3/',
                    'C:/Users/hou/Desktop/radar_competition/non/']

def process_file(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'DS1' in f:
            range_velocity_data = f['DS1'][:]
        else:
            print(f"DS3 dataset not found in {file_path}")
            range_velocity_data = None
    return range_velocity_data

def custom_ticks(ticks):
    return [f'{tick:.0f}' if tick.is_integer() else f'{tick:.1f}' for tick in ticks]

data_list = []
for file_directory in file_directory_all:
    for filename in os.listdir(file_directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(file_directory, filename)
            print(f"Processing file: {file_path}")
            range_velocity_data = process_file(file_path)

            if range_velocity_data is not None:
                sum_channels = range_velocity_data.shape[0]
                
                channel_data = range_velocity_data[0, :, :, :]
                heat_map_data = np.sum(channel_data, axis=1)
                data_list.append(heat_map_data)
                
                # plt.figure(figsize=(12, 6), dpi=50)
                # plt.imshow(heat_map_data, cmap='hot', interpolation='nearest')
                # plt.colorbar()
                # plt.title(f"Heat map for {os.path.basename(file_path)} - Channel 0")
                
                # plt.xlim(0, heat_map_data.shape[1])
                # plt.ylim(0, heat_map_data.shape[0])
                
                # plt.xticks(np.arange(0, heat_map_data.shape[1], step=20))
                # plt.yticks(np.arange(0, heat_map_data.shape[0], step=5))

                # plt.show()
                    
                for channel in range(sum_channels):
                    channel_data = range_velocity_data[channel, :, :, :]
                    heat_map_data = np.sum(channel_data, axis=0)
                    data_list.append(heat_map_data)

                    # plt.figure(figsize=(12, 6), dpi=50)
                    # plt.imshow(heat_map_data, cmap='hot', interpolation='nearest')
                    # plt.colorbar()
                    # plt.title(f"Heat map for {os.path.basename(file_path)} - Channel {channel+1}")
                    
                    # plt.xlim(0, heat_map_data.shape[1])
                    # plt.ylim(0, heat_map_data.shape[0])
                    
                    # if channel == 0:
                    #     original_ticks = np.arange(0, heat_map_data.shape[0], step=7.5)
                    #     mapped_ticks = np.linspace(-1, 1, len(original_ticks))
                    #     plt.yticks(original_ticks, custom_ticks(mapped_ticks))
                    # else:
                    #     original_ticks = np.arange(0, heat_map_data.shape[0], step=5)
                    #     mapped_ticks = np.linspace(-30, 30, len(original_ticks))
                    #     plt.yticks(original_ticks, custom_ticks(mapped_ticks))
                        
                    # plt.xticks(np.arange(0, heat_map_data.shape[1], step=20))

                    # plt.show()

data_array = np.array(data_list)
data_array = data_array / np.max(data_array)  # 正規化數據

# 重新組織數據，每三筆為一組
num_samples = data_array.shape[0] // 3
data_pairs = np.zeros((num_samples, data_array.shape[1], data_array.shape[2], 3))

for i in range(num_samples):
    data_pairs[i, :, :, 0] = data_array[3 * i]
    data_pairs[i, :, :, 1] = data_array[3 * i + 1]
    data_pairs[i, :, :, 2] = data_array[3 * i + 2]

# 正確的資料
correct_data = data_pairs[:20, :, :, :]
correct_labels = np.ones((20,))  # 標籤為1表示正確資料

# 不正確的資料
incorrect_data = data_pairs[20:, :, :, :]
incorrect_labels = np.zeros((140,))

# 選組進行訓練
incorrect_data_p1 = data_pairs[20:40, :, :, :]
incorrect_data_p2 = data_pairs[40:60, :, :, :]
incorrect_data_p3 = data_pairs[60:80, :, :, :]
incorrect_data_p4 = data_pairs[80:100, :, :, :]
incorrect_data_p5 = data_pairs[100:120, :, :, :]
incorrect_data_p6 = data_pairs[120:140, :, :, :]
incorrect_data_p7 = data_pairs[140:160, :, :, :]

# %%%%%%%%%%%%%%%%%%%%
def create_model():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(32, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

X = np.array(range(20))
kf = KFold(n_splits=4,shuffle=True)
kf.get_n_splits(X)
ave_confuse=[]

for i, (train_index, test_index) in enumerate(kf.split(X)):
    # model2
    train_data = np.concatenate([correct_data[train_index], 
                                 incorrect_data_p4[train_index],
                                 incorrect_data_p5[train_index],
                                 incorrect_data_p6[train_index],
                                 ])
    train_labels = np.concatenate([np.ones(15), np.zeros(45)])
    test_data = np.concatenate([correct_data[test_index], 
                                 incorrect_data_p4[test_index],
                                 incorrect_data_p5[test_index],
                                 incorrect_data_p6[test_index],
                                 ])
    test_labels = np.concatenate([np.ones(5), np.zeros(15)])  
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    model = create_model()

    # 訓練模型
    hist = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_val, y_val))
    # 繪製訓練過程圖表
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(len(train_acc))


    plt.figure()
    plt.plot(xc, train_acc, label='Train Accuracy')
    plt.plot(xc, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(xc, train_loss, label='Train Loss')
    plt.plot(xc, val_loss, label='Validation Loss')
    plt.legend()
    plt.show()

    # 儲存模型
    now = datetime.now()
    model.save(f"model_fold_2_{i+1}.h5")
    # allmodel
    train_data = np.concatenate([correct_data[train_index], 
                                 incorrect_data_p1[train_index],
                                 incorrect_data_p2[train_index],
                                 incorrect_data_p3[train_index],
                                 incorrect_data_p4[train_index],
                                 incorrect_data_p5[train_index],
                                 incorrect_data_p6[train_index],
                                 incorrect_data_p7[train_index],
                                 ])
    train_labels = np.concatenate([np.ones(15), np.zeros(105)])
    test_data = np.concatenate([correct_data[test_index], 
                                 incorrect_data_p1[test_index],
                                 incorrect_data_p2[test_index],
                                 incorrect_data_p3[test_index],
                                 incorrect_data_p4[test_index],
                                 incorrect_data_p5[test_index],
                                 incorrect_data_p6[test_index],
                                 incorrect_data_p7[test_index],
                                 ])
    test_labels = np.concatenate([np.ones(5), np.zeros(35)])  
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    model = create_model()

    # 訓練模型
    hist = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_val, y_val))
    # 繪製訓練過程圖表
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(len(train_acc))


    plt.figure()
    plt.plot(xc, train_acc, label='Train Accuracy')
    plt.plot(xc, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(xc, train_loss, label='Train Loss')
    plt.plot(xc, val_loss, label='Validation Loss')
    plt.legend()
    plt.show()

    # 儲存模型
    now = datetime.now()
    model.save(f"model_fold_all_{i+1}.h5")
    # %concat
    model1_name = f'C:/Users/hou/Desktop/radar_competition/model_fold_all_{i+1}.h5'
    model2_name = f'C:/Users/hou/Desktop/radar_competition/model_fold_2_{i+1}.h5'
    model1 = tf.keras.models.load_model(model1_name)
    model2 = tf.keras.models.load_model(model2_name)
    predictions1 = model1.predict(test_data)
    predictions1_labels = (predictions1 > 0.1).astype("int32")  # 二分類預測

    # 顯示模型1的混淆矩陣
    cm1 = confusion_matrix(test_labels, predictions1_labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for allModel 1')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 找出數值為1的index
    indices = np.where(predictions1_labels == 1)[0]

    print(indices)

    # 取得模型2的預測結果
    predictions2 = model2.predict(test_data[indices])
    predictions2_labels = (predictions2 > 0.5).astype("int32")  # 二分類預測

    # 顯示模型2的混淆矩陣
    cm2 = confusion_matrix(test_labels[indices], predictions2_labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Model 2')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    if np.size(cm2)==1:
        cm2=[[0,0],[0,cm2.item()]]
    if i==0:
        save_confuse1=cm1
        save_confuse2=cm2
    else:
        save_confuse1=np.hstack((save_confuse1,cm1))
        save_confuse2=np.hstack((save_confuse2,cm2))
save_confuse_final=np.vstack((save_confuse1,save_confuse2))
