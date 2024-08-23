
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, HWResultReceiver, FeatureMapReceiver
import numpy as np
import time
import tensorflow as tf
import h5py

# 載入模型
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 進行模型預測
# def predict(interpreter, input_data):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#
#     # 設定模型的輸入
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#
#     # 獲取模型的輸出
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     return output_data



# count=0
def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_60cm")  # Set the setting folder name
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)             # Start the setting process
    # ksp.startSetting(SettingConfigs)        # Start the setting process in sub_thread

def process_file(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'DS1' in f:
            radar_signal_data = f['DS1'][:]
        else:
            print(f"DS1 dataset not found in {file_path}")
            radar_signal_data = None
    return radar_signal_data

def startLoop():
    # kgl.ksoclib.switchLogMode(True)
    # R = RawDataReceiver(chirps=32)
    count = 0
    # Receiver for getting Raw data
    R = FeatureMapReceiver(chirps=32)       # Receiver for getting RDI PHD map
    # R = HWResultReceiver()                  # Receiver for getting hardware results (gestures, Axes, exponential)
    # buffer = DataBuffer(100)                # Buffer for saving latest frames of data
    R.trigger(chirps=32)                             # Trigger receiver before getting the data
    time.sleep(0.5)
    print('# ====================================== Start getting gesture ====================================')
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    stack_res = np.zeros([2, 32, 32, 150], dtype=np.float32)
    while count < 150:                             # loop for getting the data
        res = R.getResults()                # Get data from receiver
        if res is None:
            continue
        print('frame = {}'.format(count))          # Print results
        stack_res[:, :, :, count] = res
        count = count+1
        time.sleep(0.03)
        '''
        Application for the data.
        '''
    data_list = []

    # stack_res = process_file('C:/Users/hsinlei/Desktop/gesture_data/r1/Background_0004_2024_07_23_15_50_29.h5')
    # stack_res = process_file('C:/Users/hsinlei/Desktop/gesture_data/fff1/Background_0007_2024_08_05_15_54_36.h5')
    # stack_res = process_file('C:/Users/hsinlei/Desktop/gesture_data/f1/Background_0007_2024_07_23_16_01_52.h5')

    if stack_res is not None:
        sum_channels = stack_res.shape[0]

        channel_data = stack_res[0, :, :, :]
        heat_map_data = np.sum(channel_data, axis=1)
        data_list.append(heat_map_data)
        for channel in range(sum_channels):
            channel_data = stack_res[channel, :, :, :]
            heat_map_data = np.sum(channel_data, axis=0)
            data_list.append(heat_map_data)

    data_array = np.array(data_list)
    data_array = data_array / np.max(data_array)
    data_pairs = np.zeros((1, data_array.shape[1], data_array.shape[2], 3))
    data_pairs[0, :, :, 0] = data_array[0]
    data_pairs[0, :, :, 1] = data_array[1]
    data_pairs[0, :, :, 2] = data_array[2]
    data_pairs=data_pairs.astype(np.float32)
    # res = np.zeros([1, 32, 150, 3], dtype=np.float32)
    return data_pairs

def recognition(final_feature):
    # h5 模型檔案路徑
    model_1_path = r'C:\Users\hsinlei\Desktop\competition\model_fold_all_1.h5'
    model_2_path = r'C:\Users\hsinlei\Desktop\competition\model_fold_2_1.h5'

    # TFLite 模型檔案路徑
    # model_1_path = r'C:\Users\hsinlei\Desktop\pycharm_model\dataset1+non\2024-08-21_11-17-37_my_model_converted.tflite'
    # model_2_path = r'C:\Users\hsinlei\Desktop\pycharm_model\dataset2+non\2024-08-21_11-16-14_my_model_converted.tflite'

    # 載入模型
    # interpreter_1 = load_model(model_1_path)
    # interpreter_2 = load_model(model_2_path)
    interpreter_1 = tf.keras.models.load_model(model_1_path)
    interpreter_2 = tf.keras.models.load_model(model_2_path)
    input_data = final_feature
    output_1 = (interpreter_1.predict(input_data) > 0.1).astype("int32")
    output_2 = (interpreter_2.predict(input_data) > 0.5).astype("int32")

    # output_1 = (predict(interpreter_1, input_data) > 0.5).astype("int32")
    # output_2 = (predict(interpreter_2, input_data) > 0.3).astype("int32")
    result = output_1*output_2

    value = result[0][0]
    print(value)
    if value == 1:
        print("=================This is the correct gesture!=================")
    else:
        print("=================This is not the correct gesture. Please try again!=================")

def main():
    # kgl.setLib()
    # # kgl.ksoclib.switchLogMode(True)
    # connect()  # First you have to connect to the device
    # startSetting()  # Second you have to set the setting configs
    # final_feature = startLoop()                             # Last you can continue to get the data in the loop
    # recognition(final_feature)
    while True:
        kgl.setLib()
        # kgl.ksoclib.switchLogMode(True)
        connect()  # First you have to connect to the device
        startSetting()  # Second you have to set the setting configs
        final_feature = startLoop()                             # Last you can continue to get the data in the loop
        recognition(final_feature)
        print("按任意鍵開始重新識別...")
        input()  # 等待用戶按下 Enter 鍵
        print("重啟識別...")


if __name__ == '__main__':
    main()
