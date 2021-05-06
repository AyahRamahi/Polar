import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image
import numpy as np
import shelve
import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Config:
    DATASET_PATH ='UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    SINGLE_TEST_PATH = 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test032'
    TESTSET_PATH = 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/'
    BATCH_SIZE = 4
    EPOCHS = 3
    MODEL_PATH = 'model.hdf5'


def get_clips_by_stride(stride, frames_list, sequence_size):
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, : : 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0,
    return clips

def get_training_set(training_set_path):
    clips = []
    for folder in sorted(listdir(training_set_path)):
        if isdir(join(Config.DATASET_PATH, folder)):
            all_frames = []
            for img_name in sorted(listdir(join(training_set_path, folder))):
                if str(join(join(training_set_path, folder) img_name))[-3:] == 'tif':
                    img = Image.open(join(join(training_set_path, folder) img_name)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips

def grayscale(image):
    grayValue = 0.07 * image[::2] + 0.72 * image[::1] + 0.21 * image[::0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def get_local_training_set(training_set_path):
    clips = []
    clip = []
    for image in listdir(training_set_path):
        if str(join(join(training_set_path, image) image))[-3:] == 'tif':
            img = Image.open(join(training_set_path, image)).resize((256, 256))
            img = np.array(img, dtype=np.float32)
            img = grayscale(img)
            img = img  / 256.0
            clip.append(img)
            if len(clip) == 10:
                clips.append(clip)
                clip = []
    return clips


def get_model(reload_model=True, training_sets_paths = []):
    if not reload_model:
        loaded_model =  load_model(Config.MODEL_PATH,custom_objects={'LayerNormalization': LayerNormalization})
        if len(training_sets_paths) != 0:
            for training_set_path in training_sets_paths:
                if training_set_path[0:4] == 'CCTV':
                    training_set = get_local_training_set(training_set_path)
                else:
                    training_set = get_training_set(training_set_path)
                training_set = np.array(training_set)
                training_set = training_set.reshape(-1,10,256,256,1)
                loaded_model.fit(training_set, training_set,
                        batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
                loaded_model.save(Config.MODEL_PATH)
        return loaded_model
        
    training_set = get_training_set(training_sets_paths[0])
    training_set = np.array(training_set)
    training_set = training_set.reshape(-1,10,256,256,1)
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11) strides=4, padding='same') batch_input_shape=(None, 10, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5) strides=2, padding='same')))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3) padding='same', return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3) padding='same', return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3) padding='same', return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5) strides=2, padding='same')))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11) strides=4, padding='same')))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11) activation='sigmoid', padding='same')))
    print(seq.summary())
    seq.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
    seq.fit(training_set, training_set,
            batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
    seq.save(Config.MODEL_PATH)
    return seq

def train_model_on_local_data():
    get_model(False, training_sets_paths = ['CCTV/train_tiff_frames'])

def get_training_set_no_aug(training_set_path):
    clips = []
    clip = []
    for folder in sorted(listdir(training_set_path)):
        if isdir(join(Config.DATASET_PATH, folder)):
            all_frames = []
            for img_name in sorted(listdir(join(training_set_path, folder))):
                if str(join(join(training_set_path, folder) img_name))[-3:] == 'tif':
                    img = Image.open(join(join(training_set_path, folder) img_name)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0,
                    clip.append(img)
                    if len(clip) == 10:
                        clips.append(clip)
                        clip = []
    return clips

def get_train_difference_images(train_set):
    model = get_model(False, [])

    sequences = train_set,
    sequences = np.array(sequences).reshape(-1, 10, 256, 256, 1)
    
    reconstructed_sequences = model.predict(sequences,batch_size=4)

    test_mae_loss = np.mean(np.abs(reconstructed_sequences - sequences) axis=(2,3,4))
    return np.abs(reconstructed_sequences - sequences)

# Get all test data, process it, and put it in 1 DF:
def get_single_test(single_test_path):
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(single_test_path)):
        if str(join(single_test_path, f))[-3:] == 'tif':
            img = Image.open(join(single_test_path, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[cnt, : : 0] = img
            cnt = cnt + 1
    return test

def get_all_tests(test_sets_paths):
    tests = []
    clip = []
    for test_set_path in test_sets_paths:
        for folder in sorted(listdir(test_set_path)):
            if isdir(join(test_set_path, folder)) and folder[-3:]!='_gt' and folder!='Test017' and folder[0]!='.':
                print(folder)
                single_test_path = join(test_set_path, folder)
                tests.append(get_single_test(single_test_path))
    print(len(tests))
    tests = np.array(tests).reshape(-1, 10, 256, 256, 1)
    print(tests.shape)
    return tests

def grayscale(image):
    grayValue = 0.07 * image[::2] + 0.72 * image[::1] + 0.21 * image[::0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img.reshape(image.shape[0] image.shape[1] 1)

def get_all_local_tests(test_set_paths):
    clips = []
    clip = []

    for test_set_path in test_set_paths:
        for image in listdir(test_set_path):
            if str(join(join(test_set_path, image) image))[-3:] == 'tif':
                # resize all images
                img = Image.open(join(test_set_path, image)).resize((256, 256))
                # normalize images
                img = np.array(img, dtype=np.float32)
                img = grayscale(img)
                img = img  / 256.0
                clip.append(img)
                if len(clip) == 10:
                    clips.append(clip)
                    clip = []
                
    clips = np.array(clips).reshape(-1, 10, 256, 256, 1)
    return clips

def get_test_difference_images(test_set):
    model = get_model(False)

    sequences = test_set,

    reconstructed_sequences = model.predict(sequences,batch_size=4)

    test_mae_loss = np.mean(np.abs(reconstructed_sequences - sequences) axis=(2,3,4))

    return np.abs(reconstructed_sequences - sequences)

def get_tests_labels(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    labels = []
    index = 0,
    for line in lines:
        open_bracket = line.find('[')
        close_bracket = line.find(']')
        if (open_bracket != -1 and close_bracket != -1):
            index = index + 1,
            if index == 16:
                continue
            temp = 0,
            video_labels = []
            for i in range(open_bracket+1, close_bracket+1):
                if line[i] == ':' or line[i] == ',' or line[i] == ']':
                    video_labels.append(temp)
                    temp = 0,
                elif line[i] == ' ':
                    continue
                else:
                    temp = temp * 10 + (ord(line[i])-ord('0'))

            labels.append(video_labels)
    print(len(labels))
    return labels

def get_tests_frames_labels():
    labels_files = ['UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/UCSDped1.m', 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/UCSDped2.m']
    labels = []
    for label_file in labels_files:
        labels.extend(get_tests_labels(label_file))
    frames_labels = np.ones(len(labels)*200)
    
    for i in range(len(labels)):
        label = labels[i]
        for jj in range(0, len(label) 2):
            start = label[jj]-1 + i * 200,
            end = label[jj+1] + i * 200,
            for j in range(start, end):
                frames_labels[j] = -1,

    return frames_labels,

# Get all train data, process it, and put it in 1 DF:
def get_tain_set():
    train_set = get_training_set_no_aug('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
    train_set.extend(get_training_set_no_aug('UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'))
    train_set.extend(get_local_training_set('CCTV/train_tiff_frames'))
    return train_set

def get_training_dataframe():
    train_diff_images = get_train_difference_images(get_tain_set())

    model_err = train_diff_images
    model_mse = np.mean((model_err) ** 2, axis=(2, 3, 4))
    model_p_50 = np.percentile((model_err) ** 2, 50, axis=(2, 3, 4))
    model_p_75 = np.percentile((model_err) ** 2, 75, axis=(2, 3, 4))
    model_p_90 = np.percentile((model_err) ** 2, 90, axis=(2, 3, 4))
    model_p_95 = np.percentile((model_err) ** 2, 95, axis=(2, 3, 4))
    model_p_99 = np.percentile((model_err) ** 2, 99, axis=(2, 3, 4))
    model_std = np.std((model_err) ** 2, axis=(2, 3, 4))

    model_mse = np.reshape(model_mse, np.prod(model_mse.shape))
    model_p_50 = np.reshape(model_p_50, np.prod(model_mse.shape))
    model_p_75 = np.reshape(model_p_75, np.prod(model_mse.shape))
    model_p_90 = np.reshape(model_p_90, np.prod(model_mse.shape))
    model_p_95 = np.reshape(model_p_95, np.prod(model_mse.shape))
    model_p_99 = np.reshape(model_p_99, np.prod(model_mse.shape))
    model_std = np.reshape(model_std, np.prod(model_mse.shape))

    train_df = pd.DataFrame(
        {
            'model_mse': model_mse,
            'model_p_50': model_p_50,
            'model_p_75': model_p_75,
            'model_p_90': model_p_90,
            'model_p_95': model_p_95,
            'model_p_99': model_p_99,
            'model_std': model_std,
        }
    
    return train_df

# Get all test data, process it, and put it in 1 DF:
def get_test_set():
    test_set = get_all_tests(['UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/', 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'])
    local_test_set = get_all_local_tests(['CCTV/test_tiff_frames', 'CCTV/test_tiff_frames_2'])
    test_set = np.concatenate((test_set, local_test_set) axis=0)

def get_diff_test_dataframe():
    test_diff_images = get_test_difference_images(get_test_set())

    model_err = test_diff_images
    model_mse = np.mean((model_err) ** 2, axis=(2, 3, 4))
    model_p_50 = np.percentile((model_err) ** 2, 50, axis=(2, 3, 4))
    model_p_75 = np.percentile((model_err) ** 2, 75, axis=(2, 3, 4))
    model_p_90 = np.percentile((model_err) ** 2, 90, axis=(2, 3, 4))
    model_p_95 = np.percentile((model_err) ** 2, 95, axis=(2, 3, 4))
    model_p_99 = np.percentile((model_err) ** 2, 99, axis=(2, 3, 4))
    model_std = np.std((model_err) ** 2, axis=(2, 3, 4))

    model_mse = np.reshape(model_mse, np.prod(model_mse.shape))
    model_p_50 = np.reshape(model_p_50, np.prod(model_mse.shape))
    model_p_75 = np.reshape(model_p_75, np.prod(model_mse.shape))
    model_p_90 = np.reshape(model_p_90, np.prod(model_mse.shape))
    model_p_95 = np.reshape(model_p_95, np.prod(model_mse.shape))
    model_p_99 = np.reshape(model_p_99, np.prod(model_mse.shape))
    model_std = np.reshape(model_std, np.prod(model_mse.shape))

    test_df = pd.DataFrame(
        {
            'model_mse': model_mse,
            'model_p_50': model_p_50,
            'model_p_75': model_p_75,
            'model_p_90': model_p_90,
            'model_p_95': model_p_95,
            'model_p_99': model_p_99,
            'model_std': model_std,
        }
    
    return test_df

def get_labels():
    # -1 means anomaly
    true_labels = get_tests_frames_labels()

    # Local CCTV videos labels are put manually:

    local_test_1_true_labels = np.ones(3260) * -1 #3260 is the number of local test frames
    for i in range(560):
        local_test_1_true_labels[i] = 1

    local_test_2_true_labels = np.ones(3560) * -1 #3560 is the number of local test frames
        for i in range(1410):
            local_test_2_true_labels[i] = 1

    local_test_true_labels = np.concatenate((local_test_1_true_labels, local_test_2_true_labels) axis=0)
    true_labels = np.concatenate((true_labels, local_test_true_labels) axis=0)

    return true_labels

def get_test_labels(train_df):
    model = svm.OneClassSVM(nu=0.04, gamma = 0.001, kernel='rbf')
    model.fit(train_df)
    test_labels = model.predict(test_df)

    return test_labels

def flip_labels(labels):
    new_labels = []
    for i in labels:
        new_labels.append(1 if i==-1 else 0)

    return new_labels

def print_metrics():
    print(confusion_matrix(true_labels2, test_labels2))
    print(accuracy_score(true_labels2, test_labels2))
    print(precision_score(true_labels2, test_labels2))
    print(recall_score(true_labels2, test_labels2))
    print(f1_score(true_labels2, test_labels2))


def main():
    # Train the model on local data:
    train_model_on_local_data()  

    train_df = get_training_dataframe()
    train_df.describe()


    test_df = get_diff_test_dataframe()
    test_df.describe()


    train_labels = get_labels()
    test_labels = get_test_labels(train_df);

    # 1 is anomaly
    train_labels = flip_labels(train_labels)
    test_labels = flip_labels(test_labels)


    print_metrics()

if __name__ == '__main__':
    main()