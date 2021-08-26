import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup

class CsvDataset:

    def __init__(self, file):
        self.dataframe = pd.read_csv(file)
        self.val_df = None
        self.train_df = None
        self.val_ds = None
        self.train_ds = None

    def df_to_datasets(self, target):
        # frac(float): 要抽出的比例, random_state：隨機的狀態.
        self.val_df = self.dataframe.sample(frac=0.2, random_state=1337)
        # drop the colum 1 of 'class'.
        self.train_df = self.dataframe.drop(self.val_df.index)

        train_df = self.train_df.copy()
        val_df = self.val_df.copy()

        train_labels = train_df.pop(target)
        val_labels = val_df.pop(target)
        
        # tf.data.Dataset.from_tensor_slices(): 可以獲取列表或數組的切片。
        self.train_ds = tf.data.Dataset.from_tensor_slices((dict(train_df), train_labels))
        self.val_ds = tf.data.Dataset.from_tensor_slices((dict(val_df), val_labels))

        # shuffle(): 用來打亂數據集中數據順序.
        # buffer_size: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/
        self.train_ds = self.train_ds.shuffle(buffer_size=len(self.train_ds))
        self.val_ds = self.val_ds.shuffle(buffer_size=len(self.val_ds))

        return self.train_ds, self.val_ds


class EncodeFeatures:

    def __init__(self):
        self.feature_ds = None
            
    def numerical_feature(self, feature, name, dataset):
        # Create a Normalization layer for our feature
        normalizer = Normalization()

        # Prepare a Dataset that only yields our feature
        self.feature_ds = dataset.map(lambda x, y: x[name])
        self.feature_ds = self.feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the statistics of the data
        normalizer.adapt(self.feature_ds)

        # Normalize the input feature
        encoded_feature = normalizer(feature)
        return encoded_feature

    def categorical_feature(self, feature, name, dataset, is_string):
        lookup_class = StringLookup if is_string else IntegerLookup
        # Create a lookup layer which will turn strings into integer indices
        lookup = lookup_class(output_mode='binary')

        # Prepare a Dataset that only yields our feature
        self.feature_ds = dataset.map(lambda x, y: x[name])
        self.feature_ds = self.feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the set of possible string values and assign them a fixed integer index
        lookup.adapt(self.feature_ds)

        # Turn the string input into integer indices
        encoded_feature = lookup(feature)
        return encoded_feature


def input_features():
    # Categorical features encoded as integers
    sex = keras.Input(shape=(1,), name='sex', dtype='int64')
    cp = keras.Input(shape=(1,), name='cp', dtype='int64')
    fbs = keras.Input(shape=(1,), name='fbs', dtype='int64')
    restecg = keras.Input(shape=(1,), name='restecg', dtype='int64')
    exang = keras.Input(shape=(1,), name='exang', dtype='int64')
    ca = keras.Input(shape=(1,), name='ca', dtype='int64')

    # Categorical feature encoded as string
    thal = keras.Input(shape=(1,), name='thal', dtype='string')

    # Numerical features
    age = keras.Input(shape=(1,), name='age')
    trestbps = keras.Input(shape=(1,), name='trestbps')
    chol = keras.Input(shape=(1,), name='chol')
    thalach = keras.Input(shape=(1,), name='thalach')
    oldpeak = keras.Input(shape=(1,), name='oldpeak')
    slope = keras.Input(shape=(1,), name='slope')
    
    all_inputs = [
        sex,
        cp,
        fbs,
        restecg,
        exang,
        ca,
        thal,
        age,
        trestbps,
        chol,
        thalach,
        oldpeak,
        slope,
    ]
    return all_inputs


def specify_encoded_features(train_ds, all_inputs):

    encoded = EncodeFeatures()

    # Integer categorical features
    sex_encoded = encoded.categorical_feature(all_inputs[0], 'sex', train_ds, False)
    cp_encoded = encoded.categorical_feature(all_inputs[1], 'cp', train_ds, False)
    fbs_encoded = encoded.categorical_feature(all_inputs[2], 'fbs', train_ds, False)
    restecg_encoded = encoded.categorical_feature(all_inputs[3], 'restecg', train_ds, False)
    exang_encoded = encoded.categorical_feature(all_inputs[4], 'exang', train_ds, False)
    ca_encoded = encoded.categorical_feature(all_inputs[5], 'ca', train_ds, False)

    # String categorical features
    thal_encoded = encoded.categorical_feature(all_inputs[6], 'thal', train_ds, True)

    # Numerical features
    age_encoded = encoded.numerical_feature(all_inputs[7], 'age', train_ds)
    trestbps_encoded = encoded.numerical_feature(all_inputs[8], 'trestbps', train_ds)
    chol_encoded = encoded.numerical_feature(all_inputs[9], 'chol', train_ds)
    thalach_encoded = encoded.numerical_feature(all_inputs[10], 'thalach', train_ds)
    oldpeak_encoded = encoded.numerical_feature(all_inputs[11], 'oldpeak', train_ds)
    slope_encoded = encoded.numerical_feature(all_inputs[12], 'slope', train_ds)

    all_features = layers.concatenate(
        [
            sex_encoded,
            cp_encoded,
            fbs_encoded,
            restecg_encoded,
            exang_encoded,
            slope_encoded,
            ca_encoded,
            thal_encoded,
            age_encoded,
            trestbps_encoded,
            chol_encoded,
            thalach_encoded,
            oldpeak_encoded,
        ]
    )
    return all_features


if __name__ == '__main__':

    # Config TF to use GPU.
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset_csv_file = './datasets/heart.csv'
    target_value = 'target'
    all_model = './model_weights/all_model/08.25/heart_disease' # all_model: Model struct and model weights.

    heart_dataset = CsvDataset(file=dataset_csv_file)
    train_ds, val_ds = heart_dataset.df_to_datasets(target=target_value)

    # # .take(n): get n datas.
    # for x, y in train_ds.take(1):
    #     print('Input(Features):', x)
    #     print('Target:', y)

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    # Model build.
    all_inputs = input_features()
    all_features = specify_encoded_features(train_ds, all_inputs)
    x = layers.Dense(32, activation='relu')(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(all_inputs, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model train.
    model.fit(x=train_ds, epochs=50, verbose=1, validation_data=val_ds)

    # Model save.
    model.save(all_model)
    print('Model save done!')