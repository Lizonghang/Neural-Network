import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# KTF.set_session(sess)

from tools import *
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Activation
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.models import load_model

EPOCH = 10000
BATCH_SIZE = 512

train_X, train_y = load_train_data()
test_X = load_test_data()

validation_X = train_X[:100, ...]
validation_y = train_y[:100, ...]
train_X = train_X[100:, ...]
train_y = train_y[100:, ...]

"""
model = Sequential()
model.add(
    Conv2D(
        input_shape=(96, 96, 1),
        filters=32,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer='truncated_normal',
        use_bias=True,
        bias_initializer='zeros'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )
)
model.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer='truncated_normal',
        use_bias=True,
        bias_initializer='zeros'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )
)
model.add(
    Flatten()
)
model.add(
    Dense(
        units=512,
        activation='relu',
        kernel_initializer='truncated_normal',
        use_bias=True,
        bias_initializer='zeros',
        kernel_regularizer='l2',
        bias_regularizer='l2'
    )
)
model.add(
    Dropout(rate=0.5)
)
model.add(
    Dense(
        units=512,
        activation='relu',
        kernel_initializer='truncated_normal',
        use_bias=True,
        bias_initializer='zeros',
        kernel_regularizer='l2',
        bias_regularizer='l2'
    )
)
model.add(
    Dropout(rate=0.5)
)
model.add(
    Dense(
        units=30,
        activation='sigmoid',
        kernel_initializer='truncated_normal',
        use_bias=True,
        bias_initializer='zeros',
        kernel_regularizer='l2',
        bias_regularizer='l2'
    )
)
"""

"""
input_node = Input(shape=(96, 96, 1))
base_model = VGG19(weights=None, include_top=False, input_tensor=input_node)
flatten = Flatten()(base_model.output)
dense1 = Dense(
    units=512,
    activation='relu',
    kernel_initializer='truncated_normal',
    use_bias=True,
    bias_initializer='zeros',
    kernel_regularizer='l2',
    bias_regularizer='l2')(flatten)
dense2 = Dense(
    units=512,
    activation='relu',
    kernel_initializer='truncated_normal',
    use_bias=True,
    bias_initializer='zeros',
    kernel_regularizer='l2',
    bias_regularizer='l2')(dense1)
output_node = Dense(
    units=30,
    activation=None,
    kernel_initializer='truncated_normal',
    use_bias=True,
    bias_initializer='zeros',
    kernel_regularizer='l2',
    bias_regularizer='l2')(dense2)
model = Model(inputs=base_model.input, outputs=output_node)
"""

"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(30))
"""

model = load_model('ckpt/model.h5')

# from keras.utils import multi_gpu_model
# parallel_model = multi_gpu_model(model, gpus=2)
parallel_model = model

parallel_model.compile(
    optimizer=Adam(lr=1e-4),
    loss='mse',
    metrics=['mse']
)


class ImageLogCallback(Callback):
    def on_train_begin(self, logs=None):
        import shutil
        if os.path.exists('ckpt'):
            shutil.rmtree('ckpt')
        os.mkdir('ckpt')
        print 'initialize ckpt'

    def on_epoch_end(self, epoch, logs=None):
        model.save('ckpt/model.h5')
        if np.mod(epoch, 50) == 0:
            predict = parallel_model.predict(
                x=test_X,
                batch_size=BATCH_SIZE,
                verbose=0
            )
            make_submission(predict)
        # display(validation_X[0], y_true=validation_y[0], y_pred=predict[0], savefig=True)

parallel_model.fit(
    x=train_X,
    y=train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    verbose=2,
    validation_data=None,
    shuffle=True,
    callbacks=[ImageLogCallback()],
    initial_epoch=399
)

# parallel_model.fit_generator(
#     generator=ImageDataGenerator().flow(train_X, train_y, batch_size=128),
#     steps_per_epoch=train_X.shape[0] / BATCH_SIZE,
#     epochs=EPOCH
# )