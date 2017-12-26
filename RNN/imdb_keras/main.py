import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from utils import *
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

MAX_LEN = 200
BATCH_SIZE = 256
EPOCH = 50
OPTIMIZER = 'adam'

data = load_data(MAX_LEN, shuffle=False)

# model = build_lstm_model(data.max_features, MAX_LEN)
# model = build_bidirectional_lstm_model(data.max_features, MAX_LEN)
model = build_gru_model(data.max_features, MAX_LEN)

print model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=OPTIMIZER,
    metrics=['accuracy']
)

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')
if not os.path.exists('logs'):
    os.makedirs('logs')

model.fit(
    x=data.X_train,
    y=data.y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    validation_data=(data.X_test, data.y_test),
    callbacks=[
        ModelCheckpoint('ckpt/model.h5', save_best_only=True),
        TensorBoard('logs'),
        ReduceLROnPlateau(verbose=1)
    ]
)
