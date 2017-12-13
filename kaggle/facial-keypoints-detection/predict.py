import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from tools import *
from keras.models import load_model

model = load_model('ckpt/model.h5')

test_data = load_test_data()

predict = model.predict(
    x=test_data,
    batch_size=64,
    verbose=1
)

make_submission(predict)

# if not os.path.exists('img'):
#     os.mkdir('img')
# for i in range(test_data.shape[0]):
#     display(test_data[i], y_pred=predict[i], savefig=True)
