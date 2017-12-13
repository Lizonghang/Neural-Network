import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from tools import *
from keras.models import load_model


def make_submission(test_labels):
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv('dataset/IdLookupTable.csv')
    values = []

    cols = ["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y",
            "left_eye_inner_corner_x", "left_eye_inner_corner_y", "left_eye_outer_corner_x",
            "left_eye_outer_corner_y", "right_eye_inner_corner_x", "right_eye_inner_corner_y",
            "right_eye_outer_corner_x", "right_eye_outer_corner_y", "left_eyebrow_inner_end_x",
            "left_eyebrow_inner_end_y", "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
            "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y", "right_eyebrow_outer_end_x",
            "right_eyebrow_outer_end_y", "nose_tip_x", "nose_tip_y", "mouth_left_corner_x",
            "mouth_left_corner_y", "mouth_right_corner_x", "mouth_right_corner_y", "mouth_center_top_lip_x",
            "mouth_center_top_lip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][cols.index(row.FeatureName)],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('dataset/submission.csv', index=False)


if __name__ == '__main__':
    model = load_model('ckpt/model.h5')
    make_submission(model.predict(load_test_data()))
