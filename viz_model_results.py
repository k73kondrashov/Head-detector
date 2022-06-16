import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow import keras
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

#Выбрать нужное
type = 'Video'
# type = 'Image'
media_path = 'E:/Desktop/Work/scripts/Collect_heads/Videos/queue_test_01.mp4'
data_folder = 'first_train'
data_folder = os.path.join('data', data_folder)
model_name = 'ssd_resnet101_v1_fpn_1024x1024'
ckpt = 'ckpt-39'
ckpt_path = os.path.join('pretrained_models',model_name, 'checkpoints', ckpt)

det_thresh = 0.1

configs = config_util.get_configs_from_pipeline_file(os.path.join('pretrained_models', model_name, 'train_pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(ckpt_path).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(data_folder,'label_map.pbtxt'))
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def detect(img):
    print('Загрузка модели')
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = img.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=det_thresh,
        agnostic_mode=False)
    return image_np_with_detections

def play_video(media_path):
    cap = cv2.VideoCapture(media_path)

    while True:
        ret, img = cap.read()

        if ret:
            img_with_detection = detect(img)
            cv2.imshow('detections', img_with_detection)
            cv2.waitKey(1)

def main():
    if type == 'Video':
        play_video(media_path)
    if type == 'Image':
        img = cv2.imread(media_path)
        cv2.imshow('detections', detect(img))
        cv2.waitKey(0)





if __name__ == "__main__":
    main()

