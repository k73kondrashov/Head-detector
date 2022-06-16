import os
import cv2
import time
import tensorflow as tf
import numpy as np
import random
from string import ascii_lowercase
import shutil
import pickle
from utils import (seconds_to_m_s, start_processing_time, calculated_left_time,
                   create_xml_box, create_xml_file, draw_box)


import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

start_work = time.time()

# ----------------------------------------------
#            ***  DIRECTORY  ***
# Путь к папке с входными видео.
videos_dir = 'E:/Desktop/Work/data/Вход Суши'
# Путь к папке с выходными изображениями
# (создастся автоматически).
images_dir = 'Images'
# Папка для моделей
models_dir = 'pretrained_models'
# Название модели
model_name = 'ssd_resnet101_v1_fpn_1024x1024'

#Название checkpoints
ckpt = 'ckpt-31'
# Название меток основного класса (головы)
main_class_name = 'head'

# Название текущего видео
video_name = '10_06_2022_11_13_37.mp4'

# Порог вероятности, ниже которой
# предсказание отфильтровывается.
SCORE_THRESH = 0.3

# Какой каждый кадр брать.
# Если False, то программа будет
# запрашивать каждый раз при запуске.
# Чтобы не спрашивала, вместо False
# нужно указать значение.
# serial_number = False
serial_number = 15

# Сколько кадров должно пройти перед
# сохранением контрольной точки
# с учётом пропуска кадров.
checkpoint_size = 100

# Использовать как выходные метки:
# False - использовать только словарь меток
# True - использовать дополнительно метки xml
USE_XML = True

# Начать с начала, даже если
# есть контрольная точка.
RESET = False

# Нужно ли создавать проверочные
# изображения в папке Draw
# с нарисованными рамками голов
DRAW_BOXES = True
# ==============================================

# ----------------------------------------------
# Путь к .config
config_path = os.path.join(models_dir, model_name, 'train_pipeline.config')

# Путь к checpoint
ckpt_path = os.path.join('pretrained_models', model_name, 'checkpoints', ckpt)

prefix = ''.join(random.choice(ascii_lowercase) for i in range(6))
video_name_short = video_name.split('.')[0]

print('\nВидео:', video_name)
video_in_path = os.path.join(videos_dir, video_name)

cap = cv2.VideoCapture(video_in_path)

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Ширина и высота кадра:', vid_width, vid_height)

vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
print('Количество кадров в секунду:', vid_fps)

vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Всего кадров в видео:', vid_length)

length_sec = vid_length/vid_fps
print('Длительность видео в секундах: {:.2f}'.format(length_sec))
# ==============================================

if not serial_number:
    print('\nНиже введите цифру какой каждый кадр нужно взять с видео.')
    print('Например, 5, пограмма возьмёт каждый пятый кадр.')
    print('Тогда получится {} выходных изображений.'.format(int(vid_length // 5)))
    print('Если нужно сохранить все изображения введите "1".')
    serial_number = int(input('\nВведите цифру какой каждый кадр нужно взять: '))

print('\nИз видео будет взят каждый {} кадр'.format(serial_number))
total_save = int(vid_length // serial_number)
print('Всего получится изображений из видео:', total_save)
# ==============================================

configs = config_util.get_configs_from_pipeline_file(config_path)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(ckpt_path).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
# ==============================================

# ----------------------------------------------
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

data_out_dir = os.path.join(images_dir, video_name_short)
if not os.path.exists(data_out_dir):
    os.mkdir(data_out_dir)
else:
    if RESET:
        shutil.rmtree(data_out_dir)
        os.mkdir(data_out_dir)

images_xml_dir = os.path.join(data_out_dir, 'Images_xml')
if not os.path.exists(images_xml_dir):
    os.mkdir(images_xml_dir)

if DRAW_BOXES:
    draw_dir = os.path.join(data_out_dir, 'Draw')
    if not os.path.exists(draw_dir):
        os.mkdir(draw_dir)

labels_dir = os.path.join(data_out_dir, 'Labels')
if not os.path.exists(labels_dir):
    os.mkdir(labels_dir)
# ==============================================

# ----------------------------------------------
info_pickles_path = os.path.join(labels_dir, '{}_info.p'.format(video_name_short))
labels_dict_path = os.path.join(labels_dir, '{}_labels_dict.p'.format(video_name_short))

if os.path.exists(info_pickles_path):
    with open(info_pickles_path, 'rb') as file:
        info_pickles = pickle.load(file)

    video_name_pickle = info_pickles[0]
    frame_ind = info_pickles[1]
    frame_ind_load = info_pickles[1]
    serial_number = info_pickles[2]
    count_save = info_pickles[3]
    prefix = info_pickles[4]
    class_name_load = info_pickles[5]

    assert video_name_pickle == video_name, 'Wrong video name'

    assert frame_ind != 'done', 'Video already processed'

    assert os.path.exists(labels_dict_path), "Doesn`t exist label_dict file"

    assert class_name_load == main_class_name, 'Wrong main class name'
    main_class_name = class_name_load

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

    with open(labels_dict_path, 'rb') as file:
        labels_dict = pickle.load(file)

    print('\nЗагрузка с контрольной точки!')
else:
    frame_ind = 0
    frame_ind_load = 0
    count_save = 0
    labels_dict = {}
    print('\nЗагрузка с нуля!')
# ==============================================

processing_start = start_processing_time()
print('\nВыполняется обработка и сохранение кадров...')

while True:
    ret, image = cap.read()

    if not ret:
        break
    saved_frame = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(tf.expand_dims(image.astype(np.float32), 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    boxes = detections['detection_boxes'].numpy()[0]
    scores = detections['detection_scores'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0]

    boxes[:, ::2] *= h
    boxes[:, 1::2] *= w
    boxes = boxes.astype(int)
    all_boxes_form = ''

    image_name_short = '{}_{:05d}'.format(prefix, count_save)
    image_name = image_name_short + '.jpg'
    boxes_to_dict = []
    for box, score, label in zip(boxes, scores, classes):
        if score > SCORE_THRESH:
            y1, x1, y2, x2 = box
            boxes_to_dict.append((x1,y1,x2,y2))
            all_boxes_form += create_xml_box('head', [x1, y1, x2, y2])
            saved_frame = True
        else:
            break
    if saved_frame:
        form_xml = create_xml_file(video_name_short,
                                   os.path.join(images_xml_dir, image_name),
                                   image_name,
                                   all_boxes_form)
        with open(os.path.join(images_xml_dir, image_name_short + '.xml'), 'w') as file:
            file.write(form_xml)
        cv2.imwrite(os.path.join(images_xml_dir, image_name), image)
        count_save += 1
        if DRAW_BOXES:
            image = draw_box(image, boxes[scores>SCORE_THRESH])
            cv2.imwrite(os.path.join(draw_dir, image_name), image)

        labels_dict[image_name] = boxes_to_dict


    frame_ind += serial_number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)


    if frame_ind % (serial_number * checkpoint_size) == 0:
        info_pickles = (video_name, frame_ind, serial_number, count_save, prefix, main_class_name)

        with open(info_pickles_path, 'wb') as file:
            pickle.dump(info_pickles, file)

        with open(labels_dict_path, 'wb') as file:
            pickle.dump(labels_dict, file)

        print('\nОбработано кадров:', frame_ind)
        calculated_left_time(total_save, processing_start, frame_ind, serial_number, frame_ind_load)

cap.release()
cv2.destroyAllWindows()
print('Сохранение кадров завершено.')
# ==============================================

# ----------------------------------------------
info_pickles = (video_name, 'done', serial_number, count_save, prefix, main_class_name)

with open(info_pickles_path, 'wb') as file:
    pickle.dump(info_pickles, file)

with open(labels_dict_path, 'wb') as file:
    pickle.dump(labels_dict, file)

duration = time.time() - start_work
processing_duration = time.time() - processing_start
print('\nВсего обработка видео заняла {} минут и {} секунд'.format(*seconds_to_m_s(duration)))
print('Точнее:', round(duration, 2))
print('Только обработка:', round(processing_duration, 2))
# ==============================================