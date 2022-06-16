import os
import time
import datetime
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import sys

def seconds_to_m_s(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return m, s
# ==============================================

# ----------------------------------------------
def create_xml_file(folder, path, file_name, all_boxes):
    return f'''<?xml version="1.0" ?>
<annotation>
    <folder>{folder}</folder>
    <filename>{file_name}</filename>
    <path>{path}</path>
    <source>
        <database>Unknown</database>
    </source>
    <segmented>0</segmented>
    {all_boxes}
</annotation>'''

def create_xml_box(class_name, box):
    xmin,ymin,xmax,ymax = box
    box_form = f'''<object>
    <name>{class_name}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
        <xmin>{xmin}</xmin>
        <ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax>
        <ymax>{ymax}</ymax>
    </bndbox>
</object>'''
    return box_form
# ==============================================

# ----------------------------------------------
def get_poits(image, bboxes, main_class_name, use_xml=True, draw_box=True):
    points = []
    all_boxes_form = ''

    for i, box in enumerate(bboxes):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        box = (xmin, ymin, xmax, ymax)
        points.append(box)

        if use_xml:
            box_form = create_xml_box(main_class_name, box)
            all_boxes_form += box_form

        if draw_box:
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            hh = int((ymax - ymin) / 2)
            ww = int((xmax - xmin) / 2)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.ellipse(image, (cx, cy), (ww, hh), 0, 0, 360, (255, 0, 0), 2)

    return points, all_boxes_form
# ==============================================

# ----------------------------------------------
def draw_box(image, boxes):
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        hh = int((ymax - ymin) / 2)
        ww = int((xmax - xmin) / 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.ellipse(image, (cx, cy), (ww, hh), 0, 0, 360, (255, 0, 0), 2)
    return image

# ----------------------------------------------
def start_processing_time():
    start_time_hms = datetime.datetime.now()
    print('Начало работы в {}:{:02d}:{:02d}'.format(start_time_hms.hour, start_time_hms.minute, start_time_hms.second))
    return time.time()

def calculated_left_time(total_save, start_pocessing, frame_ind, serial_number, frame_ind_load):
    pass_images = int((frame_ind - frame_ind_load) / serial_number)
    left_images = total_save - int(frame_ind / serial_number)
    if left_images > 0:
        left_time_sec = int((time.time() - start_pocessing) * left_images / pass_images)
        print('Осталось обработать изображений:', left_images)
        left_time = datetime.timedelta(seconds=left_time_sec)
        print('Это займёт:', str(left_time))
        finish_time = datetime.datetime.now() + left_time
        print('Примерно обработка закончится в {}:{:02d}:{:02d}'.format(finish_time.hour, finish_time.minute, finish_time.second))

def bar_progress(current, total, width=80):
    cur = int(current/total * 25)
    lost = 25 - cur
    line = '[' + '='*cur + '>' +  ' '*lost + ']'
    progress_message = f"Downloading:  {line} %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()
# ==============================================