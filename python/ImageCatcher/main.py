import cv2
import numpy as np
import tensorflow as tf

from models import YOLOv3
from utils import postprocessing
from config import config
from utils.load_yolov3_weights import load_yolov3_weights
import time
import os
import requests
import urllib.request

from scrapy.selector import Selector
from icrawler.builtin import GoogleImageCrawler


def removeExtensionFile(filePath, fileExtension):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            if file.name.endswith(fileExtension):
                os.remove(file.path)
        return 'Remove File : ' + fileExtension
    else:
        return 'Directory Not Found'


#https://www.google.com/search?q=[검색어]&source=lnms&tbm=isch&sa=X&dpr=2&sourch=Int&tbs=sur:fc
#재사용가능 라이센스 구글 이미지 스크립트.

## BBox 밖 cropping 하기

google_crawler = GoogleImageCrawler(parser_threads=100, downloader_threads=100,
                                    storage={'root_dir': 'input/car'})

filters = dict(license='commercial')

google_crawler.crawl(keyword='car', filters=filters, max_num=10000,
                     min_size=(70,70), max_size=None)



removeExtensionFile('input/car', '.gif')

with tf.variable_scope('model'):
    print("Constructing computational graph...")
    model = YOLOv3(config)
    print("Done")

    print("Loading weights...")
    global_vars = tf.global_variables(scope='model')
    assign_ops = load_yolov3_weights(global_vars, config['WEIGHTS_PATH'])

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    print("Done")
    print("=============================================")

print("Loading class names...")
classes = []
colours = {}
f = open(config['CLASS_PATH'], 'r').read().splitlines()
for i, line in enumerate(f):
    classes.append(line)
    colours[i] = tuple([int(z) for z in np.random.uniform(0, 255, size=3)])
print(classes)
print("Done")
print("=============================================")

print("Running YOLOv3...")


def resize_bbox_to_original(original_image, bbox):
    """Resize a detected bounding box to fit the original image.

    :param original_image: The original image.
    :param bbox: The bounding box.
    :return: The resized bounding box.
    """
    original_size = np.array(original_image.shape[:2][::-1])
    resized_size = np.array([config['IMAGE_SIZE'], config['IMAGE_SIZE']])
    ratio = original_size/resized_size
    bbox = bbox.reshape(2, 2)*ratio
    bbox = list(bbox.reshape(-1))
    bbox = [int(z) for z in bbox]
    return bbox


def label_bboxes(original_image, bbox, class_id, score):
    """Draw a bounding box on the original image with a label.

    :param original_image: The original iamge.
    :param bbox: The bounding box.
    :param class_id: The class ID of the bounding box.
    :param score: The objectness score.
    :return: The labeled image.
    """
    x1, y1, x2, y2 = resize_bbox_to_original(original_image, bbox)
    label = '{}: {}%'.format(classes[class_id], int(score*100))

    cv2.rectangle(frame, (x1, y1), (x2, y2), colours[class_id], 2)

    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    w, h = text_size
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 - h), colours[class_id], cv2.FILLED)

    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return original_image


with tf.Session() as sess:
    sess.run(assign_ops)
    filenames = os.listdir('input/car') # 내가 저장할 폴더를 미리 지정해두고 시작하면 된다.
    filenames.sort()
    #filenames.remove('.DS_Store')

    if len(filenames) == 0:
        print
        ("Not exist")
    count = 0
    for filename in filenames:
        # image input
        # TODO : compatibility 'jpg / png / jpeg'

        frame = cv2.imread('input/car/' + filename, )
        before = time.time()
        resized_frame = cv2.resize(frame, (config['IMAGE_SIZE'], config['IMAGE_SIZE']))
        detected_bboxes = sess.run(model.outputs, feed_dict={model.inputs: np.expand_dims(resized_frame, axis=0)})
        filtered_bboxes = postprocessing.nms(detected_bboxes, conf_thresh=config['CONF_THRESH'],iou_thresh=config['IOU_THRESH'])
        filtered_bboxes_re = postprocessing.nms(detected_bboxes, conf_thresh=config['CONF_THRESH_HIGH'],iou_thresh=config['IOU_THRESH'])

        countBoxes = 0
        countBoxes_re = 0
        for class_id, v in filtered_bboxes.items():
            for detection in v:
                countBoxes = countBoxes + 1
                #label_bboxes(frame, detection['bbox'], class_id, detection['score'])

        for class_id, v in filtered_bboxes_re.items():
            for detection in v:
                countBoxes_re = countBoxes_re + 1
                #label_bboxes(frame, detection['bbox'], class_id, detection['score'])

        if (countBoxes == 1) and (countBoxes_re == 1)and (2 in filtered_bboxes) :


           # print(str(countBoxes))
            now = time.time()
            print(filename + " processed " +str(now-before))
            title, expander = filename.split(".")


            cv2.imwrite('output/' + 'output_'+ title+ "." + expander,frame)
            count = count + 1

    print("number of total processed image is " + str(len(filenames)) + ". and "+ str(count) + " images are saved.")

