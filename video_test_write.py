from face_detector import YoloV5FaceDetector

import os
import cv2
import glob2
import numpy as np
import pandas as pd
import tensorflow as tf

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import normalize
from skimage.io import imread
from tqdm import tqdm

def init_det_and_emb_model(model_file):
    det = YoloV5FaceDetector()
    if model_file is not None:
        face_model = tf.keras.models.load_model(model_file, compile=False)
    else:
        face_model = None
    return det, face_model

def embedding_images(det, face_model, known_user, batch_size=32, force_reload=False):
    while known_user.endswith("/"):
        known_user = known_user[:-1]
    dest_pickle = os.path.join(known_user, os.path.basename(known_user) + "_embedding.npz")

    if force_reload == False and os.path.exists(dest_pickle):
        aa = np.load(dest_pickle)
        image_classes, embeddings = aa["image_classes"], aa["embeddings"]
    else:
        if not os.path.exists(known_user):
            return [], [], None
        image_names = glob2.glob(os.path.join(known_user, "*/*.jpg"))

        """ Detct faces in images, keep only those have exactly one face. """
        nimgs, image_classes = [], []
        for image_name in tqdm(image_names, "Detect"):
            img = imread(image_name)
            nimg = do_detect_in_image(img, det, image_format="RGB")[-1]
            if nimg.shape[0] > 0:
                nimgs.append(nimg[0])
                image_classes.append(os.path.basename(os.path.dirname(image_name)))

        """ Extract embedding info from aligned face images """
        steps = int(np.ceil(len(image_classes) / batch_size))
        nimgs = (np.array(nimgs) - 127.5) * 0.0078125
        embeddings = [face_model(nimgs[ii * batch_size : (ii + 1) * batch_size]) for ii in tqdm(range(steps), "Embedding")]

        embeddings = normalize(np.concatenate(embeddings, axis=0))
        image_classes = np.array(image_classes)
        np.savez_compressed(dest_pickle, embeddings=embeddings, image_classes=image_classes)

    print(">>>> image_classes info:")
    print(pd.value_counts(image_classes))
    return image_classes, embeddings, dest_pickle

def do_detect_in_image(image, det, image_format="BGR"):
    # Преобразование формата изображения
    if image_format == "BGR":
        image_to_detect = image[:, :, ::-1]
    else:
        image_to_detect = image

    # Здесь мы используем параметры по умолчанию функции `detect_in_image`.
    # Вы можете изменить их, если это необходимо для вашего случая.
    max_output_size = 15
    iou_threshold = 0.45
    score_threshold = 0.25

    bboxes, pps, ccs, nimgs = det.detect_in_image(image_to_detect, max_output_size, iou_threshold, score_threshold,
                                                  image_format="RGB")

    if bboxes.ndim > 1:
        bbs = bboxes[:, :4].astype("int")
        ccs = bboxes[:, -1]
    else:
        bbs = np.array([])
        ccs = np.array([])

    return bbs, ccs, nimgs

def image_recognize(image_classes, embeddings, det, face_model, frame, image_format="BGR"):
    if isinstance(frame, str):
        frame = imread(frame)
        image_format = "RGB"
    bbs, ccs, nimgs = do_detect_in_image(frame, det, image_format=image_format)
    if len(bbs) == 0:
        return [], [], [], []

    emb_unk = face_model((nimgs - 127.5) * 0.0078125).numpy()
    emb_unk = normalize(emb_unk)
    dists = np.dot(embeddings, emb_unk.T).T
    rec_idx = dists.argmax(-1)
    rec_dist = [dists[id, ii] for id, ii in enumerate(rec_idx)]
    rec_class = [image_classes[ii] for ii in rec_idx]

    return rec_dist, rec_class, bbs, ccs

def draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh):
    for dist, label, bb, cc in zip(rec_dist, rec_class, bbs, ccs):
        # Red color for unknown, green for Recognized
        color = (0, 0, 255) if dist < dist_thresh else (0, 255, 0)
        label = "Unknown" if dist < dist_thresh else label

        left, up, right, down = bb
        cv2.line(frame, (left, up), (right, up), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, up), (right, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, down), (left, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (left, down), (left, up), color, 3, cv2.LINE_AA)

        xx, yy = np.max([bb[0] - 10, 10]), np.max([bb[1] - 10, 10])
        cv2.putText(frame, "Label: {}, dist: {:.4f}".format(label, dist), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return frame

def video_recognize_writer(image_classes, embeddings, det, face_model, video_source=0, frames_per_detect=5, dist_thresh=0.6,
                    output_video_path="output.mp4"):
    cap = cv2.VideoCapture(video_source)

    # Получаем параметры видео и инициализируем VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Используем mp4v-кодек, который поддерживает .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cur_frame_idx = 0
    while True:
        grabbed, frame = cap.read()
        if grabbed != True:
            break
        if cur_frame_idx % frames_per_detect == 0:
            rec_dist, rec_class, bbs, ccs = image_recognize(image_classes, embeddings, det, face_model, frame)
            cur_frame_idx = 0

        draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh)

        # Записываем обработанный кадр в выходной файл
        video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite("{}.jpg".format(cur_frame_idx), frame)
        if key == ord("q"):
            break

        cv2.imshow("", frame)
        cur_frame_idx += 1

    cap.release()
    video_writer.release()  # Освобождаем ресурсы VideoWriter
    cv2.destroyAllWindows()


# Инициализация и настройка параметров
model_file = "GN_W1.3_S1_ArcFace_epoch46.h5"  # замените на ваш путь к файлу модели
known_user = r"D:\Vf\GhostFaceNets_my\datasets\faces_emore_112x112_folders_2"  # замените на путь к папке с изображениями известного пользователя
known_user_force = None  # замените, если вы хотите принудительно перезагрузить
embedding_batch_size = 4
video_source = 0  # 0 для веб-камеры, или строка для пути к файлу видео
dist_thresh = 0.6
frames_per_detect = 5
output_video_path = r"D:\Vf\GhostFaceNets_my\video_res\video_res_1.mp4"  # замените на ваш путь к выходному файлу видео

# Настройка GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Инициализация моделей
det, face_model = init_det_and_emb_model(model_file)
force_reload = known_user_force is not None
known_user = known_user_force if force_reload else known_user

if known_user is not None and face_model is not None:
    image_classes, embeddings, _ = embedding_images(det, face_model, known_user, embedding_batch_size, force_reload)
    video_source = int(video_source) if str.isnumeric(str(video_source)) else video_source
    video_recognize_writer(image_classes, embeddings, det, face_model, video_source, frames_per_detect, dist_thresh, output_video_path)
