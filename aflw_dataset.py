import os
import sqlite3
import hashlib
import random
import numpy as np

from PIL import Image
from scipy.misc import imread


_SQL_FACE_DATA = r"""
SELECT
  f.file_id AS file_id,
  fr.x as x,
  fr.y as y,
  fr.w as w,
  fr.h as h
FROM
  facerect AS fr
  JOIN faces AS f ON fr.face_id = f.face_id
WHERE
  fr.annot_type_id = 1
"""


def _read_face_data(aflw_sqlite_file):
    with sqlite3.connect(aflw_sqlite_file) as conn:
        c = conn.cursor()
        c.execute(_SQL_FACE_DATA)
        data = c.fetchall()
        face_data = {}
        for file_id, x, y, w, h in data:
            face_data.setdefault(file_id, []).append((x, y, w, h))
        # face_data has the format [(file_id, [(x, y, w, h), ...]), ...]
        return face_data.items()


def _line_i_u(la, lb):
    if la[0] > lb[0]:
        la, lb = lb, la
    if la[1] > lb[0]:
        return min(la[1], lb[1]) - lb[0], lb[1] - la[0]
    return 0, la[1] - la[0] + lb[1] - lb[0]


def _rect_iou(ra, rb):
    xa, ya, wa, ha = ra
    xb, yb, wb, hb = rb
    
    xi, xu = _line_i_u((xa, xa + wa), (xb, xb + wb))
    yi, yu = _line_i_u((ya, ya + ha), (yb, yb + hb))

    return xi * yi * 1.0 / (xu * yu)


def _crop_image(im, rect):
    x, y, w, h = rect
    return im[y:y+h, x:x+w]


def _scale_image(im, new_w, new_h):
    im = Image.fromarray(im).resize([new_w, new_h], Image.ANTIALIAS)
    return np.asarray(im)


def _gen_pos_face_impl(img, face_rect, min_iou):
    """Generate a positive face example.
    
    Args:
      img: a numpy array of image
      face_rect: (x, y, w, h) of a ground truth face
      min_iou: a positive face example must have a IOU >= min_iou with the
          ground truth face.
    Returns:
      A random generated (x, y, a, a) of a positive face.
    """
    # To overlap a ground truth face (x, y, w, h) with MIN_IOU using 
    # a rectangle (xp, yp, a, a), we have
    # 0) Let max_s = max(a * a, w * h)
    # 1) bounds for a
    #   MIN_IOU <= IOU = I / U <= a * a / (w * h)
    #                          <= w * h / (a * a)
    #   so sqrt(MIN_IOU * w * h) <= a <= sqrt(w * h / MIN_IOU)
    # 2) bounds for xp
    #   MIN_IOU <= (xp + a - x) * a / max_s
    #           <= (x + w - xp) * a / max_s
    #   so xp >= MIN_IOU * max_s / a + x - a
    #      xp <= x + w - MIN_IOU * max_s / a
    # 3) bounds for yp is similar
    #      yp >= MIN_IOU * max_s / a + y - a
    #      yp <= y + h - MIN_IOU * max_s / a
    img_h, img_w = img.shape[0], img.shape[1]
    x, y, w, h = face_rect
    s = w * h

    a_lb = np.ceil(np.sqrt(min_iou * s))
    a_ub = np.floor(np.sqrt(s / min_iou))
    a = np.random.randint(min(a_lb, img_w, img_h), min(a_ub, img_w, img_h) + 1)
    
    max_s = max(a * a, s)
    xp_lb = max(np.ceil(min_iou * max_s / a + x - a), 0)
    xp_ub = min(np.floor(x + w - min_iou * max_s / a), img_w - a)
    if xp_lb > xp_ub:
        return
    xp = np.random.randint(xp_lb, xp_ub + 1)
    
    yp_lb = max(np.ceil(min_iou * max_s / a + y - a), 0)
    yp_ub = min(np.floor(y + h - min_iou * max_s / a), img_h - a)
    if yp_lb > yp_ub:
        return
    yp = np.random.randint(yp_lb, yp_ub + 1)
    
    if _rect_iou((xp, yp, a, a), face_rect) > min_iou:
        return xp, yp, a, a

    
def _gen_pos_faces(img, face_rects, num_examples, min_iou=0.5, max_try=10):
    pos_faces = []
    for i in range(num_examples):
        face_rect = face_rects[np.random.randint(len(face_rects))]
        for _ in range(max_try):
            pos_face = _gen_pos_face_impl(img, face_rect, min_iou)
            if pos_face is not None:
                pos_faces.append(pos_face)
                break
    return pos_faces


def _gen_neg_face_impl(img, face_rects, max_iou):
    img_h, img_w = img.shape[0], img.shape[1]
    a = np.random.randint(1, min(img_h, img_w) + 1)
    xp = np.random.randint(0, img_w - a + 1)
    yp = np.random.randint(0, img_h - a + 1)
    for face_rect in face_rects:
        if _rect_iou((xp, yp, a, a), face_rect) > max_iou:
            return None
    return xp, yp, a, a


def _gen_neg_faces(img, face_rects, num_examples, max_iou=0.5, max_try=10):
    neg_faces = []
    for i in range(num_examples):
        for _ in range(max_try):
            neg_face = _gen_neg_face_impl(img, face_rects, max_iou)
            if neg_face is not None:
                neg_faces.append(neg_face)
                break
    return neg_faces


def _gen_face_examples(im, face_rects, scale_to_w, scale_to_h):
    ans = []
    for r in face_rects:
        ex = _scale_image(_crop_image(im, r), scale_to_w, scale_to_h)
        if ex.shape == (scale_to_h, scale_to_w, 3):
            ans.append(ex)
    return ans


def _random_flip_examples(examples):
    flip_flags = np.random.randint(3, size=len(examples))
    ans = []
    for flip_flag, ex in zip(flip_flags, examples):
        if flip_flag == 1:
            ans.append(np.fliplr(ex))
        elif flip_flag == 2:
            ans.append(np.flipud(ex))
        else:
            ans.append(ex)
    return ans


class FaceDb(object):

    def __init__(self, aflw_sqlite_path, aflw_image_dir):
        self._aflw_sqlite_path = aflw_sqlite_path
        self._aflw_image_dir = aflw_image_dir
        self._face_data = _read_face_data(aflw_sqlite_path)
        random.shuffle(self._face_data)

    def read_image(self, file_id):
        return imread(os.path.join(self._aflw_image_dir, file_id))

    @property
    def face_data(self):
        return self._face_data


class ExamplePool(object):

    def __init__(self, batch_size, capacity):
        self._batch_size = batch_size
        self._capacity = capacity
        self._examples = []
        self._epoch = 0
        self._curr_idx = 0

    def reload(self, examples):
        self._examples.extend(examples)
        random.shuffle(self._examples)
        self._examples = self._examples[0:self._capacity]
        self._epoch = 0
        self._curr_idx = 0
        
    def next_batch(self):
        batch_size = min(len(self._examples), self._batch_size)
        end_idx = self._curr_idx + batch_size
        if end_idx > len(self._examples):
            self._epoch += 1
            self._curr_idx = 0
            end_idx = self._batch_size
            random.shuffle(self._examples)
        ans = self._examples[self._curr_idx:end_idx]
        self._curr_idx = end_idx
        return ans

    @property
    def epoch(self):
        return self._epoch

    @property
    def progress(self):
        return self._curr_idx * 1.0 / len(self._examples)
    
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def capacity(self):
        return self._capacity
    

class Dataset(object):

    def __init__(self, face_db, face_data_range=None,
                 face_iou_threshold=0.5,
                 reload_every_num_batches=6,
                 reload_num_images=200,
                 image_refresh_batch_percent = 0.06,
                 pos_face_pool=ExamplePool(
                     batch_size=32,
                     capacity=384),
                 neg_face_pool=ExamplePool(
                     batch_size=96,
                     capacity=1152)):
        self._face_db = face_db
        self._face_data_range = face_data_range
        self._face_iou_threshold = face_iou_threshold
        self._reload_every_num_batches = reload_every_num_batches
        self._reload_num_images = reload_num_images
        self._image_refresh_batch_percent = image_refresh_batch_percent
        self._pos_face_pool = pos_face_pool
        self._neg_face_pool = neg_face_pool

        face_data = face_db.face_data
        if face_data_range is not None:
            self._face_data = face_data[face_data_range[0]:face_data_range[1]]
        self._face_data_pool = ExamplePool(batch_size=reload_num_images,
                                           capacity=len(face_data))
        self._face_data_pool.reload(face_data)

        self._batch_count = 0
        
    def _reload(self, scale_to_w, scale_to_h):
        face_data = self._face_data_pool.next_batch()
        pos_examples = []
        neg_examples = []
        for file_id, face_rects in face_data:
            new_im = self._face_db.read_image(file_id)
            if new_im.ndim != 3:
                continue
            num_pos_faces = int(np.ceil(self._pos_face_pool.batch_size *
                                        self._image_refresh_batch_percent))
            num_neg_faces = int(np.ceil(self._neg_face_pool.batch_size *
                                        self._image_refresh_batch_percent))
            pos_face_rects = _gen_pos_faces(new_im, face_rects, num_pos_faces,
                                            self._face_iou_threshold)
            neg_face_rects = _gen_neg_faces(new_im, face_rects, num_neg_faces,
                                            self._face_iou_threshold)
            pos_examples.extend(_gen_face_examples(
                new_im, pos_face_rects, scale_to_w, scale_to_h))
            neg_examples.extend(_gen_face_examples(
                new_im, neg_face_rects, scale_to_w, scale_to_h))

        self._pos_face_pool.reload(_random_flip_examples(pos_examples))
        self._neg_face_pool.reload(_random_flip_examples(neg_examples))
        
    def next_batch(self, scale_to_w, scale_to_h):
        if self._batch_count % self._reload_every_num_batches == 0:
            self._reload(scale_to_w, scale_to_h)
        self._batch_count += 1

        pos_examples = self._pos_face_pool.next_batch()
        neg_examples = self._neg_face_pool.next_batch()
        examples = pos_examples + neg_examples
        examples = [np.expand_dims(ex, axis=0) for ex in examples]
        labels = [1] * len(pos_examples) + [0] * len(neg_examples)
        return np.concatenate(examples), labels

    @property
    def epoch(self):
        return self._face_data_pool.epoch

    @property
    def progress(self):
        return self._face_data_pool.progress
