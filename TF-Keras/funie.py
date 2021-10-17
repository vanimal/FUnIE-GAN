import cv2
import numpy as np
import time

from keras.models import model_from_json
from os.path import exists
from PIL import Image
from utils.data_utils import deprocess, preprocess, read_image, split_chunks

img_size = 256
gutter = 64

class funie:
  def __init__(self, model_path = 'models/gen_p/model_15320_'):
    model_h5 = model_path + ".h5"
    model_json = model_path + ".json"

    assert(exists(model_h5) and exists(model_json))

    with open(model_json, "r") as json_file:
      loaded_model_json = json_file.read()

    self.funie_gan_generator = model_from_json(loaded_model_json)
    self.funie_gan_generator.load_weights(model_h5)

  def process_image_file(self, img_path, out_path):
    if (exists(out_path)):
      print('Output {0} already exists, skipping.'.format(out_path))
      return

    s = time.time()

    img = read_image(img_path)
    out_img = self.process_image(img)
    out_img.save(out_path)

    print("Processed {0} in {1:.2f} sec".format(img_path, time.time() - s))

  def process_video_file(self, vid_path, out_path):
    if (exists(out_path)):
      print('Output {0} already exists, skipping.'.format(out_path))
      return

    print('Processing {0}'.format(vid_path))

    vid = cv2.VideoCapture(vid_path)

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) - 2 * gutter
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 2 * gutter

    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    i = 0
    round_fps = int(fps)
    s = time.time()

    try:
      while vid.isOpened():
        ret, frame = vid.read()

        if not ret:
          if i == frame_count:
            break
          continue

        img = self._frame_to_img(frame)
        new_img = self.process_image(img)
        new_frame = self._img_to_frame(new_img)

        out.write(new_frame)

        i += 1
        if (i % round_fps == 0):
          t = time.time() - s; s = time.time()
          print("{0} - {1}/{2} processed {3} frames in {4:.2f} sec ({5:.2f} fps)".format(time.strftime('%I:%M:%S %p', time.localtime()), i, frame_count, round_fps, t, round_fps / t))
    except KeyboardInterrupt:
      pass

    print('{0}/{1} frames of {2} processed'.format(i, frame_count, vid_path))

    vid.release()
    out.release()

  def process_image(self, orig_img):
    chunks = split_chunks(orig_img, img_size, gutter)

    img_data = None

    for i, row in enumerate(chunks):
      processed_row = None

      for j, img in enumerate(row):
        im = preprocess(img)
        im = np.expand_dims(im, axis=0) # (1,256,256,3)
        gen = self.funie_gan_generator.predict(im)
        gen_img = deprocess(gen)[0]
        trim_img = gen_img[np.ix_(range(gutter, img_size - gutter), range(gutter, img_size - gutter))]
        processed_row = trim_img if processed_row is None else np.hstack((processed_row, trim_img)).astype('uint8')

      img_data = processed_row if img_data is None else np.vstack((img_data, processed_row))

    return Image.fromarray(img_data) \
      .crop((0, 0, orig_img.width - 2 * gutter, orig_img.height - 2 * gutter))

  def _img_to_frame(self, img):
    img_data = np.asarray(img)
    return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

  def _frame_to_img(self, frame):
    frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_data)
