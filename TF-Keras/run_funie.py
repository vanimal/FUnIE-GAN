from funie import funie
from os.path import basename, join
from utils.data_utils import getPaths
from wakepy import set_keepawake, unset_keepawake

set_keepawake()

f = funie()

in_dir = '../data/test/next/'
out_dir = '../data/output/'

print('Processing images')

for img_path in getPaths(in_dir):
  img_name = basename(img_path)
  out_path = join(out_dir, img_name)
  f.process_image_file(img_path, out_path)

print('Processing videos')

for vid_path in getPaths(in_dir, ['*.mp4']):
  vid_name = basename(vid_path)
  out_path = join(out_dir, vid_name)
  f.process_video_file(vid_path, out_path)

unset_keepawake()
