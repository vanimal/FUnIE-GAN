
import os
import time
import ntpath
import numpy as np
from scipy import misc
from keras.models import model_from_json

## local libs
from nets.funieGAN import FUNIE_GAN
from utils.plot_utils import save_test_samples_funieGAN
from utils.data_utils import getPaths, read_and_resize, preprocess

# for testing arbitrary local data
data_dir = "data/selected/"
from utils.data_loader import get_local_test_data
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))


## create dir for log and (sampled) validation data
samples_dir = "data/samples/funieGAN_gp/"
checkpoint_dir = "checkpoints/funieGAN_gp/DURV_combined/"
model_name_by_epoch = "model_40520_"
model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (os.path.exists(model_h5) and os.path.exists(model_json))

## load model arch
funie_gan = FUNIE_GAN()

# load json and create model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)

# load weights into new model
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")

times = []; s = time.time()
for img_path in test_paths:
    img_name = ntpath.basename(img_path).split('.')[0]
    im = read_and_resize(img_path, (256, 256))
    im = preprocess(im)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)

    s = time.time()
    gen = funie_gan_generator.predict(im)
    tot = time.time()-s
    times.append(tot)

    misc.imsave(samples_dir+img_name+'_real.png', im[0])
    misc.imsave(samples_dir+img_name+'_gen.png', gen[0])
    
Ttime = sum(times)
print ("Time taken: {0} sec at {1} fps".format(Ttime, len(test_paths)/Ttime))
print("\nSaved generated images in in {0}\n".format(samples_dir))








