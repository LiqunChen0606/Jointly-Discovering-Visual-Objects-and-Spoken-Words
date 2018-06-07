import tensorflow as tf
import numpy as np
import vgg, pickle, pdb, os, wavio
from audioFeature import *
from image_util import *
from audio_util import *

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import operator

GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# Parameters
batch_size = 100
epochs = 100
lr = 1e-4

images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
audios = tf.placeholder(tf.float32, [batch_size, 1024, 40])
true_out = tf.placeholder(tf.float32, [batch_size, 1000])
train_mode = tf.placeholder(tf.bool)

'''extract image and audio features'''
# VGG = vgg.Vgg19('./tensorflow-vgg/vgg19.npy')
VGG = vgg.Vgg19()
VGG.build(images, train_mode)
imageFeatures = VGG.conv6
audioFeatures = AudioFeature(audios)

''' design loss function (Rank Loss) '''
# S_p
Similarity_matrix = tf.einsum('eabc,ecd->eabd', imageFeatures, tf.transpose(audioFeatures,[0,2,1]))
Similarity_p = tf.reduce_mean(Similarity_matrix, [1,2,3])
#S_j and S_c
Similarity_j = tf.einsum('eabc,ecd->eabd', imageFeatures[::-1], tf.transpose(audioFeatures,[0,2,1]))
Similarity_j = tf.reduce_mean(Similarity_j, [1,2,3])
Similarity_c = tf.einsum('eabc,ecd->eabd', imageFeatures, tf.transpose(audioFeatures[::-1],[0,2,1]))
Similarity_c = tf.reduce_mean(Similarity_c, [1,2,3])

# obj_loss = tf.reduce_mean(tf.maximum(Similarity_c - Similarity_p + 1, 0) \
#     + tf.maximum(Similarity_j - Similarity_p + 1, 0))

# image_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ImageFeatures')
# audio_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AudioFeatures')
# optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9)
# solver = optimizer.minimize(obj_loss, var_list = image_vars + audio_vars)

''' prepare data'''
def sample_batch(X, size):
    start_idx = np.random.randint(0, len(X)-size)
    return X[start_idx:start_idx+size]

with open('./flickr8k.pkl', 'rb') as f:
    flickr_list = pickle.load(f)

print('stage 1 begins')
root_audio = './data/flickr_audio/wavs/'
root_image = './data//Flicker8k_Dataset/'
# dataloader_list = flickr_list[0][:30000]
devdata_list = flickr_list[0][-1000:]
audio_text_dict = flickr_list[1]
# Shuffle the data
# dataloader_list = np.random.permutation(dataloader_list)
pdb.set_trace()
print('stage 1 completed')

''' Initialize '''
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
# sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Load pretrained Model
try:
    saver.restore(sess=sess, save_path="./model/model.ckpt")
    print("\n--------model restored--------\n")
except:
    print("\n--------model Not restored--------\n")
    pass


''' Training '''
# dataList = sample_batch(devdata_list, batch_size)
# audio_list = [root_audio + tmp[0] for tmp in dataList]
# image_list = [root_image + tmp[1] for tmp in dataList]
count = {'1':0, '5':0, '10':0}
length = len(devdata_list)

# for ii in range(length//batch_size - 1):
#     dataList = devdata_list[ii*batch_size : (ii+1)*batch_size]
#     audio_list = [root_audio + tmp[0] for tmp in dataList]
#     image_list = [root_image + tmp[1] for tmp in dataList]
#     toText_list = [tmp[3] for tmp in dataList]

#     sample_images = np.array([get_image(image_file, 224, is_crop=False) \
#         for image_file in image_list])
#     sample_audio = np.array([get_audio(audio_file) for audio_file in audio_list])
#     sample_sents = [audio_text_dict[idx] for idx in toText_list]
    
#     if ii == 0:
#         matrix = sess.run(
#             Similarity_matrix,
#             feed_dict={
#                 images: sample_images,
#                 audios: sample_audio
#             })
#     else:
#         tmp = sess.run(Similarity_matrix, feed_dict={images: sample_images,audios: sample_audio})
#         matrix = np.concatenate([matrix, tmp], 0)

# import scipy.io as sio
# sio.savemat('./matrix.mat', {'x': matrix})
# pdb.set_trace()

for ii in range(length):
    fix_audio_name = root_audio + devdata_list[ii][0]
    fix_audio = np.array(get_audio(fix_audio_name))
    fix_sents_name = devdata_list[ii][3]
    fix_sents = audio_text_dict[fix_sents_name]
    fix_audio = np.expand_dims(fix_audio, 0)
    order = {}

    for jj in range(length):
        #NOTE: prepare the data
        image_name = root_image + devdata_list[jj][1]
        sample_images = np.array(get_image(image_name, 224, is_crop=False))
        sample_sents_name = devdata_list[0][3]
        sample_sents = audio_text_dict[sample_sents_name]

        sample_images = np.expand_dims(sample_images, 0)

        sp = sess.run(Similarity_p, feed_dict={images: sample_images, audios: fix_audio})
        sp = np.float32(np.squeeze(sp))

        order[devdata_list[jj][1]] = sp

    order = sorted(order.items(), key=operator.itemgetter(1), reverse=True)
    order = order[:10]
    if devdata_list[ii][1] in dict(order):
        count['10'] += 1
    if devdata_list[ii][1] in dict(order[:5]):
        count['5'] += 1
    if devdata_list[ii][1] in order[0]:
        count['1'] += 1

print(count)
pdb.set_trace()
