import tensorflow as tf
import numpy as np
import vgg
import pdb
import pickle

# dataloader_list = []

# file_list = './data/flickr_audio/wav2capt.txt'
# with open(file_list) as fin:
#     for l in fin:
#         l_split = l.split()
#         tmp = l_split[-2] + l_split[-1] # data format: name.wav, name.jpg, #{0..4}, name.jpg#{0..4} 
#         l_split.append(tmp)
#         dataloader_list.append(l_split)

# text_list = './data/txt/Flickr8k.token.txt' #Flickr8k.lemma.token.txt
# audio_text_dict = {}
# with open(text_list) as fin:
#     for l in fin:
#         l_split = l.split()
#         # dataloader_list.append(l_split)
#         tmp_sent = u' '.join(l_split[1:]) + '\n'
#         audio_text_dict[l_split[0]] = tmp_sent # data format: name.jpg#{0..4}, sentence
        
# pickle.dump([dataloader_list, audio_text_dict], open('flickr8k.pkl', 'wb'))

with open('./flickr8k.pkl', 'rb') as f:
    flickr_list = pickle.load(f)
