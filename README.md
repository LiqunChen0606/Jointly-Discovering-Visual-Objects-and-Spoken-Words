# Jointly-Discovering-Visual-Objects-and-Spoken-Words


##### paper link (https://arxiv.org/pdf/1804.01452.pdf)

## Requirement
Python 3.6, Tensorflow 1.8, wavio, python_speech_features

## How to run:
    1) download flickr8k speech caption files and image files
    2) In the data folder, flickr8k.pkl provides paired information. Details of how to use this pickle file can be found in main_SISA or MISA python file.

    3) python main_SISA/MISA.py

## Experiment
### Speech captions retrieve images for Flickr8k dataset:
this result is on test dataset, which is the last 1000 images and captions

    R@1: 0.027, R@5: 0.127, R@10:0.245

##### Note: still working in progress 

## TODO list
    1) image to caption retrieval
    2) ...