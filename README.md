# Semantic Search
![Preview](https://github.com/hundredblocks/semantic-search/blob/master/assets/image_search_cover.jpeg)

This repository contains a barebones implementation of a semantic search engine. 
The implementation is based on leveraging pre-trained embeddings from VGG16 (trained on Imagenet), and GloVe (trained on Wikipedia).


It allows you to:
- Find similar images to an input image

- Find similar words to an input word
 
- Search through images using any word

- Generate tags for any image

See examples of usage by following along on this [notebook](http://insight.streamlit.io/0.13.3-8ErS/index.html?id=QAKzY9mLjr4WbTCgxz3XBX).

Read more details abbout why and how you would use this in this blog [post](https://blog.insightdatascience.com/the-unreasonable-effectiveness-of-deep-learning-representations-4ce83fc663cf).

## Setup

Clone the repository locally and then, if you only with to use the library without setting up the demo:

`pip install -r requirements.txt`

If you want to run the Streamlit demo or the downloader:

`pip install -r requirements-all.txt`

If you intend to use text, download pre-trained GloVe [vectors](http://nlp.stanford.edu/data/glove.6B.zip). Place them in `models/glove.6B/glove.6B.300d.txt`. We use the ones of length 300.

Download an example image dataset by using `python downloader.py` or your own. Image dataset must be of the format. Credit to Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier for the dataset.
```
dataset/
|
|--- class_0/
|      |-------image_0
|      |-------image_1
|      ...
|
|      |-------image_n
|--- class_1/
|     ...
|  
|--- class_n/
```
Each class name should be one word in the english language, or multiple words separated by "_". In our example dataset for example, we rename the "diningtable" folder to "dining_table".
## Usage

To make full use of this repository, feel free to import the vector_search package in your project. For added convenience, a few functions are exposed through a command line API. THey are documented below. 

### Using pre-trained models for image search

#### Search for a similar image to an input image
First, you need to index your images:

`python search.py --index_folder images --features_path feat --file_mapping index`

Then, you can search through your images:

`python search.py --input_image image.jpg --features_path feat --file_mapping index`

#### Search for an image using words

For this, you will need to have **trained a custom model** (see below) that can map images to words, then use the same command with the additional optional argument pointing to your model.

Similarly, start by indexing your images:

`python search.py --input_folder images --features_path feat --file_mapping index --model_path model.hdf5 --glove_path models/glove.6B`

Then, you can search through your images by either passing an image:

`python search.py --input_image image.jpg --features_path feat --file_mapping index --model_path model.hdf5 --glove_path models/glove.6B `

Or a word

`python search.py --input_word street --features_path feat --file_mapping index --model_path model.hdf5 --glove_path models/glove.6B `

### Training a custom model

`python train.py --model_save_path {PATH_TO_SAVE_MODEL} --checkpoint_path {PATH_TO_SAVE_CHECKPOINTS} --glove_path {DIRECOTRY CONTAINING GLOVE VECTORS} --dataset_path {PATH_TO_DATA} --num_epochs {NUMBER_OF_EPOCHS_TO_TRAIN}`

For example, if you've downloaded the pascal dataset, and placed vectors in `models/golve.6B` and want to train for 50 epochs.

`python train.py --model_save_path my_model.hdf5 --checkpoint_path checkpoint.hdf5 --glove_path models/glove.6B --dataset_path dataset --num_epochs 50`

### Running the demo

To run the demo, run `demo.py`. You may need to train your own model, so make sure to update the flags at the top of the file to match what you are trying to accomplish.
