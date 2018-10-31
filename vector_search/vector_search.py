import logging
import os
import json
import time

import h5py
import numpy as np

from annoy import AnnoyIndex
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.losses import cosine_proximity
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_headless_pretrained_model():
    """
    Loads the pretrained version of VGG with the last layer cut off
    :return: pre-trained headless VGG16 Keras Model
    """
    print ("Loading headless pretrained model...")
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)
    return model


def generate_features(image_paths, model):
    """
    Takes in an array of image paths, and a trained model.
    Returns the activations of the last layer for each image
    :param image_paths: array of image paths
    :param model: pre-trained model
    :return: array of last-layer activations, and mapping from array_index to file_path
    """
    print ("Generating features...")
    start = time.time()
    images = np.zeros(shape=(len(image_paths), 224, 224, 3))
    file_mapping = {i: f for i, f in enumerate(image_paths)}

    # We load all our dataset in memory because it is relatively small
    for i, f in enumerate(image_paths):
        img = image.load_img(f, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    logger.info("%s images loaded" % len(images))
    inputs = preprocess_input(images)
    logger.info("Images preprocessed")
    images_features = model.predict(inputs)
    end = time.time()
    logger.info("Inference done, %s Generation time" % (end - start))
    return images_features, file_mapping


def save_features(features_filename, features, mapping_filename, file_mapping):
    """
    Save feature array and file_item mapping to disk
    :param features_filename: path to save features to
    :param features: array of features
    :param mapping_filename: path to save mapping to
    :param file_mapping: mapping from array_index to file_path/plaintext_word
    """
    print ("Saving features...")
    np.save('%s.npy' % features_filename, features)
    with open('%s.json' % mapping_filename, 'w') as index_file:
        json.dump(file_mapping, index_file)
    logger.info("Weights saved")


def load_features(features_filename, mapping_filename):
    """
    Loads features and file_item mapping from disk
    :param features_filename: path to load features from
    :param mapping_filename: path to load mapping from
    :return: feature array and file_item mapping to disk

    """
    print ("Loading features...")
    images_features = np.load('%s.npy' % features_filename)
    with open('%s.json' % mapping_filename) as f:
        index_str = json.load(f)
        file_index = {int(k): str(v) for k, v in index_str.items()}
    return images_features, file_index


def index_features(features, n_trees=1000, dims=4096, is_dict=False):
    """
    Use Annoy to index our features to be able to query them rapidly
    :param features: array of item features
    :param n_trees: number of trees to use for Annoy. Higher is more precise but slower.
    :param dims: dimension of our features
    :return: an Annoy tree of indexed features
    """
    print ("Indexing features...")
    feature_index = AnnoyIndex(dims, metric='angular')
    for i, row in enumerate(features):
        vec = row
        if is_dict:
            vec = features[row]
        feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index


def build_word_index(word_vectors):
    """
    Builds a fast index out of a list of pretrained word vectors
    :param word_vectors: a list of pre-trained word vectors loaded from a file
    :return: an Annoy tree of indexed word vectors and a mapping from the Annoy index to the word string
    """
    print ("Building word index ...")
    logging.info("Creating mapping and list of features")
    word_list = [(i, word) for i, word in enumerate(word_vectors)]
    word_mapping = {k: v for k, v in word_list}
    word_features = [word_vectors[lis[1]] for lis in word_list]
    logging.info("Building tree")
    word_index = index_features(word_features, n_trees=20, dims=300)
    logging.info("Tree built")
    return word_index, word_mapping


def search_index_by_key(key, feature_index, item_mapping, top_n=10):
    """
    Search an Annoy index by key, return n nearest items
    :param key: the index of our item in our array of features
    :param feature_index: an Annoy tree of indexed features
    :param item_mapping: mapping from indices to paths/names
    :param top_n: how many items to return
    :return: an array of [index, item, distance] of size top_n
    """
    distances = feature_index.get_nns_by_item(key, top_n, include_distances=True)
    return [[a, item_mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


def search_index_by_value(vector, feature_index, item_mapping, top_n=10):
    """
    Search an Annoy index by value, return n nearest items
    :param vector: the index of our item in our array of features
    :param feature_index: an Annoy tree of indexed features
    :param item_mapping: mapping from indices to paths/names
    :param top_n: how many items to return
    :return: an array of [index, item, distance] of size top_n
    """
    distances = feature_index.get_nns_by_vector(vector, top_n, include_distances=True)
    return [[a, item_mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


def get_weighted_features(class_index, images_features):
    """
    Use class weights to re-weigh our features
    :param class_index: Which Imagenet class index to weigh our features on
    :param images_features: Unweighted features
    :return: Array of weighted activations
    """
    class_weights = get_class_weights_from_vgg()
    target_class_weights = class_weights[:, class_index]
    weighted = images_features * target_class_weights
    return weighted


def get_class_weights_from_vgg(save_weights=False, filename='class_weights'):
    """
    Get the class weights for the final predictions layer as a numpy martrix, and potentially save it to disk.
    :param save_weights: flag to save to disk
    :param filename: filename if we save to disc
    :return: n_classes*4096 array of weights from the penultimate layer to the last layer in Keras' pretrained VGG
    """
    model_weights_path = os.path.join(os.environ.get('HOME'),
                                      '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    weights_file = h5py.File(model_weights_path, 'r')
    weights_file.get('predictions').get('predictions_W_1:0')
    final_weights = weights_file.get('predictions').get('predictions_W_1:0')

    class_weights = np.array(final_weights)[:]
    weights_file.close()
    if save_weights:
        np.save('%s.npy' % filename, class_weights)
    return class_weights


def setup_custom_model(intermediate_dim=2000, word_embedding_dim=300):
    """
    Builds a custom model taking the fc2 layer of VGG16 and adding two dense layers on top
    :param intermediate_dim: dimension of the intermediate dense layer
    :param word_embedding_dim: dimension of the final layer, which should match the size of our word embeddings
    :return: a Keras model with the backbone frozen, and the upper layers ready to be trained
    """
    print ("Setting up custom model ...")
    headless_pretrained_vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    x = headless_pretrained_vgg16.get_layer('fc2').output

    # We do not re-train VGG entirely here, just to get to a result quicker (fine-tuning the whole network would
    # lead to better results)
    for layer in headless_pretrained_vgg16.layers:
        layer.trainable = False

    image_dense1 = Dense(intermediate_dim, name="image_dense1")(x)
    image_dense1 = BatchNormalization()(image_dense1)
    image_dense1 = Activation("relu")(image_dense1)
    image_dense1 = Dropout(0.5)(image_dense1)

    image_dense2 = Dense(word_embedding_dim, name="image_dense2")(image_dense1)
    image_output = BatchNormalization()(image_dense2)

    complete_model = Model(inputs=[headless_pretrained_vgg16.input], outputs=image_output)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    complete_model.compile(optimizer=sgd, loss=cosine_proximity)
    return complete_model


def load_glove_vectors(glove_dir, glove_name='glove.6B.300d.txt'):
    """
    Mostly from keras docs here https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    Download GloVe vectors here http://nlp.stanford.edu/data/glove.6B.zip
    :param glove_name: name of pre-trained file
    :param glove_dir: directory in witch the glove file is located
    :return:
    """
    f = open(os.path.join(glove_dir, glove_name))
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index
