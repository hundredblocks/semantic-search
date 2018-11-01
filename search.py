from keras.engine.saving import load_model
from argparse import ArgumentParser

from utils import load_paired_img_wrd
from vector_search import vector_search


def build_parser():
    par = ArgumentParser()
    par.add_argument('--features_path', type=str,
                     dest='features_path', help='filepath to save/load features', required=True)
    par.add_argument('--file_mapping', type=str,
                     dest='file_mapping', help='filepath to save/load file to image mapping', required=True)
    par.add_argument('--index_folder', type=str,
                     dest='index_folder', help='folder to index', required=False)
    par.add_argument('--input_image', type=str,
                     dest='input_image', help='input image path to search query', required=False)
    par.add_argument('--input_word', type=str,
                     dest='input_word', help='input word to search query', required=False)
    par.add_argument('--glove_path', type=str,
                     dest='glove_path', help='path to pre-trained GloVe vectors', required=False)
    par.add_argument('--model_path', type=str,
                     dest='model_path', help='path to custom model', required=False)
    par.add_argument('--index_boolean', type=str,
                     dest='index_boolean', help='Boolean: Create index instead of search', required=False)
    par.add_argument('--features_from_new_model_boolean', type=str,
                     dest='features_from_new_model_boolean', help='Boolean: If to create features from new model', required=False)
    return par

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def check_inputs(folder, image, word, model_path, glove_path):
    if not folder and not (image or word):
        raise ValueError(
            "You must either provide a folder to index, or an image/word to search, you provided %s folder, %s image, "
            "%s word " % (folder, image, word))
    if folder is not None and (image or word) is not None:
        raise ValueError("You must provide either a folder to index, or an image/word to search, not both or neither. "
                         "You provided %s folder, %s image, %s word" % (folder, image, word))

    if (word is True) != (model_path is True):
        raise ValueError("Ypu must provide a custom model if searching by word, you provided %s for word and %s for "
                         "model" % (word, model_path))
    if (model_path is True) != (glove_path is True):
        raise ValueError("Ypu must provide a glove path if training custom model, you provided %s for model and %s for "
                         "glove" % (model_path, glove_path))

def index_images(folder, features_path, mapping_path, model, features_from_new_model_boolean, glove_path):
    print ("Now indexing images...")
    # Use word vectors if leveraging the new model
    if features_from_new_model_boolean:
        word_vectors=vector_search.load_glove_vectors(glove_path)
    else:
        word_vectors=[]
    # Use utiliy function
    _, _, paths = load_paired_img_wrd(
        folder=folder, 
        word_vectors=word_vectors,
        use_word_vectors=features_from_new_model_boolean)
    images_features, file_index = vector_search.generate_features(paths, model)
    vector_search.save_features(features_path, images_features, mapping_path, file_index)
    return images_features, file_index


# This is an inefficient implimentation for the proof of context
def get_index(input_image, file_mapping):
    for index, file in file_mapping.items():
        if file == input_image:
            return index
    raise ValueError("Image %s not indexed" % input_image)

def generate_features(index_folder, features_path, file_mapping, loaded_model, features_from_new_model_boolean, glove_path):
    features, index = index_images(
        index_folder, 
        features_path, 
        file_mapping, 
        loaded_model, 
        features_from_new_model_boolean,
        glove_path)
    print("Indexed %s images" % len(features))

def build_index_and_search_through_it(images_features, file_index):
    # Decide whether to do only image search or hybrid search
    if not features_from_new_model_boolean:
        # This is pure image search
        image_index = vector_search.index_features(images_features, dims=4096)
        search_key = get_index(input_image, file_index)
        results = vector_search.search_index_by_key(search_key, image_index, file_index)
        print(results)
    else:
        word_vectors = vector_search.load_glove_vectors(glove_path)
        # If we are searching for tags for an image
        if not input_word:
            # Work on a single image instead of indexing
            search_key = get_index(input_image, file_index)
            word_index, word_mapping = vector_search.build_word_index(word_vectors)
            results = vector_search.search_index_by_value(images_features[search_key], word_index, word_mapping)
        # If we are using words to search through our images
        else:
            image_index = vector_search.index_features(images_features, dims=300)
            results = vector_search.search_index_by_value(word_vectors[input_word], image_index, file_index)
        print(results)

if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    features_path = options.features_path
    file_mapping = options.file_mapping
    index_folder = options.index_folder
    input_image = options.input_image
    input_word = options.input_word
    model_path = options.model_path
    glove_path = options.glove_path
    index_boolean = str2bool(options.index_boolean)
    features_from_new_model_boolean = str2bool(options.features_from_new_model_boolean)

    check_inputs(
        index_folder, 
        input_image, 
        input_word, 
        model_path, 
        glove_path)

    # Decide whether to use pre-trained VGG or custom model, if one was provided
    if model_path:
        loaded_model = load_model(model_path)
    else:
        loaded_model = vector_search.load_headless_pretrained_model()

    # Decide whether to index the images (if you already have them) or load images to disk
    if index_boolean:
        generate_features(index_folder, features_path, file_mapping, loaded_model, features_from_new_model_boolean, glove_path)
    else:
        images_features, file_index = vector_search.load_features(features_path, file_mapping)
        build_index_and_search_through_it(images_features, file_index)
