from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras_preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import load_paired_img_wrd
from vector_search import vector_search
import streamlit as st
import numpy as np
import inspect
from argparse import ArgumentParser

# Caching some slow functions
train_test_split = st.cache(train_test_split)
vector_search.load_glove_vectors = st.cache(vector_search.load_glove_vectors)


@st.cache
def to_array(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x_raw = image.img_to_array(img)
    return x_raw.astype(np.uint8)

def show_top_n(n, res, search_by_img=True):
    top_n = np.stack([to_array(res[i][1]) for i in range(min(len(res), n))])
    captions = [res[i][2] for i in range(min(len(res), n))]
    if search_by_img:
        captions[0] = "Original"
    st.image(top_n, caption=captions)

def show_source(fn):
    st.write('```\n%s\n```' % inspect.getsource(fn))

def build_parser():
    par = ArgumentParser()
    par.add_argument('--features_path', type=str,
                     dest='features_path', help='filepath to save/load features in simple model',
                     default="feat_4096")
    par.add_argument('--file_mapping_path', type=str,
                     dest='file_mapping_path', help='filepath to index file for simple model',
                     default="index_4096")
    par.add_argument('--model_path', type=str,
                     dest='model_path', help='Model path',
                     default="my_model.hdf5")
    par.add_argument('--custom_features_path', type=str,
                     dest='custom_features_path', help='filepath to save/load features in complex model',
                     default="feat_300")
    par.add_argument('--custom_features_file_mapping_path', type=str,
                     dest='custom_features_file_mapping_path', help='filepath to index in complex model',
                     default="index_300")
    par.add_argument('--search_key', type=int,
                     dest='search_key', help='Select a search key, 200 suggested',
                     default=200)
    par.add_argument('--train_model', type=str,
                     dest='train_model', help='Boolean True/False to train',
                     default="False")
    par.add_argument('--generate_image_features', type=str,
                     dest='generate_image_features', help='Boolean True/False to generate image features',
                     default="False")
    par.add_argument('--generate_custom_features', type=str,
                     dest='generate_custom_features', help='Boolean True/False to generate custom features',
                     default="False")
    par.add_argument('--training_epochs', type=int,
                     dest='training_epochs', help='Total training epochs for the complex model',
                     default=2)
    par.add_argument('--batch_size_training', type=int,
                     dest='batch_size_training', help='Total training epochs for the complex model',
                     default=32)
    par.add_argument('--glove_model_path', type=str,
                     dest='glove_model_path', help='Relative destination to glove model path',
                     default="models/glove.6B")
    par.add_argument('--data_path', type=str,
                     dest='data_path', help='Relative destination to dataset folder path',
                     default="dataset")
    return par

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def load_images_vectors_paths(glove_model_path, data_path):
    word_vectors = vector_search.load_glove_vectors(glove_model_path)
    images, vectors, image_paths = load_paired_img_wrd(data_path, word_vectors)
    return images, vectors, image_paths, word_vectors


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    features_path = options.features_path
    file_mapping_path = options.file_mapping_path
    model_path = options.model_path
    custom_features_path = options.custom_features_path
    custom_features_file_mapping_path = options.custom_features_file_mapping_path
    search_key = options.search_key
    train_model = str2bool(options.train_model)
    generate_image_features = str2bool(options.generate_image_features)
    generate_custom_features = str2bool(options.generate_custom_features)
    training_epochs = options.training_epochs
    batch_size_training = options.batch_size_training
    glove_model_path = options.glove_model_path
    data_path = options.data_path

    st.title("""Building a Semantic Search Engine""")
    st.image(Image.open('assets/image_search_cover.jpeg'), use_column_width=True)

    st.write("""
    In this post, I will outline how to use vector representations to create a **search engine**.
    
    We will go through three successive steps
    * How to search for similar images to an input image
    * How to search for similar words and synonyms
    * Generating tags from images, and searching for images using language
    
    To do this, we will use a simple approach of using **embeddings**, vector representations of 
    images and text.Once we have embeddings, searching simply becomes a matter of finding vectors close
    to our input vector.
    
    The way we do this is by calculating the **cosine distance** between our image embedding, and embeddings for
    other images. Similar images will have similar embeddings, meaning a **low cosine distance between embeddings**.
    
    To start things off, we will need a dataset to experiment with.
    """)

    st.header("Dataset")
    st.subheader("Loading the data")
    st.write("""
    Let's start by loading our dataset, which consists of a total of a **1000 images**, divided in **20 classes** 
    with 50 images for each.
    
    This dataset can be found [here](http://vision.cs.uiuc.edu/pascal-sentences/). *Credit to Cyrus 
    Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier.*
    
    In addition, we load [GloVe](https://nlp.stanford.edu/projects/glove/) vectors pre-trained on Wikipedia, 
    which we will use when we incorporate text.
    """)

    if not train_model:
        # It is possible to do this loading in the background while other things are surfaced to users
        with st.echo():
            images, vectors, image_paths, word_vectors = load_images_vectors_paths(glove_model_path, data_path)
    else:
        images, vectors, image_paths, word_vectors = load_images_vectors_paths(glove_model_path, data_path)
    st.write("Here is what our load function looks like:")
    show_source(load_paired_img_wrd)
    all_labels = [fold.split("/")[1] for fold in image_paths]
    st.write("Here is a list of our classes:", '\n'.join('`%s`' % elt for elt in sorted(list(set(all_labels)))))

    st.write(
        "We now have a tensor of images of size %s, of word vectors of size %s, and a list of corresponding file "
        "paths." % (images.shape, vectors.shape))

    st.subheader("Visualizing the data")
    st.write("Let's see what our data looks like, here is one example image from each class")

    sample_images = [to_array(image_paths[1 + i * 50]) for i in range(20)]
    captions = [all_labels[1 + i * 50] for i in range(20)]
    st.image(sample_images, caption=captions)

    st.write("""
    We can see our labels are pretty **noisy**, many photos contain multiple categories, and the label is not
    always from the most prominent one.
    """)

    st.header("Indexing the images")
    st.write("""
    We are now going to load a model that was **pre-trained** on a large data set (imagenet), and is freely available
     online.
    
    We use this model to generate **embeddings** for our images.
    
    As you can see below, once we've used the model to generate image features, we can then **store them to disk** 
    and re-use them without needing to do inference again! This is one of the reason that embeddings are so popular 
    in practical applications, as they allow for huge efficiency gains. 
    """)

    with st.echo():
        model = vector_search.load_headless_pretrained_model()
        if generate_image_features:
            print ("Generating image features...")
            images_features, file_index = vector_search.generate_features(image_paths, model)
            vector_search.save_features(features_path, images_features, file_mapping_path, file_index)
        else:
            images_features, file_index = vector_search.load_features(features_path, file_mapping_path)

    st.write("Our model is simply VGG16 without the last layer (softmax)")
    st.image(Image.open('assets/vgg16_architecture.jpg'), width=800, caption="Original VGG. Credit to Data Wow Blog")
    st.image(Image.open('assets/vgg16_chopped.jpg'), width=800, caption="Our model")
    st.write("This is how we get such a model in practice")
    show_source(vector_search.load_headless_pretrained_model)

    st.write("""
    What do we mean by generating embeddings? Well we just use our pre-trained model up to the penultimate layer, and 
    store the value of the activations.""")
    show_source(vector_search.generate_features)

    st.write('Here are what the embeddings look like for the first 20 images. Each image is now represented by a '
             'sparse vector of size 4096:')
    st.write(images_features[:20])

    st.write("Now that we have the features, we will build a fast index to search through them using Annoy.")
    with st.echo():
        image_index = vector_search.index_features(images_features)
    show_source(vector_search.index_features)
    st.header("Using our embeddings to search through images")
    st.write("""
    We can now simply take in an image, get its **embedding** (saved to disk), and look in our fast index to 
    find **similar embeddings, and thus similar images**.
    
    This is especially useful, since image labels are often noisy, so there is more to an image than it's label.
    
    In our dataset for example, we have both a class `cat`, and a class `bottle`.
    
    Which class do you think this image is labeled as?
    """)

    st.image(to_array(image_paths[search_key]), caption="Cat or bottle")
    
    
    #If you need to output your dataset_lookup keys
    '''
    count=0
    for path in image_paths:
        print(count, path)
        count+=1
    '''

    st.write("""
    The correct answer is **bottle** ... This is an actual issue that comes up often in real datasets. Labeling images as 
    unique categories is quite limiting, which is why we hope to use more nuanced representations.
    
    Luckily, this is exactly what deep learning is good at!
    
    Let's see if our image search using embeddings does better than human labels
    """)
    st.write("Searching for index `%s`, file `%s`" % (search_key, image_paths[search_key]))

    with st.echo():
        results = vector_search.search_index_by_key(search_key, image_index, file_index)
    show_source(vector_search.search_index_by_key)
    st.write('\n'.join('- `%s`' % elt for elt in results))
    show_top_n(9, results)

    st.write("""
    Great, we mostly get more images of **cats**, which seems very reasonable!. One image in there 
    however, is of a shelf of bottles. 
    
    This approach perform wells to find similar images in general, but some times we are only interested
    in **part of the image**.
    
    For example, given an image of a cat and a bottle, we might be only interested in similar cats, not similar bottles.
     
    A common approach is to use an **object detection** model first, detect our cat, and do image search on a cropped 
    version of the original image. 
    
    This adds a huge computing overhead, which we would like to avoid if possible.
    
    There is a simpler "hacky" approach, which consists of **re-weighing** the activations of the last layer using our 
    target class's weights. This cool trick initially brought to my attention by [Insight](
    http://insightdatascience.com/) Fellow [Daweon Ryu](https://www.linkedin.com/in/daweonryu/) 
    
    Let's demonstrate how this works, by weighing our activations according to class `284` in Imagenet, `Siamese cat`.
    """)
    with st.echo():
        # Index 284 is the index for the Siamese cat class in Imagenet
        weighted_features = vector_search.get_weighted_features(284, images_features)
        weighted_index = vector_search.index_features(weighted_features)
        weighted_results = vector_search.search_index_by_key(search_key, weighted_index, file_index)
    st.write("Here is how we perform our **trick** of re-weighing the features.")
    show_source(vector_search.get_weighted_features)
    st.write('\n'.join('- `%s`' % elt for elt in weighted_results))
    show_top_n(9, weighted_results)

    st.write("""
    We can see that the search has been biased to look for **Siamese cat like things**. We no longer show
    any bottles, but we do return an image of a sheep for the last image, which is much more cat-like than bottle-like!
    
    We have seen we can search for similar images in a **broad** way, or by **conditioning on a particular class**
    our model was trained on.
    
    This is a great step forward, but since we are using a model **pre-trained on Imagenet**, we are thus limited to 
    the thousand **Imagenet classes**. These classes are far far from all encompassing (they lack a category for
    human for example), so we would ideally like to find something more **flexible**
    """)

    st.header("Shifting to words")
    st.write("""
    Taking a detour to the world of NLP, could we use a similar approach to index and search for words?
    
    We loaded a set of pre-trained vectors from GloVe, which were obtaining by crawling through all of
    Wikipedia, and learning the semantic relationships between words in that dataset.
    
    Here is what our pre-trained word embeddings look like, dense vectors of size 300:
    """)
    with st.echo():
        st.write("word", word_vectors["word"])
        st.write("vector", word_vectors["vector"])
        try:
            st.write("fwjwiwejiu", word_vectors["fwjwiwejiu"])
        except KeyError as key:
            st.write("The word %s does not exist" % key)

    st.subheader("Indexing")
    st.write("Just like before, we will create an indexer, this time indexing all of the GloVe vectors")
    with st.echo():
        word_index, word_mapping = vector_search.build_word_index(word_vectors)
    show_source(vector_search.build_word_index)
    st.subheader("Searching")
    st.write("Now, we can **search our embeddings** for similar words!")

    st.write("Searching for %s" % word_mapping[16])
    results = vector_search.search_index_by_key(16, word_index, word_mapping)
    st.write('\n'.join('- `%s`' % elt for elt in results))

    st.write("Searching for %s" % word_mapping[48])
    results = vector_search.search_index_by_key(48, word_index, word_mapping)
    st.write('\n'.join('- `%s`' % elt for elt in results))

    st.write("Searching for %s" % word_mapping[23])
    results = vector_search.search_index_by_key(23, word_index, word_mapping)
    st.write('\n'.join('- `%s`' % elt for elt in results))

    st.write("""
    This is a pretty general method, but our representations seem **incompatible**.
    The embeddings for images are of size 4096, while the ones for words are of size 300, 
    how could we use one to search for the other?
    
    In addition, even if both embeddings were the same size, they were each trained in a completely different fashion, 
    so it is incredibly unlikely that images and related words would happen to have the same embeddings randomly. 
    We need to train a **joint model**.
    """)

    st.header("Worlds Collide")
    st.subheader("Creating a hybrid model")
    st.write("Let's now create a **hybrid** model that can go from words to images and vice versa.")
    st.write("""
    Here, we are actually training our own model. The way we use this, is by drawing inspiration from a great
    paper called [DeViSE](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model).
    
    The idea is to combine both representations by re-training our image model and **change its labels**.
    
    Usually, image classifiers are trained to pick a category out of many (1000 for Imagenet). What this
    translates to is that for Imagenet for example, the last layer is a vector of size 1000 representing the
    **probability of each class**. This is the layer we had previously removed.
     
    For this next part, we replace the target of our model with the **word vector of our category**. This allows our
    model to learn to map the semantics of images to the semantics of words!
     
    Here, we have added two dense layers, leading to an output layer of size 300.
    
    Here is what the model looked like when it was trained on Imagenet:
    """)

    st.image(Image.open('assets/vgg16_architecture.jpg'), width=800)

    st.write("Here is what it looks like now:")

    st.image(Image.open('assets/vgg16_extended.jpg'), width=800)

    st.subheader("Training the model")

    st.write("And here is the code to build it.")
    with st.echo():
        custom_model = vector_search.setup_custom_model()
    show_source(vector_search.setup_custom_model)

    st.write("""
    We then re-train our model on a training split of our dataset, to learn to predict **the word_vector
    associated with the label of an image**.
    
    For an image with the category cat for example, we are trying to predict the 300-length vector associated with cat.
    
    This training takes a bit of time, so I saved a model I trained on my laptop overnight.
    
    It is important to note that the training data we are using here (80% of our dataset, so 800 images) is
    minuscule, compared to usual datasets (Imagenet has **a million** images, **3 orders of magnitude more**).
    If we were using a traditional technique of training with categories, we would not expect our model to perform
    incredibly well on the test set, and would certainly not expect it to perform well on completely new examples.
    """)
    with st.echo():
        if train_model:
            print ("Training model...")
            with st.echo():
                num_epochs = training_epochs
                batch_size = batch_size_training
                st.write(
                    "Training for %s epochs, this might take a while, "
                    "change train_model to False to load pre-trained model" % num_epochs)
                x, y = shuffle(images, vectors, random_state=2)
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
                checkpointer = ModelCheckpoint(filepath='checkpoint.hdf5', verbose=1, save_best_only=True)
                custom_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                 epochs=num_epochs, batch_size=batch_size, callbacks=[checkpointer])
                custom_model.save(model_path)
        else:
            st.write("Loading model from `%s`" % model_path)
            custom_model = load_model(model_path)

    st.subheader("Building two indices (words and images)")
    st.write("Our model is trained and ready")
    st.write("For our word index, we will use the one we built previously.")

    st.write("Now, to build a fast image index, we need to run a forward pass on every image with our new model")

    # Load or generate the custom features
    if generate_custom_features:
        print ("Generating custom features...")
        hybrid_images_features, file_mapping = vector_search.generate_features(image_paths, custom_model)
        vector_search.save_features(custom_features_path, hybrid_images_features, custom_features_file_mapping_path,
                                    file_mapping)
    else:
        hybrid_images_features, file_mapping = vector_search.load_features(custom_features_path,
                                                                           custom_features_file_mapping_path)
    image_index = vector_search.index_features(hybrid_images_features, dims=300)

    st.write('Here are what our embeddings look like now for the first 20 images')
    st.write('They are of length 300, just **like our word vectors**!')
    st.write(hybrid_images_features[:20])

    st.subheader("Generating semantic tags")

    st.write("We can now easily extract tags from any image")
    st.write("Let's try with our cat/bottle image")

    st.image(to_array(image_paths[search_key]))
    st.write('Generating tags for `%s`' % file_mapping[search_key])
    with st.echo():
        results = vector_search.search_index_by_value(hybrid_images_features[search_key], word_index, word_mapping)
    show_source(vector_search.search_index_by_value)
    st.write('\n'.join('- `%s`' % elt for elt in results))

    st.write("These results are reasonable, let's try to see if we can detect more than the bottle in the "
             "messy image below.")

    st.image(to_array(image_paths[886]))
    st.write('Generating tags for `%s`' % file_mapping[886])
    with st.echo():
        results = vector_search.search_index_by_value(hybrid_images_features[886], word_index, word_mapping)
    st.write('\n'.join('- `%s`' % elt for elt in results))

    st.write("The model learns to extract **many relevant tags**, even from categories that it was not trained on!")

    st.subheader("Searching for images using text")
    st.write("""
    Most importantly, we can use our joint embedding to search through our image database using any word.
    We simply need to get our pre-trained word embedding from GloVe, and find the images that have the most similar
    embeddings! Generalized image search with minimal data.
    
    Let's start first with a word that was actually in our training set
    """)

    with st.echo():
        results = vector_search.search_index_by_value(word_vectors["dog"], image_index, file_mapping)

    st.write('\n'.join('- `%s`' % elt for elt in results))
    show_top_n(9, results, search_by_img=False)

    st.write("Now let's try with words we **did not train on**.")

    with st.echo():
        results = vector_search.search_index_by_value(word_vectors["ocean"], image_index, file_mapping)

    st.write('\n'.join('- `%s`' % elt for elt in results))
    show_top_n(9, results, search_by_img=False)

    with st.echo():
        results = vector_search.search_index_by_value(word_vectors["tree"], image_index, file_mapping)

    st.write('\n'.join('- `%s`' % elt for elt in results))
    show_top_n(9, results, search_by_img=False)

    with st.echo():
        results = vector_search.search_index_by_value(word_vectors["street"], image_index, file_mapping)

    st.write('\n'.join('- `%s`' % elt for elt in results))
    show_top_n(9, results, search_by_img=False)

    st.write("""
    Our image search seems quite robust! Combining the semantics of images and words is quite powerful.
    
    We can even search for a combination of multiple words by averaging word vectors together
    """)
    with st.echo():
        results = vector_search.search_index_by_value(np.mean([word_vectors["cat"], word_vectors["sofa"]], axis=0),
                                                      image_index, file_mapping)

    st.write('\n'.join('- `%s`' % elt for elt in results))
    show_top_n(9, results, search_by_img=False)
    st.write("""
    This is a fantastic result, as most those images contain some version of a furry animal and a sofa 
    (I especially enjoy the leftmost image on the second row, which seems like a bag of furriness next to a couch)! 
    Our model, which was only trained on single words, can handle combinations of two words. 
    We have not built Google Image Search yet, but this is definitely impressive for a relatively simple architecture.
    """)
    st.write("Hope you enjoyed it! [Let me know](https://twitter.com/EmmanuelAmeisen) "
             "if you have any questions, feedback, or comments.")
    st.header("Fin")
