"""
Imported from https://github.com/rupy/PascalSentenceDataset/blob/master/pascal_sentence_dataset.py
Updated to Python3, and removed sentence downloading code
"""
from urllib.parse import urljoin

from pyquery import PyQuery
import os
import requests


class PascalSentenceDataSet():
    DATASET_DIR = 'dataset/'
    PASCAL_DATASET_URL = 'http://vision.cs.uiuc.edu/pascal-sentences/'

    def __init__(self):
        self.url = PascalSentenceDataSet.PASCAL_DATASET_URL

    def download_images(self):
        dom = PyQuery(self.url)
        for img in dom('img').items():
            img_src = img.attr['src']
            category, img_file_name = os.path.split(img_src)

            # make category directories
            output_dir = PascalSentenceDataSet.DATASET_DIR + category
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            # download image
            output = os.path.join(output_dir, img_file_name)
            print(output)
            if img_src.startswith('http'):
                img_url = img_src
            else:
                img_url = urljoin(self.url, img_src)
            if os.path.isfile(output):
                print("Already downloaded, Skipping: %s" % output)
                continue
            print("Downloading: %s" % output)
            with open(output, 'wb') as f:

                while True:
                    result = requests.get(img_url)
                    raw = result.content
                    if result.status_code == 200:
                        f.write(raw)
                        break
                    print("error occurred while fetching img")
                    print("retry...")


if __name__ == "__main__":
    # create instance
    dataset = PascalSentenceDataSet()
    # # download images
    dataset.download_images()
