import urllib.request
import os.path
import re
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ElementTree
from stemming.porter2 import stem
from collections import Counter


def download_file_and_save(url, file_name):
    """Download the file from `url` and save it locally under `file_name`:
    Args:
        url (string): The url to download the file from
        file_name (string): The name of the file
    """
    if not os.path.exists('./' + file_name):
        print('Downloading file from {}...'.format(url))
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            data = response.read()  # a `bytes` object
            print('Saving file as {}...'.format(file_name))


def tokenise(text):
    """Removes punctuation, new lines and multiple white
    Args:
        text (string): The text provided to tokenise
    Returns:
        tokenised (list): List of tokens
    """
    no_date_in_headline = re.sub(r"^.*?/+", "", text, flags=re.MULTILINE)   # Remove date from the headlines
    no_punctuation = re.sub(r"[.?\-\",!;'/:()\[\]\(\)&\n+\t+]", " ", no_date_in_headline, flags=re.MULTILINE)  # Remove punctuation
    no_extra_spaces = re.sub(r"\s{2,}", " ", no_punctuation, flags=re.I)   #
    tokenised = no_extra_spaces.lower().strip().split(' ')
    # print('{} words after tokenisation'.format(len(tokenised)))
    return tokenised


def remove_stop_words(words):
    """Remove stop words
    Args:
        words (list): The list of tokenised words
    Returns:
        words (list): A list of words without the stop words included
    """
    words = list(filter(lambda x: x not in stop_words, words))
    # print('{} words after removing stop words'.format(len(words)))
    return words


def normalise(words):
    """Porter stemmer
    Args:
        words (TYPE): Description
    Returns:
        TYPE: Description
    """
    return list([stem(word) for word in words])


def preprocess(doc):
    return normalise(remove_stop_words(tokenise(doc)))


def load_xml(xml_file, tag):
    with open(xml_file, 'r') as xml_file:   # Read xml file
        xml = xml_file.read()

    xml = '<ROOT>' + xml + '</ROOT>'   # Add a root tag
    root = ElementTree.fromstring(xml)
    return root


def word_freq_in_doc(doc):
    word_freq = Counter(doc)
    print('{} \n'.format(word_freq))

    return word_freq


def positional_inverted_index(token_list):
    inverted_index = dict()

    for index, doc in enumerate(token_list):
        print('Document {}: {}'.format(index, doc))
        for word in doc:
            original_doc = doc_list[index]

            word_occurences = re.findall(r"\b" + word + r"\w*", original_doc, flags=re.I)
            doc_id = index+1
            doc_indices_dict = {}
            doc_indices_dict[doc_id] = word_occurences

            if word in inverted_index:
                inverted_index[word].update(doc_indices_dict)
            else:
                inverted_index.setdefault(word, doc_indices_dict)

    print(inverted_index)


if __name__ == '__main__':
    STOP_WORDS_FILE = 'stop_words.txt'
    SAMPLE_FILE = './data/sample.xml'
    TREC_SAMPLE_FILE = './data/trec.sample.xml'

    download_file_and_save(
        'http://members.unine.ch/jacques.savoy/clef/englishST.txt',
        STOP_WORDS_FILE)

    # Store stop words in a list
    with open(STOP_WORDS_FILE) as file:
        stop_words = [word.strip() for word in file]

    root = load_xml(TREC_SAMPLE_FILE, './DOC')
    doc_list = []
    token_list = []

    for doc in root:
        headline = doc.find('HEADLINE').text
        text = doc.find('TEXT').text
        doc_list.append(headline)
        doc_list.append(text)
        token_list.append(preprocess(headline))
        token_list.append(preprocess(text))

    token_list = token_list[0: 3]
    positional_inverted_index(token_list)
