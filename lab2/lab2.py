import urllib.request
import os.path
import pickle
import xml.etree.ElementTree as ElementTree
from collections import Counter
from preprocess import tokenise, remove_stop_words, normalise
from index_search import create_term_doc_collection, search_queries, save_boolean_search_results


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
            out_file.write(data)


def preprocess(doc):
    return normalise(remove_stop_words(tokenise(doc), stop_words))


def load_xml(xml_file, tag):
    with open(xml_file, 'r') as xml_file:   # Read xml file
        xml = xml_file.read()

    xml = '<ROOT>' + xml + '</ROOT>'   # Add a root tag
    root = ElementTree.fromstring(xml)
    return root


def word_freq_in_doc(doc):
    word_freq = Counter(doc)
    return word_freq


def get_word_indices_in_text(words, text):
    text_token = tokenise(text)
    word_indices = []

    for word in words:
        word_index = text_token.index(word.lower())
        word_indices.append(word_index + 1)

    return word_indices


def find_indices_of_word(doc_list, word):
    indices_list = [i + 1 for i in range(len(doc_list)) if doc_list[i] == word]
    return indices_list


# @TODO: Sort the inverted index
# @TODO: Add df (document frequency) in inverted index
def create_inverted_index(tokenised_docs):
    inverted_index = dict()

    for doc_no, token_doc_list in tokenised_docs.items():
        for word in token_doc_list:
            word_indices = find_indices_of_word(token_doc_list, word)
            doc_indices_dict = {}
            doc_indices_dict[doc_no] = word_indices
            if word in inverted_index:
                inverted_index[word].update(doc_indices_dict)
            else:
                inverted_index.setdefault(word, doc_indices_dict)

    return inverted_index


def save_inverted_index_txt(inv_index, file_name):
    f = open(file_name + '.txt', 'a+')

    for word in inv_index:
        indices_dict = inv_index[word]
        f.write(word + ':\n')

        for doc_num in indices_dict:
            indices_str = ', '.join(map(str, indices_dict[doc_num]))
            f.write('  ' + str(doc_num) + ': ' + indices_str + '\n')
        f.write('\n')

    f.close()


# @TODO: Save binary file in the required format
def save_file_binary(obj, file_name):
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file_binary(file_name):
    with open(file_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    # Directories
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'
    # Data files
    STOP_WORDS_FILE = DATA_DIR + '/stop_words.txt'
    SAMPLE_FILE = DATA_DIR + '/sample.xml'
    TREC_SAMPLE_FILE = DATA_DIR + '/trec.sample.xml'
    QUERIES_BOOLEAN = DATA_DIR + '/queries.boolean.txt'
    # Results files
    INVERTED_INDEX_FILE = RESULTS_DIR + '/inverted_index'
    RESULTS_BOOLEAN_FILE = RESULTS_DIR + '/results.boolean'

    create_directory(DATA_DIR)
    create_directory(RESULTS_DIR)
    # @TODO: Download xml files??
    download_file_and_save('http://members.unine.ch/jacques.savoy/clef/englishST.txt', STOP_WORDS_FILE)
    # @TODO: Use new url for queries file
    download_file_and_save('http://www.inf.ed.ac.uk/teaching/courses/tts/labs/lab2/queries.lab2.txt', QUERIES_BOOLEAN)

    # Save stop words and boolean queries
    with open(STOP_WORDS_FILE) as file:
        stop_words = [word.strip() for word in file]

    with open(QUERIES_BOOLEAN) as queries_file:
        queries_boolean = [query.lower().split(': ')[1] for query in queries_file]

    # Load the provided trec sample xml
    root = load_xml(TREC_SAMPLE_FILE, './DOC')
    doc_list = []
    token_doc_list = []
    tokenised_docs = {}
    doc_nums = []

    for doc in root:
        doc_no = doc.find('DOCNO').text
        headline = doc.find('HEADLINE').text
        text = doc.find('TEXT').text
        headline_with_text = headline + ' ' + text

        doc_nums.append(doc_no)
        doc_list.append(headline_with_text)
        token_doc_list.append(preprocess(headline_with_text))
        tokenised_docs[doc_no] = preprocess(headline_with_text)

    inverted_index = create_inverted_index(tokenised_docs)
    # @TODO: Load inverted index from file into memory
    save_inverted_index_txt(inverted_index, INVERTED_INDEX_FILE)
    save_file_binary(inverted_index, INVERTED_INDEX_FILE)

    # Create a term-document incident collection that shows which documents each term belongs to
    collection_table = create_term_doc_collection(inverted_index, doc_nums)

    # print(queries_boolean)
    results_boolean = []
    logical_operators = ['and', 'or', 'not']

    search_results = search_queries(queries_boolean, collection_table, inverted_index, doc_nums)
    save_boolean_search_results(search_results, queries_boolean, RESULTS_BOOLEAN_FILE)
