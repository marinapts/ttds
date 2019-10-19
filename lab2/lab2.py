import urllib.request
import os.path
import pickle
import xml.etree.ElementTree as ElementTree
import ast
from collections import Counter
from preprocess import tokenise, remove_stop_words, normalise
from index_search import create_term_doc_collection, boolean_search_queries, save_boolean_search_results, ranked_retrieval, save_ranked_retrieval_results


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


# @TODO: Add df (document frequency) and tf (term frequency) in inverted index
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

    # return inverted_index
    save_inverted_index_txt(inverted_index, INVERTED_INDEX_FILE)
    save_file_binary(inverted_index, INVERTED_INDEX_FILE)


def save_inverted_index_txt(inverted_index, file_name):
    with open(file_name + '.txt', 'w') as f:
        sorted_words = sorted(inverted_index.keys())

        for word in sorted_words:
            indices_dict = inverted_index[word]
            f.write(word + ':\n')

            for doc_num in indices_dict:
                indices_str = ','.join(map(str, indices_dict[doc_num]))
                f.write('  ' + str(doc_num) + ': ' + indices_str + '\n')
            f.write('\n')
        print('Inverted index saved at {}.txt'.format(file_name))


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
    QUERIES_RANKED = DATA_DIR + '/queries.ranked.txt'
    # Results files
    INVERTED_INDEX_FILE = RESULTS_DIR + '/inverted_index'
    RESULTS_BOOLEAN_FILE = RESULTS_DIR + '/results.boolean'
    RESULTS_RANKED_FILE = RESULTS_DIR + '/results.ranked'

    create_directory(DATA_DIR)
    create_directory(RESULTS_DIR)
    # @TODO: Download xml files??
    download_file_and_save('http://members.unine.ch/jacques.savoy/clef/englishST.txt', STOP_WORDS_FILE)
    # @TODO: Use new url for queries file
    download_file_and_save('http://www.inf.ed.ac.uk/teaching/courses/tts/labs/lab2/queries.lab2.txt', QUERIES_BOOLEAN)
    download_file_and_save('http://www.inf.ed.ac.uk/teaching/courses/tts/labs/lab3/queries.lab3.txt', QUERIES_RANKED)

    # Save stop words and boolean queries
    with open(STOP_WORDS_FILE) as file:
        stop_words = [word.strip() for word in file]

    with open(QUERIES_BOOLEAN) as queries_boolean_file:
        queries_boolean = [query.lower().split(': ')[1] for query in queries_boolean_file]

    with open(QUERIES_RANKED) as queries_ranked_file:
        queries_ranked = [query.lower().split(' ', 1)[1] for query in queries_ranked_file]

    # Load the provided trec sample xml
    root = load_xml(TREC_SAMPLE_FILE, './DOC')
    doc_list = []
    token_doc_list = []
    tokenised_docs = {}
    doc_nums = []
    test_list = []

    for doc in root:
        doc_no = doc.find('DOCNO').text
        headline = doc.find('HEADLINE').text
        text = doc.find('TEXT').text
        headline_with_text = headline + ' ' + text

        doc_nums.append(doc_no)
        doc_list.append(headline_with_text)
        test_list.append(tokenise(headline_with_text))
        token_doc_list.append(preprocess(headline_with_text))
        tokenised_docs[doc_no] = preprocess(headline_with_text)

    create_inverted_index(tokenised_docs)
    # @TODO: Load inverted index from file into memory
    # save_inverted_index_txt(inverted_index, INVERTED_INDEX_FILE)
    inverted_index = load_file_binary('./results/inverted_index')
    print(inverted_index)

    # Create a term-document incident collection that shows which documents each term belongs to
    collection_table = create_term_doc_collection(inverted_index, doc_nums)

    # Boolean, phrase and proximity search
    boolean_search_results = boolean_search_queries(queries_boolean, collection_table, inverted_index, doc_nums)
    save_boolean_search_results(boolean_search_results, queries_boolean, RESULTS_BOOLEAN_FILE)

    # Ranked search
    ranked_retrieval_results = ranked_retrieval(queries_ranked, collection_table, doc_nums, inverted_index, stop_words)
    save_ranked_retrieval_results(ranked_retrieval_results, RESULTS_RANKED_FILE)
