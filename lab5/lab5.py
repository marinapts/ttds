import urllib.request
import os.path
import pickle
import xml.etree.ElementTree as ElementTree
import ast
import operator
import numpy as np
from collections import Counter, OrderedDict
from preprocess import tokenise, remove_stop_words, stemming
from index_search import create_term_doc_collection, boolean_search_queries, save_boolean_search_results, ranked_retrieval, save_ranked_retrieval_results


def preprocess(doc):
    return stemming(remove_stop_words(tokenise(doc), stop_words))


def load_xml(xml_file, tag):
    with open(xml_file, 'r') as xml_file:   # Read xml file
        xml = xml_file.read()

    xml = '<ROOT>' + xml + '</ROOT>'   # Add a root tag
    root = ElementTree.fromstring(xml)
    return root


def load_file_binary(file_name):
    with open(file_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_docs_from_trec(root):
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

    return doc_nums, token_doc_list, tokenised_docs


def term_tfidf_score(terms, N):
    terms_freq = Counter(terms)
    term_score_dict = dict()

    for term in terms:
        tf = terms_freq[term]  # Frequency of term in terms list
        df = len(inverted_index[term].keys())  # Num of docs that contain this term
        term_score = tf * np.log10(N / df)

        if term not in term_score_dict:
            term_score_dict[term] = term_score

    sorted_scores = sorted(term_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_scores


def top_n_d_terms(n_docs, n_terms):
    with open('./results/Qm.' + str(n_docs) + '.' + str(n_terms) + '.txt', 'w') as f:
        # Preprocess the queries for the ranked retrieval
        with open(QUERIES_RANKED) as queries_ranked_file:
            for query in queries_ranked_file:
                query_id = query.split(' ')[0]
                preprocessed_query = preprocess(query.split(' ', 1)[1])
                f.write(str(query_id) + ' ' + ' '.join(preprocessed_query) + ' + ')

                # Get top ranked docs ids for each query
                query_docs = ranked_docs_for_queries[query_id][:n_docs]
                concatenated_top_docs = []
                for doc_id in query_docs:
                    # Get document text for doc_id
                    concatenated_top_docs.extend(tokenised_docs[doc_id])

                term_scores = term_tfidf_score(concatenated_top_docs, len(doc_nums))
                # Get only the first 5 terms
                first_n_terms = [i[0] for i in term_scores[:n_terms]]
                print(query, first_n_terms, '\n')
                f.write(' '.join(first_n_terms) + '\n')


if __name__ == '__main__':
    # Directories
    DATA_DIR = 'data/'
    RESULTS_DIR = 'results/'
    # Data files
    STOP_WORDS_FILE = DATA_DIR + 'stop_words.txt'
    TREC_SAMPLE_FILE = DATA_DIR + 'trec.sample.xml'
    QUERIES_BOOLEAN = DATA_DIR + 'queries.boolean.txt'
    QUERIES_RANKED = DATA_DIR + 'queries.ranked.txt'

    # Save stop words and boolean queries
    with open(STOP_WORDS_FILE) as file:
        stop_words = [word.strip() for word in file]

    inverted_index = load_file_binary('./results/inverted_index')

    # Load the provided trec sample xml
    root = load_xml(TREC_SAMPLE_FILE, './DOC')
    doc_nums, token_doc_list, tokenised_docs = load_docs_from_trec(root)

    ranked_docs_for_queries = dict()
    with open(RESULTS_DIR + 'results.ranked.txt', 'r') as f:
        lines = f.readlines()

        for line in lines:
            query_id, _, doc_id, _, doc_score, _, _ = line.split(' ')
            if query_id in ranked_docs_for_queries:
                ranked_docs_for_queries[query_id].append(doc_id)
            else:
                ranked_docs_for_queries[query_id] = [doc_id]

    # Get the TOP_N_DOCS for each query
    top_n_docs = 1
    top_n_terms = 5

    top_n_d_terms(1, 5)
