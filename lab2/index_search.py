import re
import numpy as np
from preprocess import normalise

PHRASE_REGEX = re.compile(r'\"\w+(\s+)\w+\"')
PROXIMITY_REGEX = re.compile(r'\#\d+\(.\w+\,(\s+).\w+\)')


def create_term_doc_collection(inverted_index, doc_nums):
    """Create a term-document incident collection that shows which documents each term belongs to

    Args:
        inverted_index (dict): Description
        doc_nums (list): Description

    Returns:
        collection_dict (dict): A boolean vector for each word
    """
    words = list(inverted_index.keys())
    collection_dict = dict()
    boolean_matrix = np.zeros((len(words), len(doc_nums)), dtype=np.bool)

    # Create a mapping of document numbers to continuous indices
    doc_num_dict = {}
    for ind, doc_num in enumerate(doc_nums):
        doc_num_dict[doc_num] = ind

    for i, word in enumerate(words):
        docs_for_specific_word = list(inverted_index[word].keys())

        for index, doc_num in enumerate(docs_for_specific_word):
            doc_id = doc_num_dict[doc_num]
            boolean_matrix[i][int(doc_id)] = True

        collection_dict[word] = boolean_matrix[i]

    return collection_dict


def phrase_proximity_search(terms, max_distance, keep_order, inverted_index, doc_nums):
    """Common function for phrase and proximity search
    """
    term_1_docs = inverted_index[terms[0]]
    term_2_docs = inverted_index[terms[1]]

    # Find common docs between the 2 words
    common_docs = set(term_1_docs).intersection(term_2_docs)
    common_docs_ids = []

    for doc in list(common_docs):
        # Check the distance between the words
        for i in term_2_docs[doc]:
            for j in term_1_docs[doc]:
                distance_diff = int(i) - int(j)

                if keep_order:
                    # Order of words matters in phrase search so distance diff MUST be positive
                    if distance_diff > 0 and distance_diff <= max_distance:
                        common_docs_ids.append(int(doc))
                else:
                    # Order of words does not matter in proximity search and distance diff can be negative,
                    # so we take its absolute value
                    if abs(distance_diff) <= max_distance:
                        common_docs_ids.append(int(doc))

    common_docs_ids = sorted(set([int(i) for i in common_docs_ids]))
    boolean_vector = convert_doc_ids_to_boolean(common_docs_ids, doc_nums)
    return boolean_vector


def convert_doc_ids_to_boolean(doc_list, doc_nums):
    boolean_doc_list = np.zeros(len(doc_nums), dtype=np.bool)

    for doc_id in doc_list:
        doc_index = doc_nums.index(str(doc_id))
        boolean_doc_list[doc_index] = True

    return boolean_doc_list


def boolean_search(query_str_transformed, doc_nums):
    # Map doc numbers to continuous indices
    doc_num_mapping = {}

    for ind, doc_num in enumerate(doc_nums):
        doc_num_mapping[ind] = doc_num

    boolean_vector = eval(query_str_transformed)
    documents = [doc_num_mapping[i] for i in range(len(boolean_vector)) if boolean_vector[i] == True]
    return documents


def split_query(query):
    def replace_space_with_underscore(match):
        return match.group(0).replace(' ', '_')

    def remove_space_from_proximity_search(match):
        return match.group(0).replace(' ', '')

    # Replaces spaces from inside quotes with underscore
    query = PHRASE_REGEX.sub(replace_space_with_underscore, query)
    # Removes space from the proximity search
    query = PROXIMITY_REGEX.sub(remove_space_from_proximity_search, query)
    return query.lower().split()


def array_to_string(arr):
    array_str = ''

    for i in arr:
        array_str += '{}, '.format(i)
    return array_str


def search_queries(queries, collection_table, inverted_index, doc_nums):
    logical_operators_mapping = {'and': '&', 'or': '|', 'not': '~'}
    search_results = []

    for query in queries:
        query_tokens = split_query(query)
        query_eval_string = ''

        for token in query_tokens:
            if token.startswith('#'):   # Proximity search
                # Split phrase into distance and terms
                distance, term_1, term_2 = list(filter(None, re.split(r'\W+', token)))
                terms = normalise([term_1, term_2])
                boolean_vector = phrase_proximity_search(terms, int(distance), False, inverted_index, doc_nums)
                query_eval_string += 'np.array([{}]) '.format(array_to_string(boolean_vector))

            elif token.startswith('"'):    # Phrase search
                terms = normalise(token.replace('_', ' ').replace('"', '').split())
                boolean_vector = phrase_proximity_search(terms, 1, True, inverted_index, doc_nums)
                query_eval_string += 'np.array([{}]) '.format(array_to_string(boolean_vector))

            elif token in logical_operators_mapping:    # One of AND, OR, NOT
                token = logical_operators_mapping[token]
                query_eval_string += '{} '.format(token)

            else:
                normalised_word = normalise([token])[0]
                boolean_vector = collection_table[normalised_word]
                query_eval_string += 'np.array([{}]) '.format(array_to_string(boolean_vector))

        query_search_results = boolean_search(query_eval_string, doc_nums)
        search_results.append(query_search_results)

    return search_results


def save_boolean_search_results(results_boolean, queries, file_name):
    f = open(file_name + '.txt', 'w+')

    for index, query in enumerate(queries):
        for doc_id in results_boolean[index]:
            f.write(str(index + 1) + ' 0 ' + doc_id + ' 0 1 0\n')
    f.close()
    print('Boolean search results saved at {}.txt'.format(file_name))
