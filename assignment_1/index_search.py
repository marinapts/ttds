import re
import numpy as np
from preprocess import stemming


PHRASE_REGEX = re.compile(r'\"\w+(\s+)\w+\"')
PROXIMITY_REGEX = re.compile(r'\#\d+\(.\w+\,(\s+).\w+\)')


def create_term_doc_collection(inverted_index, doc_nums):
    """Create a term-document incident collection that shows which documents each term belongs to
    Args:
        inverted_index (dict): Index of terms as keys and dict of documents with positions as values
        doc_nums (list): An array of the documents numbers
    Returns:
        collection_dict (dict): A boolean vector for each word
    """
    words = list(inverted_index.keys())
    collection_dict = dict()
    boolean_matrix = np.zeros((len(words), len(doc_nums)), dtype=np.bool)

    # Create a mapping of document numbers to continuous indices from 0 to len(doc_nums)-1
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
    """Common function for phrase and proximity search. We treat the phrase search as a proximity search with distance 1
    Args:
        terms (list): Description
        max_distance (int): The distance of the words indicated by the number after the # in the query
        keep_order (boolean): If True, it's a phrase search (where order matters) otherwise it's a proximity search
        inverted_index (dict)
        doc_nums (list)
    Returns:
        boolean_vector (list): A list of True and False for each document
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
    """Converts a list of document ids into a list of boolean values for each document, which indicates the existance
    of a search result in each dicument
    Args:
        doc_list (list)
        doc_nums (list)
    Returns:
        boolean_doc_list (list)
    """
    boolean_doc_list = np.zeros(len(doc_nums), dtype=np.bool)

    for doc_id in doc_list:
        doc_index = doc_nums.index(str(doc_id))
        boolean_doc_list[doc_index] = True

    return boolean_doc_list


def convert_booleans_to_docs_ids(bool_list, doc_nums):
    """The opposite of the function convert_doc_ids_to_boolean. It converts a list of boolean values to a list of document ids
    Args:
        bool_list (list)
        doc_nums (list)
    Returns:
        doc_id_list (list)
    """
    doc_id_list = []

    for index, boolean_val in enumerate(bool_list):
        if boolean_val is True:
            doc_id = doc_nums[index]
            doc_id_list.append(doc_id)
    return doc_id_list


def boolean_search(query_str_transformed, doc_nums):
    """Performs boolean search for one or more combinations of terms.
    Args:
        query_str_transformed (str): A string representation of the AND, OR and NOT used between numpy arrays
        doc_nums (list)
    Returns:
        documents (list): The resulting documents of the boolean search
    """
    doc_num_mapping = {}  # Map doc numbers to continuous indices

    for ind, doc_num in enumerate(doc_nums):
        doc_num_mapping[ind] = doc_num

    # Use eval to evaluate the boolean search
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

    # @TODO: Remove stop words if not a logical operator
    return query.lower().split()


def array_to_string(arr):
    array_str = ''

    for i in arr:
        array_str += '{}, '.format(i)
    return array_str


def boolean_search_queries(queries, collection_table, inverted_index, doc_nums):
    """Performs boolean, phrase or proximity search for a given set of queries
    Args:
        queries (list): Preprocessed queries
        collection_table (dict)
        inverted_index (dict)
        doc_nums (list)
    Returns:
        search_results (list): The resultsing documents for each search query
    """
    print('Starting boolean search...')
    logical_operators_mapping = {'and': '&', 'or': '|', 'not': '~'}
    search_results = []

    for query in queries:
        query_tokens = split_query(query)
        query_eval_string = ''

        for token in query_tokens:
            if token.startswith('#'):   # Proximity search
                # Split phrase into distance and terms
                distance, term_1, term_2 = list(filter(None, re.split(r'\W+', token)))
                terms = stemming([term_1, term_2])
                boolean_vector = phrase_proximity_search(terms, int(distance), False, inverted_index, doc_nums)
                query_eval_string += 'np.array([{}]) '.format(array_to_string(boolean_vector))

            elif token.startswith('"'):    # Phrase search
                terms = stemming(token.replace('_', ' ').replace('"', '').split())
                boolean_vector = phrase_proximity_search(terms, 1, True, inverted_index, doc_nums)
                query_eval_string += 'np.array([{}]) '.format(array_to_string(boolean_vector))

            elif token in logical_operators_mapping:    # One of AND, OR, NOT
                token = logical_operators_mapping[token]
                query_eval_string += '{} '.format(token)

            else:
                stem_word = stemming([token])[0]
                boolean_vector = collection_table[stem_word]
                query_eval_string += 'np.array([{}]) '.format(array_to_string(boolean_vector))

        query_search_results = boolean_search(query_eval_string, doc_nums)
        search_results.append(query_search_results)

    return search_results


def save_boolean_search_results(queries, results_boolean, file_name):
    """Saves the results for the boolean search queries in the provided file_name
    Args:
        queries (list): Queries from queries.boolean.txt
        results_boolean (list)
        file_name (str): queries.boolean.txt
    """
    f = open(file_name + '.txt', 'w+')

    for index, query in enumerate(queries):
        for doc_id in results_boolean[index]:
            f.write(str(index + 1) + ' 0 ' + doc_id + ' 0 1 0\n')
    f.close()
    print('Boolean search results saved at {}.txt\n'.format(file_name))


def ranked_retrieval(queries, collection_table, doc_nums, inverted_index, stop_words):
    """Performs ranked IR based on TFIDF
    Args:
        queries (list): Queries from queries.ranked.txt
        collection_table (dict)
        doc_nums (list)
        inverted_index (dict)
        stop_words (list)
    Returns:
        ranked_scores (list): The resultsing documents and the score for each ranked query
    """
    print('Starting ranked retrieval...')
    ranked_scores = {}

    for query_index, query_tokens in enumerate(queries):
        # Convert query into an OR boolean search and use eval to evaluate it
        boolean_vectors = []
        for token in query_tokens:
            boolean_vector = collection_table[token]
            boolean_vectors.append('np.array([{}])'.format(array_to_string(boolean_vector)))

        query_eval_string = ' | '.join(boolean_vectors)
        query_documents = boolean_search(query_eval_string, doc_nums)

        query_scores = []
        # Map query_boolean_result to a list of document ids
        for doc in query_documents:
            score = TFIDF(doc, query_tokens, len(doc_nums), inverted_index)
            query_scores.append((doc, score))

        # Sort scores for each query on a descending order
        query_scores = sorted(query_scores, key=lambda x: x[1], reverse=True)
        ranked_scores[query_index + 1] = query_scores

    return ranked_scores


def TFIDF(document, terms, N, inverted_index):
    """Calculates the retrieval score using the TFIDF (term frequency - inverse document frequency) formula
    Args:
        document (str)
        terms (list)
        N (list): Total number of documents
        inverted_index (dict)
    Returns:
        total_score (float): Retrieval score for a query and a document
    """
    total_score = 0

    # For each term calculate the tf (term frequency in doc) and df (number of docs that word appeared in)
    for term in terms:
        # Check if the document includes the term
        if document in inverted_index[term].keys():
            # Frequency of term in this document
            tf = len(inverted_index[term][document])
            # Number of documents in which the term appeared
            df = len(inverted_index[term].keys())
            term_weight = (1 + np.log10(tf)) * np.log10(N / df)
            total_score += term_weight

    return total_score


def save_ranked_retrieval_results(ranked_results, file_name):
    """Saves the results for the ranked queries in the provided file_name
    Args:
        ranked_results (list)
        file_name (str)
    """
    f = open(file_name + '.txt', 'w+')

    for query in ranked_results.keys():
        for index, (doc, score) in enumerate(ranked_results[query]):
            if index < 1000:
                printed_res = str(query) + ' 0 ' + doc + ' 0 ' + '%.4f' % score + ' 0 \n'
                f.write(printed_res)
    f.close()
    print('Ranked search results saved at {}.txt\n'.format(file_name))
