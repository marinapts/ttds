import numpy as np
from preprocess import normalise, tokenise


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


def boolean_search(collection_dict, queries, doc_nums):
    """Applies Boolean search

    Args:
        collection_dict (dict): Description
        queries (list): Description
        doc_nums (list): Description

    Returns:
        results_boolean (list): Results of the boolean search for each query
    """
    logical_operators = {'and': '&', 'or': '|', 'not': '~'}
    results_boolean = []
    doc_num_dict = {}

    for ind, doc_num in enumerate(doc_nums):
        doc_num_dict[ind] = doc_num

    # For each query find the related documents
    for index, query in enumerate(queries):
        # Preprocess the query string
        query_list = normalise(tokenise(query))
        converted_query = []

        # Convert each word except for the logical operators to a binary number
        for word in query_list:
            if word in logical_operators.keys():
                converted_query.append(logical_operators[word])
            else:
                if word not in collection_dict:
                    # Word isn't included in any of the documents
                    collection_dict_False = np.zeros(len(doc_nums), dtype=bool)
                    converted_query.append('np.%s' % repr(collection_dict_False))
                else:
                    word_vector_str = ''
                    np.set_printoptions(threshold=np.prod(collection_dict[word].shape))

                    for w in collection_dict[word]:
                        word_vector_str += '{}, '.format(w)

                    converted_query.append('np.array([%s])' % word_vector_str)

        final_query = ' '.join(converted_query)
        boolean_vector = eval(final_query)
        documents = [doc_num_dict[i] for i in range(len(boolean_vector)) if boolean_vector[i] == True]

        for doc in documents:
            results_boolean.append([index + 1, 0, doc, 0, 1, 0])

    return results_boolean


def save_boolean_search_results(results_boolean, file_name):
    f = open(file_name + '.txt', 'w+')
    for row in results_boolean:
        row_str = ' '.join(str(i) for i in row)
        f.write(row_str + '\n')
    f.close()


def proximity_search(inverted_index, queries):
    # #10(income, taxes)
    for index, query in enumerate(queries):
        query_list = normalise(tokenise(query))
        proximity_number = int(query_list[0].replace('#', ''))
        word_1 = query_list[1]
        word_2 = query_list[2]

        word_1_docs = inverted_index[word_1]
        word_2_docs = inverted_index[word_2]

        # Find common docs between the 2 words
        common_docs = set(word_1_docs).intersection(word_2_docs)
        docs_with_phrase = []

        for doc in list(common_docs):
            # Check the distance of the words
            for i in word_2_docs[doc]:
                for j in word_1_docs[doc]:
                    # @TODO: Check if order matters for proximity search (not phrase search)
                    distance_diff = int(i) - int(j)
                    if distance_diff <= proximity_number and distance_diff > 0:
                        docs_with_phrase.append(int(doc))

        proximity_doc_results = []
        for doc in sorted(set(docs_with_phrase)):
            proximity_doc_results.append([index + 1, 0, doc, 0, 1, 0])

        print(res for res in proximity_doc_results)
        return proximity_doc_results
