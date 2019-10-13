import numpy as np


def create_term_doc_collection(inverted_index, num_of_docs):
    words = [word for word in inverted_index]
    collection_dict = dict()
    boolean_vector = np.zeros((len(words), num_of_docs), dtype=np.bool)

    for i, word in enumerate(words):
        docs_with_word = inverted_index[word].keys()

        for doc in docs_with_word:
            boolean_vector[i][doc-1] = 1

        collection_dict[word] = boolean_vector[i]

    return collection_dict


def boolean_search(collection_dict, queries, token_doc_list):
    logical_operators = {'and': '&', 'or': '|', 'not': '~'}
    results_boolean = []

    # For each query find the related documents
    for index, query in enumerate(queries):
        # Split the query string into a list
        query_list = query.lower().split(' ')
        converted_query = []

        # Convert each word except for the logical operators to a binary number
        for word in query_list:
            if word in logical_operators.keys():
                converted_query.append(logical_operators[word])
            else:
                if word not in collection_dict:
                    # Word isn't included in any of the documents
                    collection_dict_False = np.zeros(len(token_doc_list), dtype=bool)
                    converted_query.append('np.%s' % repr(collection_dict_False))
                else:
                    word_vector_str = ''
                    np.set_printoptions(threshold=np.prod(collection_dict[word].shape))
                    for word in collection_dict[word]:
                        word_vector_str += '%s, ' % (word)

                    converted_query.append('np.array([%s])' % word_vector_str)

        final_query = ' '.join(converted_query)
        boolean_vector = eval(final_query)
        documents = [i+1 for i in range(len(boolean_vector)) if boolean_vector[i] == True]

        for doc in documents:
            results_boolean.append([index+1, 0, doc, 0, 1, 0])

    print('Boolean results: ', results_boolean)
    return results_boolean


def save_boolean_search_results(results_boolean, file_name):
    f = open(file_name + '.txt', 'w+')
    for row in results_boolean:
        row_str = ' '.join(str(i) for i in row)
        f.write(row_str + '\n')
    f.close()
