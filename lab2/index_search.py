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

    print(collection_dict)
    return collection_dict


def boolean_search(collection_dict, queries):
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
                converted_query.append('np.%s' % repr(collection_dict[word]))

        query_final = ' '.join(converted_query)
        boolean_vector = eval(query_final)

        for b_index, b in enumerate(boolean_vector):
            print(b_index)
            if b:
                results_boolean.append([index+1, 0, b_index+1, 0, 1, 0])
        print('\n')

    print(results_boolean)
    return results_boolean


def save_boolean_search_results(results_boolean, file_name):
    print('\n\n')
    f = open(file_name + '.txt', 'a+')
    for row in results_boolean:
        row_str = ' '.join(str(i) for i in row)
        print(row_str)
        f.write(row_str + '\n')
    f.close()
