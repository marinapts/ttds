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

    # print(collection_dict)
    return collection_dict


def boolean_search(collection_dict, queries, token_doc_list):
    logical_operators = {'and': '&', 'or': '|', 'not': '~'}
    results_boolean = []

    # For each query find the related documents
    for index, query in enumerate(queries):
        # Split the query string into a list
        query_list = query.lower().split(' ')
        print('query_list: ', query_list)
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
                    # @TODO: inverted_index for 'correct' is correct, but collection_dict[word] is NOT
                    print([i for i in collection_dict[word]])
                    converted_query.append('np.%s' % repr(collection_dict[word]))


        print('converted_query: ', converted_query)
        query_final = ' '.join(converted_query)
        print('query_final: ', query_final)
        boolean_vector = eval(query_final)

        for b_index, b in enumerate(boolean_vector):
            # print(b_index)
            if b:
                results_boolean.append([index+1, 0, b_index+1, 0, 1, 0])
        print('\n')

    print('Boolean results:')
    print(results_boolean)
    return results_boolean


def save_boolean_search_results(results_boolean, file_name):
    f = open(file_name + '.txt', 'w+')
    for row in results_boolean:
        row_str = ' '.join(str(i) for i in row)
        f.write(row_str + '\n')
    f.close()
