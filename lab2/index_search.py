import numpy as np
from preprocess import normalise, tokenise


# def map_docs_number_to_index(docs, doc_nums):
#     docs_ind = []

#     for doc in docs:


def create_term_doc_collection(inverted_index, doc_nums):
    words = list(inverted_index.keys())
    collection_dict = dict()
    boolean_matrix = np.zeros((len(words), len(doc_nums)), dtype=np.bool)

    # Create an index of the doc numbers
    doc_num_dict = {}
    for ind, doc_num in enumerate(doc_nums):
        doc_num_dict[doc_num] = ind

    for i, word in enumerate(words):
        docs_for_specific_word = list(inverted_index[word].keys())
        # print(docs_for_specific_word)

        for index, doc_num in enumerate(docs_for_specific_word):
            doc_id = doc_num_dict[doc_num]
            boolean_matrix[i][int(doc_id)] = True

        # print('\n', len(boolean_matrix[i]))
        collection_dict[word] = boolean_matrix[i]

    # print(collection_dict)
    return collection_dict


def boolean_search(collection_dict, queries, num_of_docs):
    logical_operators = {'and': '&', 'or': '|', 'not': '~'}
    results_boolean = []

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
                    collection_dict_False = np.zeros(num_of_docs, dtype=bool)
                    converted_query.append('np.%s' % repr(collection_dict_False))
                else:
                    word_vector_str = ''
                    np.set_printoptions(threshold=np.prod(collection_dict[word].shape))
                    # print(collection_dict[word])

                    for w in collection_dict[word]:
                        # word_vector_str += '%s, ' % (w)
                        word_vector_str += '{}, '.format(w)

                    # print(word_vector_str)
                    converted_query.append('np.array([%s])' % word_vector_str)

        final_query = ' '.join(converted_query)
        boolean_vector = eval(final_query)
        documents = [i+1 for i in range(len(boolean_vector)) if boolean_vector[i] == True]
        # print(documents)

        for doc in documents:
            results_boolean.append([index+1, 0, doc, 0, 1, 0])

    # print('Boolean results: ', results_boolean)
    for res in results_boolean:
        print(res)
    return results_boolean


def save_boolean_search_results(results_boolean, file_name):
    f = open(file_name + '.txt', 'w+')
    for row in results_boolean:
        row_str = ' '.join(str(i) for i in row)
        f.write(row_str + '\n')
    f.close()
