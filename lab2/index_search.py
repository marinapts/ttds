import numpy as np


def create_boolean_search_table(inverted_index, num_of_docs):
    words = [word for word in inverted_index]
    boolean_table = np.zeros((len(words), num_of_docs))

    for i, word in enumerate(words):
        docs_with_word = inverted_index[word].keys()

        print(docs_with_word)
        print(docs_with_word)

        for doc in docs_with_word:
            boolean_table[i][doc - 1] = 1

    print(boolean_table)
    return boolean_table
