import numpy as np


def create_boolean_search_table(inverted_index, num_of_docs):
    words = [word for word in inverted_index]
    boolean_table = np.zeros((len(words), num_of_docs))
    print(words)

    index = 0
    for word in words:
        docs_with_word = inverted_index[word].keys()
        print(docs_with_word.keys())
        # boolean_table[index] = [i in docs_with_word for i in ]
        index += 1
