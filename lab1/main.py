import urllib.request
import os.path
import re
import matplotlib.pyplot as plt
import numpy as np
from stemming.porter2 import stem
from collections import Counter


def download_file_and_save(url, file_name):
    """Download the file from `url` and save it locally under `file_name`:

    Args:
        url (string): The url to download the file from
        file_name (string): The name of the file
    """
    if not os.path.exists('./' + file_name):
        print('Downloading file from {}...'.format(url))
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            data = response.read()  # a `bytes` object
            print('Saving file as {}...'.format(file_name))
            out_file.write(data)


def tokenise(text):
    """Removes punctuation, new lines and multiple white spaces from text
    and then converts it to lowecase and splits it into tokens

    Args:
        text (string): The text provided to tokenise

    Returns:
        TYPE: Description
    """
    no_punctuation = re.sub(r"[.?\-\",!;':()\[\\]\n+\t+]", "", text, flags=re.I)
    no_punctuation = re.sub(r"\s{2,}", "", no_punctuation, flags=re.I)
    tokenised = no_punctuation.lower().split(' ')
    print('{} words after tokenisation'.format(len(tokenised)))
    return tokenised


def remove_stop_words(words):
    """Remove stop words

    Args:
        words (list): The list of tokenised words

    Returns:
        words (list): A list of words without the stop words included
    """
    words = list(filter(lambda x: x not in stop_words, words))
    print('{} words after removing stop words'.format(len(words)))
    return words


def normalise(words):
    """Porter stemmer

    Args:
        words (TYPE): Description

    Returns:
        TYPE: Description
    """
    return list([stem(word) for word in words])


def plot_zipfs_law(log_rank, log_frequency):
    """Plots the rank of the words with their frequency on a log scale

    Args:
        log_rank (list): Log of word rank
        log_frequency (list): Log of word frequency
    """
    plt.plot(log_rank, log_frequency)

    plt.suptitle('Zipf\'s Law')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.show()


def plot_benfords_law(first_digits_counts):
    """Plot Benford's law

    Args:
        first_digits_counts (list): Frequency of the first digits of the word frequencies
    """
    first_digits_prob = [np.log(1+(1/digit)) for digit in first_digits_counts.keys()]

    plt.bar(first_digits_counts.keys(), first_digits_prob)
    plt.suptitle('Benford\'s Law')
    plt.xlabel('First digit')
    plt.ylabel('log(1 + 1/digit)')
    plt.show()


def heaps_law(tokenised_text):
    """Plot Heap's

    Args:
        tokenised_text (list): List of words

    Returns:
        total_num_words_vector (list): Number of words that were read
        vocab_size (list): Number of unique words
    """
    vocab_size = list()
    vocab = dict()
    num_words_read = 0

    for idx, word in enumerate(tokenised_text):
        num_words_read += 1

        if word in vocab:
            vocab[word] += 1
            vocab_size.append(vocab_size[-1])   # vocab size stays the same
        else:
            vocab[word] = 1
            vocab_size.append(vocab_size[-1] + 1 if len(vocab_size) > 0 else 1)   # Increase the vocab size

    total_num_words_vector = np.array(np.arange(num_words_read)).reshape(-1, 1)
    plt.plot(total_num_words_vector, vocab_size)
    plt.suptitle('Heap\'s Law')
    plt.xlabel('Number of words')
    plt.ylabel('Vocabulary size')
    # plt.show()

    return total_num_words_vector, np.array(vocab_size).reshape(-1, 1)


def fit_heaps_law_parameters(num_words_read_vector, vocab_size):
    """Fit linear regression to Heap's Law

    Args:
        num_words_read_vector (list): Vector of inputs
        vocab_size (list): Vector of outputs

    Returns:
        params (list): Vector of params b and log_k
    """
    transformed_v = np.ones(len(num_words_read_vector)).reshape(-1, 1)
    log_v_vector = np.concatenate((np.log(num_words_read_vector), np.log(transformed_v)), axis=1)

    params = np.linalg.lstsq(np.log(vocab_size), log_v_vector, rcond=None)[0]
    b, log_k = params[0]

    pred_vocab = np.exp(log_k) * (num_words_read_vector**b)

    # Revisit this - predictions don't seem correct!
    plt.plot(num_words_read_vector, pred_vocab)
    plt.show()


if __name__ == '__main__':
    BIBLE_FILE = 'bible.txt'
    WIKI_FILE = 'abstracts_wiki.txt'
    STOP_WORDS_FILE = 'stop_words.txt'

    WIKI_PREPROCESSED = 'wiki_preprocess.txt'
    BIBLE_PREPROCESSED = 'bible_preprocess.txt'

    download_file_and_save(
        'http://www.gutenberg.org/cache/epub/10/pg10.txt',
        BIBLE_FILE)
    download_file_and_save(
        'https://www.inf.ed.ac.uk/teaching/courses/tts/labs/lab1/abstracts.wiki.txt.gz',
        WIKI_FILE)
    download_file_and_save(
        'http://members.unine.ch/jacques.savoy/clef/englishST.txt',
        STOP_WORDS_FILE)

    # Store stop words in a list
    with open(STOP_WORDS_FILE) as file:
        stop_words = [word.strip() for word in file]

    with open(BIBLE_FILE, 'r') as f:
        lines = f.readlines()
        print('{} lines of text'.format(len(lines)))

        tokenised_text = tokenise(' '.join(lines))
        text_with_no_stop_words = remove_stop_words(tokenised_text)
        normalised_text = normalise(text_with_no_stop_words)

        # Save preprocessed text to a new file
        with open(BIBLE_PREPROCESSED, 'w+') as new_file:
            normalised_text_string = ' '.join(normalised_text)
            new_file.write(normalised_text_string)

        # Calculate frequency of words
        word_freq = Counter(normalised_text)
        counts = dict(word_freq.most_common())
        print('Frequency of words: {}\n', counts)

        # Zipf's law
        labels, frequency = zip(*counts.items())
        rank = np.arange(len(labels)) + 1
        plot_zipfs_law(np.log(rank), np.log(frequency))

        # Benford's law
        first_digits = [int(str(freq)[0]) for freq in frequency]
        first_digits_counts = dict(sorted(Counter(first_digits).items()))
        plot_benfords_law(first_digits_counts)

        total_num_words_vector, vocab_size = heaps_law(tokenised_text)
        fit_heaps_law_parameters(total_num_words_vector[1::], vocab_size[1::])
