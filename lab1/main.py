import urllib.request
import os.path
import re
from stemming.porter2 import stem


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
    """
    no_punctuation = re.sub(r"[.?\-\",!;'\n+]", "", text, flags=re.I)
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
    """
    return list([stem(word) for word in words])


if __name__ == '__main__':
    # @TODO#2: Write code to download txt.gz

    BIBLE_FILE = 'bible.txt'
    WIKI_FILE = 'abstracts.wiki.txt'
    STOP_WORDS_FILE = 'stop_words.txt'

    download_file_and_save(
        'http://www.gutenberg.org/cache/epub/10/pg10.txt',
        'bible.txt')
    download_file_and_save(
        'http://members.unine.ch/jacques.savoy/clef/englishST.txt',
        'stop_words.txt')

    with open(STOP_WORDS_FILE) as file:
        stop_words = [word.strip() for word in file]

    with open(BIBLE_FILE, 'r') as f:
        lines = f.readlines()
        print('{} lines of text'.format(len(lines)))

        tokenised_text = tokenise(' '.join(lines))
        text_with_no_stop_words = remove_stop_words(tokenised_text)
        normalised_text = normalise(text_with_no_stop_words)

        # Save preprocessed text to a new file
        with open('bible_preprocess.txt', 'w+') as new_file:
            print(type(normalised_text))
            normalised_text_string = ' '.join(normalised_text)
            new_file.write(normalised_text_string)
            # new_file.close()
