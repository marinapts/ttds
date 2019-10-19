import re
from stemming.porter2 import stem


def tokenise(text):
    """Removes punctuation, new lines, multiple whitespaces and the initial
    FT in front of the headlines and then splits it into tokens
    Args:
        text (string): The text provided to tokenise
    Returns:
        tokenised (list): List of tokens
    """
    reg_1 = r'(^FT\s{2})|([^\w\s])|(\_)'
    reg_2 = r'\s+'

    # Replace punctuation marks and the abbreviation FT at the beginning of the headline with empty string
    no_punctuation_text = re.sub(reg_1, ' ', text, flags=re.MULTILINE)
    # Replace new lines and multiple spaces with empty string
    no_spaces_text = re.sub(reg_2, ' ', no_punctuation_text, flags=re.MULTILINE)
    # Lowecase and split text
    tokenised = no_spaces_text.lower().strip().split(' ')
    return tokenised


def remove_stop_words(words, stop_words):
    """Remove stop words
    Args:
        words (list): The list of tokenised words
    Returns:
        words (list): A list of words without the stop words included
    """
    words = list(filter(lambda x: x not in stop_words, words))
    return words


def normalise(words):
    """Porter stemmer
    Args:
        words (list): Description
    Returns:
        (list): Normalised list of prepeocessed words
    """
    return list([stem(word) for word in words])
