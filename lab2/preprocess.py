import re
from stemming.porter2 import stem


def tokenise(text):
    """Removes punctuation, new lines and multiple white
    Args:
        text (string): The text provided to tokenise
    Returns:
        tokenised (list): List of tokens
    """
    # @TODO: Remove only FT, not the date as well
    no_date_in_headline = re.sub(r"^.*?/+", "", text, flags=re.MULTILINE)   # Remove date from the headlines
    no_punctuation = re.sub(r"[.?\-\",!;'/:()\[\]\(\)&\n+\t+]", " ", no_date_in_headline, flags=re.MULTILINE)  # Remove punctuation
    no_extra_spaces = re.sub(r"\s{2,}", " ", no_punctuation, flags=re.I)   #
    tokenised = no_extra_spaces.lower().strip().split(' ')
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
