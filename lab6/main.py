import re
import requests
from stemming.porter2 import stem
from bs4 import BeautifulSoup


def remove_stop_words(words, stop_words):
    """Remove stop words
    Args:
        words (list): The list of tokenised words
    Returns:
        words (list): A list of words without the stop words included
    """
    words = list(filter(lambda x: x not in stop_words, words))
    return words


def stemming(words):
    """Applies Porter stemmer
    Args:
        words (list): Description
    Returns:
        (list): Normalised list of prepeocessed words
    """
    return list([stem(word) for word in words])


def get_title_from_link(links, tweet):
    ignore_urls = ['http://t.co/foXvuGfnaN', 'http://t.co/3aKDXDk4Vh']
    for link in links:
        try:
            if link not in ignore_urls:
                response = requests.get(link, timeout=None)

                print('ori', tweet)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = getattr(soup.title, 'string', '')
                    tweet = tweet.replace(link, title)
                    # print('new', tweet.replace(link, title))
                    # print('\n')
                else:
                    print('.....................Not 200............................................')
        except:
            print('.....................Not found............................................')
    return tweet


def preprocess_tweet(tweet):
    # Remove links, unicode characters and extra spaces from tweet
    reg_1 = r'http\S+'
    reg_2 = r'[\x85]'
    reg_3 = r'(RT\s{1})|([^\w\s\#\@])'
    reg_3 = r'[^\w\s\#\@]'
    reg_4 = r'\s+'

    # Replace links with titles
    links = re.findall(reg_1, tweet)
    tweet = get_title_from_link(links, tweet)

    tweet = re.sub(reg_1, '', tweet, flags=re.MULTILINE)  # Remove links
    tweet = re.sub(reg_2, '', tweet, flags=re.MULTILINE)  # Remove unicode character
    tweet = re.sub(reg_3, '', tweet, flags=re.MULTILINE)  # Remove RT that is present on most tweets and most punctuation
    tweet = re.sub(reg_4, ' ', tweet, flags=re.MULTILINE)  # Remove extra spaces
    print('new', tweet)
    tweet = tweet.lower().strip().split(' ')

    # Duplicate words with # or @
    for t in tweet:
        if t.startswith('#'):
            tweet.append(t.split('#')[1])
        elif t.startswith('@'):
            tweet.append(t.split('@')[1])

    return stemming(remove_stop_words(tweet, stop_words))  # Apply stemming and stop words removal


def map_classes_to_ids():
    with open('./class_ids.txt', 'r') as f:
        lines = f.readlines()
        class_ids = dict()

        for line in lines:
            category, category_id = line.replace('\n', '').split('\t')
            class_ids[category] = category_id

        return class_ids


def load_dataset(dataset_type, stop_words):
    with open('./tweetsclassification/Tweets.14cat.' + dataset_type, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        tweets_dict = dict()
        tweet_words = []  # all tweet words
        tweets_with_no_links = []
        ids = []

        if '\n' in lines:
            lines.remove('\n')

        for line in lines:
            line = line.strip().split('\t')
            tweet_id, tweet, target = line

            links_in_tweet = re.findall(r'http\S+', tweet)
            tweet = get_title_from_link(links_in_tweet, tweet)
            tweets_with_no_links.append([tweet_id, tweet, target])

            preprocessed_tweet = preprocess_tweet(tweet)
            print('\n')
            tweets_dict[tweet_id] = tuple([preprocessed_tweet, target])
            tweet_words.extend(preprocessed_tweet)
            ids.append(tweet_id)

    with open('./tweetsclassification/Tweets_new.' + dataset_type, 'w') as f:
        for tweet_row in tweets_with_no_links:
            f.write('\t'.join(tweet_row))
            f.write('\n')

    # Remove duplicates
    unique_words = list(set(tweet_words))
    return tweets_dict, unique_words, ids


def create_feature_vector_file(tweets_dict, unique_words, class_ids, dataset_type):
    with open('./results/feats.' + dataset_type, 'w') as f:
        for tweet_id in tweets_dict:
            words, category = tweets_dict[tweet_id]
            feature_vector = []

            for w in words:
                if w in unique_words:
                    feature_vector.append(unique_words.index(w) + 1)

            f.write(class_ids[category])

            for word_id in sorted(list(set(feature_vector))):
                f.write(' ' + str(word_id) + ':' + '1')

            f.write(' #' + tweet_id + '\n')

        print('File ./results/feats.' + dataset_type, 'created')


if __name__ == '__main__':
    with open('./englishST.txt') as file:
        stop_words = [word.strip() for word in file]

    class_ids = map_classes_to_ids()
    tweets_dict_train, unique_words_train, ids_train = load_dataset('train', stop_words)
    tweets_dict_test, _, ids_test = load_dataset('test', stop_words)

    # Print words with unique id in a file
    with open('./results/feats.dic', 'w') as f:
        for idx, word in enumerate(unique_words_train):
            f.write(str(idx + 1) + ' ' + word + '\n')
        print('File ./results/feats.dic created')

    # Create feature vectors for training and test set
    create_feature_vector_file(tweets_dict_train, unique_words_train, class_ids, 'train')
    create_feature_vector_file(tweets_dict_test, unique_words_train, class_ids, 'test')
