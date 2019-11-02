import re


def preprocess_tweet(tweet):
    # Remove links, unicode characters and extra spaces from tweet
    reg_1 = r'http\S+'
    reg_2 = r'[\x85]'
    reg_3 = r'[^\w\s\#\@]'
    reg_4 = r'\s+'

    tweet = re.sub(reg_1, '', tweet, flags=re.MULTILINE)
    tweet = re.sub(reg_2, '', tweet, flags=re.MULTILINE)
    tweet = re.sub(reg_3, '', tweet, flags=re.MULTILINE)
    tweet = re.sub(reg_4, ' ', tweet, flags=re.MULTILINE)
    tweet = tweet.lower().strip().split(' ')
    return tweet


def map_classes_to_ids():
    with open('./class_ids.txt', 'r') as f:
        lines = f.readlines()
        class_ids = dict()

        for line in lines:
            category, category_id = line.replace('\n', '').split('\t')
            class_ids[category] = category_id

        return class_ids


def load_dataset(dataset_type):
    with open('./tweetsclassification/Tweets.14cat.' + dataset_type, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        tweets_dict = dict()
        tweet_words = []  # all tweet words

        if '\n' in lines:
            lines.remove('\n')

        for line in lines:
            line = line.strip().split('\t')
            tweet_id, tweet, target = line
            tweets_dict[tweet_id] = tuple([preprocess_tweet(tweet), target])
            tweet_words.extend(preprocess_tweet(tweet))

    # Preprocess tweets - remove links and tokenise
    unique_words = list(set(tweet_words))

    return tweets_dict, unique_words


def create_feature_vector_file(tweets_dict, unique_words, class_ids, dataset_type):
    with open('./results/feats.' + dataset_type, 'w') as f:
        for tweet_id in tweets_dict:
            words, category = tweets_dict[tweet_id]
            feature_vector = []

            for w in words:
                if w in unique_words:
                    feature_vector.append(unique_words.index(w) + 1)
            # feature_vector = [unique_words.index(w) + 1 for w in words]
            f.write(class_ids[category])

            for word_id in sorted(list(set(feature_vector))):
                f.write(' ' + str(word_id) + ':' + '1')

            f.write(' #' + tweet_id + '\n')
        print('File ./results/feats.' + dataset_type, 'created')


if __name__ == '__main__':
    class_ids = map_classes_to_ids()
    tweets_dict_train, unique_words_train = load_dataset('train')
    tweets_dict_test, unique_words_test = load_dataset('test')

    # Print words with unique id in a file
    # with open('./results/feats.dic', 'w') as f:
    #     for idx, word in enumerate(unique_words_train):
    #         f.write(str(idx + 1) + ' ' + word + '\n')

    # Create feature vectors for training and test set
    create_feature_vector_file(tweets_dict_train, unique_words_train, class_ids, 'train')
    create_feature_vector_file(tweets_dict_test, unique_words_train, class_ids, 'test')
