import re
from stemming.porter2 import stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, tree
from sklearn.metrics import classification_report, f1_score


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


def preprocess_tweet(tweet):
    # Remove links, unicode characters and extra spaces from tweet
    reg_1 = r'http\S+'
    reg_2 = r'[\x85]'
    reg_3 = r'(RT\s{1})|([^\w\s\#\@])'
    reg_3 = r'[^\w\s\#\@]'
    reg_4 = r'\s+'

    tweet = re.sub(reg_1, '', tweet, flags=re.MULTILINE)  # Remove links
    tweet = re.sub(reg_2, '', tweet, flags=re.MULTILINE)  # Remove unicode character
    tweet = re.sub(reg_3, '', tweet, flags=re.MULTILINE)  # Remove RT that is present on most tweets and most punctuation
    tweet = re.sub(reg_4, ' ', tweet, flags=re.MULTILINE)  # Remove extra spaces
    tweet = tweet.lower().strip().split(' ')

    # Duplicate words with # or @
    for t in tweet:
        if t.startswith('#'):
            tweet.append(t.split('#')[1])
        elif t.startswith('@'):
            tweet.append(t.split('@')[1])

    return stemming(remove_stop_words(tweet, stop_words))  # Apply stemming and stop words removal


def load_dataset(dataset_type):
    # with open('./tweetsclassification/Tweets.14cat.' + dataset_type, 'r', encoding='ISO-8859-1') as f:
    with open('./tweetsclassification/tweets.' + dataset_type, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()

        if '\n' in lines:
            lines.remove('\n')

        tweets = [preprocess_tweet(line.strip().split('\t')[1]) for line in lines]
        tweets = [' '.join(t) for t in tweets]
        targets = [line.split('\t')[2].replace('\n', '') for line in lines]
        return tweets, targets

    # # Remove duplicates
    # unique_words = list(set(tweet_words))
    # return tweets_dict, unique_words


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


def map_classes_to_ids():
    with open('./class_ids.txt', 'r') as f:
        lines = f.readlines()
        class_ids = dict()

        for line in lines:
            category, category_id = line.replace('\n', '').split('\t')
            class_ids[category] = category_id

        return class_ids


# =================================================
#               Main
# =================================================

with open('./englishST.txt') as file:
    stop_words = [word.strip() for word in file]

target_names = map_classes_to_ids()

X_train, y_train = load_dataset('train')
X_test, y_test = load_dataset('test')

vectorizer = CountVectorizer(encoding='ISO-8859-1', strip_accents='unicode')
X_train_transform = vectorizer.fit_transform(X_train)
X_test_transform = vectorizer.transform(X_test)

# Fit Multinomial NB
multinomial_clf = MultinomialNB().fit(X_train_transform, y_train)
y_pred = multinomial_clf.predict(X_test_transform)
y_pred_prob = multinomial_clf.predict_proba(X_test_transform)
acc = multinomial_clf.score(X_test_transform, y_test)
print('Multinomial NB accuracy', acc)
print('Multinomial NB f1_score', f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, target_names=target_names))

# Fit SVM
svm_clf = svm.LinearSVC().fit(X_train_transform, y_train)
y_pred = svm_clf.predict(X_test_transform)
acc = svm_clf.score(X_test_transform, y_test)
print('\nSVM accuracy', acc)
print('SVM f1_score', f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, target_names=target_names))

# Fit DecisionTreeClassifier
decision_tree_clf = tree.DecisionTreeClassifier().fit(X_train_transform, y_train)
y_pred = decision_tree_clf.predict(X_test_transform)
acc = decision_tree_clf.score(X_test_transform, y_test)
print('\nDecision tree accuracy', acc)
print('Decision tree f1_score', f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, target_names=target_names))