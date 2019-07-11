from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer
import classification

n_samples = 2000
n_features = 1000
n_components = 3
n_top_words = 100


def isExists(word_list, target_word):
    for word in word_list:
        if word == target_word:
            return True

    return False


def words_tag_freq_calculation(container_path, truthful):
    # load data set from given directory path
    training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    filter_data = []
    for index in range(0, len(training_data.data)):
        if (training_data.target[index] == truthful):
            # print('Target: ', training_data.target[index], 'Content: ', training_data.data[index])
            filter_data.append(training_data.data[index])

    return filter_data


def print_top_words(model, feature_names, n_top_words):
    full_str = ""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        full_str += " ".join([feature_names[i]
                              for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

    print()
    full_str = full_str.split()
    return full_str


def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list


def get_top_words_from_topic_modeling(container_path, n_topic, n_top_words):
    training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf_review = tf_vectorizer.fit_transform(training_data.data)

    lda_review = LatentDirichletAllocation(n_components=n_topic, max_iter=20,
                                           learning_method='online',
                                           learning_offset=50.,
                                           random_state=0)
    lda_review.fit(tf_review)
    tf_feature_names = tf_vectorizer_pos_review.get_feature_names()

    return print_top_words(lda_review, tf_feature_names, n_top_words)


print("Loading dataset...")

container_path_neg = "../data/negative_polarity/"
container_path_pos = "../data/positive_polarity/"
container_path_comb = "../data/combined/"

neg_truthful = words_tag_freq_calculation(container_path_neg, True)
neg_deceptive = words_tag_freq_calculation(container_path_neg, False)

pos_truthful = words_tag_freq_calculation(container_path_pos, True)
pos_deceptive = words_tag_freq_calculation(container_path_pos, False)

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")

tf_vectorizer_neg_review = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_neg_review = tf_vectorizer_neg_review.fit_transform(neg_deceptive + neg_truthful)

tf_vectorizer_pos_review = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_pos_review = tf_vectorizer_pos_review.fit_transform(pos_deceptive + pos_truthful)

tf_vectorizer_neg_deceptive = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_neg_deceptive = tf_vectorizer_neg_deceptive.fit_transform(neg_deceptive)

tf_vectorizer_neg_truthful = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_neg_truthful = tf_vectorizer_neg_truthful.fit_transform(neg_truthful)

tf_vectorizer_pos_deceptive = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_pos_deceptive = tf_vectorizer_pos_deceptive.fit_transform(pos_deceptive)

tf_vectorizer_pos_truthful = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_pos_truthful = tf_vectorizer_pos_truthful.fit_transform(pos_truthful)

tf_vectorizer_truthful = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_truthful = tf_vectorizer_truthful.fit_transform(pos_truthful + neg_truthful)

tf_vectorizer_deceptive = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_deceptive = tf_vectorizer_deceptive.fit_transform(pos_deceptive + neg_deceptive)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda_neg_review = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                           learning_method='online',
                                           learning_offset=50.,
                                           random_state=0)
lda_neg_review.fit(tf_pos_review)

# print("\nTopics in LDA model of negative deceptive:")
tf_feature_names = tf_vectorizer_pos_review.get_feature_names()
top_word_list = print_top_words(lda_neg_review, tf_feature_names, n_top_words)

''' Start '''

lda_neg_deceptive = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                              learning_method='online',
                                              learning_offset=50.,
                                              random_state=0)
lda_neg_deceptive.fit(tf_pos_deceptive)

print("\nTopics in LDA model of negative deceptive:")
tf_feature_names = tf_vectorizer_pos_deceptive.get_feature_names()
neg_topic_list = print_top_words(lda_neg_deceptive, tf_feature_names, n_top_words)

lda_neg_truthful = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                             learning_method='online',
                                             learning_offset=50.,
                                             random_state=0)
lda_neg_truthful.fit(tf_pos_truthful)

print("\nTopics in LDA model of negative truthful:")
tf_feature_names = tf_vectorizer_pos_truthful.get_feature_names()
neg_topic_list = neg_topic_list + print_top_words(lda_neg_truthful, tf_feature_names, n_top_words)

neg_topic_list = Remove(neg_topic_list)

''' End '''

top_word_list = Remove(top_word_list)
# print(top_word_list)

forward_bigram = []
backward_bigram = top_word_list

neg_review = neg_truthful + neg_deceptive

for each_review in neg_review:
    word_list = each_review.split()

    prev_word = ""
    for index in range(len(word_list) - 1):
        if word_list[index] in top_word_list:
            to = None
            # if prev_word+ " "+ word_list[index] not in backward_bigram:
            # backward_bigram.append(prev_word + " " + word_list[index])
            # if word_list[index] + " "+ word_list[index+1] not in backward_bigram:
            # backward_bigram.append(word_list[index] + " "+ word_list[index+1])

            # if prev_word + " " + word_list[index] +" "+ word_list[index+1] not in backward_bigram:
            # backward_bigram.append(prev_word + " " + word_list[index] +" "+ word_list[index+1])

        prev_word = word_list[index]

print(len(backward_bigram))
# backward_bigram = None

categories = ['deceptive_from_MTurk', 'truthful_from_Web']

n_fold = 5
# classifiy_review will return classification result
result = classification.classifiy_review(container_path_pos, categories, n_fold, top_word_list)
# precision, recall, f1_score of each categories
for k, value in result.items():
    if k != 'accuracy':
        print(k, ": ", categories[0], " = ", value[0], ",", categories[1], " = ", value[1])
    else:
        print('Accuracy : ', result[k])

'''
training_data = load_files(container_path_neg, description=None, categories=categories, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)
print(training_data.target_names)


tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2),stop_words='english', vocabulary=backward_bigram)


X_train_counts = tf_vectorizer.fit_transform(training_data.data)
#print(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
'''

'''

lda_neg_deceptive = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda_neg_deceptive.fit(tf_neg_deceptive)

print("\nTopics in LDA model of negative deceptive:")
tf_feature_names = tf_vectorizer_neg_deceptive.get_feature_names()
print_top_words(lda_neg_deceptive, tf_feature_names, n_top_words)

lda_neg_truthful = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda_neg_truthful.fit(tf_neg_truthful)

print("\nTopics in LDA model of negative truthful:")
tf_feature_names = tf_vectorizer_neg_truthful.get_feature_names()
print_top_words(lda_neg_truthful, tf_feature_names, n_top_words)


lda_pos_deceptive = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda_pos_deceptive.fit(tf_pos_deceptive)

print("\nTopics in LDA model of positive deceptive:")
tf_feature_names = tf_vectorizer_pos_deceptive.get_feature_names()
print_top_words(lda_pos_deceptive, tf_feature_names, n_top_words)


lda_pos_truthful = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda_pos_truthful.fit(tf_pos_truthful)

print("\nTopics in LDA model of positive truthful:")
tf_feature_names = tf_vectorizer_pos_truthful.get_feature_names()
print_top_words(lda_pos_truthful, tf_feature_names, n_top_words)


lda_truthful = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda_truthful.fit(tf_truthful)

print("\nTopics in LDA model of truthful review :")
tf_feature_names = tf_vectorizer_truthful.get_feature_names()
print_top_words(lda_truthful, tf_feature_names, n_top_words)


lda_deceptive = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda_deceptive.fit(tf_deceptive)

print("\nTopics in LDA model of deceptive review :")
tf_feature_names = tf_vectorizer_deceptive.get_feature_names()
print_top_words(lda_deceptive, tf_feature_names, n_top_words)
'''

