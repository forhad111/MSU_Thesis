from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer
import classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk import sent_tokenize, word_tokenize
import  random
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def get_unbalance_data_set(container_path, truthful_percentage, deceptive_percentage):
    training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    filter_truthful_index = []
    filter_deceptive_index = []
    for index in range(0, len(training_data.data)):
        if (training_data.target[index] == True):
            filter_truthful_index.append(index)
        else:
            filter_deceptive_index.append(index)

    filter_data_truthful = random.sample(filter_truthful_index, (int)(len(training_data.data)/2 * truthful_percentage))
    filter_data_deceptive = random.sample(filter_deceptive_index, (int)(len(training_data.data)/2 * deceptive_percentage))

    filter_data = filter_data_truthful+filter_data_deceptive

    list_need_to_delete = []
    new_data_list = []

    for index in range(0, len(training_data.data)):
        if index not in filter_data:

            list_need_to_delete.append(index)
        else:
            new_data_list.append(training_data.data[index])
    training_data.target = np.delete(training_data.target, list_need_to_delete)
    training_data.data =  new_data_list

    return training_data


def words_tag_freq_calculation(container_path, truthful):

    #load data set from given directory path
    training_data = load_files(container_path, description=None,  load_content=True,
                              shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    filter_data = []
    for index in range(0, len(training_data.data)) :
        if(training_data.target[index] == truthful):
            #print('Target: ', training_data.target[index], 'Content: ', training_data.data[index])
            filter_data.append(training_data.data[index])

    return filter_data

def topic_word_distribution(topic_word_list, container_path):
    training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    filter_data = []
    for each_word in topic_word_list:
        count_T = 0
        count_F = 0
        for index in range(0, len(training_data.data)):
            # print('Target: ', training_data.target[index], 'Content: ', training_data.data[index])
            str_data = training_data.data[index]
            if str_data.find(each_word) != -1 :
                if (training_data.target[index] == True):
                    count_T += 1
                else:
                    count_F += 1

        print(each_word, " appears in Truthful review : ", count_T, " and Deceptive review : ", count_F)

def get_data_sentence_containing_topic_model_words(container_path, topic_word_list, data):
    training_data = data
    if (training_data == None):
        training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)
    filter_data = []
    #print(training_data.data[2])
    for index in range(0, len(training_data.data)):
        #print("Before: ", training_data.data[index])
        document = sent_tokenize(training_data.data[index])
        new_document = ""
        for sentence in document:
            word_list = word_tokenize(sentence.lower())
            for word in topic_word_list:
                if word.lower() in word_list:
                    new_document += sentence+" "
                    break

        training_data.data[index] = new_document
        #print("After: ", new_document)

    #print(training_data.data[2])
    return training_data

def get_lemmatize_data_set(container_path):
    training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    lemmatizer = WordNetLemmatizer()

    for index in range(0, len(training_data.data)):
        # print("Before: ", training_data.data[index])
        #document = word_tokenize(training_data.data[index])
        #print(training_data.data[index])
        word_list = word_tokenize(training_data.data[index])
        new_document = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list])
       # print(new_document)

        training_data.data[index] = new_document

    return training_data

def print_top_words(container_path, model, feature_names, n_top_words):
    full_str = ""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        full_str += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        full_str += " "
        str_topic_word = " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        #topic_word_list = str_topic_word.split()
        #topic_word_distribution(topic_word_list, container_path)
        print(message)

    print()
    full_str = full_str.split()
    return full_str


def Remove(duplicate, fullRemove):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
        else:
            final_list.remove(num)
            if fullRemove == False:
                final_list.append(num)
    return final_list

def lemmatize_word_list(wordList):
    lemmatize_list = []
    lemmatizer = WordNetLemmatizer()
    for word in wordList:
        lemmatize_list.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

    return lemmatize_list


def get_top_words_from_topic_modeling(container_path, n_topic, n_top_words, data):
    training_data = data
    if(training_data == None):
        training_data = load_files(container_path, description=None, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf_review = tf_vectorizer.fit_transform(training_data.data)

    lda_review = LatentDirichletAllocation(n_components=n_topic, max_iter=20,
                                               learning_method='online',
                                               learning_offset=50.,
                                               random_state=0)
    lda_review.fit(tf_review)
    tf_feature_names = tf_vectorizer.get_feature_names()

    return print_top_words(container_path, lda_review, tf_feature_names, n_top_words)

def get_top_words_from_topic_modeling_(data, n_topic, n_top_words):

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf_review = tf_vectorizer.fit_transform(data)

    lda_review = LatentDirichletAllocation(n_components=n_topic, max_iter=20,
                                               learning_method='online',
                                               learning_offset=50.,
                                               random_state=0)
    lda_review.fit(tf_review)
    tf_feature_names = tf_vectorizer.get_feature_names()

    return print_top_words("", lda_review, tf_feature_names, n_top_words)


print("Loading dataset...")

container_path_neg = "../data/negative_polarity/"
container_path_pos = "../data/positive_polarity/"
container_path_comb = "../data/combined/"

container_path_temp = "../data/amazon/temp/"


categories = ['deceptive_from_MTurk', 'truthful_from_Web']

n_components = 2
n_top_words = 1000


data_path = container_path_neg

data = None#get_unbalance_data_set(container_path=data_path, truthful_percentage=1, deceptive_percentage=0.3)
#data = get_lemmatize_data_set(data_path)
top_word_list = get_top_words_from_topic_modeling(data_path,n_components,n_top_words, data)
#top_word_list = lemmatize_word_list(top_word_list)
top_word_list = Remove(top_word_list, False)

print("Total top word list:", len(top_word_list))


data = get_data_sentence_containing_topic_model_words(data_path,top_word_list, data)

vocabularyList = top_word_list
#vocabularyList = None

n_fold = 5
# classifiy_review will return classification result
hidden_layer_node = 3
model =  None#MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_layer_node), random_state=1);
#param: container_path, categories, number_of_fold, vocabularyList = None, model =None,training_data=None
result = classification.classifiy_review(data_path, categories, n_fold, vocabularyList, model, data)
# precision, recall, f1_score of each categories
for k, value in result.items():
    if k != 'accuracy':
        print(k, ": ", categories[0], " = ", value[0], ",", categories[1], " = ", value[1])
    else:
        print('Accuracy : ', result[k])

