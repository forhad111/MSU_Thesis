from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer
import classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk import sent_tokenize, word_tokenize
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

def classifiy_review(container_path, categories, number_of_fold, vocabularyList = None, model =None,training_data=None):
    '''
        This function takes data set path, categories and fold value.
        It returns precision, recall, f1_score of each categories.

    '''
    #load data set from given directory path
    if(training_data == None):
        training_data = load_files(container_path, description=None, categories=categories, load_content=True,shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)


    print(training_data.target_names)

    #Attribute values for each tuple
    X = np.array(training_data.data)
    #Target output for each tuple
    Y = np.array(training_data.target)

    #KFold condition setting
    kf = KFold(n_splits=number_of_fold, random_state=None, shuffle=False)
    result ={}
    index = 1
    accuracy = 0
    #Split data into n parts. Use n-1 parts for training and remaining part use for tesing
    for train_index, test_index in kf.split(X):
        print("Fold ",index, " is started...")
        #Splitted training and testing attribute values
        X_train, X_test = X[train_index], X[test_index]
        #Output value of traing and testing attribute values
        y_train, y_test = Y[train_index], Y[test_index]

        n_components = 2
        n_top_words = 1000

        top_word_list_truth = get_top_words_from_topic_modeling(X_train, y_train, n_components, n_top_words, True)
        top_word_list_truth = Remove(top_word_list_truth)

        top_word_list_deceptive = get_top_words_from_topic_modeling(X_train, y_train, n_components, n_top_words, False)
        top_word_list_deceptive = Remove(top_word_list_deceptive)

        output = []
        for i in range(len(X_test)):
            output.append(isTruthFul_Review(top_word_list_truth, top_word_list_deceptive,X_test[i]))

        predicted = np.array(output)
        accuracy += np.mean(predicted == y_test)
        result[index] = precision_recall_fscore_support(y_test, predicted)
        index = index + 1

        print(metrics.classification_report(y_test, predicted, target_names=training_data.target_names))

    precision = 0
    recall = 0
    f1_score = 0
    #Adds up each fold precesion, recallm f1_score
    for key, value in result.items():
        #print(key, " = ",  value)
        precision += value[0]
        recall += value[1]
        f1_score += value[2]

    avg_accuracy = accuracy/number_of_fold
    avg_precision = precision/number_of_fold
    avg_recall = recall/number_of_fold
    avg_f1_score = f1_score/number_of_fold

    calculated_result = {}
    calculated_result['accuracy'] = avg_accuracy
    calculated_result['precision'] = avg_precision
    calculated_result['recall'] = avg_recall
    calculated_result['f1_score'] = avg_f1_score
    #return mean of accuracy, precision, recall, f1_score
    return calculated_result

def isTruthFul_Review(top_truthful_list, top_deceptive_list, document):
    word_list = word_tokenize(document)
    count_truthful_words = 0
    count_deceptive_words = 0

    for word in word_list:
        if word in top_truthful_list:
            count_truthful_words = count_truthful_words + 1
        if word in top_deceptive_list:
            count_deceptive_words = count_deceptive_words + 1

    return count_truthful_words > count_deceptive_words


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

def get_data_sentence_containing_topic_model_words(container_path, topic_word_list):
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


def get_top_words_from_topic_modeling(X, Y, n_topic, n_top_words, truthful):

    training_data = []

    for index in range(len(X)):
        if Y[index] == truthful:
            training_data.append(X[index])

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf_review = tf_vectorizer.fit_transform(training_data)

    lda_review = LatentDirichletAllocation(n_components=n_topic, max_iter=20,
                                               learning_method='online',
                                               learning_offset=50.,
                                               random_state=0)
    lda_review.fit(tf_review)
    tf_feature_names = tf_vectorizer.get_feature_names()

    return print_top_words(lda_review, tf_feature_names, n_top_words)



print("Loading dataset...")

container_path_neg = "../data/negative_polarity/"
container_path_pos = "../data/positive_polarity/"
container_path_comb = "../data/combined/"


categories = ['deceptive_from_MTurk', 'truthful_from_Web']

n_components = 4
n_top_words = 150

data_path = container_path_pos


n_fold = 5
# classifiy_review will return classification result
hidden_layer_node = 3
model =  None#MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_layer_node), random_state=1);
result = classifiy_review(data_path, categories, n_fold, None, model, None)
# precision, recall, f1_score of each categories
for k, value in result.items():
    if k != 'accuracy':
        print(k, ": ", categories[0], " = ", value[0], ",", categories[1], " = ", value[1])
    else:
        print('Accuracy : ', result[k])

