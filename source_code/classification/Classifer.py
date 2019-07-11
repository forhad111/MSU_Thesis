
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
import pprint
import operator


def classifiy_review(container_path, categories, model, number_of_fold, n_gram, stop_words):
    '''
        This function takes data set path, categories and fold value.
        It returns precision, recall, f1_score of each categories.

    '''

    #load data set from given directory path
    training_data = load_files(container_path, description=None, categories=categories, load_content=True,
                              shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    #print(training_data.target_names)

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
        #print("Fold ",index, " is started...")
        #Splitted training and testing attribute values
        X_train, X_test = X[train_index], X[test_index]
        #Output value of traing and testing attribute values
        y_train, y_test = Y[train_index], Y[test_index]

        '''
            vectorize condition for traing data. CountVectorizer supports counts of N-grams of words or
            consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices. 
            Here we use both uni-gram and bi-gram.
        '''
        #count_vect = CountVectorizer(ngram_range=(1,n_gram))
        vec = TfidfVectorizer(ngram_range=(1, n_gram), stop_words=stop_words)
        #X = vec.fit_transform(training_data.data)
        '''
            Text preprocessing, tokenizing and filtering of stopwords are included in a
            high level component that is able to build a dictionary of features and transform documents to feature vectors:
        '''
        #X_train_counts = vec.fit_transform(X_train)

        '''
            Giving longer and shorter documents similar priority and  downscale weights.
            This downscaling is called tf–idf.
        '''
        #tfidf_transformer = TfidfTransformer()
        X_train_tfidf = vec.fit_transform(X_train)

        '''
            Building a classifier using linear SVM with stochastic gradient descent (SGD) learning
        '''
        clf = model.fit(X_train_tfidf, y_train)

        #Test data set transformation for testing
        #X_test_counts = vec.transform(X_test)
        docs_test = vec.transform(X_test)

        #Prediction result using test data set
        predicted = clf.predict(docs_test)

        accuracy += np.mean(predicted == y_test)
        #print(f1_score(y_test, predicted))
        result[index] = precision_recall_fscore_support(y_test, predicted)

        '''
        print("                                 P                       R                       F1")
        print('')
        print(categories[0],'   ',result[index][0][0],'   ',result[index][1][0],'   ',result[index][2][0])
        print(categories[1], '      ', result[index][0][1], '   ', result[index][1][1], '   ', result[index][2][1])
        print('')
        '''

        index = index +1
        #print(metrics.classification_report(y_test, predicted,target_names=training_data.target_names))
        #print("Confusion Matrix : ")
        #print(metrics.confusion_matrix(test_data.target, predicted))

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




def classify_review_using_heldout(container_path, categories, container_path_test, model, ngram, stopwords):
    # load data set from given directory path
    training_data = load_files(container_path, description=None, categories=categories, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)
    print(training_data.target_names)

    vec = TfidfVectorizer(ngram_range=(1, n_gram), stop_words=stop_words)
    X_train_tfidf = vec.fit_transform(training_data.data)

    clf = model.fit(X_train_tfidf, training_data.target)

    test_data = load_files(container_path_test, description=None, categories=categories, load_content=True,
                           shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    docs_test = vec.transform(test_data.data)

    predicted = clf.predict(docs_test)

    #print('Accuracy : ',np.mean(predicted == test_data.target))
    #print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
    #print("Confusion Matrix : ")
    #print(metrics.confusion_matrix(test_data.target, predicted))

    result = precision_recall_fscore_support(test_data.target, predicted)

    calculated_result = {}
    calculated_result['accuracy'] = np.mean(predicted == test_data.target)
    calculated_result['precision'] = result[0]
    calculated_result['recall'] = result[1]
    calculated_result['f1_score'] = result[2]
    #print("P: ",result[0])
    # return mean of accuracy, precision, recall, f1_score
    return calculated_result


categories = ['deceptive_from_MTurk', 'truthful_from_Web']

container_path_neg = "../data/negative_polarity/"
container_path_pos = "../data/positive_polarity/"
container_path_comb = "../data/combined/"

stop_words = set(stopwords.words('english'))
#print(stop_words)
#training_data = load_files(container_path_neg, description=None, categories=categories, load_content=True, shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)
#print(training_data.target_names)
''' 
print(training_data.target_names)
count_vect = CountVectorizer(ngram_range=(1, 2))
X_train_counts = count_vect.fit_transform(training_data.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



vec = TfidfVectorizer(ngram_range=(1,2), stop_words=stop_words)
X = vec.fit_transform(training_data.data)
#print(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, training_data.target, random_state =1)


#model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None);
model.fit(Xtrain, ytrain)
labels = model.predict(Xtest)
print(accuracy_score(labels, ytest))
'''




n_fold = 5
n_gram = 2
hidden_layer_node = 3
#model =  MultinomialNB()
best_accuracy = 0
hidden_layer_node_number = 0
all_result = {}
for hidden_layer_node in range(2,50):
    print('MLPClassifier\'s performance with hidden layer one and each hidden layer contains ',hidden_layer_node,' nodes')
    model =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_layer_node), random_state=1);

    # classifiy_review will return classification result
    result = classifiy_review(container_path_comb, categories, model, n_fold, n_gram, stop_words)
    #result = classify_review_using_heldout(container_path_neg,categories,container_path_pos,model,n_gram,stop_words)
    # precision, recall, f1_score of each categories

    if(result['accuracy']> best_accuracy):
        best_accuracy = result['accuracy']
        hidden_layer_node_number = hidden_layer_node

    all_result[hidden_layer_node] = result['accuracy']

    print('Classification performance with 5-Fold cross validation: ')
    print('Accuracy : ', result['accuracy'])
    print('')
    print("                                 P                       R                       F1")
    print('')
    print(categories[0], '   ', result['precision'][0], '   ', result['recall'][0], '   ', result['f1_score'][0])
    print(categories[1], '      ', result['precision'][1], '   ', result['recall'][1], '   ', result['f1_score'][1])
    print('')


    '''
    for k, value in result.items():
        if k != 'accuracy':
            print(k, ": ", categories[0], " = ", value[0], ",", categories[1], " = ", value[1])
        else:
            print('Accuracy : ', result[k])
    '''

    ''' 
    print('Train with positive review and test with negative review')
    classify_review_using_heldout(container_path_pos,categories,container_path_neg, model, n_gram, stop_words)
    
    print('Train with negative review and test with positive review')
    classify_review_using_heldout(container_path_neg, categories, container_path_pos, model, n_gram, stop_words)
    '''

print('Best accuracy : ', best_accuracy, ' when hidden layer was one and number of node in hidden layer was ',hidden_layer_node_number)
all_result = sorted(all_result.items(), key=operator.itemgetter(1), reverse= True)
pprint.pprint(all_result)