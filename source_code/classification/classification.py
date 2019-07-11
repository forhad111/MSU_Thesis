from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB




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

        '''
            vectorize condition for training data. CountVectorizer supports counts of N-grams of words or
            consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices. 
            Here we use both uni-gram and bi-gram.
        '''
        #count_vect = CountVectorizer(ngram_range=(1,2))
        count_vect = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2), stop_words='english',vocabulary=vocabularyList)

        '''
            Text preprocessing, tokenizing and filtering of stopwords are included in a
            high level component that is able to build a dictionary of features and transform documents to feature vectors:
        '''
        X_train_counts = count_vect.fit_transform(X_train)


        '''
            Giving longer and shorter documents similar priority and  downscale weights.
            This downscaling is called tf–idf.
        '''
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        print(X_train_tfidf.shape)
        '''
            Building a classifier using linear SVM with stochastic gradient descent (SGD) learning
        '''
        if(model == None):
            clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None).fit(X_train_tfidf, y_train)
        else:
            clf = model.fit(X_train_tfidf, y_train)

        #Test data set transformation for testing
        X_test_counts = count_vect.transform(X_test)
        docs_test = tfidf_transformer.transform(X_test_counts)

        #Prediction result using test data set
        predicted = clf.predict(docs_test)

        accuracy += np.mean(predicted == y_test)
        #print(f1_score(y_test, predicted))
        result[index] = precision_recall_fscore_support(y_test, predicted)
        #print(result)
        index = index +1
        print(metrics.classification_report(y_test, predicted,target_names=training_data.target_names))
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

def classify_review_using_heldout(container_path, categories, container_path_test):
    # load data set from given directory path
    training_data = load_files(container_path, description=None, categories=categories, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)
    print(training_data.target_names)

    count_vect = CountVectorizer(ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(training_data.data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None).fit(
        X_train_tfidf, training_data.target)

    test_data = load_files(container_path_test, description=None, categories=categories, load_content=True,
                           shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    X_test_counts = count_vect.transform(test_data.data)
    docs_test = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(docs_test)
    print('Accuracy : ',np.mean(predicted == test_data.target))
    print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
    print("Confusion Matrix : ")
    print(metrics.confusion_matrix(test_data.target, predicted))

def classify_review_using_NB(container_path,categories):
    training_data = load_files(container_path, description=None, categories=categories, load_content=True,
                               shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)
    print(training_data.target_names)

    count_vect = CountVectorizer(ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(training_data.data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)