
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
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pprint
import operator
from sklearn.utils import shuffle


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


def probability_of_deceptive_truthful(review, wordDict, maxNumberOfBacket):
    truthful_count = 0
    deceptive_count = 0
    neutral = 0
    for list_Info in wordDict:
        start_index = 0
        #list_Info = str(list_Info)
        while review.find(list_Info[0], start_index) >= 0:
            start_index = review.find(list_Info[0], start_index) + 1
            if(list_Info[1] >= maxNumberOfBacket-1):
                truthful_count += 1
            elif(list_Info[1] <2):
                deceptive_count += 1
            #else:
                #neutral += 1
    #print("Review : "+ str(review) + ", maxNumberOfBacket : "+ str(maxNumberOfBacket))
    #print("Total truthful count : "+ str(truthful_count), " Deceptive count : "+ str(deceptive_count) + ", Neutral count : "+ str(neutral))
    if(truthful_count + deceptive_count <= 0):
        return -1

    #print("Probability of truthful : " +str( (truthful_count/(truthful_count+ deceptive_count))))
    return truthful_count/(truthful_count+ deceptive_count)

def create_data_set(truthful_path, deceptive_path, df, threshold, max_limit):

    #max_limit = len(df.index)/2
    truthful_count = 0
    deceptive_count = 0
    print("Total len : " + str(df.index))
    for i in range(len(df.index)):
        helpful_votes = 0
        body_val = str(df.at[i,"Body"]).split(" ")
        #if(len(body_val) < 25):
         #   continue
        helpful_votes = int((df.at[i, 'Helpful Votes']).replace(",", ""))
        #helpful_votes = int(df.at[i, 'Helpful Votes'])

        if(helpful_votes >= threshold):
            if(truthful_count > max_limit):
                continue
            write_to_file(truthful_path+"/t_"+str(truthful_count)+".txt", df.at[i,"Body"])
            truthful_count = truthful_count + 1
        elif (helpful_votes <= 0) :
            if(deceptive_count > max_limit):
                continue
            write_to_file(deceptive_path + "/t_" + str(deceptive_count)+".txt", df.at[i, "Body"])
            deceptive_count = deceptive_count + 1


    print("Total truthful count : "+ str(truthful_count))
    print("Total deceptive count : " + str(deceptive_count))

def write_to_file(data_path, data):
    f = open(data_path, "w");
    f.write(str(data))
    f.close()

input_file = "../data/amazon/iPhone6.csv"

# comma delimited is the default
df = pd.read_csv(input_file, header = 0)

# for space delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = " ")

# for tab delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = "\t")

# put the original column names in a python list
original_headers = list(df.columns.values)
print(original_headers)

truthful_data_path = "../data/amazon/temp/truthful_from_Web"
deceptive_data_path = "../data/amazon/temp/deceptive_from_MTurk"

#create_data_set(truthful_data_path, deceptive_data_path, df, 1, 445)


df['Date'] = pd.to_datetime(df.Date)
df = df.sort_values(by = 'Date')
#df = shuffle(df)
#df = df[801:1000]
n_components = 2
n_top_words = 170
print(len(df))
word_list = []
word_freq = {}
each_chunck_size = (int)(len(df)/5)-1
for i in range(5):
    temp_df = df[(i*each_chunck_size)+1:(i+1)*each_chunck_size]
    topic_word_list = get_top_words_from_topic_modeling_(temp_df['Body'].values.astype('U'),n_components,n_top_words)
    topic_word_list = set(topic_word_list)
    topic_word_list = list(topic_word_list)
    for word in topic_word_list:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] = word_freq[word] +1
    word_list += topic_word_list
counts = Counter(word_list)

#print(counts)

sorted_x = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)

maxVal = 0
for k in sorted_x:
    maxVal = k[1]
    break

print("Maxval : " + str(maxVal))
#df = df["Body"].values.astype('U')
#print(df['Body'][0])
#print(df['Body'][1])
truthful_labeled = 0
deceptive_labeled = 0
neutral_labeled = 0
correctly_Tructhful_detected = 0
correctly_Deceptive_detected = 0
False_Negative = 0
False_Positive = 0
for i in range (each_chunck_size* 5):
    #print(df['Body'][i])
    x = probability_of_deceptive_truthful(str((df["Body"][i])), sorted_x, maxVal)
    #helpful_vote = int((df["Helpful Votes"][i]).replace(",", ""))
    helpful_vote = int((df["Helpful Votes"][i]))
    if(helpful_vote >= 3): #x >= 0.65):
        truthful_labeled += 1
        #print("Labeled truthful and received helpful votes : "+str(df["Helpful Votes"][i]))

        #if(helpful_vote >= 3):
        if(x >= 0.65):
           correctly_Tructhful_detected += 1
        else :
            False_Negative += 1
    #elif(x <= 0.35):
    elif(helpful_vote <= 0):
        deceptive_labeled += 1
        #print(df["Helpful Votes"][i])
        #if(helpful_vote >= 3):
        if(x <= 0.35):
           correctly_Deceptive_detected += 1
        else:
            False_Positive += 1
    else:
        neutral_labeled += 1

True_Positive = correctly_Tructhful_detected
True_Negative = correctly_Deceptive_detected
precision = True_Positive/(True_Positive+False_Positive)
recall = True_Positive / (True_Positive + False_Negative)
F1_score = 2 * (precision * recall)/(precision+recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1_Score: ", F1_score)

print("Truthful Labeled : "+ str(truthful_labeled))
print("Correctly Truthful Labeled : "+ str(correctly_Tructhful_detected))
print("Correctly Truthful Labeled in percentage : "+ str(correctly_Tructhful_detected/truthful_labeled))
print("Deceptive Labeled : "+ str(deceptive_labeled))
print("Correctly Deceptive Labeled : " + str(correctly_Deceptive_detected))
print("Correctly Deceptive Labeled in percentage : " + str(correctly_Deceptive_detected/deceptive_labeled))
print("Neutral Labeled : "+ str(neutral_labeled))
print("Truthful Labeled lower: "+ str(truthful_labeled/(truthful_labeled+deceptive_labeled+neutral_labeled)))
print("Truthful Labeled upper : "+ str(truthful_labeled/(truthful_labeled+deceptive_labeled)))



''' 
labels, values = zip(*counts.items())
# sort your values in descending order
indSort = np.argsort(values)[::-1]
# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

bar_width = 0.30
'''
#plt.bar(indexes, values)

# add labels
#plt.xlabel('x',rotation=180)
#plt.xticks(indexes + bar_width, labels)
#plt.show()


# remove the non-numeric columns
#df = df._get_numeric_data()

# put the numeric column names in a python list
#numeric_headers = list(df.columns.values)

# create a numpy array with the numeric values for input into scikit-learn
#numpy_array = df.as_matrix()

# reverse the order of the columns
#numeric_headers.reverse()
#reverse_df = df[numeric_headers]

# write the reverse_df to an excel spreadsheet
#reverse_df.to_excel('path_to_file.xls')
