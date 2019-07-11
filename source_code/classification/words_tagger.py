from sklearn.datasets import load_files
from nltk.corpus import brown
import nltk
import pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
import string, re

regex = re.compile('[%s]' % re.escape(string.punctuation))
stop_words = set(stopwords.words('english'))

def test_re(s):
    return regex.sub('', s)

def words_tag_freq_calculation(container_path):

    #load data set from given directory path
    training_data = load_files(container_path, description=None,  load_content=True,
                              shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    print(training_data.target_names)
    return training_data.data

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


def words_tag_freq_calculation_by_each_document(container_path, truthful):

    #load data set from given directory path
    training_data = load_files(container_path, description=None,  load_content=True,
                              shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

    filter_data = []
    result = {'NOUN':0,'VERB':0,'.':0,'PRON':0,'ADJ':0,'ADV':0,'DET':0, 'NUM':0,'ADP':0,'CONJ':0,'PRT':0,'X':0 }
    count = 0
    for index in range(0, len(training_data.data)) :
        if(training_data.target[index] == truthful):
            #print('Target: ', training_data.target[index], 'Content: ', training_data.data[index])
            count = count +1
            text = training_data.data[index]
            text = word_tokenize(text)
            filtered_string = [w for w in text if not w in stop_words]

            total_filtered_word = len(filtered_string)

            text = nltk.pos_tag(filtered_string, tagset='universal')
            tag_fd = nltk.FreqDist(tag for (word, tag) in text)

            for most_feq_word in tag_fd.most_common():
                #print(most_feq_word[0], '===>', round(most_feq_word[1] / total_filtered_word, 2))
                result[most_feq_word[0]] = result[most_feq_word[0]] + (most_feq_word[1] / total_filtered_word)

            filter_data.append(training_data.data[index])

    for key, value in result.items():
        print(key,"===>",round(value/count, 2))

    return filter_data


if __name__ == '__main__':
    container_path_neg = "../data/negative_polarity/"
    container_path_pos = "../data/positive_polarity/"
    container_path_comb = "../data/combined/"

    print('Negative deceptive review frequency analysis considering individual document')
    neg_deceptive_ind = words_tag_freq_calculation_by_each_document(container_path_neg, False)

    print('Negative truthful review frequency analysis considering individual document')
    neg_truthful_ind = words_tag_freq_calculation_by_each_document(container_path_neg,True)

    print('Positive deceptive review frequency analysis considering individual document')
    pos_deceptive_ind = words_tag_freq_calculation_by_each_document(container_path_pos, False)

    print('Positive deceptive review frequency analysis considering individual document')
    pos_truthful_ind = words_tag_freq_calculation_by_each_document(container_path_pos, True)

    neg_truthful = words_tag_freq_calculation(container_path_neg,True)
    neg_deceptive = words_tag_freq_calculation(container_path_neg,False)

    pos_truthful = words_tag_freq_calculation(container_path_pos,True)
    pos_deceptive = words_tag_freq_calculation(container_path_pos, False)

    #result = words_tag_freq_calculation(container_path_neg)
    #result = neg_truthful + pos_truthful
    result = neg_deceptive + pos_deceptive
    stop_words = set(stopwords.words('english'))

    full_data_set = ""
    for data in result:
        data = test_re(data)
        #data = data.translate("#.,")
        #print(data)
        full_data_set += data


    text = word_tokenize(full_data_set)
    #pprint.pprint(nltk.pos_tag(text))
    print('Total number of words: ', len(text))

    filtered_string = [w for w in text if not w in stop_words]
    total_filtered_word = len(filtered_string)
    print('Total number of words after removing stopwords:',total_filtered_word)

    text = nltk.pos_tag(filtered_string, tagset='universal')
    tag_fd = nltk.FreqDist(tag for (word, tag) in text)
    pprint.pprint(tag_fd.most_common())
    print('Fraction of each parts of speech compared to total words.')
    for most_feq_word in tag_fd.most_common():
        print(most_feq_word[0],'===>',round(most_feq_word[1]/total_filtered_word,2))