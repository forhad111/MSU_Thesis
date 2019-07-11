import classification


if __name__ == '__main__':
    #class categories
    categories = ['deceptive_from_MTurk', 'truthful_from_Web']
    #categories = ['deceptive_from_MTurk', 'truthful_from_TripAdvisor']
    #data set path
    container_path_neg = "../data/negative_polarity/"
    container_path_pos = "../data/positive_polarity/"
    container_path_comb = "../data/combined/"
    #number of fold wants to use for classification
    n_fold = 5
    #classifiy_review will return classification result
    result = classification.classifiy_review(container_path_pos, categories, n_fold, None)
    #precision, recall, f1_score of each categories
    for k, value in result.items():
        if k != 'accuracy':
            print(k,": ",categories[0]," = ", value[0],",",categories[1], " = ",value[1])
        else:
            print ('Accuracy : ', result[k])

    #for heldout
    #print('Train with positive review and test with negative review')
    #classification.classify_review_using_heldout(container_path_pos,categories,container_path_neg)

    #print('Train with negative review and test with positive review')
    #classification.classify_review_using_heldout(container_path_neg, categories, container_path_pos)


