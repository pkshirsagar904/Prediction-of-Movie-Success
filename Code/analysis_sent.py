import matplotlib.pyplot as plt
import os
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Variables Initialization

Neg = []
Pos = []
vPos = []
vNeg = []
f_X = ""
f_Y = ""
test_pos=0
test_neg = 0
neut=0


"""
used to generate stopwords list
"""
def gSW():

    #get the Text File which has all the stopwords from the file
    get_stopWords = "Data/stopwords.txt"

    #list for stopwords
    SW = []

    #open the file of stopwords open it and store it in a list
    o = open(get_stopWords, 'r')
    s_line = o.readline()
    while s_line:
        SW.append(s_line.strip())
        s_line = o.readline()
    o.close()
    return SW



"""
This function is used to create a Dictionary of words according to the polarities
We have taken 4 Categories:
Very Positive Words, Positive Words, Negative Words, Very Negative Words
"""
def genDictFrmPolarity(a_list):
    # Create list to store the words and its score i.e. polarity
    w = []
    sc = []

    #   for word in affine_list, generate the Words with their scores (polarity)
    for i in a_list:
        w.append(i.split("\t")[0].lower())
        sc.append(int(i.split("\t")[1].split("\n")[0]))

    #   categorize words into different categories
    for j in range(len(w)):
        if sc[j] == 1 or sc[j] == 2 or sc[j] == 3:
            Pos.append(w[j])
        elif sc[j] == 4 or sc[j] == 5:
            vPos.append(w[j])
        elif sc[j] == -1 or sc[j] == -2 or sc[j] == -3:
            Neg.append(w[j])
        elif sc[j] == -4 or sc[j] == -5:
            vNeg.append(w[j])


"""
Here preprocessing of the data and dimensionality steps is done
returns processed_data LIST
"""
def data_pre(dataS):

    data_pre= []

    #create a list of all the Stopwords to be removed
    stopWords = gSW()
    for tweet in dataS:

        temp_tweet = tweet

        #Convert @username to USER_MENTION
        tweet = re.sub('@[^\s]+', 'USER_MENTION', tweet).lower()
        tweet.replace(temp_tweet , tweet)

        #Remove the unnecessary white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet, tweet)

        #Replace #HASTAG with only the word by removing the HASH (#) symbol
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        #Replace all the numeric terms
        tweet = re.sub('[0-9]+', "", tweet)
        tweet.replace(temp_tweet, tweet)

        #Remove all the STOP WORDS
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)

        tweet.replace(temp_tweet, tweet)

        #Replace all Punctuations
        tweet = re.sub('[^a-zA-z ]',"",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove additional white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Save the Processed Tweet after data cleansing
        data_pre.append(tweet)

    return data_pre

"""
Tbis function is used to create feature vector and assign class label accordingly
it returns Feature Vector
"""
def f_trainingData(dataset, type_class):

    neu = []
    k=0

    # split every word of the Tweet for each tweet
    data = [tweet.strip().split(" ") for tweet in dataset]

    # used to store feature of tweets
    f_vector = []
    train_pos = 0
    train_neg = 0
    # for every tweet find words and their category
    for s in data:
        # Category count for every Sentence or TWEET
        vNeg_count = 0
        Neg_count = 0
        Pos_count = 0
        vPos_count = 0


        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in s:
            if word in Neg:
                Neg_count = Neg_count + 1
            elif word in vNeg:
                vNeg_count = vNeg_count + 1
            elif word in Pos:
                Pos_count = Pos_count + 1
            elif word in vPos:
                vPos_count = vPos_count + 1
        k+=1

        #Assign Class Label
        if Neg_count == vNeg_count == Pos_count == vPos_count:
            f_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "neutral"])
            neu.append(k)
        else:
            if type_class == "positive":
                train_pos = train_pos + 1
            else:
                train_neg = train_neg + 1
            f_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, type_class])

    #print(neutral_list)
    return f_vector, train_pos, train_neg

"""
here we get the Feature Vectors for the Test Data
"""
def f_testData(datas):
    global neut,test_pos,test_neg
    data = [tweet.strip().split(" ") for tweet in datas]

    f_vector = []


    for s in data:
        vNeg_count = 0
        Neg_count = 0
        Pos_count = 0
        vPos_count = 0


        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for i in s:
            if i in Pos:
                Pos_count = Pos_count + 1
            elif i in vPos:
                vPos_count = vPos_count + 1
            elif i in Neg:
                Neg_count = Neg_count + 1
            elif i in vNeg:
                vNeg_count = vNeg_count + 1


        if (vPos_count + Pos_count) < (vNeg_count + Neg_count):
            f_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "negative"])
            test_neg = test_neg+1

        elif (vPos_count + Pos_count) > (vNeg_count + Neg_count):
            f_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "positive"])
            test_pos = test_pos + 1

        else:
            f_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "neutral"])
            neut = neut+1
    return (f_vector, test_pos, test_neg, neut)

"""
This function is used to classify data using Naive Bayes
"""
def classify_gnb(train_X, train_Y, test_X):

    print("Classifying using Gaussian Naive Bayes ...")
    return GaussianNB().fit(train_X,train_Y).predict(test_X)


"""
It is used to classify data using SVM
"""
def classify_svm(train_X, train_Y, test_X):

    print("Now we Classify using SVM")

    #clf = SVC()
    SVC().fit(train_X,train_Y)

    return SVC().predict(test_X)

def classify_gnb_twitter(train_X, train_Y, test_X, test_Y):

    print("Now We Classify using Gaussian Naive Bayes")
    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)

    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = float(sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuray: ", Accuracy)
    evaluate_classifier(conf_mat, Accuracy)



def classify_svm_twitter(train_X, train_Y, test_X, test_Y):

   print("We are Classifying SVM")
   clf = SVC()
   clf.fit(train_X, train_Y)
   yHat = clf.predict(test_X)
   print(confusion_matrix(test_Y, yHat))
   Accuracy = float(sum(confusion_matrix(test_Y, yHat).diagonal())) / np.sum(confusion_matrix(test_Y, yHat))
   print("Accuracy: ", Accuracy)
   evaluate_classifier(confusion_matrix(test_Y, yHat), Accuracy)



"""
 This function is used to classify tweets based on algorithm to classify
"""
def classify_twitter_data(file_name):


    test_data = data_pre(open(dirPath+"/Data/"+file_name).readlines())
    test_data,test_pos,test_neg,neu = f_testData(test_data)
    test_data = np.reshape(np.asarray(test_data),newshape=(len(test_data),5))
    print("Positive tweets:",test_pos)
    print("Negative tweets:", test_neg)
    #print(neu)

    #Split the Data into features and classes
    f_X_test = test_data[:,:4].astype(int)
    f_Y_test = test_data[:,4]

    print("Classifying", file_name)

    classify_svm_twitter(f_X, f_Y, f_X_test, f_Y_test)
    classify_gnb_twitter(f_X, f_Y, f_X, f_Y)

    objects1 = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    y_pos1 = np.arange(len(objects1))
    plt.ylabel('COUNT')
    per1 = [test_pos, test_neg, neu]
    plt.bar(y_pos1, per1, align='center', alpha=0.5)
    plt.xticks(y_pos1, objects1)
    plt.show()



"""
It is used to evaluate the classifier's performance. 
Also calculate Precision, Recall, F-measure and Accuracy
"""
def evaluate_classifier(conf_mat,Accuracy):
    Precision = conf_mat[0,0]/float(sum(conf_mat[0]))
    Recall = conf_mat[0,0] / float(sum(conf_mat[:,0]))
    F_Measure = (2 * (Precision * Recall))/ (Precision + Recall)

    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F-Measure: ", F_Measure)
    objects = ["ACCURACY", "PRECISION", "RECALL", "F-MEASURE"]
    y_pos=np.arange(len(objects))
    per=[Accuracy*100,Precision*100,Recall*100,F_Measure*100]
    plt.bar(y_pos,per,align='center',alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel('VALUE')
    plt.show()




if __name__ == "__main__":
    #get the current directory
    os.chdir('../')
    dirPath = os.getcwd()
    #print(dirPath)

    # STEP 1: here affinity list is generated
    print("We start to classify the data")

    #used to generate lexicon of sentiment
    a_list = open(dirPath+"/Data/Affin_Data.txt").readlines()

    # STEP 2: here dictionary is created from lexicons
    genDictFrmPolarity(a_list)

    # STEP 3: read positive and negative tweets and preprocessing is done
    print("We Read the data")
    positive_data = open(dirPath+"/Data/rt-polarity-pos.txt").readlines()
    print("We wait for Preprocessing")
    positive_data = data_pre(positive_data)
    #print(positive_data)

    negative_data = open(dirPath+"/Data/rt-polarity-neg.txt").readlines()
    negative_data = data_pre(negative_data)
    #print(negative_data)

    # STEP 4: feature vector is created and class label is assigned for training data
    print("Now Feature Vector are generated")
    positive_sentiment, q, w = f_trainingData(positive_data, "positive")
   # print("Number of Positive Words: ",len(positive_sentiment))
    negative_sentiment, e, r = f_trainingData(negative_data, "negative")
    #print("Number of Negative Words: ",len(negative_sentiment))
    final_data = positive_sentiment + negative_sentiment
   # print("Positive tweets tr:", q+e)
    #print("Negative tweets tr:", w+r)
    final_data = np.reshape(np.asarray(final_data),newshape=(len(final_data),5))

    #data is split into features and classes
    f_X = final_data[:,:4].astype(int)
    f_Y = final_data[:,4]



    #entire dataset is classified
    print("Now we Train the classfier as per the data")
    print("Classification of Test Data in progress")
    print("We Wait for Evaluation Results to be displayed")
    file_name = "deadpool2.txt"
    classify_twitter_data(file_name)