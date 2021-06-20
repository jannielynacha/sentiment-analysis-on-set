import nltk
from nltk.metrics import scores
import random
import collections
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

from nltk.tokenize import word_tokenize

from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

all_words = []

all_words_f = open("pickled_data/allwords.pickle","rb")
all_words = pickle.load(all_words_f)
all_words_f.close()

wordfeatures_f = open("pickled_data/wordfeatures.pickle", "rb")
word_features = pickle.load(wordfeatures_f)
wordfeatures_f.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets_f = open("pickled_data/featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

print("Size of featuresets: ", len(featuresets))
training_set = featuresets[:3600] 		#training set
testing_set = featuresets[3600:4000]	#testing set

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

def testing(sent_classifier):

	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	 
	for i, (feats, category) in enumerate(testing_set):
	    refsets[category].add(i)
	    observed = sent_classifier.classify(feats)
	    testsets[observed].add(i)

	print ('Classifier Accuracy: ', (nltk.classify.accuracy(sent_classifier, testing_set))*100, "%")
	print ('Classifier pos Precision:', scores.precision(refsets['pos'], testsets['pos'])*100, "%")
	print ('Classifier pos Recall:', scores.recall(refsets['pos'], testsets['pos'])*100, "%")
	print ('Classifier pos F-measure:', scores.f_measure(refsets['pos'], testsets['pos'])*100, "%")
	print ('Classifier neg Precision:', scores.precision(refsets['neg'], testsets['neg'])*100, "%")
	print ('Classifier neg Recall:', scores.recall(refsets['neg'], testsets['neg'])*100, "%")
	print ('Classifier neg F-measure:', scores.f_measure(refsets['neg'], testsets['neg'])*100, "%")
	print ('\n')


# NAIVE BAYES CLASSIFIER
print("Loading Naive Bayes Classifier...")
classifier_f = open("pickled_algorithms10/naivebayes.pickle","rb")
NB_classifier = pickle.load(classifier_f)
classifier_f.close()

# Show Most Informative Word Features
NB_classifier.show_most_informative_features(30)

testing(NB_classifier)

# MULTINOMIAL NB CLASSIFIER
print("Loading Multinomial Naive Bayes Classifier...")
classifier_f = open("pickled_algorithms10/multinomial_naivebayes.pickle","rb") #open it to read (the bytes)
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()
testing(MNB_classifier)

# LOGISTIC REGRESSION CLASSIFIER
print("Loading Logistic Regression Classifier...")
classifier_f = open("pickled_algorithms10/logistic_regression.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()
testing(LogisticRegression_classifier)

# SVC
print("Loading SVM Classifier...")
classifier_f = open("pickled_algorithms10/svc.pickle","rb")
SVC_classifier = pickle.load(classifier_f)
classifier_f.close()
testing(SVC_classifier)

# LINEAR SVC
print("Loading Linear SVM Classifier...")
classifier_f = open("pickled_algorithms10/linear_svc.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()
testing(LinearSVC_classifier)