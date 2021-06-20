import nltk
from nltk.metrics import scores
import collections
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

from nltk.tokenize import word_tokenize

from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

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

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)


# NAIVE BAYES CLASSIFIER
print("Loading Naive Bayes Classifier...")
classifier_f = open("pickled_algorithms/naivebayes.pickle","rb")
NB_classifier = pickle.load(classifier_f)
classifier_f.close()
# Show Top 30 Most Informative Features
NB_classifier.show_most_informative_features(30)

# MULTINOMIAL NB CLASSIFIER
print("Loading Multinomial Naive Bayes Classifier...")
classifier_f = open("pickled_algorithms/multinomial_naivebayes.pickle","rb") #open it to read (the bytes)
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()

# LOGISTIC REGRESSION CLASSIFIER
print("Loading Logistic Regression Classifier...")
classifier_f = open("pickled_algorithms/logistic_regression.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()

# SVC
print("Loading SVM Classifier...")
classifier_f = open("pickled_algorithms/svc.pickle","rb")
SVC_classifier = pickle.load(classifier_f)
classifier_f.close()

# LINEAR SVC
print("Loading Linear SVM Classifier...")
classifier_f = open("pickled_algorithms/linear_svc.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

# Functions for determining sentiment polarity using the classifiers
def NB_sentiment(text):
	feats = find_features(text)
	return NB_classifier.classify(feats)

def MNB_sentiment(text):
	feats = find_features(text)
	return MNB_classifier.classify(feats)

def LR_sentiment(text):
	feats = find_features(text)
	return LogisticRegression_classifier.classify(feats)

def SVM_sentiment(text):
	feats = find_features(text)
	return SVC_classifier.classify(feats)

def LSVM_sentiment(text):
	feats = find_features(text)
	return LinearSVC_classifier.classify(feats)