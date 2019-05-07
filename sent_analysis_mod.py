import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode #MODE - How we're going to choose which category has the most votes

from nltk.tokenize import word_tokenize

from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


# FOR RETRIEVING THE NEW DATASET
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

documents = []
all_words = []

documents_f = open("pickled_algorithms/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()	

all_words_f = open("pickled_algorithms/allwords.pickle","rb")
all_words = pickle.load(all_words_f)
all_words_f.close()

wordfeatures_f = open("pickled_algorithms/wordfeatures.pickle", "rb")
word_features = pickle.load(wordfeatures_f)
wordfeatures_f.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets_f = open("pickled_algorithms/featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

training_set = featuresets[:10000] #training set is first 10000
testing_set = featuresets[10000:] #testing set is the rest

# posterior = (prior occurrences*likelihood)/evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("pickled_algorithms/naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

# MULTINOMIAL NB CLASSIFIER
classifier_f = open("pickled_algorithms/multinomial_naivebayes.pickle","rb") #open it to read (the bytes)
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()
print("MNB_classifier Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# BERNOULLI NB CLASSIFIER
classifier_f = open("pickled_algorithms/bernoulli_naivebayes.pickle","rb") #open it to read (the bytes)
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()
print("BernoulliNB_classifier Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# LogisticRegression CLASSIFIER
classifier_f = open("pickled_algorithms/logistic_regression.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()
print("LogisticRegression_classifier Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDC CLASSIFIER
classifier_f = open("pickled_algorithms/sgd.pickle","rb")
SGDClassifier_classifier = pickle.load(classifier_f)
classifier_f.close()
print("SGDClassifier_classifier Accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# SVC
classifier_f = open("pickled_algorithms/svc.pickle","rb")
SVC_classifier = pickle.load(classifier_f)
classifier_f.close()
print("SVC_classifier Accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LINEAR SVC
classifier_f = open("pickled_algorithms/linear_svc.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()
print("LinearSVC_classifier Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC CLASSIFIER
classifier_f = open("pickled_algorithms/nu_svc.pickle","rb")
NuSVC_classifier = pickle.load(classifier_f)
classifier_f.close()
print("NuSVC_classifier Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


#NEW CLASSIFIER WITH VOTING CAPABILITIES
voted_classifier = VoteClassifier(classifier, 
									MNB_classifier, 
									BernoulliNB_classifier, 
									LogisticRegression_classifier, 
									SGDClassifier_classifier,
									SVC_classifier,
									LinearSVC_classifier, 
									NuSVC_classifier)

print("voted_classifier Accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
	feats = find_features(text)

	return voted_classifier.classify(feats)