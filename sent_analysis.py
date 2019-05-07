import nltk
import random
# from nltk.corpus import movie_reviews
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

# POS tagging
# allowed word types: adjective (J), adverb (R), verb (V)
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
	documents.append( (p, "pos") )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for p in short_neg.split('\n'):
	documents.append( (p, "neg") )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

save_documents = open("pickled_algorithms/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
# documents_f = open("documents.pickle","rb")
# documents = pickle.load(documents_f)
# documents_f.close()	

#convert to nltk frequency distribution
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["stupid"])

save_allwords = open("pickled_algorithms/allwords.pickle","wb")
pickle.dump(all_words, save_allwords)
save_allwords.close()
# all_words_f = open("allwords.pickle","rb")
# all_words = pickle.load(all_words_f)
# all_words_f.close()

word_features = list(all_words.keys())[:5000] #5000 words can be enough to encompass words commonly used
# and then we can train against these top 5000 words and find out which words are most common and negative
# and which words are most common and positive
save_wordfeatures = open("pickled_algorithms/wordfeatures.pickle","wb")
pickle.dump(word_features, save_wordfeatures)
save_wordfeatures.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

save_featuresets = open("pickled_algorithms/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()
# featuresets_f = open("featuresets.pickle","rb")
# featuresets = pickle.load(featuresets_f)
# featuresets_f.close()

training_set = featuresets[:10000] #training set is first 10000
testing_set = featuresets[10000:] #testing set is the rest

# posterior = (prior occurrences*likelihood)/evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)

save_classifier = open("pickled_algorithms/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
# classifier_f = open("naivebayes.pickle","rb") #open it to read (the bytes)
# classifier = pickle.load(classifier_f)
# classifier_f.close()

print("Original Naive Bayes Accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

# MULTINOMIAL NB CLASSIFIER
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/multinomial_naivebayes.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("multinomial_naivebayes.pickle","rb") #open it to read (the bytes)
# MNB_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("MNB_classifier Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# BERNOULLI NB CLASSIFIER
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/bernoulli_naivebayes.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("bernoulli_naivebayes.pickle","rb") #open it to read (the bytes)
# BernoulliNB_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("BernoulliNB_classifier Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# LogisticRegression CLASSIFIER
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/logistic_regression.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("logistic_regression.pickle","rb")
# LogisticRegression_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("LogisticRegression_classifier Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDC CLASSIFIER
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/sgd.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("pickled_algorithms/sgd.pickle","rb")
# SGDClassifier_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("SGDClassifier_classifier Accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier Accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/svc.pickle","wb")
pickle.dump(SVC_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("pickled_algorithms/svc.pickle","rb")
# SVC_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("SVC_classifier Accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LINEAR SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/linear_svc.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("linear_svc.pickle","rb")
# LinearSVC_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("LinearSVC_classifier Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC CLASSIFIER
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algorithms/nu_svc.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()
# classifier_f = open("nu_svc.pickle","rb")
# NuSVC_classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("NuSVC_classifier Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


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

print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0])*100)

def sentiment(text):
	feats = find_features(text)

	return voted_classifier.classify(feats)