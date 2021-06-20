import nltk
import nltk.metrics
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

from nltk.tokenize import word_tokenize

from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

short_pos = open("set_data/pos_set.txt","r").read()
short_neg = open("set_data/neg_set.txt","r").read()

documents = []
all_words = []

# Load the documents
print("Loading Documents...")
for p in short_pos.split('\n'):
	documents.append( (p, "pos") )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		all_words.append(w[0].lower())

for p in short_neg.split('\n'):
	documents.append( (p, "neg") )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		all_words.append(w[0].lower())

# Pickle documents
save_documents = open("pickled_data/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

# Convert to nltk frequency distribution
print("Transforming all_words...")
all_words = nltk.FreqDist(all_words)

# Pickle all_words
save_allwords = open("pickled_data/allwords.pickle","wb")
pickle.dump(all_words, save_allwords)
save_allwords.close()

# Get top 3000 words as features
word_features = list(all_words.keys())[:3000]

# Pickle word features
save_wordfeatures = open("pickled_data/wordfeatures.pickle","wb")
pickle.dump(word_features, save_wordfeatures)
save_wordfeatures.close()

# Function that returns the word features that are in the document passed
def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

# Creates a list of features with their category (pos or neg)
print("Creating featuresets...")
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffle the featuresets
random.shuffle(featuresets)

# Pickle featuresets
save_featuresets = open("pickled_data/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

# dataset partition into training set and testing set
training_set = featuresets[400:]	#training set
testing_set = featuresets[:400]		#testing set

# Train the Naive Bayes Classifier from NLTK
print("Training the standard Naive Bayes model...")
classifier = nltk.NaiveBayesClassifier.train(training_set)

save_classifier = open("pickled_algorithms/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

classifier.show_most_informative_features(30)

# Train MULTINOMIAL NB CLASSIFIER
print("Training the Multinomial Naive Bayes model...")
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
# Pickle MULTINOMIAL NB CLASSIFIER
save_classifier = open("pickled_algorithms/multinomial_naivebayes.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# Train  LogisticRegression CLASSIFIER
print("Training Logistic Regression model...")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
# Pickle LogisticRegression CLASSIFIER
save_classifier = open("pickled_algorithms/logistic_regression.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

# Train  SVC
print("Training SVM model...")
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
# Pickle SVC
save_classifier = open("pickled_algorithms/svc.pickle","wb")
pickle.dump(SVC_classifier, save_classifier)
save_classifier.close()

# Train  LINEAR SVC
print("Training Linear SVM model...")
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
# Pickle LINEAR SVC
save_classifier = open("pickled_algorithms/linear_svc.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()