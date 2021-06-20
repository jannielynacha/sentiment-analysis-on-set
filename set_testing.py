import sent_analysis_mod as s

test_neg = open("set_data/neg_final_testing.txt","r").read()
test_pos = open("set_data/pos_final_testing.txt","r").read()

positive_documents = []
negative_documents = []

for p in test_pos.split('\n'):
	positive_documents.append(p)

for p in test_neg.split('\n'):
	negative_documents.append(p)

def test_classifier(sent_classifier, pos_docs, neg_docs):
	TP = [] # True Positives
	FP = [] # False Positives
	TN = [] # True Negatives
	FN = [] # False Negatives

	for s in pos_docs:
		if sent_classifier(s) == 'pos':
			TP.append(s) # true positive
		else:
			FN.append(s) # false negative

	for s in neg_docs:
		if sent_classifier(s) == 'neg':
			TN.append(s) # True Negative
		else:
			FP.append(s) # False Positive
	
	accuracy = ((len(TP)+len(TN))/(len(pos_docs)+len(neg_docs)))*100
	error = ((len(FP)+len(FN))/(len(pos_docs)+len(neg_docs)))*100
	positive_precision = ((len(TP))/(len(TP)+len(FP)))*100
	positive_recall = ((len(TP))/(len(TP)+len(FN)))*100
	positive_fscore = 2 * ((positive_precision*positive_recall)/(positive_precision+positive_recall))
	negative_precision = ((len(TN))/(len(TN)+len(FN)))*100
	negative_recall = ((len(TN))/(len(TN)+len(FP)))*100
	negative_fscore = 2 * ((negative_precision*negative_recall)/(negative_precision+negative_recall))

	print("Total Test Documents:\t", len(pos_docs)+len(neg_docs))
	print("True Positives (TP):\t", len(TP))
	print("True Negatives (TN):\t", len(TN))
	print("False Positives (FP):\t", len(FP))
	print("False Negatives (FN):\t", len(FN))
	print("Correctly classified:\t", len(TP)+len(TN))
	print("Incorrectly classified:\t", len(FP)+len(FN))
	print("Accuracy:\t\t", accuracy, " %")
	print("Error percentage:\t", error, " %")

	print("Positive Precision:\t", positive_precision, " %") 	# correct positive:predicted as positive
	print("Positive Recall:\t", positive_recall, " %")			# correct positive:actual positive
	print("Positive F-Score:\t", positive_fscore, " %")

	print("Negative Precision:\t", negative_precision, " %")	# correct negative:predicted as negative
	print("Negative Recall:\t", negative_recall, " %")			# correct negative:actual negative
	print("Negative F-Score:\t", negative_fscore, " %")
	print("\n")

print("Testing classifiers on SET...\n\n")

print("Standard Naive Bayes Classifier test: ")
test_classifier(s.NB_sentiment, positive_documents, negative_documents)

print("Multinomial Naive Bayes Classifier test: ")
test_classifier(s.MNB_sentiment, positive_documents, negative_documents)

print("Logistic Regression Classifier test: ")
test_classifier(s.LR_sentiment, positive_documents, negative_documents)

print("SVM Classifier test: ")
test_classifier(s.SVM_sentiment, positive_documents, negative_documents)

print("Linear SVM Classifier test: ")
test_classifier(s.LSVM_sentiment, positive_documents, negative_documents)

