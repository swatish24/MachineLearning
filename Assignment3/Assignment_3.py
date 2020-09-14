import sys
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score

def predict_test_output(classifier,test_data):

	confidence_scores = classifier.predict_proba(test_data)
	pd.DataFrame(confidence_scores).to_csv("predictions.txt",header=None,index=None)


def grid_search(clf,param,training_data, training_labels):

	#use random forest for feature selection
	feature_selection_clf = RandomForestClassifier(n_estimators=10,random_state=42) 
	rfecv = RFECV(estimator=feature_selection_clf, step=1, cv=2, scoring = 'average_precision', n_jobs=-1, verbose=True)
	rfecv.fit(training_data,training_labels)

	#Grid search for all parameters
	grid_search = GridSearchCV(estimator=clf,param_grid = param,cv=2,verbose=True,n_jobs=-1)
	grid_search.fit(training_data,training_labels)

	# print(grid_search.best_estimator_)

	cv_score = cross_val_score(grid_search,training_data,training_labels,cv=2,scoring="average_precision")

	print( "CV Score : Mean: %.3f  Std: %.3f  Min: %.3f  Max: %.3f"
          % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))


	return np.mean(cv_score),grid_search.best_estimator_


if __name__ == "__main__":


	training_data_file = sys.argv[1]
	testing_data_file = sys.argv[2]

	score_list = list()

	training_file = pd.read_csv(training_data_file,sep="\t")
	testing_file = pd.read_csv(testing_data_file,sep="\t")
	# training_file = sys.argv[1]
	# testing_file= sys.argv[2]

	training_data = training_file.drop('group', axis=1)
	training_labels = training_file['group']

	classifier_list = [GradientBoostingClassifier(),RandomForestClassifier(),svm.SVC()]
	# print(classifier_list)

	parameters = [{
	    "n_estimators":[50,100,200,300],
	    "max_depth":[3,4,5],
	    "max_features":["log2","sqrt"],
	},
	{

		"max_depth":[2,3,5],
		"n_estimators":[50,100,200,300],
		"criterion":["gini","entropy"]
	},
	{
	    "C":[10,50,100],
	    "gamma":[0.001,0.0001],
	    "kernel":['linear','rbf']
	}]

	for i in range(len(classifier_list)):
		score, estimator= grid_search(classifier_list[i],parameters[i],training_data,training_labels)
		score_list.append([score,estimator] ) 
						
	score_list = sorted(score_list, key=lambda x: x[0])

	# print(score_list)

	predict_test_output(score_list[0][1], testing_file)








	





