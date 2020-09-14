import sys
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict



def knn_classifier(training_data,unseen_data,training_labels,k,distance_function):
	''' Performs k-NN Classification
	
	training_data: pandas dataframe of training data
	unseen_data: pandas dataframe of testing data
	training_labels: pandas dataframe of training labels
	k: Integer number of neighbours
	distance_function: "hamming" or "edit_distance"
	
	Returns output profile in a pandas dataframe
	'''

	dist_list = list()
	predicted_output_dict = dict()

	for unseen_index,unseen_row in unseen_data.iterrows():
		sorted_dist_list = list()
		dist_list_for_each_unseen = list()

		for training_index,training_row in training_data.iterrows():
			dist = compute_sequence_distance(distance_function,training_row['sequences'], unseen_row['sequences'])
			dist_list_for_each_unseen.append([dist,training_row['names']])      
		
		#sort list and pick out k nearest neighbours
		sorted_dist_list = sorted(dist_list_for_each_unseen)
		nearest_neighbours = sorted_dist_list[0:k]
		#retrieve sequence names of nearest neighbours and profile for all of these neighbours
		nearest_neighbours_labels = [x[1] for x in nearest_neighbours]
		nearest_neighbours_profile = training_labels[nearest_neighbours_labels]

		mean_output = nearest_neighbours_profile.mean(axis = 1)
		predicted_output_dict[unseen_row['names']] = mean_output
		predicted_output_dict = pd.DataFrame(data = predicted_output_dict)
		

	return predicted_output_dict



def compute_sequence_distance(distance_function_name,seq1,seq2):
	''' This function computes the distance between two sequences.
		
	Returns an integer
	 '''

	distance_count = 0
	distance_matrix = np.zeros((len(seq1)+1,len(seq2)+1),dtype = np.int)
	substitution = 0

	if distance_function_name =="hamming":
		length_of_seq = 0
		if len(seq1) > len(seq2):
			length_of_seq = len(seq2)
		else:
			lenth_of_seq = len(seq1)
		for i in range(length_of_seq):
			if seq1[i] != seq2[i]:
				distance_count = distance_count + 1

	elif distance_function_name == "edit_distance":
		distance_matrix = np.zeros((len(seq1)+1,len(seq2)+1),dtype = np.int)
		for i in range(len(seq1)+1):
			for j in range(len(seq2)+1):
				if i == 0:
					distance_matrix[i][j] = j
				elif j==0:
					distance_matrix[i][j] = i
				else:
					distance_matrix[i][j] = min(distance_matrix[i-1][j]+1,
												distance_matrix[i][j-1]+1,
												distance_matrix[i-1][j-1] +2 if seq1[i-1] != seq2[j-1] else distance_matrix[i-1][j-1]+0 )

		distance_count = distance_matrix[len(seq1)][len(seq2)]

	return distance_count


def cross_validation(data,labels,list_of_k,distance_function,splits=5):
	#splitting into folds
	kf = KFold(n_splits=splits, shuffle=True)
	kf.get_n_splits(data)
	folds = 1
	
	results_matrix = np.zeros((len(list_of_k)+1,len(distance_function)+1),dtype="<U16")
	mean_dict = dict()
	std_dict = dict()
	spearman_scores_dict = defaultdict(list)
	
	for train_index, test_index in kf.split(data):
					
		x_train, x_test = data.iloc[train_index], data.iloc[test_index] 
		y_train, y_test = labels.iloc[:,train_index], labels.iloc[:,test_index]
		

		for i in range(len(list_of_k)):
			for j in range(len(distance_function)):
				spearman_scores_for_param_combination = list()
				dict_key = f'{i},{j}'
				
				prediction_labels = knn_classifier(x_train,x_test,y_train,list_of_k[i],distance_function[j])
				spearmancorr = y_test.corrwith(prediction_labels,axis = 0,method='spearman')
				mean_spearman = spearmancorr.mean()
				spearman_scores_dict[dict_key].append(mean_spearman)
		folds = folds+1
	for key in spearman_scores_dict:

		[i,j] = key.split(",")
		mean_dict[key] = round(np.mean(spearman_scores_dict[key]),2)		
		results_matrix[int(i)+1][int(j)+1] = "%.2f\u00B1%.2f"%(mean_dict[key],round(np.std(spearman_scores_dict[key]),2))


	#find key in dictionary with highest mean value
	chosen_keys = [key for m in [max(mean_dict.values())] for key,val in mean_dict.items() if val == m]
	if(len(chosen_keys)>1):
		for key in chosen_keys:
			std_dict[key] = round(np.std(spearman_scores_dict[key]),2)
			chosen_keys = [key for m in [min(std_dict.values())] for key,val in std_dict.items() if val == m]

	
	[m,n] = sorted(chosen_keys)[len(chosen_keys)-1].split(",")

	for i in range(len(list_of_k)):
		results_matrix[i+1][0] = "K=%d"%list_of_k[i]

	for j in range(len(distance_function)):
		results_matrix[0][j+1] = distance_function[j]


	parameter_choice = [list_of_k[int(m)],distance_function[int(n)]]

	

	return results_matrix,parameter_choice


def write_results_to_file(results_matrix,parameter_choice):

	f = open('model_selection_table.txt','w+')
	for row in results_matrix:
		for x in row:
			f.write(x+'\t')
		f.write("\n")	

	f.write("Model Chosen: K="+str(parameter_choice[0])+", "+str(parameter_choice[1]))	
	f.close()

	return



if __name__ == "__main__":

	
	training_data_file = sys.argv[1]
	training_labels_file= sys.argv[2]

	#read training labels into pandas dataframe
	training_labels = pd.read_csv(training_labels_file,delimiter = "\t")
	training_data = pd.read_csv(training_data_file,delimiter="\t", header=None,names=["names","sequences"])
	list_of_k = [11,13,15,17]
	distance_function = ["hamming_distance","edit_distance"]

	[results_matrix,parameter_choice] = cross_validation(training_data,training_labels,list_of_k,distance_function)


	write_results_to_file(results_matrix,parameter_choice)
    







