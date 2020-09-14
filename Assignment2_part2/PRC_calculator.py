import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# Computes proportion of predicted positive instances that are acutal positives


def calculate_precision(confidence_values,true_class,threshold_list):
	precision_results_list = list()
	
	for t in range(len(threshold_list)):
		predicted_positive_count= 0
		tp_count = 0
		for i in range(len(confidence_values)):
			if float(confidence_values[i])>= threshold_list[t]:
				predicted_positive_count = predicted_positive_count + 1
				if int(true_class[i]) == 1:
					tp_count = tp_count + 1
				
		precision_results_list.append(tp_count/predicted_positive_count)

	return precision_results_list



# How many actual positive instances were classified as positive



def calculate_recall(confidence_values,true_class,threshold_list):
	recall_results_list = list()
	actual_positive_count = 0
	tp_count = 0

	for t in range(len(threshold_list)):
		actual_positive_count = 0
		tp_count = 0
		for i in range(len(true_class)):
			if int(true_class[i]) == 1:
				actual_positive_count = actual_positive_count +1
				if float(confidence_values[i]) >= threshold_list[t]:
					tp_count = tp_count + 1

		recall_results_list.append(tp_count/actual_positive_count)

	return recall_results_list

def calculate_AUPRC(x1,x2):
		value = 0
		val = 0
		
		i=0
		val = x2[i]*x1[i]
		value = val
		for i in range(len(x1)-1):
			value += (x2[i+1] - x2[i])*x1[i+1]
		
		return value


if __name__ == '__main__':
	
	input_file = sys.argv[1]

	confidence_values = list() 
	true_class = list()

	f = open(input_file)
	lines = f.readlines()
	for line in lines:
		confidence_values.append(line.rstrip('\n').split('\t')[0])
		true_class.append(line.rstrip('\n').split('\t')[1])

	threshold_list = np.arange(0.9,0,-0.01) 

	precision_results = calculate_precision(confidence_values,true_class,threshold_list)
	recall_results = calculate_recall(confidence_values,true_class,threshold_list)

	pr = np.array((precision_results,recall_results,threshold_list),dtype=float).transpose()
	pr_df = pd.DataFrame(pr,index=None)
	pr_df.to_csv("PR_table.txt",header=None,sep="\t",index=None)
	
	pr_df.columns = ["precision","recall","threshold"] 

	AUPRC_value = calculate_AUPRC(pr_df['precision'],pr_df['recall']) 

	plt.plot(pr_df['recall'],pr_df['precision'],label="AUPRC = {:.3f}".format(AUPRC_value),linestyle='-',marker='o')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('PR curve')
	plt.legend()
	plt.savefig('PRC.png')


	

