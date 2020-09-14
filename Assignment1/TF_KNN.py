import sys
import pandas as pd


#TF_mappings = dict()
    

# #extracts the 15 binding
def extract_aminoacids(transcription_factor):
    residue_idx = [3,5,6,25,31,44,46,47,48,50,51,53,54,55,57]
    seq_aminoacids = ""  
    seq_aminoacids = [transcription_factor[i] for i in residue_idx if i<len(transcription_factor)]
    
     
    return seq_aminoacids


#How different they are. The lower the value, the more similar the sequences  
def compute_sequence_distance(seq1,seq2):
    # seq1 = extract_aminoacids(seq1)
    # seq2 = extract_aminoacids(seq2)
    length_of_seq = 0
    if(len(seq1) > len(seq2)):
        length_of_seq = len(seq2)
    else:
        lenth_of_seq = len(seq1)
    distance_count = 0
    for i in range(length_of_seq):
        if seq1[i] != seq2[i]:
            distance_count = distance_count + 1
            #distance_count = 1            
      
    return distance_count
  

def knn_classifier(training_data,unseen_data,output_profile,k=15):
    dist_list = list()
    predicted_output_dict = dict()
    for unseen_seq in unseen_data:
        sorted_dist_list = list()
        dist_list_for_each_unseen = list()
        for training_seq in training_data:
            dist = compute_sequence_distance(training_seq[1], unseen_seq[1])
            dist_list_for_each_unseen.append([dist,training_seq[0]])

        
        sorted_dist_list = sorted(dist_list_for_each_unseen)
        nearest_neighbours = sorted_dist_list[0:k]
        
        nearest_neighbours_labels = [x[1] for x in nearest_neighbours]
        nearest_neighbours_profile = output_profile[nearest_neighbours_labels]
        
        mean_output = nearest_neighbours_profile.mean(axis = 1)
        predicted_output_dict[unseen_seq[0]] = mean_output

    return predicted_output_dict
    

        
       
if __name__ == "__main__":
     
    train_data_file = sys.argv[1]
    train_labels_file= sys.argv[2]
    test_sequences_file= sys.argv[3]
    
    TF_unseen_sequences = list()
    TF_training_sequences = list()
    
    #read training data
    f = open(train_data_file)
    lines = f.readlines()
    for line in lines:
        TF_training_sequences.append([line.strip().split("\t")[0],line.strip().split("\t")[1]])
    
    #read testing data
    f = open(test_sequences_file)
    lines = f.readlines()
    for line in lines:
        TF_unseen_sequences.append([line.strip().split("\t")[0],line.strip().split("\t")[1]])

    #read training labels
    output_profile = pd.read_csv(train_labels_file,delimiter = "\t")

    predicted_output = knn_classifier(TF_training_sequences,TF_unseen_sequences,output_profile)

    predicted_output = pd.DataFrame(data = predicted_output)
    predicted_output.to_csv('predicted_profile.txt', sep="\t",index=False)



   
    

    
    


        
    
        
    
    

