import pandas as pd 
import numpy as np 
#import random
from IPython import embed 

# The data import class. Behaves the same as "CNN_DATA_WINDOW_TRAINING.py", see "CNN_DATA_WINDOW_TRAINING.py" for comments. 
class Data_PreProcessing_validation():
	def __init__(self, padding="True", task="Training", limit_length="False", seq_length=200, traning_size=5700):
		# Declared global variables
		self.task=task
		self.traning_size=traning_size # Traning size is the number of patients used when creating the data set. 
		self.limit_length=limit_length
		self.seq_len_limit=seq_length
		self.batch_id=0
		self.task=task
		self.padding=padding
		print("Start of loading data")
		self.main()
		#embed()
	def main(self):
		self.get_data()
		self.cut_length_of_sequence()
		self.padd_data()
		self.padd_label()
		self.one_hot_encode_two_classes()
		self.split_data_set()
		#self.padd_label()#print("data loading done!")
		

	def get_data(self):
		patient_data=[]
		patient_labels=[]
		patient_seq_length=[]
		# innan data/thesis_labels
		#self.in_data=pd.read_pickle("data/thesis_data.pickle")
		#self.in_data=pd.read_pickle("/home/niklas/Code/RNN/Merdata/dataset_SUPER_TEST_3.pickle") # get form data frame
		self.in_data=pd.read_pickle("validationDataStand.pickle") 
		# innan data/thesis_data
		self.in_labels=pd.read_pickle("validationLabels.pickle")
		#self.in_labels=pd.read_pickle("/home/gemensam/Projects/Master_Thesis/Delad_data/orginal_labels.pickle") # get from data frame
		self.get_patient_references()
		#print(self.patient_references)
		###
		###
		#self.patient_references=[383,384,389,393,398,401,402,403,404,409]
		for patients in self.patient_references:
			if len(self.in_labels.xs(patients,level=1).as_matrix()) >19:
				patient_data.append(self.in_data.xs(patients,level=1).as_matrix()) # Gets the patients
				patient_labels.append(self.in_labels.xs(patients,level=1).as_matrix()) # Gets the patients
				patient_seq_length.append(len(self.in_labels.xs(patients,level=1).as_matrix())) # this was the problem!
		self.data_out=patient_data
		self.label_out=patient_labels
		self.seqlen=patient_seq_length
		self.labelNotEncoded=self.label_out


	def cut_length_of_sequence(self):
		# This function should cut the lenght of the sequences if they exced the limit!
     	# If they are below it shouldn't do anything at all!
		if (self.limit_length):
			temp_list_data=[]
			temp_list_labels=[]
			new_seqlen=[]
			#self.seq_len_limit=20
			for seq_data,seq_label in zip(self.data_out,self.label_out):
				#if seq_data.size < 20:
				#	continue
				if seq_data.size > self.seq_len_limit:
					seq_data=seq_data[:self.seq_len_limit]
					seq_label=seq_label[:self.seq_len_limit]
				temp_list_data.append(seq_data)
				temp_list_labels.append(seq_label)
			self.data_out=temp_list_data
			self.label_out=temp_list_labels
			for seq in self.seqlen: # loop of the seqlen vector
				#if seq <20:
				#	continue 
				if seq > self.seq_len_limit:
					new_seqlen.append(self.seq_len_limit) # put all lengths to the limitign vector
				else:
					new_seqlen.append(int(seq))
			self.seqlen=new_seqlen
		#print("This is the shape after cut {}".format(self.data_out.shape))

	def get_patient_references(self):
		temp_list=[]
		if (self.in_data.index.levels[1].size) > self.traning_size:
			for i in range(self.traning_size):
				temp_list.append(self.in_data.index.levels[1][i])
		else:
			for i in range((self.in_data.index.levels[1].size)):
				temp_list.append(self.in_data.index.levels[1][i])
		temp_list=list(set(temp_list))
		#random.shuffle(temp_list)
		self.patient_references=temp_list

	def get_max_seq(self,seq_list):
		# gets the longest sequence this is used as the baseline for the padding
		max_seq=0
		for seq in seq_list:
			if seq.shape[0]>max_seq:
				max_seq=seq.shape[0]
		return(max_seq) 

	def padd_data(self):
		#print("The max lenght of all sequences is {}".format(self.seq_len_limit))
		for index,seq_step in enumerate(self.data_out):
			if seq_step.shape[0]<self.seq_len_limit:
				nbr_features=seq_step.shape[1]
				padd_length=self.seq_len_limit-seq_step.shape[0]
				padd_seq=self.padding_sequence(padd_length,nbr_features)
				self.data_out[index]=np.concatenate((seq_step,padd_seq),axis=0)

	def padd_label(self):
		max_seq=self.get_max_seq(self.label_out)
		for index,seq_step in enumerate(self.label_out):
			if seq_step.shape[0]<self.seq_len_limit:
				#print("label padd")
				nbr_features=1
				#embed()
				padd_length=self.seq_len_limit-seq_step.shape[0]
				padd_seq=self.padding_sequence(padd_length,nbr_features)
				self.label_out[index]=np.append(seq_step,padd_seq)

	def padding_sequence(self,padd_length, nbr_features):
		padd_seq=np.zeros((padd_length, nbr_features),dtype=np.int)
		return padd_seq

	def one_hot_encoder_five_classes(self):
		# One hot encode the labels
		temp_list_label_patients=[]
		for patients in self.label_out:
			temp_list_label=[]
			for label in patients:
			#print(labels)
				if label==0:
					temp_list_label.append([1.0,0.0,0.0,0.0,0.0])
				if label==1:
					temp_list_label.append([0.0,1.0,0.0,0.0,0.0])
				if label==2:
					temp_list_label.append([0.0,0.0,1.0,0.0,0.0])
				if label==3:
					temp_list_label.append([0.0,0.0,0.0,1.0,0.0])
				if label==4:
					temp_list_label.append([0.0,0.0,0.0,0.0,1.0])
			temp_list_label_patients.append(temp_list_label)
		self.label_out=temp_list_label_patients

	def one_hot_encode_two_classes(self):
		# One hot encode the labels
		temp_list_label_patients=[]
		for patients in self.label_out:
			temp_list_label=[]
			for label in patients:
			#print(labels)
				if label==0:
					temp_list_label.append([1.0,0.0])
				else:
					temp_list_label.append([0.0,1.0])
			temp_list_label_patients.append(temp_list_label)
		self.label_out=temp_list_label_patients
		#### Added this!
		self.label_out=np.asarray(self.label_out)

	def get_validation_set(self):
		# method to retutn the test data
		array=np.array([])
		for i in self.seqlen:
			test=np.ones(i)
			test_2=np.zeros(self.seq_len_limit-i)
			result=np.append(test,test_2)
			array=np.append(array,result)
		mask=array.reshape(-1,self.seq_len_limit)
		return (self.data_out, self.label_out,self.seqlen,mask)


	def split_data_set(self):
		# generate a test and traning set! 
		# Get the test set first since i save over the 
		# data variable for the traning data
		#print(len(self.data_out))
			
		self.data_out=self.data_out
		self.label_out=self.label_out
		self.seqlen=self.seqlen
		#embed()
		

#a=Data_PreProcessing()
#batch_data,batch_labels,batch_seqlen=a.next(2)
#print(len(batch_labels[0]))
#print(len(batch_data[0]))
#print(batch_seqlen[0])
#print(batch_data[0])
#print(len(batch_labels[0]))

#### Ther is currently a problem with the size of the vectors of the labels! This is since i have a label for each step
#### I have to solve this whan i like to use this class for RNN. 
if __name__ == "__main__":
	a=Data_PreProcessing()



