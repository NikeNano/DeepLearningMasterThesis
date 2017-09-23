import pandas as pd 
import numpy as np 
import random 
from IPython import embed

# This model is equal to the "DataWndowTrain.py". See "DataWndowTrain.py" for comments of the code. 

class DataWindowValid():
	def __init__(self,pathLabel="validationLabels.pickle", pathData="validationData.pickle",windowSize=20):
		print("Updated 31 Mars")
		self.pathLabel=pathLabel
		self.pathData=pathData
		self.windowSize=windowSize
		self.batch_id=0
		self.main()

	def main(self):

		self.importData()
		self.generateWindows()
		self.oneHotEncoded2Labels()
		self.splitData()
		self.countLabels2Labels()

	def importData(self):
		patient_data=[]
		patient_labels=[]
		patient_seq_length=[]
		print("test")
		
		self.in_labels=pd.read_pickle(self.pathLabel) # get from data frame
		#	print("here")
		#	print(self.pathLabel)
		#except:
		#	print("second loading option")
		#	self.in_labels=pd.read_pickle("")
		###self.in_labels.reset_index(inplace=True)
		###self.in_labels=self.in_labels[self.in_labels['Time']<'2016-12-13']
		###self.in_labels.set_index(['Time', 'PatientID'], inplace=True)
		try:
			#[['AverageBloodFlowRate', 'StartSittingPulse', 'LitersProcessed', 'TimeDialyzed', 'FluidRemoved', 'PatientTempStart', 'PatientTempEnd', 'StartSittingSystolicBP', 'StartSittingDiastolicBP', 'EndSittingSystolicBP', 'EndSittingDiastolicBP', 'time_since_hosp_0', 'time_since_hosp_1', 'time_since_hosp_2', 'time_since_hosp_3', 'time_since_hosp_4', 'time_since_hosp_5', 'time_since_hosp_6', 'time_since_hosp_7', 'time_since_hosp_8', 'time_since_hosp_9', 'time_since_hosp_10', 'time_since_hosp_11', 'time_since_hosp_12', 'time_since_hosp_13', 'time_since_hosp_14', 'time_since_hosp_15', 'time_since_hosp_16', 'time_since_hosp_17', 'time_since_hosp_18', 'time_since_hosp_19']]#print("here")
			self.in_data=pd.read_pickle(self.pathData) # get form data frame
			#self.in_data=self.in_data[['AverageBloodFlowRate', 'StartSittingPulse', 'LitersProcessed', 'TimeDialyzed', 'FluidRemoved', 'PatientTempStart', 'PatientTempEnd', 'StartSittingSystolicBP', 'StartSittingDiastolicBP', 'EndSittingSystolicBP', 'EndSittingDiastolicBP', 'time_since_hosp_0', 'time_since_hosp_1', 'time_since_hosp_2', 'time_since_hosp_3', 'time_since_hosp_4', 'time_since_hosp_5', 'time_since_hosp_6', 'time_since_hosp_7', 'time_since_hosp_8', 'time_since_hosp_9', 'time_since_hosp_10', 'time_since_hosp_11', 'time_since_hosp_12', 'time_since_hosp_13', 'time_since_hosp_14', 'time_since_hosp_15', 'time_since_hosp_16', 'time_since_hosp_17', 'time_since_hosp_18', 'time_since_hosp_19']]
		except:
			self.in_data= pd.read_pickle(pathData)
		###self.in_data.reset_index(inplace=True)
		###self.in_data=self.in_data[self.in_data['Time']<'2016-12-13']
		###self.in_data.set_index(['Time', 'PatientID'], inplace=True)
		self.patientReferences()
		###
		###
		#self.patient_references=[383,384,389,393,398,401,402,403,404,409]
		for patients in self.patient_references:
			patient_data.append(self.in_data.xs(patients,level=1).as_matrix()) # Gets the patients
			patient_labels.append(self.in_labels.xs(patients,level=1).as_matrix()) # Gets the patients
		self.data_out=patient_data
		self.label_out=patient_labels

	def splitData(self):
		# split the data in test and training set
		split=int(np.rint(len(self.window_data)*0.7))
		self.test_data=self.window_data
		self.test_label=self.window_label
		#self.data_out=self.window_data[:self.windowSize+split]
		#self.label_out=self.window_label[:self.windowSize+split]

	def nextWindowBatch(self, batch_size):
		if self.batch_id == len(self.data_out):
			self.batch_id = 0
		batch_data = (self.data_out[self.batch_id:min(self.batch_id +batch_size, len(self.data_out))])
		batch_labels = (self.label_out[self.batch_id:min(self.batch_id + batch_size, len(self.data_out))])
		self.batch_id = min(self.batch_id + batch_size, len(self.data_out))
		return (np.asarray(batch_data), np.asarray(batch_labels))		

	def generateWindows(self):
		temp_window_data=[]
		temp_window_label=[]
		for data,label in zip(self.data_out,self.label_out):
			steps=1
			if len(data) > 40 : # now also drop all the frames where we have nan due to that all values are set to nan and forward/backward fill cant solve that
				#if np.isnan(data).sum() >0 :
				#		print("data")
				while steps < len(data)-self.windowSize:
					window_data=np.concatenate(data[steps:steps+self.windowSize])
					window_label=label[steps+self.windowSize]
					temp_window_data.append(window_data)
					temp_window_label.append(window_label)
					steps=steps+1
		print("The number of generated windows is : {}".format(len(temp_window_data)))
		print("The number of generated labels is : {}".format(len(temp_window_label)))
		#print("The data is:{} ".format(window_data))
		self.window_data=temp_window_data
		self.window_label=temp_window_label

	def testWindowBatch(self):
		return (self.test_data, self.test_label)

	def oneHotEncoded4Labels(self):
		temp_list_label=[]
		for label in self.window_label:
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
		self.window_label=temp_list_label

	def oneHotEncoded2Labels(self):
		temp_list_label=[]
		#embed()
		for label in self.window_label:
			if label==0:
				temp_list_label.append([1.0 ,0.0])
			else:
				temp_list_label.append([0.0 ,1.0])
		self.window_label=temp_list_label

	def patientReferences(self):
		temp_list=[]
		for i in range((self.in_data.index.levels[1].size)):
			temp_list.append(self.in_data.index.levels[1][i])
		temp_list=list(set(temp_list))
		#random.shuffle(temp_list)
		print("The number of patients is {} ".format(len(temp_list)))
		self.patient_references=temp_list

	def countLabels4Labels(self):
		# This is a function to return the distribution of the sequences! Good to know
		# Have how ever been throuing out a lot of data! This is pretty sad. 
		count0=count1=count2=count3=count4=0
		labels=self.window_label
		for label in labels:
			#print(label)
			if label==([1.0,0.0,0.0,0.0,0.0]):
				count0=count0+1
			if label==[0.0,1.0,0.0,0.0,0.0]:
				count1=count1+1
			if label==[0.0,0.0,1.0,0.0,0.0]:
				count2=count2+1
			if label==[0.0,0.0,0.0,1.0,0.0]:
				count3=count3+1
			if label==[0.0,0.0,0.0,0.0,1.0]:
				count4=count4+1
		numberOfLabels=len(labels)
		print("The number of 0 labels is :{}".format(count0))	
		print("The relative amount of 0 is :{0:.2f}% ".format(100*count0/numberOfLabels))
		print("The relative amount of 1 is :{0:.2f}% ".format(100*count1/numberOfLabels))
		print("The relative amount of 2 is :{0:.2f}% ".format(100*count2/numberOfLabels))
		print("The relative amount of 3 is :{0:.2f}% ".format(100*count3/numberOfLabels))
		print("The relative amount of 4 is :{0:.2f}% ".format(100*count4/numberOfLabels))

	def countLabels2Labels(self):
		# This is a function to return the distribution of the sequences! Good to know
		# Have how ever been throuing out a lot of data! This is pretty sad. 
		count0=count1=count2=count3=count4=0
		labels=self.window_label
		for label in labels:
			if label==([1.0,0.0]):
				count0=count0+1
			else:
				count1=count1+1
		numberOfLabels=len(labels)
		print("The number of 0 labels is :{}".format(count0))	
		print("The relative amount of 0 is :{0:.2f}% ".format(100*count0/numberOfLabels))
		print("The relative amount of 1 is :{0:.2f}% ".format(100*count1/numberOfLabels))


#a=DataWindow()

#data,labels=a.nextWindowBatch(1)
#print(len(data[0]))
#print(np.asarray(a.window_label).shape)
#print(a.window_label[1])
#print(data)
#print(len(data[0]))
#print(len(data))
#countLabels(a.window_label)




