from IPython import embed
import pandas as pd
from tqdm import tqdm as tq
import pickle
import numpy as np
import random
from sklearn.utils import shuffle

# This class works the same as "CNN_DATA_WINDOW_TRAINING.py", see "CNN_DATA_WINDOW_TRAINING.py" for comments. 
class DataWindowCNNValid():
	def __init__(self,windowSize=20):
		self.windowSize=windowSize
		self.batch_id=0
		#self.dataPath="/home/niklas/Code/RNN/Merdata/dataset_featuresAdded.pickle"
		#self.labelPath="/home/niklas/Code/RNN/Merdata/SplitData/thesis_labels.pickle"
		self.dataPath="validationDataStand.pickle" # "/home/niklas/Code/RNN/Merdata/SplitData/validationDataStand.pickle"
		self.labelPath="validationLabels.pickle" # "/home/niklas/Code/RNN/Merdata/SplitData/validationLabels.pickle"
		self.labelColumnName=0
		#self.dataPath="/home/niklas/Code/RNN/Merdata/dataset_SUPER_TEST_3.pickle"
		#self.labelPath="/home/gemensam/Projects/Master_Thesis/Delad_data/orginal_labels.pickle"
		#self.labelColumnName="label"
		self.main()

	def main(self):
		self.getPatientID()
		self.importData()
		#self.testData()
		self.generateWinwods()
		self.splitData()
		#self.makeBatches()
		#self.saveData()
		#self.loadData()
		#embed()
	def testData(self):
		self.data=[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]
		self.label=[[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,0,0]]


	def importData(self):
		self.data=pd.read_pickle(self.dataPath)
		self.labels=pd.read_pickle(self.labelPath)
		data=self.data.reset_index()
		label=self.labels.reset_index()
		data_gropus=data.groupby("PatientID")
		label_groups=label.groupby("PatientID")
		all_data=[]
		all_label=[]
		#embed()
		print(self.patientID)
		for patients in self.patientID:
			data=data_gropus.get_group(patients)
			data=data.drop(['Time','PatientID'],1)
			all_data.append(data.values)
			all_label.append(label_groups.get_group(patients)[self.labelColumnName].values)
		self.data=all_data
		self.label=all_label

	def getPatientID(self):
		data=pd.read_pickle(self.dataPath)
		data.reset_index(inplace=True)
		listPatients=data['PatientID'].unique()
		random.shuffle(listPatients)
		self.patientID=listPatients

	def splitData(self):
		split1=int(len(self.WindowsData)*0.8)
		split2=int((split1)*0.8)

		#self.traindata=np.asarray(self.WindowsData[:split2])
		#shapeTrainData=self.traindata.shape
		#self.traindata=self.traindata.reshape(shapeTrainData[0],shapeTrainData[2],shapeTrainData[3],shapeTrainData[4])
		#self.trainlabel=np.asarray(self.WindowsLabels[:split2])
		#self.trainlabel=self.oneHot(self.trainlabel)

		#self.testdata=np.asarray(self.WindowsData[split2:split1])
		#shapeTestData=self.testdata.shape
		#self.testdata=self.testdata.reshape(shapeTestData[0],shapeTestData[2],shapeTestData[3],shapeTestData[4])
		#self.testlabel=np.asarray(self.WindowsLabels[split2:split1])
		#self.testlabel=self.oneHot(self.testlabel)

		self.validationdata=np.asarray(self.WindowsData)		
		shapeValidationData=self.validationdata.shape
		self.validationdata=self.validationdata.reshape(shapeValidationData[0],shapeValidationData[2],shapeValidationData[3],shapeValidationData[4])
		self.validationlabel=np.asarray(self.WindowsLabels)
		#self.validationlabel=self.oneHot(self.validationlabel)
		#embed()

	def oneHot(self,input):
		temp_list_label=[]
		for label in input: 
			if label==0:
				temp_list_label.append([1.0 ,0.0])
			else: 
				temp_list_label.append([0.0 ,1.0])
		output=np.asarray(temp_list_label)
		return output

	def makeBatches(self):
		data=self.traindata
		label=self.trainlabel
		# positive is the rare label....
		# This is different between new and old...
		dataPositiveLabel=data[label==0]
		labelPositiveLabel=label[label==0]
		dataNegativeLabel=data[label==1]
		labelNegativeLabel=label[label==1]
		testSet=int(len(dataNegativeLabel)/len(dataPositiveLabel))

		if testSet>1:
			dataArray=np.concatenate((dataNegativeLabel,dataPositiveLabel), axis=0)
			labelArray=np.concatenate((labelNegativeLabel,labelPositiveLabel), axis=0)
			testSet=testSet-1
		for i in range(testSet):
			dataArray=np.concatenate((dataArray, dataPositiveLabel), axis=0)
			labelArray=np.concatenate((labelArray, labelPositiveLabel), axis=0)
			#newPosData.append(dataPositiveLabel)
			#newPosLabel.append(labelPositiveLabel)
		#newPosData=np.asarray(newPosData)
		#newPosLabel=np.asarray(newPosLabel)

		#newPosData=np.reshape(newPosData,(-1,1,20,11))
		#newPosLabel=np.reshape(newPosLabel,(-1))
		#embed()
		#data=np.concatenate((newPosData, dataNegativeLabel), axis=0)
		#label=np.concatenate((newPosLabel, labelNegativeLabel), axis=0)
		data,label= shuffle(dataArray, labelArray, random_state=0)
		#embed()
		self.traindata=data
		self.trainlabel=label


	def generateWinwods(self):
		print("generator")
		temp_window_data=[]
		temp_window_label=[]
		#embed()
		for data,label in (zip(self.data,self.label)): # took away tq
			if len(data)>25:
				step=0
				#embed()
				while step+self.windowSize-1 < len(data):
					window_data=data[step:+step+self.windowSize]
					#embed()
					window_label=label[step+self.windowSize-1] # wrong otherwise!!!!!! 
					if window_label!=0:
							window_label=1
					window_data=window_data.transpose()
					#### CHANGE HERE FOR OTHER DATA
					#### CHANGE HERE IT IS IMPORTANT
					#### DO IT!!! / NIKLAS 
					#### 31 is for the 31 features
					#### Might be different!!!!
					

					#### CHANGE TO 11 NOW
					window_data=window_data.reshape(31,20,1)

					temp_window_data.append([window_data])
					temp_window_label.append(window_label)
					step=step+1
		self.WindowsData=temp_window_data
		self.WindowsLabels=temp_window_label
		#embed()

	def saveData(self):
		with open('traindata.pickle', 'wb') as handle:
			pickle.dump(self.traindata, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print("hej")

if __name__ == '__main__':
	a=DataWindowCNN()
	print((a.traindata))
	print((a.trainlabel))
	print((a.WindowsData))
	print((a.WindowsLabels))
	a.makeBatches()
	embed()
