This is the read me file for the 3 predictive model developed by Niklas Hansson in his master thesis at Lytics Health Care AB
2017. The read me will discuss the three models "FFNN.py", "CNN.py" and "RNN.py". The models are developed based on native tensorflow. 
Thus it is requaired to have tensorflow installed to be able to run the models. The models should represent the following:

FFNN.py - This is an implementenation of a Feed forward neural network for classification, to predict if a patient will be hospitalised within 30 days.
	  This model uses a slidingwindow model in order to work with the sequantial data, see "Master_thesis_Report.pdf". 
          THis model uses the "DataWindowTrain.py" and "DataWindowValid.py" to import of the data. They work in
          the same way but import training,test and validation(test) set sepreatly. "DataWindowTrain.py" is commented. 

CNN.py - This is an implementation of a Convolutional neural networkf for classification, to predict if a patient will be hospitalised within 30 days. 
	 This model uses a sliding window approach in order to allow the model to work on the sequantila data, see "Master_thesis_Report.pdf"
         This model uses the "CNN_DATA_WINDOW_TRAINING.py" and "CNN_DATA_WINDOW_VALID.py" to import of the data. They work in
         the same way but import training,tes and validation(test) set sepreatly. "CNN_DATA_WINDOW_TRAINING.py" is commented. 

RNN.py - This is an implementation of a	Reccurent neural networkf for classification, to predict if a patient will be hospitalised within 30 days. 
         This model is sequencial and works with the seperate sequences. The model uses "Data_Import_V4_test_version2_train.py" and "Data_Import_V4_test_version2_validation.py"
         they work the same but import training,tes and validation(test) set sepreatly. "Data_Import_V4_test_version2_train.py" is commented.

To run the models, run the "RNN.py", "CNN.py" or "FFNN.py" to run the respective models. 
All the models work with daily data. The different models handles the data differently and thus have different import steps. The data files are contained in the folder
in order to allow for the models to be run. It should be noted that the code is written with the intent for investigation only and many small tweks are used. 
It could be noted that the validation set and test set are switched compare to the traditional naming. 
/ Niklas Hansson 3 July 2017
