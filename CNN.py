# CNN model. 

from sklearn.utils import shuffle
import matplotlib
import numpy as np

import sklearn.datasets
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from IPython import embed
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from CNN_DATA_WINDOW_TRAINING import DataWindowCNNTraning
from CNN_DATA_WINDOW_VALID import DataWindowCNNValid
#from DataWindowCNNTest import DataWindowCNN
from confusionmatrix import ConfusionMatrix

# resetting the graph ...
reset_default_graph()


def onehot(t, num_classes): # funtion to one-hot-encode the labels, if the MINIST data is used for test
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, int(col)] = 1
    return out

def myOneHot(input): # Funtion to do one-hot-encoding of the labels, this is used for the dialysis data. 
    temp_list_label=[]
    for label in input: 
        if label==0:
            temp_list_label.append([1.0 ,0.0])
        else: 
            temp_list_label.append([0.0 ,1.0])
    output=np.asarray(temp_list_label)
    return output

#LOAD the mnist data. To speed up training we'll only work on a subset of the data.
#Note that we reshape the data from (nsamples, num_features)= (nsamples, nchannels*rows*cols)  -> (nsamples, nchannels, rows, cols)
# in order to retain the spatial arrangements of the pixels
data = np.load('TestData/mnist.npz') # load minist data if it should be used for testing. 
ministData=False # decide to used dialysis data of MINIST data
myData=True # decide to used dialysis data	of MINIST data
GPU=False  # If GPU should be used, never fully implemented due to lack of memory in the GPU units

if ministData==True:  # loads the data correctly for the MINIST 
	num_classes = 10
	nchannels,rows,cols = 1,28,28
	x_train = data['X_train'][:10000].astype('float32')
	x_train = x_train.reshape((-1,nchannels,rows,cols))
	targets_train = data['y_train'][:10000].astype('int32')

	x_valid = data['X_valid'][:5000].astype('float32')
	x_valid = x_valid.reshape((-1,nchannels,rows,cols))
	targets_valid = data['y_valid'][:5000].astype('int32')

	x_test = data['X_test'][:5000].astype('float32')
	x_test = x_test.reshape((-1,nchannels,rows,cols))
	targets_test = data['y_test'][:5000].astype('int32')
#embed()

if myData==True: # laods the dialysis data. This is done using the import classes. 

	dataGeneratorTrain=DataWindowCNNTraning(20) # imports the training data, window size of 20 sequence steps is used
	dataGeneratorValid=DataWindowCNNValid(20) # imports the validaiton data, window	size of	20 sequence steps is used
	num_classes=2
	x_train = dataGeneratorTrain.traindata[:] # 5 works
	targets_train = dataGeneratorTrain.trainlabel[:] # 5 works
	
	#x_train,targets_train= shuffle(x_train, targets_train, random_state=0)

	x_valid = dataGeneratorValid.validationdata[:]
	targets_valid = dataGeneratorValid.validationlabel[:]

	x_test = dataGeneratorTrain.testdata[:]
	targets_test = dataGeneratorTrain.testlabel[:]

#embed()

print("Information on dataset")
print("x_train", x_train.shape)
print("targets_train", targets_train.shape)
print("x_valid", x_valid.shape)
print("targets_valid", targets_valid.shape)
print("x_test", x_test.shape)
print("targets_test", targets_test.shape)


# import layers from tensorflow, see tensorflow documentation for further information 
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax


# define a simple feed forward neural network

# hyperameters of the model is defined and the chape of the fileters are set based upon the size of the input data. 
#embed()
### original
if ministData==True:
	channels = x_train.shape[1]
	height = x_train.shape[2]
	width = x_train.shape[3]
	num_classes = 10
### my 
if myData==True:
	channels = x_train.shape[1]
	height = x_train.shape[3]
	width = x_train.shape[2]
	num_classes = 2

num_filters_conv1=  24 
num_filters_conv2= 18 
num_filters_conv3= 24
kernel_size_conv1 = [31,3] # [height, width]
stride_conv1 = [1, 1] # [stride_height, stride_width]
kernel_size_conv2=[1,3]
kernel_size_conv2=[1,1]
kernel_size_conv3=[1,6] # should change this due to the first filter size! 
num_l1 = 512


# Setting up placeholder, this is where your data enters the graph!

#### Original
if ministData==True:
	print("The minist data is running!")
	x_pl = tf.placeholder(tf.float32, [None, channels, height, width])
	l_reshape = tf.transpose(x_pl, [0, 2, 3, 1]) # TensorFlow uses NHWC instead of NCHW
	is_training = tf.placeholder(tf.bool)
if myData==True:
	#### My
	print("The Dialysis data is running!")
	x_pl = tf.placeholder(tf.float32, [None, 31, 20,1])
	l_reshape = tf.transpose(x_pl, [0, 1, 2, 3]) # TensorFlow uses NHWC instead of NCHW
	is_training = tf.placeholder(tf.bool)


#is_training = tf.placeholder(tf.bool)#used for dropout

# Building the layers of the neural network
# we define the variable scope, so we more easily can recognise our variables later
l_conv1 = convolution2d(l_reshape, num_filters_conv1, kernel_size=[31,3], stride=[1,1], scope="l_conv1")
l_maxPool2 = max_pool2d(l_conv1, [1,3], scope="l_maxPool2")
l_conv3 = convolution2d(l_maxPool2, num_filters_conv2, kernel_size_conv2, stride_conv1, scope="l_conv3")
l_maxPool4 = max_pool2d(l_conv3, [1,3], scope="l_maxPool4")
l_conv4=convolution2d(l_maxPool4, num_filters_conv3, kernel_size_conv3, stride_conv1, scope="l_conv4")
l_flatten = flatten(l_conv4, scope="flatten") # use l_conv1 instead of l_reshape
l1 = fully_connected(l_flatten, num_l1, activation_fn=relu, scope="l1")
l1 = dropout(l1 ,keep_prob=0.99, scope="dropout")
y = fully_connected(l1, num_classes, activation_fn=softmax, scope="y")

# y_ is a placeholder variable taking on the value of the target batch.
y_ = tf.placeholder(tf.float32, [None, num_classes])
y_test=tf.placeholder(tf.int32, [None])

# computing cross entropy per sample
cross_entropy = -tf.reduce_sum(y_ * tf.log(y+1e-8), reduction_indices=[1])

# averaging over samples
cross_entropy = tf.reduce_mean(cross_entropy)

# defining our optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

# applying the gradients
train_op = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Changed batch size form 10 to 1000
batch_size = 10000
num_epochs = 75
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
num_samples_test = x_test.shape[0]
num_batches_test = num_samples_test // batch_size

# list to store the accuracy and losses. 
train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0

loss = []

### GPU, This was never fully used to lack or memory at the GPU:s
if GPU==True:
	gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
###
sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=60)) # Creates the tensorflow session and sets the parallisation to run 60 threads
sess.run(tf.global_variables_initializer()) # intializes the variables that are created. 
sess.run(tf.initialize_local_variables())
print("Time to run! ")


# lists to store the loss, acc and auc score
trainLossList=[]
trainAccList=[]
trainAucList=[]
testLossList=[]
testAccList=[]
testAucList=[]
validLossList=[]
validAccList=[]
validAucList=[]

# the model is run during which it is trained and evaluated. 
try:
    for epoch in range(num_epochs):
        print("Ephoch number {}".format(1+ epoch))
        #Forward->Backprob->Update params
        cur_loss = 0
        accuracyList=[]
        aucList=[]
        aucListTest=[]
        #embed()
        for i in tqdm(range(num_batches_train)): # The model is trained
            index = range(i*batch_size, (i+1)*batch_size)
            x_batch = x_train[index]
            target_batch = targets_train[index]
            #feed_dict_train = {x_pl: x_batch, y_: onehot(target_batch, num_classes)}
            #feed_dict_train_test={x_pl: x_batch, y_test: onehot(target_batch, num_classes),is_training: True}
            if ministData==True:
                feed_dict_train = {x_pl: x_batch, y_: onehot(target_batch, num_classes), is_training: True}
            if myData==True:
                feed_dict_train = {x_pl: x_batch, y_: myOneHot(target_batch), is_training: True}
            fetches_train = [train_op, cross_entropy]
            res = sess.run(fetches=[train_op, cross_entropy], feed_dict=feed_dict_train)
            #acc = sess.run(accuracy, feed_dict=feed_dict_train)
            #print("The batch loss is : {} for itteration number : {}".format(res[1],i))
            #embed()######new test

            #pred_network= sess.run(fetches=[y], feed_dict={x_pl: x_batch,is_training: False})
            #pred=np.asarray(pred_network)
            #pred=pred[0][:,1]
            #fpr, tpr, thresholds = metrics.roc_curve(target_batch, pred, pos_label=0)
            #aucScore=metrics.auc(fpr, tpr)
            #aucList.append(aucScore)
            #print("The auc score is {} ".format(aucScore))
            #accuracyList.append(acc)
            batch_loss = res[1] #this will do the complete backprob pass
            cur_loss += batch_loss
        loss += [cur_loss/batch_size]

        confusion_valid = ConfusionMatrix(num_classes)
        confusion_train = ConfusionMatrix(num_classes)
        print("Training done")

        trainLoss=[]
        trainAcc=[]
        trainAuc=[]
        for i in range(num_batches_train): # The model is evaluated
            index = range(i*batch_size, (i+1)*batch_size)
            x_batch = x_train[index]
            targets_batch = targets_train[index]
            fetches_train = [cross_entropy]
            feed_dict_train = {x_pl: x_batch, y_: myOneHot(targets_batch), is_training: False}
            res = sess.run(fetches=[cross_entropy], feed_dict=feed_dict_train)
            trainLoss.append(res[0])
            feed_dict_train = {x_pl: x_batch, y_: myOneHot(target_batch), is_training: False}
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            trainAcc.append(acc)
            pred_network= sess.run(fetches=[y], feed_dict={x_pl: x_batch,is_training: False})
            prob=np.asarray(pred_network)
            _,b,c=prob.shape
            prob=prob.reshape(b,c)
            target=myOneHot(targets_batch)
            auc=roc_auc_score(y_true=target, y_score=prob)
            trainAuc.append(auc)
        #embed()
        print("The traning loss is:{}".format(np.mean(trainLoss)))
        print("The traning accuracy is:{}".format(np.mean(trainAcc)))
        print("The traning auc is {}".format(np.mean(trainAuc)))
        trainAccList.append(np.mean(trainAcc))
        trainAucList.append(np.mean(trainAuc))
        trainLossList.append(np.mean(trainLoss))
        testLoss=[]
        testAcc=[]
        testAuc=[]
        for i in range(num_batches_test): # The model is evaluated
            index = range(i*batch_size, (i+1)*batch_size)
            x_batch = x_test[index]
            targets_batch = targets_test[index]
            fetches_train = [cross_entropy]
            feed_dict_train = {x_pl: x_batch, y_: myOneHot(targets_batch), is_training: False}
            res = sess.run(fetches=[cross_entropy], feed_dict=feed_dict_train)
            testLoss.append(res[0])
            feed_dict_train = {x_pl: x_batch, y_: myOneHot(target_batch), is_training: False}
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            testAcc.append(acc)
            pred_network= sess.run(fetches=[y], feed_dict={x_pl: x_batch,is_training: False})
            prob=np.asarray(pred_network)
            _,b,c=prob.shape
            prob=prob.reshape(b,c)
            target=myOneHot(targets_batch)
            auc=roc_auc_score(y_true=target, y_score=prob)
            testAuc.append(auc)
        print("The test loss is:{}".format(np.mean(testLoss)))    
        print("The test accuracy is:{}".format(np.mean(testAcc)))
        print("The test auc is :{}".format(np.mean(testAuc)))
        testLossList.append(np.mean(testLoss))
        testAccList.append(np.mean(testAcc))
        testAucList.append(np.mean(testAuc))
        validLoss=[]
        validAuc=[]
        validAcc=[]
        for i in range(num_batches_valid): # The model is evaluated
            index = range(i*batch_size, (i+1)*batch_size)
            x_batch = x_valid[index]
            targets_batch = targets_valid[index]
            fetches_train = [cross_entropy]
            feed_dict_train = {x_pl: x_batch, y_: myOneHot(targets_batch), is_training: False}
            res = sess.run(fetches=[ cross_entropy], feed_dict=feed_dict_train)
            validLoss.append(res[0])            
            feed_dict_train = {x_pl: x_batch, y_: myOneHot(target_batch), is_training: False}
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            validAcc.append(acc)
            pred_network= sess.run(fetches=[y], feed_dict={x_pl: x_batch,is_training: False})
            prob=np.asarray(pred_network)
            _,b,c=prob.shape
            prob=prob.reshape(b,c)
            target=myOneHot(targets_batch)
            auc=roc_auc_score(y_true=target, y_score=prob)
            validAuc.append(auc)
        print("The validation loss is:{}".format(np.mean(validLoss)))
        print("The validation accuracy is:{}".format(np.mean(validAcc)))
        print("The validation auc is:{}".format(np.mean(validAuc)))
        validLossList.append(np.mean(validLoss))
        validAccList.append(np.mean(validAcc))
        validAucList.append(np.mean(validAuc))
        #train_acc += [train_acc_cur]
        #valid_acc += [valid_acc_cur]
        #print("My mean accuracy is {:.5f}".format(np.mean(accuracyList)))
        #print("My mean auc score is on training data {:.5f}".format(np.mean(aucList)))
        #print("My mean auc score is on testing data  {:.5f}".format(np.mean(aucListTest)))
        #print("Epoch %i : Train Loss %e , Train acc %f,  Valid acc %f " \
        #% (epoch+1, loss[-1], train_acc_cur, valid_acc_cur))
    embed()
except KeyboardInterrupt:
    pass
    
#embed()
print("Done") # The model is finished

