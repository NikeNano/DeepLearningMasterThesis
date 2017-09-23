# For the RNN the sequences are zero padded. This is done in the import class, however this has to be tackled then the acc,auc and loss is calculated. 
# For the loss and acc this is done based upon a mask. For the AUC the sequence lenght is used. 

from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm
from IPython import embed # Give a strange error for tensorflow....000
# add del IPython at the end of the file solves the problem it seams like....
from tensorflow.python.framework.ops import reset_default_graph
from Data_Import_V4_test_version2_train import Data_PreProcessing_train
from Data_Import_V4_test_version2_validation import Data_PreProcessing_validation
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from sklearn.metrics import roc_auc_score 
# Resetting the graph
reset_default_graph()

#### MODEL #####
print("This is the updated version")
learning_rate = 0.001 # changed from 0.005
### CHANGED THE NUMBER OF ITTERATIONS FROM 300000 to 3000000
### Changed to number to from 3000000 to 30000
# defines how long the model should run. 
training_iters = 3000000
batch_size = 100
display_step = 10
train_model=True

# Network Parameters
#### used to be 200
#### used to be 500
#### used to be no input to the Data_preProcessing()
seq_max_len = 1000 # Sequence max length, longer sequences are cut to this sequence lenght if they are longer! This is importance to notice.  
n_hidden = 20 # memory # increased memory from 80 to 160   # Changed to 120     
n_classes = 2 # linear sequence or not          
nbr_features=31#11 # Number of features in the input data  

#trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
#testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)
trainset=Data_PreProcessing_train(seq_length=seq_max_len) 
validationset=Data_PreProcessing_validation(seq_length=seq_max_len)
#embed()
datalen=trainset.data_out
iter_ephoc=len(datalen)
iter_ephoc=int(iter_ephoc/batch_size)
num_ephocs=60

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len,nbr_features]) ####### CHANGE HERE!!!!
y = tf.placeholder("float", [None, seq_max_len,n_classes])
myMask=tf.placeholder("float",[None,seq_max_len])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# The defined weights and biases defiend below are note used in the finished model but was used during evaluation. 
weights = {
    'out2': tf.Variable(tf.random_normal([  256,n_classes])), # Should add to that we get all sequence steps in here by having [hidden, n_classes, max_seq_len],
    'out1': tf.Variable(tf.random_normal([  n_hidden,256])) # Should add to that we get all sequence steps in here by having [hidden, n_classes, max_seq_len]
}   # Base case = [256, n_classes,seq_max_len]

biases = {
    'out2': tf.Variable(tf.random_normal([ n_classes])), # Should add so that all sequence steps is fitted in here and thus the bias should be [n_classes,max_seq_len] Then we should have for all steps!!
    'out1': tf.Variable(tf.random_normal([ 256])) # Should add so that all sequence steps is fitted in here and thus the bias should be [n_classes,max_seq_len] 
}   # Base case= [n_classes,seq_max_len]


def unpack_sequence(tensor): # Unpack the sequences
    ## Split the single tensor in to a list of frames, sequecne
    return tf.unpack(tf.transpose(tensor, perm=[1,0,2]))

def pack_sequence(sequence): # pack the sequences
    # Combine a list of rame sin to a single tensor
    return tf.transpose(tf.pack(sequence),perm=[1,0,2])

def cost(output,target,seqlen,mask): # calcualate the cost 
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = mask #tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    #embed() Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)

def accuracy(output, target, seqlen,mask): # calculate the accuracy
    mistakes = tf.not_equal(tf.argmax(target, 2), tf.argmax(output, 2))
    mistakes = tf.cast(mistakes, tf.float32)
    mask =mask #tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    mistakes *= mask #mask
    # Average over actual sequence lengths.
    mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
    mistakes /= tf.reduce_sum(mask, reduction_indices=1)
    return (1-tf.reduce_mean(mistakes))

def error(output, target, seqlen,mask): # calculate the cost
    mistakes = tf.not_equal(tf.argmax(target, 2), tf.argmax(output, 2))
    mistakes = tf.cast(mistakes, tf.float32)
    mask = mask #tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    mistakes *= mask
    # Average over actual sequence lengths.
    mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
    mistakes /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(mistakes)

def dynamicRNN(x, seqlen, weights, biases): # defines the RNN model and the layers used. Some of the hyperparamters are also set in the layers
    x=unpack_sequence(x) 
        # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    ##### CHANGED THE OUTPUT_KEEP_PROB=0.8 to 0.9
    #cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.9) #####
    cell = tf.contrib.rnn.AttentionCellWrapper(cell=lstm_cell,attn_length=20,state_is_tuple=True)
    outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32, sequence_length=seqlen)
    output=pack_sequence(outputs)
    output = tf.reshape(output, [-1, n_hidden]) # This is a key step to remember Niklas!!!!
    l1 = fully_connected(output, num_outputs=26, activation_fn=relu,normalizer_fn=batch_norm, scope="l1")
    l0 = dropout(l1 ,keep_prob=0.75, scope="l1_dropout") # Used to be 0.95
    #l2 = fully_connected(l0, num_outputs=32, activation_fn=relu,normalizer_fn=batch_norm, scope="l2")
    prediction = fully_connected(l0, 2, activation_fn=softmax, scope="y")
    prediction = tf.reshape(prediction, [-1, seq_max_len, n_classes])
    return prediction

def outputForAuc(output,target,seqlen): # this function was not used in the end!

    #prob=tf.reshape(output,[-1,n_classes])
    #target=tf.reshape(target,[-1,n_classes])
    prob=output
    target=target
    return prob,target

pred = dynamicRNN(x, seqlen, weights, biases)

cost=cost(pred,y,seqlen,myMask)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
auc=tf.contrib.metrics.streaming_auc(pred, y)

accuracy=accuracy(pred,y,seqlen,myMask)
error=error(pred,y,seqlen,myMask)
forAuc=outputForAuc(pred,y,seqlen)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# defines lists for the loss, acc and auc
trainAllAuc=[]
trainAllLoss=[]
trainAllAcc=[]
testAllAuc=[]
testAllLoss=[]
testAllAcc=[]
validAllAuc=[]
validAllLoss=[]
validAllAcc=[]

# Launch the graph
with tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=40)) as sess:
    sess.run(init)
    step = 1
    print("Start")
    #print("The number of steps is: {}".format(training_iters/(batch_size)))
    #embed()
    for epoch in range(num_ephocs): # train the model 
        print("Epoch number {}".format(epoch+1))
        for itter in tqdm(range(iter_ephoc)):
            batch_x, batch_y, batch_seqlen,mask = trainset.next(batch_size)
            #print("here and running!")#embed()# Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen,myMask: mask})
            #step=step+1
        # Calculate batch accuracy
        lossList=[]
        accList=[]
        # Evaluation of the traing data. 
        for itter in range(iter_ephoc): # evaluate the model
            batch_x, batch_y, batch_seqlen,mask = trainset.next(batch_size)
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen,myMask: mask})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen,myMask: mask})
            accList.append(acc)
            lossList.append(loss)
        print("The traning loss is {}".format(np.mean(lossList)))
        print("The traning accuracy is {}".format(np.mean(accList)))
        batch_x, batch_y, batch_seqlen,mask = trainset.get_train_set()
        prob_in,target_in=sess.run(forAuc, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
        temp_list_target_new=[]
        temp_list_prob_new=[]
        #To calculate the auc correctly the padded sequences have to be removed. This is done below and then the AUC is calculated. 
        for index,seeeqlen in enumerate(batch_seqlen):
            step=0
            while step < seeeqlen:
                temp_list_prob_new.append(prob_in[index,step,:])
                temp_list_target_new.append(target_in[index,step,:])
                step=step+1
        prob=np.asarray(temp_list_prob_new)
        target=np.asarray(temp_list_target_new)
        print("The traning auc score is {}".format(roc_auc_score(y_true=target, y_score=prob)))
        trainAllAuc.append(roc_auc_score(y_true=target, y_score=prob))
        trainAllLoss.append(np.mean(lossList))
        trainAllAcc.append(np.mean(accList))
       
        #Evaluation of the test set for the model, this is done as for the validaiton set and training set. 
        test_data,test_label,test_seqlen,mask =trainset.get_test_set()
        loss=sess.run(cost, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen, myMask:mask})
        acc=sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen, myMask:mask})
        print("The test loss is {}".format(loss))
        print("The test accuracy is {} " .format(acc))
        prob_in,target_in=sess.run(forAuc, feed_dict={x: test_data, y: test_label,seqlen: test_seqlen})
        temp_list_target_new=[]
        temp_list_prob_new=[]
        for index,seeeqlen in enumerate(test_seqlen):
            step=0
            while step < seeeqlen:
                temp_list_prob_new.append(prob_in[index,step,:])
                temp_list_target_new.append(target_in[index,step,:])
                step=step+1
        prob=np.asarray(temp_list_prob_new)
        target=np.asarray(temp_list_target_new)
        print("The test auc score is {}".format(roc_auc_score(y_true=target, y_score=prob)))
        testAllAuc.append(roc_auc_score(y_true=target, y_score=prob))
        testAllLoss.append(loss)
        testAllAcc.append(acc)

        #Evaluation of the validation set for the model, this	is done	as for the test set and training set.
        valid_data,valid_label,valid_seqlen,mask =validationset.get_validation_set()
        loss=sess.run(cost, feed_dict={x: valid_data, y: valid_label, seqlen: valid_seqlen, myMask:mask})
        acc=sess.run(accuracy, feed_dict={x: valid_data, y: valid_label, seqlen: valid_seqlen, myMask:mask})
        print("The validation loss is {}".format(loss))
        print("The validation accuracy is {} " .format(acc))
        prob_in,target_in=sess.run(forAuc, feed_dict={x: valid_data, y: valid_label,seqlen: valid_seqlen})
        temp_list_target_new=[]
        temp_list_prob_new=[]
        for index,seeeqlen in enumerate(valid_seqlen):
            step=0
            while step < seeeqlen:
                  temp_list_prob_new.append(prob_in[index,step,:])
                  temp_list_target_new.append(target_in[index,step,:])
                  step=step+1
        prob=np.asarray(temp_list_prob_new)
        target=np.asarray(temp_list_target_new)
        print("The validation auc score is {}".format(roc_auc_score(y_true=target, y_score=prob)))
        validAllAuc.append(roc_auc_score(y_true=target, y_score=prob))
        validAllLoss.append(loss)
        validAllAcc.append(acc) 
    embed()




   #embed()
   #validatonSet=Data_PreProcessing_validation(seq_length=seq_max_len)# this section drops all the padded outputs and all the padded labels
   # valid_data,valid_label,valid_seqlen,valid_mask=validatonSet.get_validation_set()# This have to be done seperatly for the AUC, done with a mask for the
    #prob_in,target_in=sess.run(forAuc, feed_dict={x: valid_data, y: valid_label,seqlen: valid_seqlen})# Accuracy and the cost, have to dubble check these however
    #temp_list_target_new=[]
    #temp_list_prob_new=[]
    #for index,seeeqlen in enumerate(test_seqlen):
     #   step=0
     #   while step < seeeeqlen:
     #       temp_list_prob_new.append(prob_in[index,step,:])
     #       temp_list_target_new.append(target_in[index,step,:])
     #       step=step+1
    #print("AUC?")
    #prob=np.asarray(temp_list_prob_new)
    #target=np.asarray(temp_list_target_new)
    #print("The auc score is {}".format(roc_auc_score(y_true=target, y_score=prob)))
    #embed()
    #print(test_label)
