"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import sys

from IPython import embed
from tensorflow.examples.tutorials.mnist import input_data
from DataWindowTrain import DataWindowTrain
from DataWindowValid import DataWindowValid
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from sklearn.metrics import roc_auc_score 


#FLAGS = None
#STATIONARY_PARAMTERS=20 # This should be age, gender ans so on later!
#NUM_FEATURES=11
#WINDOW_SIZE=20
#N_CLASSES=2
#iterations=300
#learning_rate=0.01
#batch_size=100000

#dataGenerator=DataWindow(windowSize=WINDOW_SIZE)

def main(_):
  # The hyperparamters are defined. 
  FLAGS = None
  STATIONARY_PARAMTERS=20 # This should be age, gender ans so on later!
  NUM_FEATURES=11
  WINDOW_SIZE=20
  N_CLASSES=2
  iterations=600
  learning_rate=0.01
  batch_size=100000
  dataGenerator=DataWindowTrain(windowSize=WINDOW_SIZE)
  dataGeneratorValid=DataWindowValid(windowSize=WINDOW_SIZE)
  display=int((len(dataGenerator.data_out)/batch_size)+1)
  #embed()
  x = tf.placeholder(tf.float32, [None, (NUM_FEATURES+STATIONARY_PARAMTERS)*WINDOW_SIZE])
  y_ = tf.placeholder(tf.float32, [None, N_CLASSES])
  is_training_pl = tf.placeholder(tf.bool, name="is_training_pl")
  # The different layers and the model is defined. Some of the hyperparamters are also set in the layers 
  pre=batch_norm(x,decay=0.9,is_training=is_training_pl,)
  # Used to be taken away
  l1 = fully_connected(x, num_outputs=26, activation_fn=relu,normalizer_fn=batch_norm, scope="l1") 
  l1_dropout = dropout(l1,keep_prob=0.90, is_training=is_training_pl, scope="l1_dropout")
  # used to be taken away
  l2=fully_connected(l1_dropout, num_outputs=105, activation_fn=relu,normalizer_fn=batch_norm, scope="l2")
  # Used to be taken away
  l2_dropout = dropout(l2,keep_prob=0.95 ,is_training=is_training_pl, scope="l2_dropout")
  y=fully_connected(l2_dropout, N_CLASSES, activation_fn=softmax, scope="y")

  # Define loss and optimizer

  cross_entropy= -tf.reduce_sum(y_ * tf.log(y+1e-10), reduction_indices=[1])
  loss=tf.reduce_mean(cross_entropy)
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # The tensorflow sessions is defined. The model is set to run 40 threads in parallel to utalize more of the servers 
  sess = tf.InteractiveSession(config=tf.ConfigProto(
  intra_op_parallelism_threads=40))
  # Initialize the variables that are defined earlier. 
  tf.global_variables_initializer().run()

  # Train
  # Defined list to store values during evaluation. 
  trainLossList=[]
  trainAccList=[]
  trainAucList=[]
  testLossList=[]
  testAccList=[]
  testAucList=[]
  validLossList=[]
  validAccList=[]
  validAucList=[]  
  epoch_nbr=0
  temp_loss=[]
  temp_train_acc=[]
  temp_test_acc=[]
  for _ in range(iterations): # the model are trained. 
    batch_xs,batch_ys = dataGenerator.nextWindowBatch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, is_training_pl:True})
    #embed()
    if _% display==0: # When the model have been trained for a epoch the model is evaluated. 
      epoch_nbr=epoch_nbr+1 
      x_train=dataGenerator.data_out
      y_train=dataGenerator.label_out
      # The train loss,auc and acc is evaluated. 
      lossTrain=sess.run(loss, feed_dict={x: x_train, y_: y_train, is_training_pl:False})
      accTrain=sess.run(accuracy, feed_dict={x: x_train, y_: y_train , is_training_pl:False})
      pred=sess.run(y,feed_dict={x: x_train,y_: y_train, is_training_pl:False})
      aucTrain=roc_auc_score(y_true=np.asarray(y_train),y_score=np.asarray(pred))
      print("Epoch number {}".format(epoch_nbr))
      print("The traning  loss is: {0:.8f} ".format(lossTrain)) 
      print("The training acc is {0:.8f}".format(accTrain*100))
      print("The traning auc is: {0:.8f} ".format(aucTrain))
      trainLossList.append(lossTrain)
      trainAccList.append(accTrain)
      trainAucList.append(aucTrain)
      x_test,y_test=dataGenerator.testWindowBatch()
      # The test loss, auc and acc is evaluated
      lossTest=sess.run(loss, feed_dict={x: x_test, y_: y_test, is_training_pl:False})
      accTest=sess.run(accuracy, feed_dict={x: x_test, y_: y_test , is_training_pl:False})
      predTest=sess.run(y,feed_dict={x: x_test,y_: y_test, is_training_pl:False})
      aucTest=roc_auc_score(y_true=np.asarray(y_test),y_score=np.asarray(predTest))

      print("The test loss is: {0:.8f} ".format(lossTest))
      print("The test acc is {0:.8f}".format(accTest*100))
      print("The test auc is: {0:.8f} ".format(aucTest))
      testLossList.append(lossTest)
      testAccList.append(accTest)
      testAucList.append(aucTest)
      x_valid,y_valid=dataGeneratorValid.testWindowBatch()
      # The validation loss,acc and auc is evaluated. 
      lossValid=sess.run(loss, feed_dict={x: x_valid, y_: y_valid, is_training_pl:False})
      accValid=sess.run(accuracy, feed_dict={x: x_valid, y_: y_valid , is_training_pl:False})
      predValid=sess.run(y,feed_dict={x: x_valid,y_: y_valid, is_training_pl:False})
      aucValid=roc_auc_score(y_true=np.asarray(y_valid),y_score=np.asarray(predValid))
      
      print("The valid loss is: {0:.8f} ".format(lossValid))
      print("The valid acc is {0:.8f}".format(accValid*100))
      print("The valid auc is: {0:.8f} ".format(aucValid))
      validLossList.append(lossValid)
      validAccList.append(accValid)
      validAucList.append(aucValid)

      #print("The test accuracy is {0:.2f} %".format(100*sess.run(accuracy, feed_dict={x: x_test,y_: y_test, is_training_pl:False})))
      #embed()
      #pred=sess.run(y,feed_dict={x: x_test,y_: y_test, is_training_pl:False})
  
  # I wont to have the graphs, wont to have the paramters ad also new roc plot
  print("I wont to have the graphs, wont to have the paramters ad also new roc plot")
  embed()# Test trained model
  x_test,y_test=dataGenerator.testWindowBatch()
  print("The final test accuracy is {0:.2f} %".format(100*sess.run(accuracy, feed_dict={x: x_test,y_: y_test, is_training_pl:False})))
###


# The code below generates a ROC curve using the scikit learn libary. 


#from sklearn.metrics import roc_curve, auc  
#import matplotlib as mpl  
#
#fpr=dict() 
#tpr=dict() 
#roc_auc=dict() 
#lw=2 
#
#mpl.use('Agg') 
#import matplotlib.pyplot as plt
#
#
#for i in range(2):                                                                               
#	 fpr[i], tpr[i], _ = roc_curve(np.array(y_test)[:, i], pred[:, i])
#	 roc_auc[i] = auc(fpr[i], tpr[i])     
#plt.plot(fpr[0], tpr[0], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate') 
# plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right") 
#plt.savefig('blablabla.png')  

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]]) 
