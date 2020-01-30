import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from google.colab import files
uploaded=files.upload()
df=pd.read_csv('sonar.csv')
dataset=df.values
X=dataset[:0,60]
y=dataset[:,60]
encoder=LabelEncoder()
encoder.fit(y)
y=encoder.transform(y)
Y = OneHotEncoder(y)

#Define the encoder function
def OneHotEncoder(labels):
  n_labels= len(labels)
  n_unique_labels=len(np.unique(labels))
  OneHotEncoder=np.zeros((n_labels,n_unique_labels))
  OneHotEncoder[np.arrange(n_labels),labels]=1
  return OneHotEncoder

  X,Y = read_dataset()
  X,Y = shuffle(X,Y,random_state=1)
  train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415 )
  print(train_x.shape)
  print(train_y.shape)
  print(test.shape)
  learning_rate = 0.3
  training_epochs=1000
  cost_history = np.empty(shape=[1], dtype=float)
  n_dim=X.shape[1]
  print("n_dim",n_dim)
  n_class=2

  n_hidden1=60
  n_hidden2=60
  n_hidden_3=60
  n_hidden_4=60
  x=tf.placeholder(tf.float32,[None,n_dim])
  W=tf.Variable(tf.zeros([n_dim, n_class]))
  b=tf.Variable(tf.zeros([ n_class]))
  y_ = tf.placeholder(tf.float32,[None,n_class])

  #Define the model
  def multilayer_perceptron (x,weights,biases):
    #Hidden layer with RELU activation function
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.sigmoid(layer_1)
    #Hidden Layer with sigmoid activastion function
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.sigmoid(layer_2)
    #Hidden Layer with sigmoid activastion function
    layer_3=tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3=tf.nn.sigmoid(layer_3)
    #Hidden Layer with relu function
    layer_4=tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
    layer_4=tf.nn.relu(layer_4)
    #output layer with linear activation
    out_layer=tf.matmul(layer_4,weights['out'])+biases['out']
    return out_layer
    #Define weights and biases
    weights={
        'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
        'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
        'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
        'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
        'out':tf.Variable(tf.truncated_normal([n_hidden_4,n_class])),

    }
    biases={
        'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out':tf.Variable(tf.truncated_normal([n_class])),
    }
    #Initialise all the variables
    init =tf.global_variables_initializer()
    saver=tf.train.Saver()
    y=multilayer_perceptron(x,weights,biases)
    cost_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logists=y,labels=y_))
    training_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    sess=tf.Session()
    sess.run(init)

    mse_history=[]
    accuracy_history=[]
    for epoch in range(training_epochs):
      sess.run(training_step, feed_dict={x:train_x,y:train_y})
      cost_history=np.append(cost_history,cost)
      correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
      accuracy=tf.reduce_mean(tf.cast.(correct_prediction,tf.float32))
      pred_y=sess.run(y, feed_dict={x:test_x})
      mse=tf.reduce_mean(tf.square(pred_y-test_y))

