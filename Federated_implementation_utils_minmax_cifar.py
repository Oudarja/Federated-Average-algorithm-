#!/usr/bin/env python
# coding: utf-8

# 
# ## 1)Python OS module provides the facility to establish the interaction between the user and the operating system. It offers many useful OS functions that are used to perform OS-based tasks and get related information about operating system.The OS comes under Python's standard utility modules.The OS module in Python provides functions for creating and removing a directory (folder), fetching its contents, changing and identifying the current directory, etc.Os module need to be imported  to interact with the underlying operating system.
# 
# ## 2)Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.The pickle module can transform a complex object into a byte stream and it can transform the byte stream into an object with the same internal structure.
# ## Pickle is a standard Python library for serializing and deserializing Python objects to and from binary data streams. This enables you to save complex data structures like lists, dictionaries, and custom classes to a file or send them over the network to another program.
# 
# ## 3)Scikit-learn's LabelBinarizer converts input labels into binary labels, each example belongs to a single class or not. Scikit-learn's MultiLabelBinarizer converts input labels into multilabel labels, each example can belong to multiple classes
# 
# ## 4) shuffle: Shuffle arrays or sparse matrices in a consistent way.
# ## 5) Python Utils is a collection of small Python functions and classes which make common           patterns shorter and easier
# 
# ## 6) A CNN can be instantiated as a Sequential model because each layer has exactly one       input and output and is stacked together to form the entire network.The core idea of Sequential API is simply arranging the Keras layers in a sequential order and so, it is called Sequential API. Most of the ANN also has layers in sequential order and the data flows from one layer to another layer in the given order until the data finally reaches the output layer.
# 
# ## 7) Tensors are multi-dimensional arrays with a uniform type (called a dtype ). You can see all supported dtypes at tf.dtypes.DType . If you're familiar with NumPy, tensors are (kind of) like np.arrays
# 
# ## 8) Max pooling: Max Pooling is a pooling operation that calculates the maximum value for patches of a feature map, and uses it to create a downsampled (pooled) feature map. It is usually used after a convolutional layer.
# 
# ## 9) Tensorflow flatten is the function available in the tensorflow library and reduces the input data into a single dimension instead of 2 dimensions. While doing so, it does not affect the batch size.
# ## 10)Sharding is a way of horizontally partitioning your data by storing different rows of the same table in multiple tables across multiple databases.
# 
# 
# 
# 

# In[36]:


import numpy as np
import random
import os
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from collections import Counter
import itertools
from collections import defaultdict

# In[48]:


def load(paths):
    data = list()
    rawData = pickle.load(open(paths, 'rb'))
    minmaxscaler = MinMaxScaler()
    # Reshape data to 2D array for scaling
    flattenedData = rawData.reshape(rawData.shape[0],-1)
    minmaxscaler.fit(flattenedData)
    # Transform flattened data
    transformedData = minmaxscaler.transform(flattenedData)
    # Reshape transformed data back to original shape
    data = np.reshape(transformedData, rawData.shape)
    return data
  
        
# In[38]:

def create_clients_IID(data_list, label_list, num_clients, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of datas and label lists.
        args: 
            data_list: a list of numpy arrays of training data
            label_list:a list of binarized labels for each data
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    #randomize the data
    data = list(zip(data_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))
    #Here a dictionary has been returned shard[i] has data set which has been taken after shufling
    return {client_names[i] : shards[i] for i in range(len(client_names))}

# In this modified function, we calculate the mean number of samples per client and use it as the lambda parameter for a Poisson distribution to sample the number of samples for each client. We also define a range of values to sample from for the Poisson distribution and use it to bound the number of samples per client. Finally, we assign data samples to each client based on the number of samples we sampled for them.
def create_clients_NONIID(data_list, label_list, num_clients, initial='clients'):
    # Create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    # Randomize the data
    data = list(zip(data_list, label_list))
    random.shuffle(data)
    # Shard data and assign to each client
    shards = [[] for _ in range(num_clients)]
    # Calculate the mean number of samples per client
    mean_samples_per_client = len(data) // num_clients
    # Sample the distribution parameters for the Dirichlet distribution
    alpha = np.random.dirichlet([1] * num_clients)
    # Distribute data based on the Dirichlet distribution
    for i in range(num_clients):
        # Calculate the number of samples for the current client
        num_samples = int(alpha[i] * mean_samples_per_client)
        # Assign data samples to the current client
        shards[i] = data[:num_samples]
        data = data[num_samples:]
    # Convert shards to tuples
    shards = [tuple(x) for x in shards]
    # Create the dictionary of client data shards
    client_data = {client_names[i]: shards[i] for i in range(num_clients)}
    return client_data



def batch_data(data_shard, bs):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


# In[40]:

# In[41]:


def weight_scalling_factor(clients_trn_data, client_name,client_names):
    #get the bs batch size
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    #tf.data.experimental.cardinality counts the data point 
    #here for each client this is summed up
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    # Nk/N (Number of data point of k'th client/ total number of data point)
    return local_count/global_count


# In[42]:


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    #Nk/N*WK[0] Nk/N * Wk[1]........
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    #scaled model weights list
    return weight_final


# In[43]:


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    #avg_grad[i]is averaged weight for i'th feature for gloabl model
    #avg_grad[0]=W1[0]+W2[0].....+Wk[0], avg_grad[1]=w1[1]+w2[1]+.....wk[1]
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad



#This function is for testing the Global model for each communication round
def test_model(X_test, Y_test,  model, comm_round):
    '''
    from_logits = True signifies the values of the loss obtained by the model are not normalized and is
    basically used when we don't have any softmax function in our model.
    '''
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    #print(logits)
    loss = cce(Y_test, logits)
    acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc,loss
