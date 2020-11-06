#!/usr/bin/env python
# coding: utf-8

# In[2]:


try:

    import tensorflow as tf
    import cv2
    import os
    import pickle
    import numpy as np
    print("Library Loaded Successfully ..........")
except:
    print("Library not Found ! ")


class MasterImage(object):

    def __init__(self,PATH='', IMAGE_SIZE = 50):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        # This will get List of categories
        self.list_categories = []

    def get_categories(self):
        for path in os.listdir(self.PATH):
            if '.DS_Store' in path:
                pass
            else:
                self.list_categories.append(path)
        print("Found Categories ",self.list_categories,'\n')
        return self.list_categories

    def Process_Image(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:        # if any image is corrupted
                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE)                 # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize,class_index])
                    except:
                        pass

            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_data.append(x[0])        # Get the X_Data
                self.y_data.append(x[1])        # get the label

            X_Data = np.asarray(self.x_data) / (255.0)      # Normalize Data
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

            return X_Data, Y_Data
        except:
            print("Failed to run Function Process Image ")

    def pickle_image(self):

        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        X_Data,Y_Data = self.Process_Image()

        # Write the Entire Data into a Pickle File
        pickle_out = open('X_Data','wb')
        pickle.dump(X_Data, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('Y_Data', 'wb')
        pickle.dump(Y_Data, pickle_out)
        pickle_out.close()

        print("Pickled Image Successfully ")
        return X_Data,Y_Data

    def load_dataset(self):

        try:
            # Read the Data from Pickle Object
            X_Temp = open('X_Data','rb')
            X_Data = pickle.load(X_Temp)

            Y_Temp = open('Y_Data','rb')
            Y_Data = pickle.load(Y_Temp)

            print('Reading Dataset from PIckle Object')

            return X_Data,Y_Data

        except:
            print('Could not Found Pickle File ')
            print('Loading File and Dataset  ..........')

            X_Data,Y_Data = self.pickle_image()
            return X_Data,Y_Data


if __name__ == "__main__":
    path = '/home/jesterrexx/Documents/Dataset/RPS/Rock-Paper-Scissors/train'
    a = MasterImage(PATH=path,
                    IMAGE_SIZE=80)

    X_Data,Y_Data = a.load_dataset()
    print(X_Data.shape)


# In[3]:


def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    cache=Z
    return A,cache


# In[4]:


def sigmoid_backward(dA,cache):
    Z=cache
    s=1/(1 + np.exp(-Z))
    dZ=dA*s*(1-s)
    assert (dZ.shape==Z.shape)
    return dZ
    


# In[5]:


def relu(Z):
    A=np.maximum(0,Z)
    assert (A.shape==Z.shape)
    cache=Z
    return A,cache


# In[6]:


def relu_backward(dA,cache):
    Z=cache
    dZ=np.array(dA,copy=True)
    dZ[Z<=0]=0
    assert (dZ.shape==Z.shape)
    return dZ


# In[ ]:





# In[ ]:





# In[7]:


print('X_Data: ',X_Data.shape)
print('Y_Data: ',Y_Data.shape)


# In[26]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y, = train_test_split(X_Data, Y_Data, test_size=0.2, random_state=42)


# In[27]:


# Explore your dataset 
m_train = train_x.shape[0]
m_test = test_x.shape[0]
num_px = train_x.shape[1]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))


# In[33]:


# Reshape the training and test examples 
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


print('--------------------------------------------------')

 
print ("train_set_x_flatten shape: " + str(train_x_flatten.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_set_x_flatten shape: " + str(test_x_flatten.shape))
print ("test_y shape: " + str(test_y.shape))
print ("sanity check after reshaping: " + str(train_x_flatten[0:5,0]))


# In[35]:


print('train_x shape: ',train_x.shape)
print('train_y shape: ',train_y.shape)
print('test_x shape: ',test_x.shape)
print('test_y shape: ',test_y.shape)


# # Architecture of the MODEL 

# In[36]:


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    


# In[37]:


parameters = initialize_parameters(2,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# # Forward Prop

# In[38]:


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# In[39]:


import numpy as np
def linear_forward_test_case():
    np.random.seed(1)
    A=np.random.randn(3,2)
    W=np.random.randn(1,3)
    b=np.random.randn(1,1)
    return A,W,b


# In[40]:


A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))


# In[41]:


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# In[42]:


def linear_activation_forward_test_case():
    np.random.seed(2)
    A_prev=np.random.randn(3,2)
    W=np.random.randn(1,3)
    b=np.random.randn(1,1)
    return A_prev,W,b


# In[43]:


A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))


# In[44]:


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A 

        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches


# In[45]:


def L_model_forward_test_case():
    np.random.seed(1)
    X=np.random.randn(4,2)
    W1=np.random.randn(3,4)
    b1=np.random.randn(3,1)
    W2=np.random.randn(1,3)
    b2=np.random.randn(1,1)
    parameters={'W1':W1,
                'b1':b1,
                'W2':W2,
                'b2':b2}
    return X, parameters


# In[46]:


X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# In[47]:


def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# In[48]:


def compute_cost_test_case():
    Y=np.array([[1,1,1]])
    aL=np.array([[.8,.9,0.4]])
    return Y, aL


# In[49]:


Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))


# # Backward prop
# 

# In[50]:


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]


    dW = np.dot(dZ, cache[0].T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (isinstance(db, float))
    
    return dA_prev, dW, db


# In[51]:


def linear_backward_test_case():
    np.random.seed(1)
    dZ=np.random.randn(1,2)
    A=np.random.randn(3,2)
    W=np.random.randn(1,3)
    b=np.random.randn(1,1)
    linear_cache=(A,W,b)
    return dZ, linear_cache


# In[52]:


dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# In[53]:


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[54]:


def linear_activation_backward_test_case():
    np.random.seed(2)
    dA=np.random.randn(1,2)
    A=np.random.randn(3,2)
    W=np.random.randn(1,3)
    b=np.random.randn(1,1)
    Z=np.random.randn(1,2)
    linear_cache=(A,W,b)
    activation_cache=Z
    linear_activation_cache=(linear_cache,activation_cache)
    return dA, linear_activation_cache


# In[55]:


AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# In[ ]:





# In[ ]:





# In[96]:


def L_model_backward(AL, Y, caches):
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    
    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    dA_prev,dW,db=linear_activation_backward(dAL, caches[L-1],'sigmoid')
    grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)]=dA_prev,dW,db
    
    for l in reversed(range(L-1)):
        dA=dA_prev
        dA_prev,dW,db=linear_activation_backward(dA,caches[l],'relu')
        grads['dA'+str(l+1)]=dA_prev
        grads['dW'+str(l+1)]=dW
        grads['db'+str(l+1)]=db
    return grads
    


# In[97]:


def L_model_backward_test_case():
    np.random.seed(3)
    AL=np.random.rand(1,2)
    Y=np.array([[1,0]])
    
    A1=np.random.randn(4,2)
    W1=np.random.randn(3,4)
    b1=np.random.randn(3,1)
    Z1=np.random.randn(3,2)
    linear_cache_activation_1=((A1,W1,b1),Z1)
    
    A2=np.random.randn(3,2)
    W2=np.random.randn(1,3)
    b2=np.random.randn(1,1)
    Z2=np.random.randn(1,2)
    linear_cache_activation_2=((A2,W2,b2),Z2)
     
    caches=(linear_cache_activation_1, linear_cache_activation_2)
    
    return AL, Y, caches
    
    


# In[98]:


AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL,Y_assess, caches)
print_grads(grads)


# In[99]:


# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache =caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],  current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# In[ ]:





# In[100]:


AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)


# In[ ]:





# In[76]:


# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters


# In[77]:


def update_parameters_test_case():
    np.random.seed(2)
    W1=np.random.randn(3,4)
    b1=np.random.randn(3,1)
    W2=np.random.randn(1,3)
    b2=np.random.randn(1,1)
    parameters={'W1':W1,
                'b1':b1,
                'W2':W2,
                'b2':b2}
    
    np.random.seed(3)
    dW1=np.random.randn(3,4)
    db1=np.random.randn(3,1)
    dW2=np.random.randn(1,3)
    db2=np.random.randn(1,1)
    grads={'dW1':dW1,
          'db1':db1,
           'dW2':dW2,
           'db2':db2}
    
    return parameters, grads


# In[78]:


parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = " + str(parameters["W1"]))
print ("b1 = " + str(parameters["b1"]))
print ("W2 = " + str(parameters["W2"]))
print ("b2 = " + str(parameters["b2"]))


# In[88]:


n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


# In[89]:


# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):

    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[90]:


parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


# In[ ]:




