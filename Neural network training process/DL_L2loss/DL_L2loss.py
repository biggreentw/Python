
# coding: utf-8

# ## Import the library

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


# ## layer definition (Need to do!!!)

# In[3]:


def InnerProduct_For(x,W,b):
    y=np.dot(x,W)+b
    return y

def InnerProduct_Back(dEdy,x,W,b):
    dEdx=np.dot(dEdy,np.transpose(W))
    dEdW=np.dot(np.transpose(x),dEdy)
    dEdb=np.dot(np.ones((1,dEdy.shape[0])),dEdy)
    return dEdx,dEdW,dEdb

def Softmax_For(x):
    sum=np.sum(np.exp(x),axis=1)
    softmax=np.exp(x)
    for q in range(x.shape[1]):
        softmax[:,q]/sum
    return softmax

def L2Loss_BackProp(y,t):
    dEdx=y-t
    return dEdx

def Sigmoid_For(x):
    y=1/(1+np.exp(-x))
    return y

def Sigmoid_Back(dEdy,x):
    dEdx=dEdy*(np.exp(-x)/((1+np.exp(-x))**2))
    return dEdx

def ReLu_For(x):
     return y

def ReLu_Back(dEdy,x):
    return dEdx

def loss_For(y,y_pred):
    loss=np.sum((y-y_pred)**2)
    return loss


# ## Setup the Parameters and Variables (Can tune that!!!)

# In[4]:


eta = 0.00001       #learning rate
Data_num = 784      #size of input data   (inputlayer)
W1_num = 15         #size of first neural (1st hidden layer)
Out_num = 10        #size of output data  (output layer)
iteration = 500         #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)
valid_num = int(image_num*0.2)


# ## Setup the Data (Create weight array here!!!)

# In[5]:


w_1= (np.random.normal(0,0.1,Data_num*W1_num)).reshape(Data_num,W1_num)
w_out  = (np.random.normal(0,0.1,W1_num*Out_num)).reshape(W1_num, Out_num)
b_1, b_out = randn(1,W1_num),randn(1,Out_num)
print("w1 shape:", w_1.shape)
print("w_out shape:", w_out.shape)
print("b_1 shape:", b_1.shape)
print("b_out shape:", b_out.shape)
#print(w_1)


# ## Prepare all the data

# ### Load the training data and labels from files

# In[6]:


df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.values
print("Training data:",fmnist_train_images.shape[0])
print("Training data shape:",fmnist_train_images.shape)


df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.values
print("Testing data:",fmnist_test_images.shape[0])
print("Testing data shape:",fmnist_test_images.shape)

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.values
print("Training labels shape:",fmnist_train_label.shape)


# ### Show the 100 testing images

# In[7]:

"""
plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1,)
    plt.imshow(image)
plt.show() 
"""
# ### Convert the training labels data type to one hot type

# In[8]:


label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)
print("Training labels shape:",train_labels_onehot.shape)
#print(label_temp)

# ### Separate train_images, train_labels into training and validating 

# In[13]:


train_data_img = np.copy(fmnist_train_images[:train_num,:])
train_data_lab = np.copy(train_labels_onehot[:train_num,:])
valid_data_img = np.copy(fmnist_train_images[train_num:,:])
valid_data_lab = np.copy(train_labels_onehot[train_num:,:])
#print(train_data_img)


# Normalize the input data between (0,1)
train_data_img = train_data_img/255.
valid_data_img = valid_data_img/255.
test_data_img = fmnist_test_images/255.

print("Train images shape:",train_data_img.shape)
print("Train labels shape:",train_data_lab.shape)
print("Valid images shape:",valid_data_img.shape)
print("Valid labels shape:",valid_data_lab.shape)
print("Test  images shape:",test_data_img.shape)


# In[12]:
iteration=100
valid_accuracy = []
Loss = []

for i in range(iteration):
    z_1=InnerProduct_For(train_data_img,w_1,b_1)
    z_2=InnerProduct_For(Sigmoid_For(z_1),w_out,b_out)
    L= loss_For(train_data_lab,z_2)
    print("Loss:",L)
    Loss.append(L)
    dEdz_2=L2Loss_BackProp(z_2,train_data_lab)
    dEdz_1_ac,dEdw_out,dEdb_out=InnerProduct_Back(dEdz_2,Sigmoid_For(z_1),w_out,b_out)
    dEdz_1= Sigmoid_Back(dEdz_1_ac,z_1)
    dEdx,dEdw_1,dEdb_1=InnerProduct_Back(dEdz_1,train_data_img,w_1,b_1)
    
    # Parameters Updating (Gradient descent)    
    w_1=np.copy(w_1-eta*dEdw_1)
    w_out=np.copy(w_out-eta*dEdw_out)
    b_1=np.copy(b_1-eta*dEdb_1)
    b_out=np.copy(b_out-eta*dEdb_out)
    
    # Do cross-validation to evaluate model
    v_1=InnerProduct_For(valid_data_img,w_1,b_1)
    v_2=InnerProduct_For(Sigmoid_For(v_1),w_out,b_out)
    
    
    #Calculate the accuracy
    a=0.0000
    for iii in range(valid_data_lab.shape[0]):
        if np.argmax(v_2[iii,:])==np.argmax(valid_data_lab[iii,:]):              # Compare the Prediction and validation
            a=a+1.00
    A=a/valid_data_lab.shape[0]*100
    print(" ","accuracy:", "%.2f" %A)
    valid_accuracy.append(A)
    
   
    
   
  
    
   
    


# ## Testing Stage
# In[10]:
t_1=InnerProduct_For(test_data_img,w_1,b_1)
test_Out_data=InnerProduct_For(Sigmoid_For(t_1),w_out,b_out)


# ### Convert results to csv file (Input the (10000,10) result array!!!)
# In[12]:


# Convert "test_Out_data" (shape: 10000,10) to "test_Prediction" (shape: 10000,1)
test_Prediction      = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("DL_LAB1_prediction_ID.csv",index=True, index_label="index")


# ## Convert results to csv file

# In[16]:


accuracy = np.array(valid_accuracy)
plt.subplot(1,2,1)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')


loss = np.array(Loss)
plt.subplot(1,2,2)
plt.plot(loss, label="$iter-loss$")
plt.plot(loss, label="$loss-accuracy$")
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()

