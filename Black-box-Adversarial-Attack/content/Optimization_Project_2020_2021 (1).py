#!/usr/bin/env python
# coding: utf-8

# Federico Zanotti (ID: 2007716) - Lorenzo Corrado (ID: 2020623)
# 
# # Zeroth Order Methods for Adversarial Machine Learning
# ### Optimization for Data Science Project
# 
# September 16, 2021
# 

# ## Import and Data preparation
# 

# In[ ]:


# Import the libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.python.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.models import Model
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm
import random
import gzip
import urllib.request
import time
import warnings
import h5py
from prettytable import PrettyTable
import torch
import gc

# Remove warnings in output
warnings.filterwarnings("ignore")

# Set seed
np.random.seed([2021])


# In[ ]:


# Download MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data('./.keras/datasets/mnist/mnist.npz')
(x_train, y_train), (x_test, y_test) = mnist.load_data('C:\\Users\\Acer\\Desktop\\朱亚杰-论文审稿\\小论文新增实验\\小论文新增实验\\content\\.keras\\datasets\\mnist\\mnist.npz')

# Model/data parameters
num_classes = 10
input_shape = (784,)

# Scale images in [0,1] range
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# Make sure images have shape (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

# Splitting training set in training and valid
validation_size = 5000

x_val = x_train[:validation_size]
y_val = y_train[:validation_size]
x_train = x_train[validation_size:]
y_train = y_train[validation_size:]

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


# Define DNN architecture
model = keras.Sequential()
model.add(tf.keras.Input(shape=input_shape))
model.add(tf.keras.layers.Reshape((28,28,1), input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(200, activation="relu"))
model.add(tf.keras.layers.Dense(200, activation="relu"))
model.add(tf.keras.layers.Dense(num_classes, name="last_dense"))
model.add(tf.keras.layers.Activation("softmax"))


# # Take the output of the last layer before the softmax operation
dnn = Model(inputs=model.input, outputs=model.get_layer("last_dense").output)



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


## Loads the weights
path = "C:\\Users\\Acer\\Desktop\\朱亚杰-论文审稿\\小论文新增实验\\小论文新增实验\\content\\content\\mnist"
# path = "./content/mnist"
if os.path.isfile(path):
  model.load_weights(path) 

  # Evaluate the model on the test set
  score = model.evaluate(x_test, y_test, verbose=0)
  print("Test loss:", score[0])
  print("Test accuracy: {} %".format(score[1]*100))
else:
  print(f"file {path} does not exist")


# ## Utility Function


# Definition of the objective function to be minimized #(FOR ALL THE SAMPLES)


def F(x, y_true):
  """
  Loss function for all the examples

  Input:
  - x: images
  - y_true: true labels of the images

  """
  f = dnn.predict(x)   # (100,10)
  f_yi = np.max(f*y_true, axis=1)  # (100)   真实分类值
  f_j = np.max(f*np.where(y_true == 1, -1e10, 1), axis=1)  # (100)  获取非真实分类最大值
  
  return np.mean(np.where(f_yi - f_j > 0, f_yi - f_j, 0))


def F_order(x,y_true):   

  x = tf.Variable(x)
  
  with tf.GradientTape as g:
    g.watch(x)

    
    f = dnn.predict(x)
    f_yi = (f * y_true).sum(dim=1)
    mask = torch.where(y_true == 1, -1e10, 1)
    f_j = (f * mask).max(dim=1).values

    # 计算损失
    loss = torch.mean(torch.where(f_yi - f_j > 0, f_yi - f_j, 0))

  return loss



def F_Par(x, y_true):
  """
  Loss function for only one example

  Input:
  - x: image
  - y: true label of the image

  """
  f = dnn.predict(x)
  f_yi = np.max(f*y_true, axis=1)
  f_j = np.max(f*np.where(y_true == 1, -1e10, 1), axis=1)

  return np.where(f_yi-f_j > 0, f_yi-f_j, 0)



# Extract n images from the same class
def extract_images(n, c):
  """
  Extract some images of the same class

  Input:
  - n: number of images to extract
  - c: label

  """
  x_extr = np.copy(x_test[y_test.argmax(axis=1)==c][:n])
  y_extr = np.copy(y_test[y_test.argmax(axis=1)==c][:n])
  
  return x_extr, y_extr



# Set the number of examples in the same class
def get_data(n, c):
  """
  return x, x_ori, y_true_in

  """
  img_in, y_true_in = extract_images(n,c)
  x_ori = np.copy(img_in)
  x = np.copy(x_ori)
  return x, x_ori, y_true_in 



def stop_attack(x, y_true):
  success = dnn.predict(x).argmax(axis=1)
  # print("Label predicted:", success)
  return sum(success==y_true.argmax(axis=1))==0


## Zeroth-order gradient estimator

def RandGradEst(x, y_true, v, d):
  """
  Two-point (gaussian) random gradient estimator

  Input:
  - x: image
  - y_true: true label of the image
  - v: smoothing parameter 
  """
  u = np.random.standard_normal((1,d))
  F_plus = F(x + v*u, y_true)
  F_ = F(x, y_true)
  
  return (d/v)*(F_plus - F_)*u

def Avg_RandGradEst(x, y_true, q, v, d):
  """
  Averaged (gaussian) random gradient estimator

  Input:
  - x: image
  - y_true: true label of the image
  - q: number of random directions
  - v: smoothing parameter
  """
  g = 0
  u = np.random.standard_normal((q,d))
  F_ = F(x, y_true)
  for j in range(q):
    F_plus = F(x + v*u[j], y_true)
    g = g + (F_plus - F_)*u[j]

  return (d/(v*q))*g

def CoordGradEst(x, y_true, mu, d):
  """
  Coordinate-wise gradient estimator
  
  Input:
  - x: images
  - y_true: true labels of the images
  - mu: smoothing parameter

  """
  q = 0
  for j in tqdm(range(d)):
    F_plus = F(x + mu*e(j,d), y_true)
    F_minus = F(x - mu*e(j,d), y_true)
    diff = F_plus - F_minus
    q = q + (diff)*e(j,d)
   
  return q/(2*mu)



## 一阶梯度估计器
@tf.function
def auto_grad(x, y_true):
 
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
    # path = "C:\\Users\\Acer\\Desktop\\朱亚杰-论文审稿\\小论文新增实验\\小论文新增实验\\content\\content\\mnist"
    with tf.GradientTape() as tape:
      tape.watch(x_tensor)
      f = dnn(x_tensor)
    # compute loss
      f_yi = tf.reduce_max(tf.multiply(f, y_true_tensor), axis=1)
      f_j = tf.reduce_max(tf.multiply(f, (1 - y_true_tensor)), axis=1)
      loss = tf.where(f_yi - f_j > 0, f_yi - f_j, [0.0])  # 这里使用了0.0而不是tf.Variable([0.0])
    # compute gradient
    grad = tape.gradient(loss, x_tensor)

  
    
    return grad  #




def attack_success_rate(x, y):
  predicted=softmax(dnn.predict(x)).argmax(axis=1)
  # print(predicted)
  true_values=y.argmax(axis=1)
  adversarial=len(predicted)-sum(predicted==true_values)
  wrong_label=(adversarial/len(predicted))*100
  return round(wrong_label,1)


'''
def plot_all(loss_ZSCG, loss_FZFW, loss_FZCGS, epochs, n, savefig=''):
  plt.figure(figsize=(10,8))
  plt.plot(loss_ZSCG, label=f"ZSCG with {n} examples")
  plt.plot(loss_FZFW, label=f"FZFW with {n} examples")
  plt.plot(loss_FZCGS, label=f"FZCGS with {n} examples")
  plt.grid("on")
  plt.legend()
  plt.xticks(np.arange(0,epochs+10,10))
  plt.xlabel("# iterations")
  plt.ylabel("loss")
  if savefig != '':
    plt.savefig(savefig)
  plt.show()
'''


# In[ ]:

'''
def nice_table(x_mod, y_true, param1, param2):
  param1_list = param1[1]
  param2_list = param2[1]
  j=0
  i=0
  t= PrettyTable([param1[0], param2[0], "ASR (%)"])
  for el in x_mod:
    asr=attack_success_rate(el, y_true)
    if i==len(param2_list):
      i=0
      p2 = param2_list[i]
      j +=1
      p1=param1_list[j]
    else:
      p2 = param2_list[i]
      p1=param1_list[j]
    i +=1
    
    t.add_row([f" {p1}", f" {p2}", asr])
  return t
'''

# ## Algorithm 1. Zeroth-Order Stochastic Conditional Gradient Method (ZSCG)
# 
# K. Balasubramanian et al., 2018.

# In[ ]:


def Avg_RandGradEst_Par(x, y_true, q, v, d):
  """
  Averaged (gaussian) random gradient estimator in parallel

  Input:
  - x: image
  - y_true: true label of the image
  - q: number of random directions
  - v: smoothing parameter
  """
  g = 0
  u = np.random.standard_normal((q,d))
  x_par_plus = np.array([x + v*u[j] for j in range(q)]).reshape((q,d))
  diff = F_Par(x_par_plus, y_true) - F(x, y_true)

  for j in range(q):
    g = g + (diff[j]/v) * u[j] 
  
  return (d/(q*v))*g


# In[ ]:


def ZSCG(N, d, s, m_k, x, y_true_in,v=-1,alpha=-1, B=1,verbose=True, clip=False):
  if v==-1:
    v = np.sqrt(2/(N*(d+3)**3))
  if alpha==-1:
    alpha = 1/np.sqrt(N)

  x_ori=np.copy(x)
  loss_ZSCG = []
  perturbations = []
  loss_ZSCG.append(F(x, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x-x_ori)))
  for k in range(N):
      
    # Get the gradient estimate
    v_k = 0
    for i in tqdm(range(x.shape[0]), disable= not verbose):
      # v_k = v_k + RandGradEst(x[i:i+1], y_true_in[i:i+1], v)
      # v_k = v_k + Avg_RandGradEst(x[i:i+1], y_true_in[i:i+1], 30, v)
      v_k = v_k + Avg_RandGradEst_Par(x[i:i+1], y_true_in[i:i+1], m_k, v, d)    
    v_k = (1/n)*v_k
    
    x_k = -s * np.sign(v_k) + x_ori # Solve the LMO
    x = (1 - alpha)*x+ alpha*x_k
    if clip:
      x= x_ori + np.clip((x-x_ori), 0, 1)
    perturbations.append(x)
    loss_ZSCG.append(F(x, y_true_in))
    if verbose:
      print("-"*100)
      print("Epoch:", k+1, "Loss:", loss_ZSCG[k], "Distortion:", np.round(np.max(np.abs(x-x_ori)),5), "Elapsed Time:")
      filename = './results/ZSCG/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k+1))
        f.write('{}\n'.format(loss_ZSCG[k]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_ZSCG, x
  x = np.clip(x, 0, 1)

  ZSCG_x_perturbated = x

  print("ZSCG Final loss = ", loss_ZSCG[-1])
  return loss_ZSCG, ZSCG_x_perturbated, perturbations


# ## Algorithm 2. Faster Zeroth-Order Frank-Wolfe Method (FZFW)
# Gao et al., 2020.

def e(i, d):
  """
  Orthogonal basis vector

  Input:
  - i: index
  - d: dimensions
  """
  
  ei = np.zeros(d)
  ei[i] = 1
  return ei



def CoordGradEst_Par(x, y_true, mu,d):
  """
  Coordinate-wise gradient estimator in parallel

  Input:
  - x: image
  - y_true: true label of the image
  - mu: smoothing parameter
  """
  
  x_par_plus = np.array([x + mu*e(j,d) for j in range(d)]).reshape(d,d)
  x_par_minus = np.array([x - mu*e(j,d) for j in range(d)]).reshape(d,d)
  diff = F_Par(x_par_plus, y_true) - F_Par(x_par_minus, y_true)
  
  q = 0
  for j in range(d):
    q = q + (diff[j])*(e(j,d)) 
    
  return (1/(2*mu))*q


def FZFW(K,d,n,s,gamma, mu,x,y_true_in, verbose=True, clip=False): 
  s_1=n
  q = s_2 = int(np.sqrt(n))
  if gamma==-1:
    gamma = 1/np.sqrt(K)
  if mu==-1:
    mu = 1/np.sqrt(d*K)
  x_ori=np.copy(x)


  loss_FZFW = []
  perturbations=[]
  loss_FZFW.append(F(x_ori, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x-x_ori)))
    print("-"*100)


  for k in range(K):
    if (k % q == 0):
      

      ########################################################
      ####### Get the gradient estimate with S1>S2 samples #######
      v_k = 0
      for i in tqdm(range(s_1), disable= not verbose):
        v_k = v_k + CoordGradEst_Par(x[i:i+1], y_true_in[i:i+1], mu,d)
      v_k=v_k/s_1
      v_k_1 = v_k
      ########################################################

    else:

      ########################################################
      ##### Get the gradient estimate with S2<S1 samples #####

      v_k = 0
      s2_idx = np.random.randint(0, n, s_2) 
      b = 15
      for idx in tqdm(s2_idx,  disable= not verbose):
        v_k = v_k + CoordGradEst_Par(x[idx:idx+1], y_true_in[idx:idx+1], mu,d) - CoordGradEst_Par(x_k_1[idx:idx+1], y_true_in[idx:idx+1], mu,d) + v_k_1
      v_k = (1/b) * v_k
      v_k_1 = v_k

      ########################################################

    #########################################
    ############# Update x ##################

    x_k_1 = np.copy(x)
    u_k = -s * np.sign(v_k) + x_ori # Solve the LMO
    d_k = u_k - x
    x = x + gamma*d_k

    #########################################    

    if clip:
      x= x_ori + np.clip((x-x_ori), 0, 1)
    perturbations.append(x)

    loss_FZFW.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k+1, "Loss:", loss_FZFW[k+1], "Distortion:", np.round(np.max(np.abs(x-x_ori)),5))
      print("-"*100)
      filename = './results/FZFW/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k+1))
        f.write('{}\n'.format(loss_FZFW[k+1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_FZFW, x
    

 
  FZFW_x_perturbated = x

  print("FZFW Final loss = ", loss_FZFW[-1])

  return loss_FZFW, FZFW_x_perturbated, perturbations


# ## Algorithm 3. Zeroth-Order Adaptive Frank-Wolfe Method (ZO-AdaSFW)
# Zhu et al., 2023.

def AdaSFW(K, T,d, n, s, gamma, x, y_true_in, verbose=True, clip=False):

  # K 是迭代次数
  # T 是内部迭代次数
  # d 是参数维度

  s_1 = n  # 批次
  q = 1
  s_2 = int(np.sqrt(n))
  esp = 1e-8
  lr = 0.99

  if gamma == -1:
    gamma = 1 / np.sqrt(K)
  
  x_ori = np.copy(x)
  loss_AdaSFW = []
  perturbations = []
  loss_AdaSFW.append(F(x_ori, y_true_in))   # 损失
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  accumulator = 0
  for k in range(K):
    #######################################################
      ####### Get the gradient estimate with S1>S2 samples #######
    v_k = 0
    for i in tqdm(range(s_1), disable=not verbose):
      v_k = v_k + auto_grad(x[i:i + 1], y_true_in[i:i + 1]) 
      v_k = v_k / s_1
      v_k_1 = v_k
      ########################################################

   

      ########################################################

    #########################################
    ############# Update x ##################
    x_k_1 = np.copy(x)
    accumulator += np.vdot(v_k, v_k)
    H2 = esp + np.sqrt(accumulator)
    H = np.diag(np.full(d, H2))
    z = x
    for t in range(T):
      deltaQ = v_k + (1 / lr) * np.dot(z - x, H)
      v_k = deltaQ
      v = -s * np.sign(v_k) + x_ori
      vz = v - z
      gamma_k = min(-lr * np.vdot(v_k, vz) / np.vdot(vz, np.dot(vz, H)), gamma)
      z = z + gamma_k * vz
    x = z

    # x_k_1 = np.copy(x)
    # u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    # d_k = u_k - x
    # x = x + gamma * d_k

    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_AdaSFW.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_AdaSFW[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = './results/AdaSFW/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_AdaSFW[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_AdaSFW, x

  ZO_AdaSFW_x_perturbated = x

  print("AdaSFW Final loss = ", loss_AdaSFW[-1])

  return loss_AdaSFW


def ZO_AdaSFW(K, T, d, n, s, gamma, mu, x, y_true_in, verbose=True, clip=False):
  s_1 = n
  q = 1
  s_2 = int(np.sqrt(n))
  esp = 1e-8
  lr = 0.99
  if gamma == -1:
    gamma = 1 / np.sqrt(K)
  if mu == -1:
    mu = 1 / np.sqrt(d * K)
  x_ori = np.copy(x)

  loss_ZO_AdaSFW = []
  perturbations = []
  loss_ZO_AdaSFW.append(F(x_ori, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  accumulator = 0
  for k in range(K):
    if (k % q == 0):

      ########################################################
      ####### Get the gradient estimate with S1>S2 samples #######
      v_k = 0
      for i in tqdm(range(s_1), disable=not verbose):
        v_k = v_k + CoordGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], mu, d)
      v_k = v_k / s_1
      v_k_1 = v_k
      ########################################################

    else:

      ########################################################
      ##### Get the gradient estimate with S2<S1 samples #####

      v_k = 0
      b_t = 1
      s2_idx = np.random.randint(0, n, s_2)

      for idx in tqdm(s2_idx, disable=not verbose):
        v_k = v_k + CoordGradEst_Par(x[idx:idx + 1], y_true_in[idx:idx + 1], mu, d) - CoordGradEst_Par(
          x_k_1[idx:idx + 1], y_true_in[idx:idx + 1], mu, d)
      v_k = (1 / b_t) * v_k + v_k_1
      v_k_1 = v_k

      ########################################################

    #########################################
    ############# Update x ##################
    x_k_1 = np.copy(x)
    accumulator += np.vdot(v_k, v_k)
    H2 = esp + np.sqrt(accumulator)
    H = np.diag(np.full(d, H2))
    z = x
    for t in range(T):
      deltaQ = v_k + (1 / lr) * np.dot(z - x, H)
      v_k = deltaQ
      v = -s * np.sign(v_k) + x_ori
      vz = v - z
      gamma_k = min(-lr * np.vdot(v_k, vz) / np.vdot(vz, np.dot(vz, H)), gamma)
      z = z + gamma_k * vz
    x = z

    # x_k_1 = np.copy(x)
    # u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    # d_k = u_k - x
    # x = x + gamma * d_k

    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_ZO_AdaSFW.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_ZO_AdaSFW[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = 'C:\\Users\\Acer\\Desktop\\朱亚杰-论文审稿\\小论文新增实验\\小论文新增实验\\content/results/ZO-AdaSFW/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_ZO_AdaSFW[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      # return loss_ZO_AdaSFW, x

  ZO_AdaSFW_x_perturbated = x

  print("ZO_AdaSFW Final loss = ", loss_ZO_AdaSFW[-1])

  return loss_ZO_AdaSFW, ZO_AdaSFW_x_perturbated, perturbations


# ## Algorithm 4. Zeroth-Order Stochastic Frank-Wolfe Method (ZO-SFW)
# Sahu et al., 2019.
def ZO_SFW(N, d, s, m_k, x, y_true_in, v=-1, alpha=-1, B=1, verbose=True, clip=False):
  if v == -1:
    v = np.sqrt(2 / (N * (d + 3) ** 3))
  if alpha == -1:
    alpha = 1 / np.sqrt(N)

  x_ori = np.copy(x)
  loss_ZO_SFW = []
  perturbations = []
  loss_ZO_SFW.append(F(x, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
  for k in range(N):

    # Get the gradient estimate
    v_k = 0
    beta = 4 / (1 + d / N) ** (1 / 3) / (k + 8) ** (2 / 3)
    for i in tqdm(range(x.shape[0]), disable=not verbose):
      # v_k = v_k + RandGradEst(x[i:i+1], y_true_in[i:i+1], v)
      # v_k = v_k + Avg_RandGradEst(x[i:i+1], y_true_in[i:i+1], 30, v)
      v_k = v_k + Avg_RandGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], m_k, v, d)
    v_k = (1 / n) * v_k

    if k ==0:
      m = v_k
    m = (1 - beta) * m + beta * v_k
    v_k = m

    x_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    x = (1 - alpha) * x + alpha * x_k
    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)
    loss_ZO_SFW.append(F(x, y_true_in))
    if verbose:
      print("-" * 100)
      print("Epoch:", k + 1, "Loss:", loss_ZO_SFW[k], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5),
            "Elapsed Time:")
      filename = './results/ZO-SFW/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_ZO_SFW[k]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_ZO_SFW, x
  x = np.clip(x, 0, 1)

  ZO_SFW_x_perturbated = x

  print("ZO_SFW Final loss = ", loss_ZO_SFW[-1])
  return loss_ZO_SFW, ZO_SFW_x_perturbated, perturbations

# ## Algorithm 5. Accelerated Stochastic Zeroth-Order Frank-Wolfe Algorithm(Acc-SZOFW(unige))
# Huang et al., 2020.
def Acc_SZOFW_Uni(K, n, d, s, m_k, x, y_true_in, v=-1, B=1, verbose=True, clip=False):
  s_1 = n
  q = 1
  s_2 = int(np.sqrt(n))
  x_ori = np.copy(x)
  a = np.copy(x)
  # b = np.copy(x)
  eta = 1 / np.sqrt(K)

  loss_Acc_SZOFW_Uni = []
  perturbations = []
  loss_Acc_SZOFW_Uni.append(F(x_ori, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  for k in range(K):
    gamma = (1 + (1 / (k + 1) / (k + 2))) * eta
    alpha = 1 / (k + 1)
    if (k % q == 0):

      ########################################################
      ####### Get the gradient estimate with S1>S2 samples #######
      v_k = 0
      for i in tqdm(range(s_1), disable=not verbose):
        v_k = v_k + Avg_RandGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], m_k, v, d)
      v_k = v_k / s_1
      v_k_1 = v_k
      ########################################################

    else:

      ########################################################
      ##### Get the gradient estimate with S2<S1 samples #####

      v_k = 0
      s2_idx = np.random.randint(0, n, s_2)
      b_k = 6
      for idx in tqdm(s2_idx, disable=not verbose):
        v_k = v_k + Avg_RandGradEst_Par(x[idx:idx + 1], y_true_in[idx:idx + 1], m_k, v, d) - Avg_RandGradEst_Par(
          x_k_1[idx:idx + 1], y_true_in[idx:idx + 1], m_k, v, d) + v_k_1
      v_k = (1 / b_k) * v_k
      v_k_1 = v_k

      ########################################################

    #########################################
    ############# Update x ##################

    x_k_1 = np.copy(x)
    u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    a = a + gamma * (u_k - a)
    b = x + eta * (u_k - x)
    x = (1 - alpha) * b + alpha * a


    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_Acc_SZOFW_Uni.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_Acc_SZOFW_Uni[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = './results/Acc-SZOFW(UniGe)/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_Acc_SZOFW_Uni[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_Acc_SZOFW_Uni, x

  Acc_SZOFW_Uni_x_perturbated = x

  print("Acc_SZOFW_Uni Final loss = ", loss_Acc_SZOFW_Uni[-1])

  return loss_Acc_SZOFW_Uni, Acc_SZOFW_Uni_x_perturbated, perturbations

# ## Algorithm 6. Accelerated Stochastic Zeroth-Order Frank-Wolfe Algorithm(Acc-SZOFW(cooge))
# Huang et al., 2020.
def Acc_SZOFW_Coo(K, n, d, s, mu, x, y_true_in, v=-1, B=1, verbose=True, clip=False):
  s_1 = n
  q = 30
  s_2 = int(np.sqrt(n))
  x_ori = np.copy(x)
  if mu == -1:
    mu = 1 / np.sqrt(d * K)
  a = np.copy(x)
  # b = np.copy(x)
  eta = 1 / np.sqrt(K)

  loss_Acc_SZOFW_Coo = []
  perturbations = []
  loss_Acc_SZOFW_Coo.append(F(x_ori, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  for k in range(K):
    gamma = (1 + (1 / (k + 1) / (k + 2))) * eta
    alpha = 1 / (k + 1)
    if (k % q == 0):

      ########################################################
      ####### Get the gradient estimate with S1>S2 samples #######
      v_k = 0
      for i in tqdm(range(s_1), disable=not verbose):
        v_k = v_k + CoordGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], mu, d)
      v_k = v_k / s_1
      v_k_1 = v_k
      ########################################################

    else:

      ########################################################
      ##### Get the gradient estimate with S2<S1 samples #####

      v_k = 0
      s2_idx = np.random.randint(0, n, s_2)
      b_k = 15
      for idx in tqdm(s2_idx, disable=not verbose):
        v_k = v_k + CoordGradEst_Par(x[idx:idx + 1], y_true_in[idx:idx + 1], mu, d) - CoordGradEst_Par(
          x_k_1[idx:idx + 1], y_true_in[idx:idx + 1], mu, d)
      v_k = (1 / b_k) * v_k + v_k_1
      v_k_1 = v_k

      ########################################################

    #########################################
    ############# Update x ##################

    x_k_1 = np.copy(x)
    u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    a = a + gamma * (u_k - a)
    b = x + eta * (u_k - x)
    x = (1 - alpha) * b + alpha * a


    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_Acc_SZOFW_Coo.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_Acc_SZOFW_Coo[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = './results/Acc-SZOFW(CooGe)/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_Acc_SZOFW_Coo[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_Acc_SZOFW_Coo, x

  Acc_SZOFW_Coo_x_perturbated = x

  print("Acc_SZOFW_Coo Final loss = ", loss_Acc_SZOFW_Coo[-1])

  return loss_Acc_SZOFW_Coo, Acc_SZOFW_Coo_x_perturbated, perturbations

# ## Algorithm 7. Accelerated Stochastic Zeroth-Order Frank-Wolfe Algorithm(Acc-SZOFW*(UniGe))
# Huang et al., 2020.
def Acc_SZOFWX_Uni(K, n, d, s, m_k, x, y_true_in, v=-1, B=1, verbose=True, clip=False):
  s_1 = n
  q = 1
  s_2 = int(np.sqrt(n))
  x_ori = np.copy(x)
  a = np.copy(x)
  # b = np.copy(x)
  eta = 1 / (K)**(2/3)

  loss_Acc_SZOFWX_Uni = []
  perturbations = []
  loss_Acc_SZOFWX_Uni.append(F(x_ori, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  for k in range(K):
    gamma = (1 + (1 / (k + 1) / (k + 2))) * eta
    alpha = 1 / (k + 1)
    rho = 1 / (1 + k)**(2/3)
    if (k % q == 0):

      ########################################################
      ####### Get the gradient estimate with S1>S2 samples #######
      v_k = 0
      for i in tqdm(range(s_1), disable=not verbose):
        v_k = v_k + Avg_RandGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], m_k, v, d)
      v_k = v_k / s_1
      v_k_1 = v_k
      ########################################################

    else:

      ########################################################
      ##### Get the gradient estimate with S2<S1 samples #####

      v_k = 0
      s2_idx = np.random.randint(0, n, s_2)
      b_k = 6
      for idx in tqdm(s2_idx, disable=not verbose):
        v_k = v_k + Avg_RandGradEst_Par(x[idx:idx + 1], y_true_in[idx:idx + 1], m_k, v, d) + (1 - rho) * (v_k_1 - Avg_RandGradEst_Par(
          x_k_1[idx:idx + 1], y_true_in[idx:idx + 1], m_k, v, d))
      v_k = (1 / b_k) * v_k
      v_k_1 = v_k

      ########################################################

    #########################################
    ############# Update x ##################

    x_k_1 = np.copy(x)
    u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    a = a + gamma * (u_k - a)
    b = x + eta * (u_k - x)
    x = (1 - alpha) * b + alpha * a


    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_Acc_SZOFWX_Uni.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_Acc_SZOFWX_Uni[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = './results/Acc-SZOFWX(UniGe)/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_Acc_SZOFWX_Uni[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_Acc_SZOFWX_Uni, x

  Acc_SZOFWX_Uni_x_perturbated = x

  print("Acc_SZOFWX_Uni Final loss = ", loss_Acc_SZOFWX_Uni[-1])

  return loss_Acc_SZOFWX_Uni, Acc_SZOFWX_Uni_x_perturbated, perturbations


# ## Algorithm 8. Accelerated Stochastic Zeroth-Order Frank-Wolfe Algorithm(Acc-SZOFW*(CooGe))
# Huang et al., 2020.
def Acc_SZOFWX_Coo(K, n, d, s, mu, x, y_true_in, v=-1, B=1, verbose=True, clip=False):
  s_1 = n
  q = 6
  s_2 = int(np.sqrt(n))
  x_ori = np.copy(x)
  if mu == -1:
    mu = 1 / np.sqrt(d) * (K)**(4/5)
  a = np.copy(x)
  # b = np.copy(x)
  eta = 1 / (K)**(4/5)

  loss_Acc_SZOFWX_Coo = []
  perturbations = []
  loss_Acc_SZOFWX_Coo.append(F(x_ori, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  for k in range(K):
    gamma = (1 + (1 / (k + 1) / (k + 2))) * eta
    alpha = 1 / (k + 1)
    rho = 1 / (1 + k)**(2/3)
    if (k % q == 0):

      ########################################################
      ####### Get the gradient estimate with S1>S2 samples #######
      v_k = 0
      for i in tqdm(range(s_1), disable=not verbose):
        v_k = v_k + CoordGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], mu, d)
      v_k = v_k / s_1
      v_k_1 = v_k
      ########################################################

    else:

      ########################################################
      ##### Get the gradient estimate with S2<S1 samples #####

      v_k = 0
      s2_idx = np.random.randint(0, n, s_2)
      b_k = 6
      for idx in tqdm(s2_idx, disable=not verbose):
        v_k = v_k + CoordGradEst_Par(x[idx:idx + 1], y_true_in[idx:idx + 1], mu, d) + (1 - rho) * (v_k_1 - CoordGradEst_Par(
          x_k_1[idx:idx + 1], y_true_in[idx:idx + 1], mu, d))
      v_k = (1 / b_k) * v_k
      v_k_1 = v_k

      ########################################################

    #########################################
    ############# Update x ##################

    x_k_1 = np.copy(x)
    u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    a = a + gamma * (u_k - a)
    b = x + eta * (u_k - x)
    x = (1 - alpha) * b + alpha * a


    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_Acc_SZOFWX_Coo.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_Acc_SZOFWX_Coo[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = './results/Acc-SZOFWX(CooGe)/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_Acc_SZOFWX_Coo[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_Acc_SZOFWX_Coo, x

  Acc_SZOFWX_Coo_x_perturbated = x

  print("Acc_SZOFWX_Coo Final loss = ", loss_Acc_SZOFWX_Coo[-1])

  return loss_Acc_SZOFWX_Coo, Acc_SZOFWX_Coo_x_perturbated, perturbations

# ## Algorithm 9. Mini-batch Stochastic Gradient-Free Frank-Wolfe Method (SFW-Grad)
# Guo et al., 2022.
def SFW_Grad(N, d, s, m_k, x, y_true_in, v=-1, alpha=-1, B=1, verbose=True, clip=False):

  x_ori = np.copy(x)
  loss_SFW_Grad = []
  perturbations = []
  loss_SFW_Grad.append(F(x, y_true_in))
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
  for k in range(N):
    v = 2 * np.sqrt(N) / d ** 1.5 / (k + 8) ** (1 / 3)
    alpha = 1 / (k + 2)
    # Get the gradient estimate
    v_k = 0
    for i in tqdm(range(x.shape[0]), disable=not verbose):
      # v_k = v_k + RandGradEst(x[i:i+1], y_true_in[i:i+1], v)
      # v_k = v_k + Avg_RandGradEst(x[i:i+1], y_true_in[i:i+1], 30, v)
      v_k = v_k + Avg_RandGradEst_Par(x[i:i + 1], y_true_in[i:i + 1], m_k, v, d)
    v_k = (1 / n) * v_k

    x_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    x = (1 - alpha) * x + alpha * x_k
    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)
    loss_SFW_Grad.append(F(x, y_true_in))
    if verbose:
      print("-" * 100)
      print("Epoch:", k + 1, "Loss:", loss_SFW_Grad[k], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5),
            "Elapsed Time:")
      filename = './results/SFW-Grad/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_SFW_Grad[k]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_SFW_Grad, x
  x = np.clip(x, 0, 1)

  SFW_Grad_x_perturbated = x

  print("SFW_Grad Final loss = ", loss_SFW_Grad[-1])
  return loss_SFW_Grad, SFW_Grad_x_perturbated, perturbations


# ## Algorithm 10. Zeroth-Order Adaptive Frank-Wolfe Method (ZO-AdaSFW)
# Zhu et al., 2023.

def AdaSFW(K, T,d, n, s, gamma, x, y_true_in, verbose=True, clip=False,sess=None):

  # K 是迭代次数
  # T 是内部迭代次数
  # d 是参数维度

  s_1 = n  # 批次
  q = 1
  s_2 = int(np.sqrt(n))
  esp = 1e-8
  lr = 0.99

  if gamma == -1:
    gamma = 1 / np.sqrt(K)
  
  x_ori = np.copy(x)
  loss_AdaSFW = []
  perturbations = []
  loss_AdaSFW.append(F(x_ori, y_true_in))   # 损失
  if verbose:
    print("Epoch:", 0, "Loss:", F(x_ori, y_true_in), "Distortion:", np.max(np.abs(x - x_ori)))
    print("-" * 100)

  accumulator = 0
  for k in range(K):
    #######################################################
      ####### Get the gradient estimate with S1>S2 samples #######
    v_k = 0
    for i in tqdm(range(s_1), disable=not verbose):
      grad = sess.run(auto_grad(x[i:i + 1], y_true_in[i:i + 1])) 
      v_k = v_k + grad
   
      
    v_k = v_k/s_1
    x_k_1 = np.copy(x)
    accumulator += np.vdot(v_k, v_k)
    H2 = esp + np.sqrt(accumulator)
    H = np.diag(np.full(d, H2))
    z = x
    for t in range(T):
      deltaQ = v_k + (1 / lr) * np.dot(z - x, H)
      v_k = deltaQ
      v = -s * np.sign(v_k) + x_ori
      vz = v - z
      gamma_k = min(-lr * np.vdot(v_k, vz) / np.vdot(vz, np.dot(vz, H)), gamma)
      z = z + gamma_k * vz
    x = z.astype(np.float32)

    # x_k_1 = np.copy(x)
    # u_k = -s * np.sign(v_k) + x_ori  # Solve the LMO
    # d_k = u_k - x
    # x = x + gamma * d_k

    #########################################

    if clip:
      x = x_ori + np.clip((x - x_ori), 0, 1)
    perturbations.append(x)

    loss_AdaSFW.append(F(x, y_true_in))
    if verbose:
      print("Epoch:", k + 1, "Loss:", loss_AdaSFW[k + 1], "Distortion:", np.round(np.max(np.abs(x - x_ori)), 5))
      print("-" * 100)
      filename = 'C:\\Users\\Acer\\Desktop\\朱亚杰-论文审稿\\小论文新增实验\\小论文新增实验\\content\\results/AdaSFW/data.txt'
      with open(filename, 'a') as f:
        f.write('{},'.format(k + 1))
        f.write('{}\n'.format(loss_AdaSFW[k + 1]))
    if stop_attack(x, y_true_in):
      print("Attack successful! stopping computation...")
      return loss_AdaSFW, x
    gc.collect()
  ZO_AdaSFW_x_perturbated = x

  print("AdaSFW Final loss = ", loss_AdaSFW[-1])

  return loss_AdaSFW




## Experiments


n=36
x, _, y_true_in = get_data(n, 4)
epochs=50

## Select algorithm to run

# loss_Z, x_Z,_=ZSCG(epochs, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)
# loss_FZ, x_FZ,_=FZFW(epochs, 784,n, 0.1,10**-1,10**-3 ,x, y_true_in, verbose=True)
loss_ZOAda, ZOAda,_=ZO_AdaSFW(epochs, 2, 784,n, 0.1,10**-1,10**-3 ,x, y_true_in, verbose=True)
# loss_ZOSFW, x_Z,_=ZO_SFW(epochs, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)
# loss_Acc_SZOFW_U, x_Z,_=Acc_SZOFW_Uni(epochs, n, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)
# loss_Acc_SZOFW_C, x_Z,_=Acc_SZOFW_Coo(epochs, n, 784, 0.1, -1, x, y_true_in,10**-5,10**-1, verbose=True)
# loss_Acc_SZOFWX_U, x_Z,_=Acc_SZOFWX_Uni(epochs, n, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)
# loss_Acc_SZOFWX_C, x_Z,_=Acc_SZOFWX_Coo(epochs, n, 784, 0.1, -1, x, y_true_in,10**-5,10**-1, verbose=True)
# loss_SFW_Grad, x_Z,_=SFW_Grad(epochs, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)
""" 
# AdaSFW
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化变量
    dnn.load_weights(path)
    loss_AdaSFW = AdaSFW(epochs, 2,784,n, 0.1,10**-1 ,x, y_true_in, verbose=True,sess=sess)
"""


