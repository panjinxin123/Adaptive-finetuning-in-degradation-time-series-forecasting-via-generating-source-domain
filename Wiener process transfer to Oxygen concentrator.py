# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:03:40 2022

@author: pjx
"""
# -*- coding: utf-8 -*-


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Dropout
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics import mean_squared_error



test_num=500

time_step=10

def generate_sinusoid_batch():
    # Note train arg is not used (but it is used for omniglot method.
    # input_idx is used during qualitative testing --the number of examples used for the grad update
    
    gen_data=[1]
    
    while gen_data[-1]>0:
        
        gen_data.append(gen_data[-1]+np.random.uniform(-0.0005, 0.0004))
        
    else:
        return np.array(gen_data)


test_size=20

def evaluate_data():
    gen_data=[1]

    while gen_data[-1]>0:

        gen_data.append(gen_data[-1]+np.random.uniform(-0.05, 0.04))

    else:
        return np.array(gen_data)
    

def load_data(data):
    
    

    data_x,data_y=[],[]  #训练集

    for i in range(len(data)-time_step-1):

        x=data[i:i+time_step]

        y=data[i+time_step+1]

        data_x.append(x)

        data_y.append(y)
        
    return np.array(data_x),np.array(data_y)






model = Sequential()
model.add(LSTM(input_dim=1, output_dim=1, return_sequences=True))
#model.add(Dropout(1))
model.add(LSTM(100, return_sequences=False))
#model.add(Dropout(1))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mse', optimizer='rmsprop')
for i in range (test_num):
    train_data=generate_sinusoid_batch()
    train_x,train_y=load_data(train_data)
    
    train_x = np.reshape(train_x, (train_x.shape[0],time_step, 1))
    
    lenth=20#len(train_y)
    
    lower=np.random.randint(0,len(train_y)-lenth)
    
    #lower=0

    model.fit(train_x[lower:lower+lenth,:,:], train_y[lower:lower+lenth], batch_size=10, nb_epoch=2)#validation_split=0.1

              
not_trainable_num=93

num=0

for layer in model.layers:
    
    if num<not_trainable_num:
    
        layer.trainable=False
    num+=1
    
pd.set_option('max_colwidth',-1)
layers=[(layer,layer.name,layer.trainable) for layer in model.layers]
a=pd.DataFrame(layers,columns=['Layer Type','Layer Name','Layer Trainable'])
#print("Trainable Layers:",model.trainable_weights)
#print(a)

test_data=np.array(np.loadtxt('D:\\Oxygen concentrator.csv',delimiter=','))
test_data=(test_data-5)/20
test_x,test_y=load_data(test_data)
test_x= np.reshape(test_x, (test_x.shape[0],time_step, 1))

model2 = Sequential()
model2.add(LSTM(input_dim=1, output_dim=1, return_sequences=True))
#model.add(Dropout(1))
model2.add(LSTM(100, return_sequences=False))
#model.add(Dropout(1))
model2.add(Dense(output_dim=1))
model2.add(Activation('linear'))
model2.summary()

model2.compile(loss='mse', optimizer='rmsprop')

interval_size=200

err_sum_1=[]
err_sum_2=[]

for i in range (interval_size):
    predict_y = model.predict(test_x)
    predict_y_2 = model2.predict(test_x)
    
    
    err_1=mean_squared_error(predict_y,test_y)
    err_2=mean_squared_error(predict_y_2,test_y)
    err_sum_1.append(err_1)
    err_sum_2.append(err_2)
    
    
    fig2 = plt.figure(2)
    plt.xlabel('x')  # y轴标题
    plt.ylabel('y') 
    plt.plot(test_y, 'g', label='true y')
    plt.plot(predict_y, '#FF5200', label='prediction')
    plt.plot(predict_y_2, 'b', label='prediction')
    plt.show()
    
    lenth=20#len(test_y)
    
    lower=np.random.randint(0,len(test_y)-lenth)
    
    #lower=0

    #model.fit(test_x[lower:lower+lenth,:,:],test_y[lower:lower+lenth], batch_size=10, nb_epoch=1, validation_split=0.1)
    #model2.fit(test_x[lower:lower+lenth,:,:],test_y[lower:lower+lenth], batch_size=10, nb_epoch=1, validation_split=0.1)
    model.fit(test_x[i%6::6,:,:],test_y[i%6::6], batch_size=10, nb_epoch=1, validation_split=0.1)
    model2.fit(test_x[i%6::6,:,:],test_y[i%6::6], batch_size=10, nb_epoch=1, validation_split=0.1)
#Trainable layers:[]


predict_y = model.predict(test_x)

#predict_y = np.reshape(predict_y, (predict_y.size,))

#predict_y=predict_y*std+mean
#test_y=test_y*std+mean
#predict_y = scaler.inverse_transform([[i] for i in predict_y])
#test_y = scaler.inverse_transform(test_y)
fig2 = plt.figure(2)
plt.plot(test_y, 'g', label='true y')
plt.plot(predict_y, '#FF5200', label='prediction')
plt.legend(loc='upper right')  # 图例位置
plt.xlabel('time point')  # y轴标题
plt.ylabel('value') 
plt.title('LSTM')
#plt.savefig('D:\\辛烷预测LSTM.png', dpi=1000)
plt.show()
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
