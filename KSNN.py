from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
	return (1/(1+np.exp(-x)))
def derivative(y):
	return y * (1 - y)

min=1

df= pd.read_csv("Trial.csv")
msk = np.random.rand(len(df)) < 0.7
train = df[msk]
test= df[~msk]
y= train["subject"]
B = np.reshape(y, (-1, 1))
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(B)

x = train.iloc[:,3:].as_matrix()
print(x)

total_iterations = 2000
learning_rate = 0.01
hidden_layer_1_neurons = 15
output_neurons = train["subject"].nunique()

weight_from_0_to_1 = np.random.uniform(size=(x.shape[1],hidden_layer_1_neurons)) #wh
bias_for_0_layer = np.random.uniform(size=(1,hidden_layer_1_neurons)) #bh

weight_from_1_to_2 = np.random.uniform(size=(hidden_layer_1_neurons,output_neurons)) #wout
bias_for_1_layer = np.random.uniform(size=(1,output_neurons)) #bout

print ("Training....")
for i in range(total_iterations):

	# forward propogation
	layer_1 = np.dot(x,weight_from_0_to_1)
	layer_1_with_bias = layer_1 + bias_for_0_layer
	layer_1_final = sigmoid(layer_1_with_bias)
	output_layer = np.dot(layer_1_final,weight_from_1_to_2)
	output_layer_with_bias = output_layer + bias_for_1_layer
	output_layer_final = sigmoid(output_layer_with_bias)


	output_layer_error = output_layer_final.dot(np.mean(y - output_layer_final))

	if 	np.mean(np.abs(output_layer_error))<min:
		min=np.mean(np.abs(output_layer_error))
		print("Error rate: "+str(min))
	else:
	 	exit()

	# backpropogration
	slope_output_layer = derivative(output_layer_final)
	slope_layer_1 = derivative(layer_1_final)
	derivative_output_layer = output_layer_error * slope_output_layer
	layer_1_error = derivative_output_layer.dot(weight_from_1_to_2.T)
	derivative_layer_1 = layer_1_error * slope_layer_1
	weight_from_0_to_1 += x.T.dot(derivative_layer_1) * learning_rate
	# here axis = 0 is first axis of improved neurons which consists bias weights
	bias_for_0_layer += np.sum(derivative_layer_1,axis=0,keepdims=True) * learning_rate
	weight_from_1_to_2 += layer_1_final.T.dot(derivative_output_layer) * learning_rate
	bias_for_1_layer += np.sum(derivative_output_layer, axis=0, keepdims=True) * learning_rate
