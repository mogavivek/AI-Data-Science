from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Data Preparation
Data = [[[(i+j)/100] for i in range(5)] for j in range(100)] # If we devide by 100 then it will give us better result, we can also compare this value without deviding by 100
target = [(i+5)/100 for i in range(100)]

# Checking the shape of above values
data = np.array(Data, dtype=float)
target = np.array(target, dtype=float)

print(data.shape)
print(target.shape)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.2, random_state=4)

# RNN model
model = Sequential()
# LSTM(((output size), shape of input = (number of input-here we do not know hence used None, length of our I/P, length of each vecotr)
# at last True means it will return every output after number, Flase means it will return the value at last node
model.add(LSTM((1), batch_input_shape=(None,5,1), return_sequences=True))
model.add(LSTM((1), return_sequences=False))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# If we show the summary of above two lines model.summary()
history = model.fit(x_train, y_train, epochs=50,validation_data=(x_test, y_test))
# above epochs use to converge our result, we can also use this as 400, 500 as per the requirement

results = model.predict(x_test)
plt.scatter(range(20),results,c='r') # r means red = given by us
plt.scatter(range(20),y_test,c='g') # g means green = expected result
plt.show()

# checking the loss
plt.plot(history.history['loss'])
plt.show()