import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv1D  # Convolution Operation
from keras.layers import MaxPooling1D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras import backend as K
from keras.utils import plot_model
from keras import  layers

#traning data
u0_ls = []
x0_ls = []
ux0_ls = []
y0_ls = []

# testing data
u_ls = []
x_ls = []
ux_ls = []
y_ls = []

#traning
x0 = 1/2
x0_ls.append(x0)
u0 = 0
u0_ls.append(u0)
y0_ls.append(x0/(1+x0**2) + u0**3)
x0 = 1/2
for k in range(1,1001):
    u = np.random.uniform(-2.5,2.5)
    x = y0_ls[k-1]
    x0_ls.append(x)
    u0_ls.append(u)
    y0_ls.append(x/(1+x**2) + u**3)
print(len(y0_ls))
ux0_ls = np.vstack((x0_ls,u0_ls)).transpose()
ux0_ls = ux0_ls[:, :, np.newaxis]
print(ux0_ls.shape)


#testing
x0 = 1/2
x_ls.append(x0)
u0 = 0
u_ls.append(u0)
y_ls.append(x0/(1+x0**2) + u0**3)
x = 1/2
for k in range(1,101):
    u = np.sin(2*np.pi*k/25) + np.sin(2*np.pi*k/10)
    x = y_ls[k-1]
    x_ls.append(x)
    u_ls.append(u)
    y_ls.append(x/(1+x**2) + u**3)
print(len(y_ls))
ux_ls = np.vstack((x_ls,u_ls)).transpose()
ux_ls = ux_ls[:, :, np.newaxis]
print(ux_ls.shape)


# initializing CNN
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding = 'same', activation='relu', input_shape = (2,1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=64, kernel_size=2, padding = 'same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=128, kernel_size=2, padding = 'same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

loss_ls = []
def RMSE(y_true, y_pred):
    loss = K.sqrt(K.mean(K.square(y_pred - y_true)))
    return loss
model.compile(loss=RMSE, optimizer='sgd')
plot_model(model,to_file='model.png',show_shapes=True,dpi=300)


fig = plt.figure()
print('----------------Training----------------------')
lin = np.linspace(0, 1000, 1001)
plt.plot(lin, y0_ls)
history = model.fit(ux0_ls, y0_ls, epochs=60, batch_size=15)
weight_updated0 = model.layers[0].get_weights()[0]
# weight_updated1 = model.layers[1].get_weights()[0]
# weight_updated2 = model.layers[2].get_weights()[0]
# weight_updated3 = model.layers[3].get_weights()[0]
# weight_updated4 = model.layers[4].get_weights()[0]
print('輸入層: ', weight_updated0)
# print('第一層隱藏層: ', weight_updated1)
# print('第二層隱藏層: ', weight_updated2)
# print('第三層隱藏層: ', weight_updated3)
# print('輸出層: ', weight_updated4)
print('----------------Training prediction----------------------')
prediction1 = model.predict(ux0_ls)
plt.plot(lin, prediction1)
plt.show()

print('----------------Testing prediction----------------------')
prediction2 = model.predict(ux_ls)
plt.plot(np.linspace(0, 100, 101), y_ls,)
plt.plot(np.linspace(0, 100, 101), prediction2)

plt.show()

print('----------------Testing loss----------------------')
plt.plot(history.history['loss'])
plt.show()


print(model.summary())
