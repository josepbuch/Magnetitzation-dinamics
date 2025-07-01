import numpy as np 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf 
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Activation,Flatten,MaxPooling2D,Dropout,Input
from tensorflow.keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D,AlphaDropout,PReLU, GaussianNoise
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau

def param_denorm(x_norm, x_min, x_max):
    x = x_norm * (x_max - x_min) + x_min
    return x

def param_norm(x,x_min,x_max):
    x_norm = (x - x_min)/(x_max - x_min)
    return x_norm


def neural_conv():
    Net = Sequential()
    Net.add(Input((64, 64))) 
    
    Net.add(Conv1D(filters=16, kernel_size=3, padding='same'))
    Net.add(PReLU())
    Net.add(BatchNormalization())
    Net.add(MaxPooling1D(pool_size=2))
    
    Net.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    Net.add(PReLU())
    Net.add(BatchNormalization())
    Net.add(MaxPooling1D(pool_size=2))

    Net.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    Net.add(PReLU())
    Net.add(BatchNormalization())
    Net.add(MaxPooling1D(pool_size=2))


    Net.add(GlobalAveragePooling1D())  
    Net.add(Dropout(0.1))  
    Net.add(Dense(128, activation='tanh'))
    Net.add(Dense(64, activation='tanh'))
    Net.add(Dense(32, activation='tanh'))
    Net.add(Dense(16, activation='tanh'))
    
    Net.add(Dense(2, activation='linear'))  
    

    Net.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    Net.summary()
    return Net


lr_callback = ReduceLROnPlateau(
    monitor='val_loss',  # Monitors validation loss
    factor=0.5,          # Reduces learning rate by half
    patience=5,          # Waits 5 epochs without improvement before reducing
    min_lr=1e-6          # Minimum learning rate allowed
)



# Theoretical data
y_train = np.load('./train_data/Pmag_200000_ani.npy')
param_train = np.load('./train_data/params_200000_ani.npy')

Net = neural_conv()
epoch = 75
alpha_param = np.transpose(np.array([param_train[:,1],param_train[:,2]]))
history = Net.fit(y_train, alpha_param,callbacks=[lr_callback],validation_split=0.1, epochs = epoch, batch_size= 256, verbose='auto')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(history.history['loss'], label = 'train')
ax1.plot(history.history['val_loss'],label= 'val')
ax1.set_yscale('log')
ax1.legend(loc = 'best')
ax1.set_title('Loss Function Huber')

ax2.plot(history.history['mae'], label = 'train')
ax2.plot(history.history['val_mae'], label = 'val')
ax2.set_yscale('log')
ax2.legend(loc = 'best')
ax2.set_title('mae')
plt.tight_layout()
plt.savefig('./models/plot_convergence.png', dpi = 400)

Net.save('./models/conv1d.keras')


