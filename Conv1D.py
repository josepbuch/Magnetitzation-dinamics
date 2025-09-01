import numpy as np 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Activation,Flatten,MaxPooling2D,Dropout,Input,Concatenate
from tensorflow.keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D,AlphaDropout,PReLU, GaussianNoise,GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pyttsx3



def param_denorm(x_norm, x_min, x_max):
    x = x_norm * (x_max - x_min) + x_min
    return x

def param_norm(x,x_min,x_max):
    x_norm = (x - x_min)/(x_max - x_min)
    return x_norm

def neural_conv1D():
    Net = Sequential()
    Net.add(Input((88, 88))) 
    
    Net.add(Conv1D(filters=20, kernel_size=3, padding='same'))
    Net.add(PReLU())
    Net.add(BatchNormalization())
    Net.add(MaxPooling1D(pool_size=2))
    
    Net.add(Conv1D(filters=40, kernel_size=3, padding='same'))
    Net.add(PReLU())
    Net.add(BatchNormalization())
    Net.add(MaxPooling1D(pool_size=2))

    Net.add(Conv1D(filters=80, kernel_size=3, padding='same'))
    Net.add(PReLU())
    Net.add(BatchNormalization())
    Net.add(MaxPooling1D(pool_size=2))


    Net.add(GlobalAveragePooling1D())  
    Net.add(Dropout(0.1))  
    Net.add(Dense(64, activation = 'tanh'))
    #Net.add(PReLU())
    Net.add(Dense(32,activation = 'tanh'))
    #Net.add(PReLU())
    Net.add(Dense(16,activation = 'tanh'))
    #Net.add(PReLU())
    
    Net.add(Dense(2, activation='linear'))  
    

    Net.compile(loss='huber', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    Net.summary()
    return Net

lr_callback = ReduceLROnPlateau(
    monitor='val_loss',  # Monitors validation loss
    factor=0.5,          # Reduces learning rate by half
    patience=2,          # Waits 5 epochs without improvement before reducing
    min_lr=1e-6          # Minimum learning rate allowed
)

y_train = np.load('./train_data/Pmag_150000.npy')
param_train = np.load('./train_data/params_150000.npy')


gyro = 28.024*10**9
alpha_param = np.transpose(np.array([param_norm(np.log10(param_train[:,1]),-3,-1.5)
                                     ,param_norm(np.log10(param_train[:,2]),np.log10(gyro*10**(-5)),np.log10(gyro*10**(-3)))]))
           
Net = neural_conv1D()
epoch = 40

history = Net.fit(y_train, alpha_param,callbacks=[lr_callback],validation_split=0.1, epochs = epoch, batch_size= 256, verbose='auto')

# Sound alert, when the compilation has finished
engine = pyttsx3.init()
engine.say("The complilation has finished")
engine.runAndWait()

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
plt.savefig('./models/plot_convergence_conv_1.png', dpi = 400)


Net.save('./models/conv1d.keras')
