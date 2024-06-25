import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Activation,Flatten,MaxPooling2D,Dropout,Input
from tqdm.auto import tqdm
from scipy.interpolate import griddata
from scipy.stats import lognorm
from keras.optimizers import Adam

# read the data from the files --------------------------------------
def data_extractor(file_name):
    '''
    Reads a csv file with 3 columns
    Returns 3 numpy.array with each column
    '''
    real_data = pd.read_csv(file_name,sep='\t', header = None)

    field = np.array(real_data.iloc[:, [0]])
    frequency =np.array( real_data.iloc[:, [1]])
    absortion = np.array(real_data.iloc[:, [2]])

    return np.reshape(field,len(field)),np.reshape(frequency,len(frequency)),np.reshape(absortion,len(absortion))

def key_file(file):
    '''
    Get the number of the file, before the second '_'
    '''
    if file == '.DS_Store':
        return None  # Saltar el fitxer .DS_Store retornant None
    try:
        number = int(file.split('_')[-1].split('.')[0])
        return number
    except ValueError:
        print(f"No es pot convertir {file} a un enter.")
        return None  # Retornar None en cas de no poder convertir

def folder_extractor(folder_path,x_grid,y_grid):
    '''
    Input: folder_path
    Output: the Absortion/Trasnmition of all the data in the folder
    '''
    # Read the files inside the folder
    files = os.listdir(folder_path)
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    
    # Sort the name of files inside the directory
    valid_files = [file for file in files if key_file(file) is not None]
    sort_files = sorted(valid_files, key= key_file)
    path = []
    for i in sort_files:
        path.append(folder_path+'/' + i)
    final_data = []
    # Read the files and join them in a one 
    for j in tqdm(path):
        field, freq, power = data_extractor(j)
        power = reshapeator(field, freq, power,x_grid,y_grid)
        power = np.flip(power, axis = 1)
        power = np.rot90(power, 2)
        final_data.append(power)
        
    return np.array(final_data)

def reshapeator (field, freq, power, x_grid, y_grid):
    '''
    Input: field, freq and power in 3 different np.array 1D and 
           x_grid and y_grid dimensions
    Output: Creation of Grid of the power with the grid paramaters
    '''
    
    field_grid, freq_grid = np.meshgrid(np.linspace(min(field), max(field), x_grid),
                                    np.linspace(min(freq), max(freq), y_grid)) 
    power_grid = griddata((field, freq), power, (field_grid, freq_grid), method='cubic')

    return power_grid

    # data generator before introduce in NN  ------------------------

def data_generator(N_data,tp, N_random,x_grid,y_grid):
    '''
    Input: the path of the folders and Ms documents in Pc
    N_random: # of times for each parameter generates a random matrix 
    if N_random = 0, A = 0.5 constant 

    Output: Ms_train and y_train where the Num Samples is Sample_Input* N_random
    '''
    print ('Extracting and Reshaping the data: ')
    folder_path_re = 'File/path/to//Data/'+tp+'_'+str(N_data)+'/data_re_'+str(N_data)
    folder_path_im = 'File/path/to//Data/'+tp+'_'+str(N_data)+'/data_im_'+str(N_data)
    a_path = 'File/path/to//Data/'+tp+'_'+str(N_data)+'/a_data_'+str(N_data)+'.csv'
    Ms_path = 'File/path/to//Data/'+tp+'_'+str(N_data)+'/Ms_data_'+str(N_data)+'.csv'

    z_re = folder_extractor(folder_path_re,x_grid,y_grid)
    z_im = folder_extractor(folder_path_im,x_grid,y_grid)
    alpha = (np.array(pd.read_csv(a_path,sep='\t', header = None)))
    Ms = np.array(pd.read_csv(Ms_path,sep='\t', header = None))

    y_train = []
    param_train = []
    c = 0 
    for i,j in zip(z_re,z_im):
        
        for k in range(N_random):
            A = A_creation(x_grid,y_grid)
            z = A * i + (1 - A) * j
            z_norm = normalization(x_grid,y_grid, z)
            y_train.append(z_norm)
            param_train.append([param_norm(Ms[c],400000,2000000),param_norm(alpha[c],-4,-1.3010)])
        c = c + 1
    param_train = np.array(param_train)
    param_train = np.reshape(param_train, (N_data*N_random,2))
    return np.array(y_train), np.array(param_train)


def param_norm(x,x_min,x_max):
    x_norm = (x - x_min)/(x_max - x_min)
    return x_norm

def A_creation(x_grid,y_grid):
    A = np.zeros((y_grid,x_grid))

    for i in range(y_grid):
        rand = np.random.rand()
        for j in range(x_grid):
            A[i][j] = rand
    return A
    
def normalization(field_grid, freq_grid,array):
    grid_normalized = np.zeros((freq_grid,field_grid))

    for i in range(freq_grid):
        fila = array[i, :]
        average = np.average(fila)
        maxim = max(fila)
        minim = min(fila)
        
        if (average + maxim) > (average - minim):
            fila_norm = fila/maxim
            
        else:
            fila_norm = fila/np.abs(minim)

        grid_normalized[i, :] = fila_norm 
        
    return grid_normalized


# Neural Network structure -----------------------------------------

def neural_conv():
    '''
    Encoder convolutional Neural Network
    '''
    Net = Sequential()

    # Convolutional Neural Network Encoder
    Net.add(Input((64,64,1)))
    Net.add(Conv2D( filters = 4, kernel_size = [4,4], padding = 'same',activation = 'relu'))
    Net.add(Conv2D( filters = 4, kernel_size = [4,4], padding = 'same',activation = 'relu'))
    Net.add(MaxPooling2D(pool_size = 2))
    Net.add(Conv2D(filters = 8, kernel_size = [2,2], padding = 'same', activation = 'relu'))
    Net.add(Conv2D(filters = 8, kernel_size = [2,2], padding = 'same', activation = 'relu'))
    Net.add(MaxPooling2D(pool_size = 2))
    Net.add(Conv2D(filters = 16, kernel_size = [2,2], padding = 'same', activation = 'relu'))
    Net.add(MaxPooling2D(pool_size = 2))
    Net.add(Conv2D(filters = 32, kernel_size = [2,2], padding = 'same', activation = 'relu'))
    Net.add(MaxPooling2D(pool_size = 2))
    # Dense Layer
    Net.add(Flatten())
    Net.add(Dropout(0.1))
    
    Net.add(Dense(10,activation = 'relu'))
    Net.add(Dense(2, activation = 'linear'))

    #Compiling NN
    Net.compile(loss = 'mse', optimizer = Adam(learning_rate=1e-5), metrics = ['accuracy'])
    return Net 


# Neural Network parameters
tp = 'MsaDHCK0'
N_data = 10000
N_random = 20
field_grid = 64
freq_grid = 64

epoch = 300
batch = 1000
Net = neural_conv()


#Obtaining data
y_train, param_train = data_generator(N_data,tp,N_random,field_grid,freq_grid)

#y_train = np.load('75000_data.npy')
#param_train = np.load('75000_param.npy')

# training 
print('Training the Neural Network')
loss_encoder = np.zeros((4,epoch))
for i in tqdm(range(epoch)): 
    history = Net.fit(y_train,param_train,validation_split=0.1, epochs = 1, batch_size = batch,verbose=0)
    loss_encoder[0][i] = history.history['loss'][0]
    loss_encoder[1][i] = history.history['accuracy'][0]

# Loss function ----------------------------------

plt.figure(figsize=(12, 2))
plt.xlabel('Epochs')
plt.ylabel('Loss Function')
plt.plot(loss_encoder[0], label = 'Loss function')
plt.plot(loss_encoder[1], label = 'Accuracy')
plt.legend(loc = 'best')
plt.rcParams.update({'font.size': 14})
plt.savefig('lossfunc_e=500.png', dpi = 400)
print('Last Loss:' ,loss_encoder[0][epoch-1])
print('Last accuracy', loss_encoder[1][epoch-1])

