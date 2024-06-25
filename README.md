# Magnetitzation-dinamics
**This repository is for the Inhouse meeting IN2UB. The project is not finished yet**. If there is any question about the project please contact by jbuchpal22@alumnes.ub.edu

**Abstract**: Magnetic thin films are fundamental components in magnetic memories and sensors. To calculate the magnetization dynamics of these film, ferromagnetic resonance (FMR) is typically employed to extract the key dynamic parameters. However, experimental setups often introduce noise and perturbations, complicating the parameter extraction process using traditional fitting formulas. 
 In this work, we propose using a deep neural network (DNN) to model FMR in magnetic thin films. Instead of conventional methods, the DNN will be used to extract directly FMR parameters. The data for the DNN is generated through simulations based on the Landau-Lifshitz-Gilbert (LLG) equation.

This program is not yet an executable program and consists only of the program codes that we are using. It has two main components, 'data_generator.nb' (using Mathematica) and 'main.py' (using Python):

- data_generator.nb: Using the LLG equations, we calculate the colormap of a ferromagnetic resonance with preselected parameters. The program contains a loop to calculate as many datasets as needed. In the training program, we use N_data = 5000. The data is generated into two folders, the Real data and the Im data. It also generates files with the parameters used to create the colormaps.

- main.py: Reads the created folders and resizes the array data from data_generator.nb. After that, we normalize the data and add noise. We prepare the data to introduce to the Neural Network and perform the training. The Neural Network structure is detailed in the table below.



| Layer Type          | Parameters                                                                      | Output Shape          |
|---------------------|---------------------------------------------------------------------------------|-----------------------|
| Input               | Input shape: (64, 64, 1)                                                        | (64, 64, 1)           |
| Conv2D              | filters: 4, kernel_size: [4, 4], padding: 'same', activation: 'relu'            | (64, 64, 4)           |
| Conv2D              | filters: 4, kernel_size: [4, 4], padding: 'same', activation: 'relu'            | (64, 64, 4)           |
| MaxPooling2D        | pool_size: 2                                                                    | (32, 32, 4)           |
| Conv2D              | filters: 8, kernel_size: [2, 2], padding: 'same', activation: 'relu'            | (32, 32, 8)           |
| Conv2D              | filters: 8, kernel_size: [2, 2], padding: 'same', activation: 'relu'            | (32, 32, 8)           |
| MaxPooling2D        | pool_size: 2                                                                    | (16, 16, 8)           |
| Conv2D              | filters: 16, kernel_size: [2, 2], padding: 'same', activation: 'relu'           | (16, 16, 16)          |
| MaxPooling2D        | pool_size: 2                                                                    | (8, 8, 16)            |
| Conv2D              | filters: 32, kernel_size: [2, 2], padding: 'same', activation: 'relu'           | (8, 8, 32)            |
| MaxPooling2D        | pool_size: 2                                                                    | (4, 4, 32)            |
| Flatten             |                                                                                 | (512)                 |
| Dropout             | rate: 0.1                                                                       | (512)                 |
| Dense               | units: 20, activation: 'relu'                                                   | (20)                  |
| Dense               | units: 3, activation: 'linear'                                                  | (3)                   |
