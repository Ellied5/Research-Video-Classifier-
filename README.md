# Research-Video-Classifier-
This project utilizes deep learning techniques such as CNNs, LSTMs, ConvLSTMs, and Autoencoders to analyze unmarked videos of mice post spinal cord injuries. The goal of this project is to assist researchers at the Miami Project to Cure Paralysis with sorting through the massive amounts of video data in order to draw accurate and timely conclusions. 
## Project Timeline and Progress Journal
### June 7,2022-Project Begins 
The intial problem of sorting through video data and the possiblity of using a video classifier is introduced. Brainstorming on the classification system begins. 
### June 9, 2022-Background Research Begins 
With the help of computer science professors, the key types of deep learning techniques that best address video classification are selected. The project will begin with investigation into convolutional neural networks (CNNs), Long short term memory (LSTMs), ConvLSTMs (CNNs+LSTMs), and Autoencoders. 
### June 10-16, 2022-Research and Data Processing 
## Overview of CNNs
### Convolutional Neural Networks 
A CNN has an input layer, hidden layer(s), and an output layer. 
## ![image](https://files.cdn.thinkific.com/file_uploads/118220/images/9ac/ef7/edb/1583485122964.jpg?width=1920&dpr=2)
In this diagram:
x1...xn represent the inputs, either from the input layer of a value from a previous hidden layer 
x0 is the bias unit, which acts similar to an intercept term and is a constant that is added to the input of the activation function 
w1...wn are the weights that the inputs are multiplied by
A is the output, which is calculated by inputing the sum of the inputs times their weights into an activation function, f. The activation function allows neural networks to be flexible, and model linear, hyperbolic, logarithmic, or gaussian relationships. 
## ![image](https://austingwalters.com/wp-content/uploads/2019/01/image-filter-matrix-cnn.png)
A convultional neural network uses filters to obtain the features of an input. Each filter is a matrix with weights which then traverses the image and produces an output. Additionally, CNNs uses pooling to reduce the amount of parameters, essentially pools shrink the matrix, by taking, for example the max value for a given quadrant. This is called MAX pooling. CNNs are used for tasks that require computer vision because they retain the spatial relationship of data.
In the creation of the final video processer, I practiced by writing a much simpler CNN, which can be found in the Example CNN file on this repository. This example shows the simplest component of CNNs, forward propogation. 
## Video Processing 
To begin processing the videos, I wrote a program video_process.py that converted the videos into tensors and created an array of these tensors which could bee fed to the model. Unfortunately, the memory on my computer was unable to run the over 800 video tensors at once. In order to create a more memory efficient solution, I decided to save each of the RGB tensors as a csv file which is what the script video_process_csv.py achieves. In order to create a model, I will have to load the data from the csv files in batches of ~200 because this what my computer can run at once. Although this solution was not my intial idea it will allow for me to accommodate the resources that I have. 
## Training the Supervised Classification Model 
The next step in creating the video classifier was the load the csv files into a training data set and to label them as either pre or post injury using a 0 or 1 respectively. 
## LSTM, ConvLSTM, and Autoencoders: Building an unsupervised model 























