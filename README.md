# Research-Video-Classifier-
This project utilizes deep learning techniques such as CNNs, LSTMs, ConvLSTMs, and Autoencoders to analyze unmarked videos of mice post spinal cord injuries. The goal of this project is to assist researchers at the Miami Project to Cure Paralysis with sorting through the massive amounts of video data in order to draw accurate and timely conclusions. 
## Project Timeline and Progress Journal
### June 7,2022-Project Begins 
The intial problem of sorting through video data and the possiblity of using a video classifier is introduced. Brainstorming on the classification system begins. 
### June 9, 2022-Background Research Begins 
With the help of computer science professors, the key types of deep learning techniques that best address video classification are selected. The project will begin with investigation into convolutional neural networks (CNNs), Long short term memory (LSTMs), ConvLSTMs (CNNs+LSTMs), and Autoencoders. 
### June 10-16, 2022-Research and Data Processing 
## Overview of CNNs, LSTMs, ConvLSTMs, and Autoencoders 
### Convolutional Neural Networks 
A CNN has an input layer, hidden layer(s), and an output layer. 
## ![image](https://files.cdn.thinkific.com/file_uploads/118220/images/9ac/ef7/edb/1583485122964.jpg?width=1920&dpr=2)
In this diagram:
x1...xn represent the inputs, either from the input layer of a value from a previous hidden layer 
x0 is the bias unit, which acts similat to an intercept term and is a constant that is added to the input of the activation function 
w1...wn are the weights that the inputs are multiplied by
A is the output, which is calculated by inputing the sum of the inputs times their weights into an activation function, f. The activation function allows neural networks to be flexible, and model linear, hyperbolic, logarithmic, or gaussian relationships. 
## ![gif][(https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_3.png](https://i.stack.imgur.com/9Iu89.gif)) 
A convultional neural network uses filters to obtain the features of an input. Each filter is a matrix with weights which then traverses the image and produces an output. Additionally, CNNs uses pooling to reduce the amount of parameters, essentially pools shrink the matrix, by taking, for example the max value for a given quadrant. This is called MAX pooling. CNNs are used for tasks that require computer vision because they retain the spatial relationship of data. 






















