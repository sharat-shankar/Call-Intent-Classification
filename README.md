# Call_Intent_classification

#### Source and Nature of data: 
Due to the lack of data, we asked some of our classmates to write about 10 lines for each category considering  possible scenarios so as to get the most important features useful in classifying the conversations into categories.
#### Pre-processing approach: 
The sample audios were converted into text using Google Cloud Platform's Speech to Text API
#### Technology used: 
Neural Networks, Speech to Text, Lemmatization and Vectorization of text.
 
#### Training Methodology 
Performing tokenization/Lemmatization on the training sentences. Use of the feature extraction/Vectorization algorithm(Term Frequency – Inverse Document) to encode the parsed tokens as integers or floating point values for use as input to a machine learning algorithm. The data is trained using a 5 layer Neural Network consisting of 3 hidden layers. The tanh activation is applied on the hidden layers and softmax function on the output layer for the purpose of classification. The class which is assigned maximum probability by the softmax function is considered as the predicted class.

#### Testing Methodology & Accuracy attained during testing Testing method: 
We applied GridSearchcv to find the hyperparameters providing maximum accuracy. Plotting the epoch Vs accuracy graph helped us find the optimum epoch and Batch size for obtaining minimum loss and maximum accuracy.
The first round of validation was performed on the X_Test set obtained by splitting our training data in the ratio 0.25.It gave us an accuracy of 97.67%
In the next step we converted the sample audios provided from Speech to Text using Google Cloud API obtaining an accuracy of 60% (loss = 4.56) and Microsoft Azure Speech API obtaining an accuracy of 80%


 
