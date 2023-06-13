# deep-learning-challenge

## Overview of the Analysis
This exercise is to use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup, a nonprofit fpundtion.  
Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures.   
### Project Scope:  Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

## Tools
Pandas, scikit-learn, StandardScaler, TensorFlow, Keras

## Source:  
The source CSV contained more than 34,000 organizations that have received funding from Alphabet Soup over the years. The dataset included metadata about each organization, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

### Preprocessing
- Dropped non-beneficial ID columns (EIN and NAME)
- COnverted categorical data into numerical values with pd.get_dummies. 
- The initial model contained an input layer with 43 units, two hidden layers with 80 and 30 units, and an output layer with 1 unit.

### Training
- The model was trained using the "adam" optimizer, binary_crossentropy loss function, and accuracy metric. 
- The training data were split into features and target arrays.
- Features were scaled using StandardScaler. 
- Model weights were saved every five epochs using ModelCheckpoint.

### Hyperparameter Tuning
- To work towards achieving the target accuracy, different hyperparameters were tested using GridSearchCV and RandomSearchCV.
- The recommended hyperparameters from GridSearchCV include the 'relu' activation function, the 'adam' optimizer, and the 80 units in the hidden layer. 
- The recommended hyperparameters from RandomSearchCV include the 'sigmoid' activation function, the 'SGD' optimizer, and the 8 hidden layers. 
- THe highest level of acuracy was achieved with a learning rate of 0.1 and 100 epochs.

### Optimization: 
- Increased the number of bins for the classification data to include all categories over 100 rather than 1000. 
- The number of bins was incrementally reduced from  from over 500 to 555 to 52, to 51.  
- Accuracy shifted slighted with these modificatiosn from 0.72 to 0.73 to .0.7312.

### Evaluation: 
- The model's loss after training is 0.5630, and the accuracy is 0.7312. 

## Results
## Data Preprocessing
### What variable(s) are the target(s) for your model?   
IS_SUCESSFUL

### What variable(s) are the features for your model?
•	APPLICATION_TYPE
•	AFFILIATION
•	CLASSIFICATION
•	USE_CASE
•	ORGANIZATION
•	STATUS
•	INCOME_AMT
•	SPECIAL_CONSIDERATIONS
•	ASK_AMT

### What variable(s) should be removed from the input data because they are neither targets nor features?
NAME and EIN were removed. Since these are unique, they would not add value to the model.

## Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?
•	The model has 8 layers, including an input layer, 6 hidden layers, and an output layer. 
•	The number of neurons in each layer are as follows: 80 for the input layer, 50 for the hidden layers, 30 for the last hidden layer, and 1 for the output layer.
•	The activation function used for all hidden layers is ReLU, which is commonly used in deep learning networks for its ability to handle non-linear data and prevent the vanishing gradient problem. The activation function used for the output layer is sigmoid, which is commonly used for binary classification problems. These were selected based on recommendations by GridSearchCV and RandomSearchCV to try and reach an accuracy score of 75%.

Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

## Preprocess the Data
## Compile, Train and Evaluate the Model
## Optimize the Model 
## Write a Report on the Neural Network Model
