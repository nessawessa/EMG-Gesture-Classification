*Disclaimer: This repo serves as a reference––contrasting machine learning approaches––and does not contain the EMG signal files used in testing.*

LAB REPORT


This lab focuses on the classification of hand gestures using Electromyography (EMG) signals. 
The problem at hand is to accurately identify different hand postures based on the electrical activity of muscles recorded by EMG sensors, achieving above a 60% success rate. 
The EMG data was captured through contractions through the arm from multiple hand gestures. 
The goal of processing the data is to categorize the input to algorithms as a set of features extracted from EMG signals, including root mean square (RMS), standard deviation, and zero-crossing rate. 
I then use two different machine learning approaches to classify the hand gestures:
- An Artificial Neural Network (ANN)
- A Naïve Bayesian classifier
  
The ANN was chosen for its capacity to handle complex nonlinear relationships, while the NBC was included for its simplicity and interpretability.
Both models were evaluated for their performance in terms of classification accuracy on test data.

Dataset and Features:

The dataset consists of EMG recordings for 8 different hand postures, with 10 trials for each posture. 
The data is stored in separate text files, each containing the EMG signals for a specific posture and trial. 
The locations of these channels were the Brachioradialis, Palmaris longus, Extensor carpi ulnaris, Extensor carpi radialis brevis, and the Extensor carpi radialis longus muscles, with an electrode at the elbow for ground. 
Each hand posture was recorded for a length of 500 samples at a frequency of 1000Hz. With ten samples of each of the eight hand postures recorded, a total of 80 hand posture files were created. 


Preprocessing:
1. The raw EMG signals are multiplied by 1000 to amplify the values.
2. The mean of each channel is subtracted to center the data around zero.
   
Feature Extraction:

- For each EMG recording, we extracted the following features:
- Root Mean Square (RMS): Represents the signal power.
- Standard Deviation: Measures the signal variability.
- Zero-crossing Rate: Indicates frequency content of the signal.
  
 After preprocessing, the dataset was a structured array of 80 samples with 12 features and one label column (0-7).
 Features were standardized to zero mean and unit variance to ensure uniform scaling.

Methods:

Artificial Neural Network (ANN)
Implemented a feedforward neural network with one hidden layer:
- Input layer: 12 neurons (one for each feature)
- Hidden layer: 60 neurons with ReLU activation
- Output layer: 8 neurons (one for each posture) with ReLU activation
  
The network is trained using backpropagation with the following parameters:
- Learning rate (η): 0.005
- Epochs: 1000

The Naïve Bayesian classifier assumes that features are independent given the class.
It uses Bayes’ theorem to compute the posterior probability of each class given the input features.


Results and Discussion:

Artificial Neural Network (ANN)

The ANN was evaluated under varying configurations of hidden layers and epochs. 
Across multiple runs, the accuracy ranged from 20% to 80%, influenced by the small dataset and random initialization. 
The best accuracy of 80% was achieved with a hidden layer of 100 neurons and 200,000 epochs. This configuration minimized the error sum effectively, as shown in Figure 3.
Despite achieving relatively high accuracy, finding the optimal configuration was challenging due to the stochastic nature of neural networks and the sensitivity to hyperparameters like learning rate and network architecture. 
Additionally, implementing a loop to track and calculate the classification accuracy required careful handling to ensure results were correctly captured.

Naïve Bayesian Classifier (NBC)

The NBC showed an accuracy range between 10% and 80%, depending on the randomized training and testing splits. 
Unlike the ANN, this model relies on statistical measures such as mean and standard deviation rather than error minimization functions.
One challenge encountered was the misalignment of posture labels due to differences in label indexing (0–7 for postures versus 1–8 in code logic).
Correcting this mismatch resolved classification issues and allowed the model to function properly.
The NBC's performance varied due to the small training set size and feature overlap. 
Randomized splits occasionally left insufficient samples for certain postures in the training set, leading to poorer predictions for underrepresented classes.

Comparison and Limitations

Both models exhibited significant variability in accuracy due to the small dataset size and the randomness in training and testing splits. 
However, the ANN consistently performed better when properly tuned. This highlights its ability to capture complex relationships in the data, albeit at a higher computational cost.
The low dataset size limited both models' potential. When fewer samples of a posture were included in the training set, the models struggled to predict those postures accurately. 
This limitation could be addressed by increasing the dataset size, thereby providing better training examples.



Conclusion and Future Work:

The ANN outperformed the NBC, achieving a higher accuracy due to its ability to model nonlinear relationships. 
The NBC, while computationally efficient, was hindered by its simplistic assumptions.

Key findings:
1. Feature extraction plays a crucial role in classification performance.
2. The ANN’s ability to learn complex patterns gives it an edge over the Naïve Bayesian classifier for this task.
3. Some postures are more challenging to distinguish, possibly due to similarities in muscle activation patterns.

Future work could explore larger datasets, additional features such as frequency-domain attributes, and ensemble methods combining multiple classifiers. Implementing more advanced architectures like Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) for improved performance.
