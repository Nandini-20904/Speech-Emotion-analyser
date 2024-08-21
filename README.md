Project Overview
This project focuses on building an Emotion Recognition System using Deep Learning. The system analyzes speech data to detect emotions, leveraging advanced machine learning techniques, including Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and an Attention mechanism. The project is built in Python, using libraries such as TensorFlow, Keras, and Optuna for model optimization.

Objective
The primary goal is to create a model that accurately predicts the emotion expressed in an audio clip. This model could be applied in various domains, including customer service, mental health analysis, and human-computer interaction, where understanding emotional states is crucial.

Dataset
The Toronto Emotional Speech Set (TESS) dataset is used for training and testing. The dataset contains 2,800 audio files, each labeled with one of seven emotions: fear, angry, disgust, neutral, sad, ps (pleasant surprise), and happy. The dataset is loaded and preprocessed to extract features for model training.

Data Preprocessing
Label Encoding: The emotion labels are one-hot encoded using OneHotEncoder.
Waveform and Spectrogram Visualization: The waveforms and spectrograms for each emotion are plotted to visually inspect the data.
Feature Extraction:
MFCCs (Mel-Frequency Cepstral Coefficients): Extracts the essential features from the audio.
Delta and Delta-Delta MFCCs: Capture the dynamic aspects of the audio signal.
Chroma Features: Capture the harmonic and pitch content.
Model Architecture
The model architecture combines CNN layers for spatial feature extraction and Bi-directional LSTM layers to capture temporal dependencies in the audio features, augmented by an Attention mechanism to focus on the most relevant parts of the sequence.

Input Layer: The input shape corresponds to the extracted features from the audio data.
Conv1D Layers: Two convolutional layers with filters (selected using Optuna) for feature extraction.
Pooling Layer: MaxPooling1D layer to reduce the dimensionality.
Reshape Layer: Reshapes the pooled output for LSTM input.
Bidirectional LSTM Layer: Captures temporal dependencies from both directions in the sequence.
Attention Layer: Enhances the model's focus on important parts of the LSTM output.
Dense Layer: Fully connected layer with ReLU activation for classification.
Output Layer: Softmax layer to predict the probability distribution over the emotion classes.
Model Optimization
The model's hyperparameters are optimized using Optuna, a hyperparameter optimization framework. The objective function is designed to maximize validation accuracy by varying key parameters like the number of filters in convolutional layers, LSTM units, dense units, dropout rate, and learning rate.

Training and Validation
The model is trained using a categorical crossentropy loss function and the Adam optimizer. Early stopping and learning rate reduction are employed to prevent overfitting and improve model generalization.

EarlyStopping: Monitors the validation loss and stops training when it doesn't improve for a set number of epochs.
ReduceLROnPlateau: Reduces the learning rate when the validation loss plateaus, allowing the model to converge more effectively.
Model Evaluation
The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.

Confusion Matrix: A heatmap is plotted to visualize the performance across different emotion classes.
Classification Report: Detailed precision, recall, and F1-score metrics are calculated for each emotion class.
Deployment and Prediction
The trained model is saved and loaded for real-time emotion prediction. A prediction function is created, which takes an audio file as input, extracts features, and outputs the predicted emotion.

Key Achievements
Advanced Model Architecture: Successfully combined CNN, Bi-directional LSTM, and Attention mechanisms for effective emotion recognition.
Hyperparameter Optimization: Leveraged Optuna for optimizing model performance, achieving higher accuracy.
Real-time Emotion Detection: Implemented a system capable of predicting emotions from audio files in real-time.
Conclusion
This project showcases the power of deep learning in understanding and predicting human emotions from speech. The developed model, with its robust architecture and optimization techniques, demonstrates significant potential for applications in various fields, from enhancing customer experiences to aiding mental health professionals.
