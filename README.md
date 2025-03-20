# Music-Genre-Classification

## Overview
This project focuses on classifying music into different genres using machine learning techniques. The dataset is processed for feature extraction, and multiple models are trained and tested to achieve high accuracy in classification. This project uses Convolutional Neural Networks (CNN) to implement a music genre classification system. Two models were trained:

1. **Model 1** - Trained using all available features.
2. **Model 2** - Trained using only selected features based on feature importance determined by a RandomForestClassifier.

## Dataset
- The dataset contains various audio features extracted from music files.
- It includes genres as target labels and numerical features representing the audio characteristics.
- Preprocessing steps include handling irrelevant data, normalizing data, and encoding categorical labels.

## Dependencies
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Methodology
Two distinct models were developed and trained to explore the impact of feature selection:

1. **Data Collection & Preprocessing**:
   - The dataset, containing audio samples from various genres and features, was sourced from Kaggle.
   - Audio files were preprocessed by converting them into features, specifically Mel-frequency cepstral coefficients (MFCCs), chroma features, and spectral contrast, using the `librosa` library. These already extracted features were used.

2. **Feature Extraction**:
   - In the first model, **all available audio features** (MFCCs, chroma, spectral contrast, etc.) were used as input for the CNN.
   - In the second model, **feature selection** was performed to identify the most relevant features for classification. Feature importance was calculated using a **RandomForestClassifier**, and only the most important features were selected for training the CNN.

3. **Model Development**:
   - Both models were trained using **Convolutional Neural Networks (CNN)**. The first model utilized the full set of extracted features, while the second model leveraged only the selected features based on their importance score from the RandomForestClassifier.
   - The models were built using TensorFlow/Keras, with appropriate layers and activation functions optimized for audio classification tasks.

4. **Model Evaluation**:
   - The models were evaluated on a validation set using accuracy, precision, recall, and F1-score to assess performance.
   - **Cross-validation** was applied to ensure that the models generalize well to unseen data and minimize overfitting.

5. **Results & Comparison**:
   - After training, the models were compared based on performance metrics to evaluate the impact of feature selection. The second model, which used selected features, was expected to demonstrate improved performance or reduced complexity compared to the first model.


### 3. Evaluation
- Evaluated models using precision, recall, and accuracy metrics.
- Analyzed misclassifications and compared genre confusion.
- Visualized results using heatmaps and precision-recall curves.
- Compared training and validation losses to check for overfitting.

## Results
| Model | Training Accuracy | Testing Accuracy |
|-------|------------------|------------------|
| Model 1 (All Features) | 97.87% | 90.59% |
| Model 2 (Selected Features) | 97.69% | 92.99% |

Model 2, which used only the most important features, outperformed Model 1 on test accuracy, suggesting that removing irrelevant features improved generalization.

## Key Findings
- Feature selection improved model generalization.
- CNN effectively captured patterns in audio features for genre classification.
- Model 2 had a lower training accuracy than Model 1 but performed better on the test set, indicating reduced overfitting.

## Visualizations

### 1. **Feature Importance Graph**:

   ![Feature_Importance_Graph](https://github.com/user-attachments/assets/75d49514-2450-49be-9dcb-b1f037edc818)

### 2. **Model Accuracy and Loss Curves**:

   - **Model 1**:
     ![Model1_History_Plot](https://github.com/user-attachments/assets/5674166b-f279-4621-abd9-0f9b39c49f1e)
     
   - **Model 2**:
     ![Model2_History_Plot](https://github.com/user-attachments/assets/14e5c699-6279-4c6d-9fb1-51737012a6a3)

### 3. **Confusion Matrix**:

   - **Model 1**:

     ![Model1_ConfMatrix](https://github.com/user-attachments/assets/fe0e1797-4ccb-4543-b9e6-b99ee52fffbc)

   - **Model 2**:

     ![Model2_ConfMatrix](https://github.com/user-attachments/assets/10bc71dd-66ad-4a37-a89c-09a1e341485c)


## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook `Classify.ipynb`.
4. Check evaluation metrics and visualizations for insights.

## Future Improvements
- Experiment with different feature selection techniques.
- Try alternative machine learning or deep learning techniques.
- Expand the dataset for better generalization.

## Project Notebook

🔗 Check out the notebook here: [Kaggle Link](https://www.kaggle.com/code/manasibhosale/music-genre-classification)

## Acknowledgments
- Used `librosa` for audio feature extraction.
- Thanks to open-source datasets and Kaggle community insights.

---
For any questions, feel free to reach out!



