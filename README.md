# Book Recommendation System 📚✨

## Introduction 🌟

This repository contains the implementation of a **Book Recommendation System** using **Neural Networks** and **Convolutional Neural Networks (CNNs)**. The system classifies books as "Recommend" or "Not Recommend" based on user ratings, leveraging user demographics, book metadata, and rating data for recommendations.

---

## Objective 🎯

The primary goal of this project is to develop a machine learning model capable of accurately recommending books. Key challenges include:
- Handling imbalanced datasets (most users rate books positively).
- Addressing missing data, particularly in the Age feature.
- Evaluating neural network models on both balanced and unbalanced datasets.

---

## Data Preprocessing 🔧

### Input Data
The dataset consists of three tables:
- **Users**: User-ID, Age, Location.
- **Books**: ISBN, Book-Title, Book-Author, Publisher, Year-Of-Publication, Image-URL.
- **Book-Ratings**: ISBN, User-ID, Book-Rating.

### Preprocessing Steps
1. **Users Table**:
   - Imputed missing Age values with the median.
   - Simplified Location to country-level.
2. **Books Table**:
   - Filled missing Book-Title and Book-Author values with "Unknown."
   - Replaced missing Year-Of-Publication values with the median.
3. **Ratings Table**:
   - Removed implicit feedback (Book-Rating = 0).
   - Merged datasets on User-ID and ISBN.

### Feature Engineering
- Assigned unique Book-ID using factorization.
- Encoded categorical features (e.g., User-ID, Location) using Label Encoding.
- Scaled numerical features (e.g., Age, Year-Of-Publication) using MinMaxScaler.
- Created binary labels for ratings:
  - **1-6**: Label = 0 (Not Recommend).
  - **7-10**: Label = 1 (Recommend).

### Addressing Class Imbalance
- Down-sampled the majority class (Label = 1) to balance the dataset.

### Splitting Data
- Separated features and labels.
- Split data into training and test sets (80%-20%).

---

## Models 🚀

### 1. Neural Network Architecture
- **Input Layers**: Separate inputs for all features.
- **Embedding Layers**: For categorical features (e.g., User-ID, Location).
- **Concatenation Layer**: Combines embeddings with scaled numerical features.
- **Dense Layers**: Includes dropout for regularization.

### 2. CNN Architecture
- **Input Layers**: Separate inputs for all features.
- **Embedding Layers**: For categorical features.
- **Reshape Layer**: Prepares data for convolution.
- **Convolutional Layers (Conv1D)**: Extracts features with pooling.
- **Global Pooling Layer**: Followed by dense layers for final classification.

### Model Variants
- Neural Network with Unbalanced Data.
- Neural Network with Balanced Data.
- Neural Network with Modified Rating Scale (1-7: Not Recommend, 8-10: Recommend).
- Neural Network without Age Feature.
- CNN with Balanced Data.
- CNN with Unbalanced Data.

---

## Training Details 🖥️

- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Binary cross entropy.
- **Evaluation Metric**: Accuracy.
- **Split**: 80% training, 20% testing.

---

## Results ✅

| Model                                     | Accuracy | Precision | Recall | F1-Score | MAP |
|-------------------------------------------|----------|-----------|--------|----------|-----|
| Neural Network (Unbalanced Data)          | 0.68     | 0.82      | 0.74   | 0.78     | 0.85|
| Neural Network (Balanced Data)            | 0.62     | 0.63      | 0.58   | 0.61     | 0.66|
| Neural Network (Modified Rating Scale)    | 0.63     | 0.69      | 0.66   | 0.67     | 0.72|
| Neural Network (No Age Feature)           | 0.64     | 0.64      | 0.66   | 0.65     | 0.67|
| CNN (Balanced Data)                       | 0.62     | 0.60      | 0.72   | 0.66     | 0.65|
| CNN (Unbalanced Data)                     | 0.72     | 0.81      | 0.82   | 0.82     | 0.84|

---

## Conclusion 🏁

This project demonstrates the effectiveness of Neural Networks and CNNs for binary classification in book recommendation systems. Key findings:
- Addressing class imbalance is critical for model performance.
- Feature selection, such as excluding Age, can impact results.
- CNNs are particularly effective for capturing relationships between user and book embeddings.

### Future Work
- Incorporate additional metadata (e.g., Publisher, Image-URL).
- Explore advanced techniques like attention mechanisms or transformer models.
- Scale the system for real-time recommendations.

---

Explore this repository to learn more about building advanced book recommendation systems! 🚀📚
