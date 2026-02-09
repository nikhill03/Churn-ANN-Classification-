# Customer Churn Prediction using ANN

This project utilizes an Artificial Neural Network (ANN) to predict customer churn for a bank. By analyzing various customer demographics and financial behaviors, the model estimates the probability of a customer leaving the bank. The solution includes a comprehensive training pipeline, a testing notebook for local inference, and an interactive Streamlit web application for real-time predictions.

## 📂 Project Structure

* **`experiments.ipynb`**: Main Training notebook. It handles:
    * **Data Loading**: Imports `Churn_Modelling.csv` and performs initial exploration.
    * **Preprocessing**: Implements Label Encoding for 'Gender', One-Hot Encoding for 'Geography', and Standard Scaling for numerical features.
    * **Model Construction**: Builds a Sequential ANN with two hidden layers and one output layer.
    * **Training**: Trains the model using the Adam optimizer and Binary Crossentropy loss, incorporating TensorBoard logging and Early Stopping for optimization.
    * **Artifact Saving**: Exports the trained `model.h5`, `scaler.pkl`, and encoder files for use in the application.
* **`prediction.ipynb`**: A notebook designed for testing the model on new data. It loads the saved model and artifacts to run predictions on specific input vectors, verifying the inference pipeline.
* **`app.py`**: A user-friendly web interface built with **Streamlit**. It provides a form for users to input customer details and displays the calculated churn probability in real-time.
* **Artifacts** (Generated during training):
    * `model.h5`: The trained Keras model.
    * `scaler.pkl`: The `StandardScaler` object for normalizing input features.
    * `label_encoder_gender.pkl`: Encoder for the 'Gender' categorical feature.
    * `onehot_encoder_geo.pkl`: Encoder for the 'Geography' categorical feature.

## 📊 Dataset

The model is trained on the `Churn_Modelling.csv` dataset, which contains **10,000 records** of bank customers.

* **Target Variable:** `Exited` (0 = Customer Stayed, 1 = Customer Churned).
* **Key Features Used:**
    * **Demographics:** Geography (France, Spain, Germany), Gender, Age.
    * **Financials:** Credit Score, Balance, Estimated Salary.
    * **Account Details:** Tenure, Number of Products, Has Credit Card, Is Active Member.
* **Preprocessing:**
    * Irrelevant columns (`RowNumber`, `CustomerId`, `Surname`) are dropped to prevent overfitting on non-predictive data.
    * Categorical variables are encoded: Label Encoding for Gender and One-Hot Encoding for Geography.
    * Numerical features are scaled using Standard Scaling to ensure efficient model convergence.

## 🛠️ Tech Stack

* **Python 3.x**
* **Deep Learning:** TensorFlow (Keras)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** TensorBoard
* **Deployment:** Streamlit

## 🧠 Model Architecture

The model is a Feed-Forward Neural Network (Sequential) designed with the following structure:

1.  **Input Layer**: Accepts 12 preprocessed features.
2.  **Hidden Layer 1**: 64 Neurons, `ReLU` activation function.
3.  **Hidden Layer 2**: 32 Neurons, `ReLU` activation function.
4.  **Output Layer**: 1 Neuron, `Sigmoid` activation function (outputs a probability between 0 and 1).

## 🚀 Installation & Setup

1.  **Clone the repository** (if applicable) or download the source files.

2.  **Install dependencies**:
    Ensure you have the required libraries installed. You can install them via pip:
    ```bash
    pip install tensorflow pandas numpy scikit-learn streamlit
    ```

3.  **Data Requirement**:
    Ensure the `Churn_Modelling.csv` file is located in the root directory before running the training notebook.

## 💻 Usage

### 1. Training the Model
Open and run `experiments.ipynb` in Jupyter Notebook or VS Code. This will:
* Preprocess the raw data.
* Train the Artificial Neural Network.
* Generate and save `model.h5`, `scaler.pkl`, and the encoder pickle files required for the app.

### 2. Running the Web App
Once the artifacts are generated, launch the Streamlit app from your terminal:

```bash
streamlit run app.py