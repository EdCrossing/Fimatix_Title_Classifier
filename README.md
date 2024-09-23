# PDF Title Classifier

This project evaluates different machine learning architectures to classify sections of styled PDF documents as titles or non-titles based on the given features.

## Project Structure

- **main.py**: Entry point for training and evaluating models.
- **src/data_preprocessing.py**: Preprocessing module for feature extraction from text and layout.
- **src/model_training.py**: Functions for training Random Forest and MLP models.
- **src/model_evaluation.py**: Functions for evaluating trained models and visualizing metrics.
- **src/hyperparameter_optimise.py**: Code for hyperparameter optimization using Optuna.
- **models/**: Directory where trained models and preprocessing artifacts are saved.
- **data/**: Directory for storing training and test datasets.
- **report/**: Directory for storing visualizations of training metrics.
- **requirements.txt**: List of required Python packages.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EdCrossing/Fimatix_Title_Classifier
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
### Training the Model
#### Random Forest
    python main.py --model random_forest
train time: <5secs
#### Multi-layer Perceptitron (cuda or cpu)
    python main.py --model mlp --device cuda 
train time: <1min
#### Optimising MLP with optuna
    python main.py --model mlp --device cuda --optimize

### Inference and Evaluation
The model can be evaluated on both the validation and test datasets as part of the training process. Results are automatically printed in the console with plots being saved to /report

### Preprocessing
During training, the following preprocessing artifacts are saved:
- **TF-IDF Vectorizer**: Captures word-level features from text.
- **Scaler**: Normalizes numerical features.
- **Label Encoder**: Encodes class labels into numerical format.
- **Imputer**: Handles missing data by filling gaps with appropriate values.

### Results
The MLP achieves an accuracy of ~96.0% on the validation set and ~97.0% on the test set.
The RF achieves an accuracy of ~97.4% on the validation set and ~98.0% on the test set

### Model Choice + Discussion 
My first choice of model was a random forest regression model due to its ease to implement and ability to quickly test the data on. 
It is also quick to train and gives a view into what features might be more important to other models.

For the second model I chose a multilayer perceptron.
I chose the MLP as it would look for any more complex or non-linear relationships in the data and has more freedom in customising/optimising the model.

From the results the RF performed marginally better, this could be due to it being a small data set and mainly categorical features, where RF models excel.
The MLP may have performed slightly worse due to the volume of data and with the current features may not present a complex non-linear relation unlike other data sources. 
The RF may also be slightly overfitting to the data due to specific dates (2016/2015) appearing in the feature importance graph.

### Future Work
- Experiment with other features such as IsDate 
- Combine with spatial/CNN approach using the pdf images
- Combine multiple models
- Unit testing
- Error logging
- Improve readability 
