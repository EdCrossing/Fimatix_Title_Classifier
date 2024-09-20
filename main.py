import argparse
import os
import joblib
import time
from src.data_preprocessing import prepare_datasets
from src.model_training import train_random_forest, train_mlp
from src.model_evaluation import evaluate_random_forest, evaluate_mlp, plot_training_metrics
from src.hyperparameter_optimise import optimize_hyperparameters

def save_model(model, model_type, path):
    if model_type == 'random_forest':
        joblib.dump(model, path)
    elif model_type == 'mlp':
        import torch
        torch.save(model.state_dict(), path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    #Argument parsing to select model type
    parser = argparse.ArgumentParser(description="Train Machine Learning Models")
    parser.add_argument('--model', type=str, choices=['random_forest', 'mlp'], default='random_forest',
                        help='Model to train')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                        help='Device to train the model on')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization for MLP')
    args = parser.parse_args()

    #file paths
    train_path = 'data/train_sections_data.csv'
    test_path = 'data/test_sections_data.csv'
    model_save_paths = {
        'random_forest': 'models/random_forest.joblib',
        'mlp': 'models/mlp.pth',
    }
    label_encoder_path = 'models/label_encoder.pkl'
    scaler_path = 'models/scaler.pkl'
    imputer_path = 'models/imputer.pkl'
    tfidf_vectorizer_path = 'models/tfidf_vectorizer.pkl'
    if not os.path.exists(train_path):
        print(f"Training file not found at {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"Testing file not found at {test_path}")
        return

    #prepare datasets
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, X_test, y_test, label_encoder, scaler, imputer, tfidf_vectorizer = prepare_datasets(
        train_path, test_path)

    #optuna
    if args.optimize and args.model == 'mlp':
        print("Starting hyperparameter optimization for MLP...")
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, device=args.device, n_trials=50)
        print("Best hyperparameters found:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

    #training
    start_time = time.time()
    if args.model == 'random_forest':
        print("Training Random Forest Classifier...")
        model = train_random_forest(X_train, y_train)
    elif args.model == 'mlp':
        print("Training MLP...")
        model, loss_history, accuracy_history, f1_history = train_mlp(X_train, y_train, device=args.device)
        # Plot training metrics
        plot_training_metrics(loss_history, accuracy_history, f1_history, save_path='report/MLP_training_metrics.png')
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training runtime: {runtime:.2f} seconds")

    #Evaluate the model, carry through tfidf
    if args.model == 'random_forest':
        print("Evaluating on Validation Set...")
        evaluate_random_forest(model, X_val, y_val, tfidf_vectorizer=tfidf_vectorizer, dataset_name="Validation")
        print("Evaluating on Test Set...")
        evaluate_random_forest(model, X_test, y_test, tfidf_vectorizer=tfidf_vectorizer, dataset_name="Test")
    elif args.model == 'mlp':
        print("Evaluating on Validation Set...")
        evaluate_mlp(model, X_val, y_val, dataset_name="Validation", device=args.device)
        print("Evaluating on Test Set...")
        evaluate_mlp(model, X_test, y_test, dataset_name="Test", device=args.device)

    #saving model and preprocessing
    print("Saving the trained model and preprocessing artifacts...")
    save_model(model, args.model, model_save_paths[args.model])
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)

    print("Complete.")


if __name__ == "__main__":
    main()