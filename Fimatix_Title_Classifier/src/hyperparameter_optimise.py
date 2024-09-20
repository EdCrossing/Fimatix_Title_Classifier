import optuna
import joblib
from src.model_training import objective
from datetime import datetime

def optimize_hyperparameters(X_train, y_train, X_val, y_val, device='cpu', n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, device), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    #Save the study and its params
    joblib.dump(study, 'models/optuna_study.pkl')
    with open('report/optuna_metrics.txt', 'w') as f:
        f.write(f"Best Trial at {timestamp}\n")
        f.write(f"Value: {trial.value}\n")
        f.write("Params:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
    return trial.params