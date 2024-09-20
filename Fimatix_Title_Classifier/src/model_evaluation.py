from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_random_forest(model, X, y, tfidf_vectorizer, dataset_name="Dataset"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} F1 Score: {f1:.4f}")
    print(f"{dataset_name} Classification Report:\n{report}")
    print(f"{dataset_name} Confusion Matrix:\n{conf_matrix}")

    #Feature Importance Plot
    importances = model.feature_importances_
    feature_names = X.columns
    #Identify TF-IDF feature names - map back from vectors
    tfidf_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_mapping = {f"tfidf_{i}": word for i, word in enumerate(tfidf_names)}
    feature_names = [tfidf_mapping.get(name, name) for name in feature_names]

    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['feature'][:20], feature_importances['importance'][:20])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances - {dataset_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('report/RF_feature_importance.png')
    plt.close()

    #confusion matrix
    plt.figure(figsize=(6, 5))
    plt.matshow(conf_matrix, cmap='Blues', fignum=1)
    plt.colorbar()
    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'RF Confusion Matrix - {dataset_name}')
    plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.savefig(f'report/RF_confusion_matrix_{dataset_name.lower()}.png')
    plt.close()

    return accuracy, f1


def evaluate_mlp(model, X, y, dataset_name="Dataset", batch_size=64, device='cpu'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    #convert data to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.long).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_pred = []
    y_true = []
    #pytorch logic!
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} F1 Score: {f1:.4f}")
    print(f"{dataset_name} Classification Report:\n{report}")
    print(f"{dataset_name} Confusion Matrix:\n{conf_matrix}")

    plt.figure(figsize=(6, 5))
    plt.matshow(conf_matrix, cmap='Blues', fignum=1)
    plt.colorbar()
    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'MLP Confusion Matrix - {dataset_name}')
    plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.savefig(f'report/MLP_confusion_matrix_{dataset_name.lower()}.png')
    plt.close()

    return accuracy, f1


def plot_training_metrics(loss_history, accuracy_history, f1_history, save_path='report/MLP_training_metrics.png'):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 4))

    #Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_history, 'r-', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    #Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracy_history, 'g-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    #Plot F1-Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_history, 'b-', label='F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training F1-Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()