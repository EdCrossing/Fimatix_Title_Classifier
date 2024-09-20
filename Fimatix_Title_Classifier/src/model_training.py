from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.mlp import MLP
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def train_random_forest(X_train, y_train):
    #initialise + train
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    #predict and calc
    y_pred = rf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred, average='weighted')
    print(f"Random Forest Training Accuracy: {accuracy:.4f}")
    print(f"Random Forest Training F1 Score: {f1:.4f}")

    return rf

#values based off optuna trial
def train_mlp(X_train, y_train, hidden_sizes=[316, 88], epochs=50, batch_size=64, learning_rate=0.019139,dropout_rate = 0.103850, device='cpu'):
    #cpu or gpu
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #initialize model, loss function, and optimizer
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = MLP(input_size, hidden_sizes, num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #initialise plots
    loss_history=[]
    accuracy_history=[]
    f1_history=[]

    #Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            #collecting metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        f1_history.append(f1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

    return model, loss_history, accuracy_history, f1_history


def objective(trial, X_train, y_train, X_val, y_val, device='cpu'):
    # Suggest hyperparameters
    hidden_size1 = trial.suggest_int('hidden_size1', 32, 512)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.9)

    # Initialize model with suggested hyperparameters
    model = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=[hidden_size1, hidden_size2],
        num_classes=len(np.unique(y_train)),
        dropout_rate=dropout_rate
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop (e.g., 10 epochs for faster optimization)
    model.train()
    for epoch in range(1, 21):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train.values, dtype=torch.float32).to(device))
        loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.long).to(device))
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_val.values, dtype=torch.float32).to(device))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    # Calculate F1-score
    f1 = f1_score(y_val, preds, average='weighted')

    return f1