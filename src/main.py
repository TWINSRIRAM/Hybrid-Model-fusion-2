import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import HybridModel
from data_loader import load_and_preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs, device):
    print(f"📊 Training for {epochs} epochs with {len(train_loader)} batches per epoch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)

    best_val_loss = float('inf')
    patience, wait = 5, 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

            _, predicted = torch.max(output.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{(correct / total):.4f}"
            })

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val)
                loss = criterion(output, y_val)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"🧪 Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=19)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_preprocess_data(data_dir, args.class_config)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    model = HybridModel(input_size=input_size, num_classes=num_classes)
    print("📡 Starting model training...")

    model = train_model(model, train_loader, val_loader, args.epochs, device)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    y_pred = label_encoder.inverse_transform(all_preds)
    y_true = label_encoder.inverse_transform(all_labels)

    print("\n🔍 Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
