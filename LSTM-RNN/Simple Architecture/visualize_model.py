import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import pickle
import os

# -------------------- Load the Model --------------------
model_path = 'trained_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    print("Model architecture saved to 'model.png'.")
else:
    print(f"Model file '{model_path}' not found!")

# -------------------- Load Training History --------------------
history_path = 'training_history.pkl'
if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
else:
    raise FileNotFoundError(f"Training history file '{history_path}' not found!")

# Extract values
acc = history.get('accuracy', [])
val_acc = history.get('val_accuracy', [])
loss = history.get('loss', [])
val_loss = history.get('val_loss', [])

# Handle length mismatches by truncating to the shortest length
min_epochs = min(len(acc), len(val_acc), len(loss), len(val_loss))
epochs = range(1, min_epochs + 1)

acc = acc[:min_epochs]
val_acc = val_acc[:min_epochs]
loss = loss[:min_epochs]
val_loss = val_loss[:min_epochs]

# -------------------- Plot Training History --------------------
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved to 'training_history.png'.")
plt.show()

