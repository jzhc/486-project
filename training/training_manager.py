import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# ---------------------------------------------------------
# 1. LOAD AND PREPARE THE DATA
# ---------------------------------------------------------
print("Loading extracted features...")
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# PyTorch CNNs expect images in the format: (Batch, Channels, Height, Width)
# Our MFCCs are (Batch, Height, Width). We add a "Channel" of 1 (like a grayscale image).
X = np.expand_dims(X, axis=1) 

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split the data: 80% for training, 20% for testing
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders feed the data to the model in small batches (16 chunks at a time)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------------------------------------
# 2. DEFINE THE CNN "MANAGER" ARCHITECTURE
# ---------------------------------------------------------
class ManagerCNN(nn.Module):
    def __init__(self):
        super(ManagerCNN, self).__init__()
        # First layer looks for basic patterns (edges, lines) in the MFCCs
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second layer looks for more complex combinations of those patterns
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Global Average Pooling forces the output to a consistent size, 
        # making the model very robust to slightly different audio lengths.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer (Outputs 2 numbers: Score for Clean, Score for Plane)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten the grid into a single 1D list
        x = self.fc(x)
        return x

# ---------------------------------------------------------
# 3. TRAIN AND EVALUATE THE MODEL
# ---------------------------------------------------------
model = ManagerCNN()
criterion = nn.CrossEntropyLoss() # Standard loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # The algorithm that updates the weights

print("Starting Training...")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()       # Clear old calculations
        outputs = model(batch_X)    # Guess the answer
        loss = criterion(outputs, batch_y) # Calculate how wrong the guess was
        loss.backward()             # Figure out how to fix the weights
        optimizer.step()            # Apply the fix
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# Test the model on the 20% of data it has never seen before
model.eval()
correct = 0
total = 0
with torch.no_grad(): # Turn off learning for testing
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1) # Pick the class with the highest score
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# Save the trained "brain" to your hard drive so you can use it in your routing script!
torch.save(model.state_dict(), "manager_model.pth")
print("Saved trained model to manager_model.pth")
