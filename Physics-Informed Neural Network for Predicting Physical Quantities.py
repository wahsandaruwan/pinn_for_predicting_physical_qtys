# ---------Imports---------
# pandas is a powerful library used for data manipulation and analysis
import pandas as pd

# numpy is a library for numerical computations and handling arrays/matrices
import numpy as np

# torch is the main package for deep learning with PyTorch
import torch
# nn (neural networks) is a subpackage in PyTorch to define and train models
import torch.nn as nn
# optim is used to implement optimization algorithms such as Adam
import torch.optim as optim

# train_test_split from scikit-learn splits data into training and testing sets
from sklearn.model_selection import train_test_split

# StandardScaler is used to standardize (scale) features in the dataset
from sklearn.preprocessing import StandardScaler

# ---------Read the dataset---------
file_path = '100_Data.csv'
data = pd.read_csv(file_path)

# ---------Define the feature matrix (X) and target variable (y)---------
# X: The feature matrix
# Selecting specific columns from the dataset 'data' to serve as features
# 'alpha', 'P', 'G', 'R', 'C', 'H' are the independent variables used to predict the target
X = data[['alpha', 'P', 'G', 'R', 'C', 'H']]

# y: The target variable
# Selecting the column 'F' from the dataset 'data' as the dependent variable to be predicted
y = data['F']

# Purpose:
# - `X` contains the input features required by the model for training and testing.
# - `y` contains the output variable that the model will learn to predict.

# ---------Split data into training and testing datasets---------
# X: The feature matrix (independent variables)
# y: The target variable (dependent variable)

# train_test_split: A function from sklearn.model_selection
# test_size=0.2: Specifies that 20% of the dataset will be used as the test set
# The remaining 80% will be used as the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Output:
# X_train: Feature data for training
# X_test: Feature data for testing
# y_train: Target data for training
# y_test: Target data for testing

# Purpose:
# This splits the dataset into separate training and testing subsets to evaluate the model's performance.
# The test set is kept separate to simulate unseen data and prevent overfitting.

# ---------Standardize training and testing datasets---------
# StandardScaler scales data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()

# Fit the scaler to the training feature data (X_train) and transform it
# The scaler calculates the mean and standard deviation from X_train
# Then it uses these statistics to standardize X_train
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test feature data (X_test) using the same scaler
# The scaler uses the mean and standard deviation calculated from X_train
# This ensures that the test data is scaled consistently with the training data
X_test_scaled = scaler.transform(X_test)

# Fit the scaler to the training target data (y_train) and transform it
# The target values are reshaped to a 2D array as required by StandardScaler
# Then it standardizes y_train using its mean and standard deviation
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

# Transform the test target data (y_test) using the same scaler
# Like X_test, y_test is scaled using the statistics from y_train
# This maintains consistency between training and test target data
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# ---------Create PyTorch tensors using training and testing datasets---------
# Convert the scaled training feature data (X_train_scaled) to a PyTorch FloatTensor
# This conversion is necessary because PyTorch models work with tensors as input
X_train_tensor = torch.FloatTensor(X_train_scaled)

# Convert the scaled training target data (y_train_scaled) to a PyTorch FloatTensor
# The target variable needs to be in tensor format for compatibility with PyTorch's loss functions
y_train_tensor = torch.FloatTensor(y_train_scaled)

# Convert the scaled test feature data (X_test_scaled) to a PyTorch FloatTensor
# These tensors will be used for model evaluation and predictions
X_test_tensor = torch.FloatTensor(X_test_scaled)

# Convert the scaled test target data (y_test_scaled) to a PyTorch FloatTensor
# This allows the evaluation of the model's predictions against the actual test targets
y_test_tensor = torch.FloatTensor(y_test_scaled)

# ---------Class for defining the physics informed neural network---------
# The model is a subclass of nn.Module, allowing it to utilize PyTorch's features for defining and training neural networks
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module
        super(PhysicsInformedNN, self).__init__()

        # Define the first fully connected (linear) layer
        # Input features: 6 (number of input parameters, e.g., alpha, P, G, R, C, H)
        # Output features: 50 (hidden layer neurons for learning complex representations)
        self.fc1 = nn.Linear(6, 50)

        # Define the second fully connected (linear) layer
        # Input features: 50 (from the previous layer)
        # Output features: 30 (fewer neurons in this layer to reduce complexity)
        self.fc2 = nn.Linear(50, 30)

        # Define the third fully connected (linear) layer
        # Input features: 30 (from the previous layer)
        # Output features: 1 (final output, e.g., the predicted F value)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, x):
        # Pass the input through the first fully connected layer followed by a ReLU activation function
        # ReLU introduces non-linearity to learn complex patterns in the data
        x = torch.relu(self.fc1(x))

        # Pass the result through the second fully connected layer followed by ReLU
        x = torch.relu(self.fc2(x))

        # Pass the result through the third fully connected layer to get the final output
        # No activation function here, as the output is expected to be a continuous value
        return self.fc3(x)

# ---------Initializes the model, loss function and optimizer---------
# Instantiate the Physics-Informed Neural Network (PINN) model
# This model incorporates physics-based equations into its architecture or loss function
model = PhysicsInformedNN()

# Define the Mean Squared Error (MSE) loss function
# This measures the average squared difference between predicted and true values
# Used as the primary loss function to train the model
mse_loss = nn.MSELoss()

# Define the optimizer for training the model
# Adam optimizer is chosen for its ability to handle sparse gradients and adaptive learning rates
# The learning rate is set to 0.001, which determines the step size during optimization
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------Function for calculating the physics informed loss---------
def physics_loss(alpha, P, G, R, C, H, F):
    # Small value to prevent numerical instabilities like division by zero
    epsilon = 1e-6  
    
    # Convert angles from degrees to radians for trigonometric calculations
    alpha_rad = alpha * np.pi / 180
    phi_rad = P * np.pi / 180

    # Calculate parameter A using an exponential decay model
    # Clamp alpha to ensure it stays within a stable range
    A = 10.50 * torch.exp(-0.009 * alpha.clamp(min=epsilon, max=1e2))
    
    # Calculate parameter B using a quadratic relationship with alpha
    B = 0.72 - (3.5e-5) * alpha**2 + 0.0031 * alpha

    # Calculate the tangent of phi and alpha, clamping to avoid extreme values
    tan_phi = torch.tan(phi_rad).clamp(min=epsilon, max=1e6)
    tan_alpha = torch.tan(alpha_rad).clamp(min=epsilon, max=1e6)
    
    # Compute the right-hand side (rhs) of the physics-informed equation
    rhs = (
        # First term: scaled by parameter A and raised to power B
        A * ((C / (G * H * tan_phi + epsilon)).clamp(min=epsilon, max=1e6)**B) * tan_phi +
        # Second term: ratio of tangents
        (tan_phi / tan_alpha) -
        # Third term: a combination of scaled coefficients and trigonometric terms
        ((C / (G * H * tan_alpha + epsilon)).clamp(min=epsilon, max=1e6) +
         (tan_phi / (torch.sin(alpha_rad) * torch.cos(phi_rad) + epsilon)))
        * R
    )
    
    # The left-hand side (lhs) is the observed value, F
    lhs = F

    # Compute the loss as the mean squared error (MSE) between lhs and rhs
    # Clamp the result to avoid extreme loss values
    return torch.mean((lhs - rhs).clamp(min=-1e6, max=1e6)**2)

# ---------Train the physics informed neural network---------
# Set the number of epochs for training
num_epochs = 1000

# Loop over the specified number of epochs
for epoch in range(num_epochs):
    # Set the model in training mode
    model.train()

    # Zero the gradients of the model parameters (to prevent accumulation from previous iterations)
    optimizer.zero_grad()

    # Forward pass: calculate predicted output by passing the input data through the model
    y_pred = model(X_train_tensor)

    # Calculate the Mean Squared Error (MSE) loss between predicted and true values
    mse = mse_loss(y_pred, y_train_tensor)

    # Calculate the Physics-Informed Loss using the custom loss function
    phys = physics_loss(X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2],
                        X_train_tensor[:, 3], X_train_tensor[:, 4], X_train_tensor[:, 5], y_pred)

    # Combine the MSE loss and Physics-Informed loss with a weight for the physics loss (0.01 here)
    total_loss = mse + 0.01 * phys  # The weight (0.01) can be adjusted based on the importance of physics loss

    # Backward pass: compute gradients for all parameters based on the total loss
    total_loss.backward()

    # Apply gradient clipping to avoid exploding gradients (keeping the gradients within a limit)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update the model parameters using the optimizer (after calculating gradients)
    optimizer.step()

    # Print the loss values every 100 epochs for monitoring training progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {mse.item():.4f}, Physics Loss: {phys.item():.4f}')

# ---------Evaluate the model---------
# Set the model to evaluation mode
model.eval()

# Disable gradient calculation for inference (saves memory and computation)
with torch.no_grad():
    # Perform a forward pass on the test data to get predictions
    y_pred = model(X_test_tensor)
    
    # Calculate the Mean Squared Error (MSE) loss between predicted and true values on the test set
    mse = mse_loss(y_pred, y_test_tensor)
    
    # Print the test MSE to evaluate model performance
    print(f'Test MSE: {mse.item():.4f}')

# ---------Actual and predicted values---------
# Print header for actual vs predicted values comparison
print("\nActual vs Predicted Values:")
print("Actual\t\tPredicted")

# Loop through the actual values and the predicted values to display them
# zip() pairs the actual and predicted values together for iteration
for actual, predicted in zip(y_test, y_pred.numpy().flatten()):
    # Print each pair of actual and predicted values, formatted to 4 decimal places
    print(f"{actual:.4f}\t\t{predicted:.4f}")

# ---------Feature importance (approximation based on gradient)---------
# Initialize an empty list to store feature importance values
feature_importance = []

# Loop through each feature in the training set
for i in range(X_train_tensor.shape[1]):
    # Create a clone of the training set tensor to avoid modifying the original data
    X_temp = X_train_tensor.clone()
    # Set requires_grad=True so we can compute gradients for each feature
    X_temp.requires_grad = True
    
    # Perform a forward pass through the model to get the predictions
    y_pred = model(X_temp)
    
    # Compute the gradients for the sum of the predictions with respect to the features
    y_pred.sum().backward()
    
    # Append the absolute mean of the gradients for the i-th feature to the list
    # This represents the importance of that feature based on the gradient magnitude
    feature_importance.append(X_temp.grad[:, i].abs().mean().item())

# Create a DataFrame to store feature names and their corresponding importance values
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})

# Sort the DataFrame by feature importance in descending order
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# Print the feature importance values in a readable format
print("\nFeature Importance:")
print(feature_importance_df)
