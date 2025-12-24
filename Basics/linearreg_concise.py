import torch
from torch import nn


class LinearRegression(nn.Module): 
    def __init__(self, input_dim, lr=0.03):
        super().__init__()
        self.lr = lr
        
        # nn.Linear handles weight and bias initialization automatically
        # We use 1 output for regression
        self.net = nn.Linear(input_dim, 1)
        
        # Initializing weights to normal distribution
        self.net.bias.data.fill_(0)

    def forward(self, X):
        """Standard PyTorch forward pass"""
        return self.net(X)



# 1. Setup Data 
X = torch.randn(100, 2)
true_w = torch.tensor([[2.0], [-3.4]])
true_b = 4.2
y = X @ true_w + true_b + torch.randn(100, 1) * 0.01

# 2. Instantiate Model
model = LinearRegression(input_dim=2)

# 3. Define Loss and Optimizer 
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=model.lr)

# 4. The Training Loop
for epoch in range(3):
    # Forward pass
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}')


# VERIFICATION
w = model.net.weight.data
b = model.net.bias.data
print(f'\nError in w: {true_w.T - w}')
print(f'Error in b: {true_b - b}')