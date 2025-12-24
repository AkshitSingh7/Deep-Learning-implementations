import torch
import random


# 1. DATA GENERATION 

def synthetic_data(w, b, num_examples):
    """Generates y = Xw + b + noise"""
    # Generate X from standard normal distribution
    X = torch.normal(0, 1, (num_examples, len(w)))
    
    # Generate y
    y = torch.matmul(X, w) + b
    
    # Add noise
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
  
    num_examples = len(features)
    indices = list(range(num_examples))
    
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 2. MODEL DEFINITION 

class LinearRegressionScratch:
    def __init__(self, num_inputs, sigma=0.01):
        
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        """The matrix-vector product: y = Xw + b"""
        return torch.matmul(X, self.w) + self.b


# 3. LOSS FUNCTION

def squared_loss(y_hat, y):
    """Squared loss: 0.5 * (y_hat - y)^2"""
    
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 4. OPTIMIZER 
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        """Perform the gradient descent update step"""
        with torch.no_grad(): # We don't want to track gradients during the update
            for param in self.params:
                param -= self.lr * param.grad

    def zero_grad(self):
        """Reset gradients to zero after each step"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# 5. MAIN EXECUTION & TRAINING LOOP


# --- A. Setup Data ---
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# --- B. Setup Hyperparameters ---
batch_size = 10
lr = 0.03
num_epochs = 3
net = LinearRegressionScratch(num_inputs=2, sigma=0.01)
optimizer = SGD([net.w, net.b], lr)

# --- C. Training Loop ---
print("Start Training...")
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 1. Forward Pass
        y_hat = net.forward(X)
        
        # 2. Compute Loss
        l = squared_loss(y_hat, y).mean() 
        
        # 3. Backward Pass (Compute Gradients)
       
        optimizer.zero_grad() 
        l.backward()
        
        # 4. Update Parameters
        optimizer.step()
        
    # Print progress every epoch
    with torch.no_grad():
        train_l = squared_loss(net.forward(features), labels).mean()
        print(f'Epoch {epoch + 1}, Loss: {float(train_l):.6f}')


# 6. VERIFICATION

with torch.no_grad():
    print('\n--- Results ---')
    print(f'True w: {true_w}')
    print(f'Estimated w: {net.w.reshape(true_w.shape)}')
    print(f'Error in w: {true_w - net.w.reshape(true_w.shape)}')
    
    print(f'True b: {true_b}')
    print(f'Estimated b: {net.b}')
    print(f'Error in b: {true_b - net.b}')