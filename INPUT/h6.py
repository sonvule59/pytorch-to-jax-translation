import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic

# Define a simple Language Model (e.g., an LSTM-based model)
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Use the last hidden state for prediction
        return self.softmax(output)
    
# Create synthetic training data
torch.manual_seed(42)
vocab_size = 50
seq_length = 10
batch_size = 32
X_train = torch.randint(0, vocab_size, (batch_size, seq_length))  # Random integer input
y_train = torch.randint(0, vocab_size, (batch_size,))  # Random target words

# Initialize the model, loss function, and optimizer
embed_size = 64
hidden_size = 128
num_layers = 2
model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Log progress every epoch
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Now, we will quantize the model dynamically to reduce its size and improve inference speed
# Quantization: Apply dynamic quantization to the language model
quantized_model = quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_language_model.pth")

