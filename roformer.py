from transformers import RoFormerModel, RoFormerTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "junnyu/roformer_chinese_base"  # Change model if needed
tokenizer = RoFormerTokenizer.from_pretrained(model_name)
model = RoFormerModel.from_pretrained(model_name)

# Encode a sample text
text = "Hello, this is a test for RoFormer."
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get the last hidden state
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)  # (batch_size, seq_length, hidden_size)