import random
import json
import sys
import torch
from model import NeuralNet
from preprocessing import bag_of_words, tokenize

# Check if a GPU is available and select the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and other data from the JSON file
with open('intents.json', 'r') as f:
    content = json.load(f)

FILE = 'Parameter.pth'
# Load the saved model checkpoint with map_location set to 'cpu'
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Shara"
print("Let's chat! Type 'quit' to exit")

# Read input from command line arguments
if len(sys.argv) > 1:
    input_sentence = ' '.join(sys.argv[1:])
else:
    input_sentence = input("You: ")

while input_sentence != "quit":
    sentence = tokenize(input_sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in content["intents"]:
            if tag == intent["tag"]:
                response = f"{bot_name}: {random.choice(intent['responses'])}"
                sys.stdout.flush()  # Ensure the output is flushed immediately
    else:
        response = f"{bot_name}: I do not understand..."
        sys.stdout.flush()  # Ensure the output is flushed immediately

    # Read the next input
    if len(sys.argv) <= 1:
        input_sentence = input("You: ")
    else:
        break
