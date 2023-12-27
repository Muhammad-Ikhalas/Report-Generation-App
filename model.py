import torch
import torch.nn as nn
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import io


from basecmnmodel import BaseCMNModel
from tokenizer import Tokenizer
from test import Tester


# model = torch.load('./model_iu_xray.pth', map_location=torch.device('cpu'))
# model.eval()  
# Load your PyTorch model (replace 'model.pth' with your actual model file)
# model = torch.load('model_iu_xray.pth')
# model.eval()  # Set the model to evaluation mode

def read_stdin_binary():
    return sys.stdin.buffer.read()

args = {
        'image_dir': '/content/drive/MyDrive/iu_xray/images',
        'ann_path': './annotation.json',
        'dataset_name': 'iu_xray',
        'max_seq_length': 60,
        'threshold': 3,
        'num_workers': 2,
        'batch_size': 16,
        'visual_extractor': 'resnet101',
        'visual_extractor_pretrained': True,
        'd_model': 512,
        'd_ff': 512,
        'd_vf': 2048,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'logit_layers': 1,
        'bos_idx': 0,
        'eos_idx': 0,
        'pad_idx': 0,
        'use_bn': 0,
        'drop_prob_lm': 0.5,
        'topk': 32,
        'cmm_size': 2048,
        'cmm_dim': 512,
        'sample_method': 'beam_search',
        'beam_size': 3,
        'temperature': 1.0,
        'sample_n': 1,
        'group_size': 1,
        'output_logsoftmax': 1,
        'decoding_constraint': 0,
        'block_trigrams': 1,
        'n_gpu': 0,
        'epochs': 100,
        'save_dir': 'results/iu_xray',
        'record_dir': 'records/',
        'log_period': 1000,
        'save_period': 1,
        'monitor_mode': 'max',
        'monitor_metric': 'BLEU_4',
        'early_stop': 50,
        'optim': 'Adam',
        'lr_ve': 5e-5,
        'lr_ed': 7e-4,
        'weight_decay': 5e-5,
        'adam_betas': (0.9, 0.98),
        'adam_eps': 1e-9,
        'amsgrad': True,
        'noamopt_warmup': 5000,
        'noamopt_factor': 1,
        'lr_scheduler': 'StepLR',
        'step_size': 50,
        'gamma': 0.1,
        'seed': 9233,
        'resume': None,
        # 'load': "/content/results/iu_xray/model_best.pth",
        'load': "./model_best.pth",

    }

# Create tokenizer
tokenizer = Tokenizer(args)

# Build model architecture
model = BaseCMNModel(args, tokenizer)

# Check if the model is not None
if model is not None:
    # Move the model to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # print(device)

    # Read the image data from stdin
    image_data = read_stdin_binary()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # image_2 = Image.open(image_path_2).convert('RGB')

    image = transform(image)
        # image_2 = transform(image_2)

        # Stack the images along a new dimension
    image = torch.stack([image, image], dim=0)

        # Add batch dimension
    image = image.unsqueeze(0)

        # Move the input image to the same device as the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)






    # Generate the report
    with torch.no_grad():
        output, _ = model(image, mode='sample')

    generated_report = tokenizer.decode_batch(output.cpu().detach().numpy())[0]

    # Print the generated report
    print(f"Generated Report for Image: {generated_report}")
    with open('generated_report.txt', 'w') as file:
        file.write(generated_report)
    print("generated_report_successful")

else:
    print("Error: Model is None. Check the main function.")

