
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import json
import io
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model for inference"""
    logger.info(f"Loading model from {model_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the trained model weights first to inspect what's saved
    model_path = os.path.join(model_dir, "model.pth")
    logger.info(f"Loading model weights from {model_path}")
    
    try:
        # Try loading with weights_only=False for newer PyTorch versions
        try:
            saved_data = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Fall back to old torch.load for older PyTorch versions
            saved_data = torch.load(model_path, map_location=device)
        
        logger.info(f"Saved data keys: {list(saved_data.keys())}")
        
        # Extract the actual model state dict
        if 'model_state_dict' in saved_data:
            state_dict = saved_data['model_state_dict']
            logger.info("Found model_state_dict")
        elif 'state_dict' in saved_data:
            state_dict = saved_data['state_dict']
            logger.info("Found state_dict")
        else:
            # Assume the entire saved_data is the state_dict
            state_dict = saved_data
            logger.info("Using entire saved data as state_dict")
        
        logger.info(f"Model state dict keys: {list(state_dict.keys())[:10]}...")
        
        # Check if this is a complete model or just the classifier
        has_backbone = any('conv1' in key or 'layer1' in key for key in state_dict.keys())
        
        if has_backbone:
            # Complete model - load normally
            logger.info("Complete model detected")
            model = resnet18(pretrained=False)
            
            # Check if the model uses Sequential fc layer or simple Linear
            has_sequential_fc = any('fc.1.weight' in key for key in state_dict.keys())
            
            if has_sequential_fc:
                # Model has Sequential fc layer (with dropout)
                logger.info("Sequential FC layer detected")
                num_classes = state_dict['fc.1.weight'].shape[0]
                
                # Get dropout value from hyperparameters if available
                dropout_rate = 0.25  # default
                if 'hyperparameters' in saved_data and 'dropout' in saved_data['hyperparameters']:
                    dropout_rate = saved_data['hyperparameters']['dropout']
                    logger.info(f"Using dropout rate from hyperparameters: {dropout_rate}")
                
                # Create Sequential fc layer to match the saved model
                model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(model.fc.in_features, num_classes)
                )
                logger.info(f"Created Sequential fc with dropout={dropout_rate}, num_classes={num_classes}")
                
            else:
                # Standard Linear fc layer
                logger.info("Standard Linear FC layer detected")
                if 'fc.weight' in state_dict:
                    num_classes = state_dict['fc.weight'].shape[0]
                else:
                    num_classes = 133  # default
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                logger.info(f"Created Linear fc with num_classes={num_classes}")
            
            # Load the complete state dict
            try:
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded complete model state dict")
            except Exception as load_error:
                logger.error(f"Error loading state dict: {load_error}")
                # Print available keys for debugging
                logger.info(f"Available state dict keys: {list(state_dict.keys())}")
                logger.info(f"Model keys: {list(model.state_dict().keys())}")
                raise load_error
        else:
            # Only classifier saved - use pretrained backbone
            logger.info("Only classifier detected - using pretrained backbone")
            model = resnet18(pretrained=True)  # Use pretrained backbone
            
            # Try to get num_classes from hyperparameters first
            num_classes = 2  # default
            if 'hyperparameters' in saved_data:
                hyperparams = saved_data['hyperparameters']
                logger.info(f"Hyperparameters: {hyperparams}")
                if 'num_classes' in hyperparams:
                    num_classes = hyperparams['num_classes']
                elif 'n_classes' in hyperparams:
                    num_classes = hyperparams['n_classes']
            
            # Try to infer from classifier weights
            if 'fc.weight' in state_dict:
                num_classes = state_dict['fc.weight'].shape[0]
            elif 'weight' in state_dict:
                num_classes = state_dict['weight'].shape[0]
                
            logger.info(f"Number of classes: {num_classes}")
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Load only the classifier weights
            if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
                fc_state_dict = {'weight': state_dict['fc.weight'], 'bias': state_dict['fc.bias']}
                model.fc.load_state_dict(fc_state_dict)
                logger.info("Loaded fc weights from fc.weight and fc.bias")
            elif 'weight' in state_dict and 'bias' in state_dict:
                fc_state_dict = {'weight': state_dict['weight'], 'bias': state_dict['bias']}
                model.fc.load_state_dict(fc_state_dict)
                logger.info("Loaded fc weights from weight and bias")
            else:
                logger.error(f"Cannot load classifier weights. Available keys: {list(state_dict.keys())}")
                raise ValueError("Cannot load classifier weights")
        
        logger.info("Model weights loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise e
    
    model.to(device)
    model.eval()
    
    return model

def input_fn(request_body, content_type='application/x-image'):
    """Parse input data for inference"""
    logger.info(f"Processing input with content_type: {content_type}")
    
    if content_type == 'application/x-image':
        try:
            # Handle image input
            image = Image.open(io.BytesIO(request_body)).convert('RGB')
            logger.info(f"Image loaded successfully, size: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise e
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run prediction on the input data"""
    logger.info("Running prediction")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the same transforms used during training
    # Adjust these if your training used different transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Transform the input image
        input_tensor = transform(input_data).unsqueeze(0).to(device)
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        logger.info(f"Prediction completed: class={predicted_class}, confidence={confidence}")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist()[0]
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e

def output_fn(prediction, accept='application/json'):
    """Format the prediction output"""
    logger.info(f"Formatting output with accept: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
