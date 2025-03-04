import torch
import torch.nn as nn

def model_v0_test(target, device):
    # Initialize the model
    model = target().to(device)
    
    try:
        model.layer_1
    except AttributeError:
        print("Error: 'layer_1' is not defined in the model.")
        return
    
    try:
        model.layer_2
    except AttributeError:
        print("Error: 'layer_2' is not defined in the model.")
        return

    # Testing correct layers if they exist
    assert isinstance(model.layer_1, nn.Linear), "Layer 1 should be of type nn.Linear. Make sure you have initialized it properly."
    assert model.layer_1.in_features == 2, f"Expected input features of layer 1 to be 2, but got {model.layer_1.in_features}"
    assert model.layer_1.out_features == 5, f"Expected output features of layer 1 to be 5, but got {model.layer_1.out_features}"
    
    assert isinstance(model.layer_2, nn.Linear), "Layer 2 should be of type nn.Linear. Make sure you have initialized it properly."
    assert model.layer_2.in_features == 5, f"Expected input features of layer 2 to be 5, but got {model.layer_2.in_features}"
    assert model.layer_2.out_features == 1, f"Expected output features of layer 2 to be 1, but got {model.layer_2.out_features}"
        
    print('\033[92mModel architecture is defined correctly!')

    
def grade_loss_fn(target):
    
    # Check if the loss_fn is correctly set to nn.BCEWithLogitsLoss
    assert isinstance(target, torch.nn.modules.loss.BCEWithLogitsLoss), \
        f"Error: loss_fn should be an instance of BCEWithLogitsLoss, but got {type(target)}"
    
    # Check if the class is correctly initialized
    print('\033[92mCorrectly initialized the loss function!')

def test_optimizer_configuration(target):
    """
    Tests if the optimizer is configured correctly with the model and learning rate 0.1.
    
    Parameters:
    - target: The class that configures the optimizer.
    """

    assert isinstance(target,torch.optim.SGD), \
        f"Error: loss_fn should be an instance of BCEWithLogitsLoss, but got {type(target)}"
    
    # Check if the learning rate is correctly set to 0.1
    lr = target.param_groups[0]['lr']

    # Assert that the learning rate is correctly set to 0.1
    assert lr == 0.1, f"Error: Please check the learning rate setting. The current value is:  {lr}"
    
    print('\033[92mOptimizer is correctly configured with learning rate 0.1!')
    

def test_model_v2(target, device):
    """
    Tests if the model is configured correctly with the layers and activation function.
    
    Parameters:
    - target: The model to test.
    - device: The device to which the model should be moved.
    """
  
    model = target().to(device)
    
    assert isinstance(model.layer_1, nn.Linear), "Error: layer_1 should be of type nn.Linear."
    assert model.layer_1.in_features == 2, f"Expected input features for layer_1 to be 2, but got {model.layer_1.in_features}"
    assert model.layer_1.out_features == 10, f"Expected output features for layer_1 to be 10, but got {model.layer_1.out_features}"

    assert isinstance(model.layer_2, nn.Linear), "Error: layer_2 should be of type nn.Linear."
    assert model.layer_2.in_features == 10, f"Expected input features for layer_2 to be 10, but got {model.layer_2.in_features}"
    assert model.layer_2.out_features == 10, f"Expected output features for layer_2 to be 10, but got {model.layer_2.out_features}"

    assert isinstance(model.layer_3, nn.Linear), "Error: layer_3 should be of type nn.Linear."
    assert model.layer_3.in_features == 10, f"Expected input features for layer_3 to be 10, but got {model.layer_3.in_features}"
    assert model.layer_3.out_features == 1, f"Expected output features for layer_3 to be 1, but got {model.layer_3.out_features}"

    # Check if the activation function is applied (it should be ReLU)
    assert isinstance(model.relu, nn.ReLU), "Error: The activation function should be nn.ReLU."

    # Test if the forward pass produces output with the expected shape
    sample_input = torch.randn(2, 2).to(device)  # batch of 2 samples, 2 features
    output = model(sample_input)
    
    # Assert that the output has the correct shape (1 output per sample)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), but got {output.shape}"

    print("\033[92mModel configuration is correct! All tests passed.")
