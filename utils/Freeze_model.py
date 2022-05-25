def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requries_grad = requires_grad
        
def freeze_all_layers(module):
    set_module_requires_grad_(module, False)
    
def unfreeze_all_layers(module):
    set_module_requires_grad_(module, True)
    
def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers(model)