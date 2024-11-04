import torch
from ...core.utils.config import configs 
from ...quantize.custom_quantized_format import build_quantized_network_from_cfg
from ...quantize.quantize_helper import create_scaled_head, create_quantized_head

from ...custom_op.register import register_SVD_with_var, register_SVD, register_HOSVD_with_var, register_filter, register_quantized_SVD, register_quantized_filter

__all__ = ['build_mcu_model']

def get_all_conv_with_name(model):
    conv_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            conv_layers[name] = mod
    return conv_layers

def build_mcu_model():
    cfg_path = f"assets/mcu_models/{configs.net_config.net_name}.pkl"
    cfg = torch.load(cfg_path)
    
    model = build_quantized_network_from_cfg(cfg, n_bit=8)
    
    # print("######################## MODEL BEFORE REGISTER GF ######################################")
    # print(model)

    if configs.net_config.gradfilt:
        num_of_finetune = configs.net_config.num_of_conv_finetune    

        all_convolution_layers = get_all_conv_with_name(model)
        
        finetuned_conv_layers = dict(list(all_convolution_layers.items())[-num_of_finetune:])
        filter_cfgs = {"cfgs": finetuned_conv_layers, "type": "conv", "radius": 2}
        if configs.net_config.quantized:
            register_quantized_filter(model, filter_cfgs)
        else:
            register_filter(model, filter_cfgs)
    
    # print("######################## MODEL AFTER REGISTER GF ######################################")
    
    # print(model)

    if configs.net_config.mcu_head_type == 'quantized':
        model = create_quantized_head(model)
    elif configs.net_config.mcu_head_type == 'fp':
        model = create_scaled_head(model, norm_feat=False)
    else:
        raise NotImplementedError

    # print("######################## MODEL AFTER CREATE CLASSIFIER HEAD ######################################")
    
    # print(model)
    return model
