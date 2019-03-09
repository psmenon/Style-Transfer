# Neural style transfer

# import packages
import torch
from torch import nn,optim
from torchvision import models,transforms
from argparse import ArgumentParser
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

# load the pretrained model
vgg_model = models.vgg19(pretrained=True).features

for param in vgg_model.parameters():
    param.requires_grad_(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = vgg_model.to(device)

#helper functions

def load_image(img_path,max_size=400,shape=None):
    """
        the processed image
    """
    
    image = Image.open(img_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    img_transforms = transforms.Compose([transforms.Resize(size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))])
    
    
    pro_img = img_transforms(image)[:3,:,:].unsqueeze(0)
    
    return pro_img 


def get_features(image,model):
    """
        features of interest
    """
    
    layers = ['0','5','10','19','21','28']
    features = {}
    feature = image
    
    for name,layer in vgg_model._modules.items():
        feature = layer(feature)
        if name in layers:
            features[name] = feature
    
    return features


def gram_matrix(tensor):
    """
        Calculate the Gram Matrix of a given tensor 
    """
    
    _,d,h,w = tensor.size()
    matrix = tensor.reshape(d,h*w)
    gram = torch.mm(matrix,matrix.t())
    
    return gram

def plot_image(tensor):
    
    tensor_img = tensor.clone().cpu().detach().squeeze(0)
    tensor_img =tensor_img.numpy().transpose(1,2,0)
    tensor_img = tensor_img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    
    return tensor_img.clip(0,1)

# main function

def main(config):
    
    content = load_image(img_path=config.content,max_size=config.max_size).to(device)
    style = load_image(img_path=config.style,shape=content.size()[-2:]).to(device)
    
    content_features = get_features(content,vgg_model)
    style_features = get_features(style,vgg_model)
    
    target = content.clone().requires_grad_(True).to(device)
    
    style_weights = {'0':1,'5':0.8,'10':0.5,'19':0.3,'28':0.1}
    
    content_layer = '21'
    
    optimizer = optim.Adam([target],lr = config.lr)
    
    for epoch in range(1,config.epochs+1):
        
        target_features = get_features(target,vgg_model)
        content_loss = torch.mean((target_features[content_layer] - content_features[content_layer])**2)
    
        style_loss = 0
        for layer in style_weights:
            
            target_gram = gram_matrix(target_features[layer])
            style_gram = gram_matrix(style_features[layer])
            _,d,h,w = target_features[layer].shape
            
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d*h*w)
        
        total_loss = config.content_weight * content_loss + config.style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % config.show_every == 0:
            print('Total loss = {}'.format(total_loss.item()))
            plt.imshow(plot_image(target))
            plt.show()
        
        torch.save(vgg_model.state_dict(),'checkpoint.pth')
            
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--content',type=str,help='enter path for content image')
    parser.add_argument('--style',type=str,help='enter path for style image')
    parser.add_argument('--max_size',type=int,default=400,help='enter max size of image')
    parser.add_argument('--style_weight',type=float,default=1e6,help='enter style weight')
    parser.add_argument('--content_weight',type=float,default=1,help='enter content weight')
    parser.add_argument('--lr',type=float,default=0.003,help='enter learning rate')
    parser.add_argument('--epochs',type=int,default=2000,help='enter number of epochs')
    parser.add_argument('--show_every',type=int,default=400,help='enter number of times to show image during training')
    
    config = parser.parse_args()
    main(config)