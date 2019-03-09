# Neural Style-Transfer

Neural style transfer is a technique that combines the content of one image with the style of another image using a convolutional neural network. Given a content image and a style image, the goal is to generate a target image that minimizes the content difference with the content image and the style difference with the style image.

## Content loss

The target image and content image are forward propagated to the pretrained VGG model, and feature maps are extracted from multiple convolutional layers. Then, the target image is updated to minimize the mean-squared error between the feature maps of the content image and its feature maps.

## Style Loss

The target image and style image are forward propagated to the pretrained VGG model, and feature maps are extracted from multiple convolutional layers. Then, the target image is updated to minimize the mean-squared error between the Gram matrix of the style image and the Gram matrix of the target image. 

For more details please refer original [paper](https://arxiv.org/abs/1508.06576)

## Training in Linux

(1) Create and activate a Python 3.6 environment using Anaconda:
   
  ```bash
  conda create --name name_of_environment python=3.6
  source activate name_of_environment
  ```
  
(2) Clone repository and install dependencies

```bash
git clone https://github.com/psmenon/Style-Transfer.git
pip install -r requirements.txt
```

(3) Run the script 
```bash
$ python main.py --content=name_of_content_image --style=name_of_style_image
```

