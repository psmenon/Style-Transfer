# Neural Style-Transfer

Neural style transfer is a technique that combines the content of one image with the style of another image using a convolutional neural network. Given a content image and a style image, the goal is to generate a target image that minimizes the content difference with the content image and the style difference with the style image.

## Content loss

The target image and content image are forward propagated to the pretrained VGG model, and feature maps are extracted from multiple convolutional layers. Then, the target image is updated to minimize the mean-squared error between the feature maps of the content image and its feature maps.

# Style Loss

The target image and style image are forward propagated to the pretrained VGG model, and feature maps are extracted from multiple convolutional layers. Then, the target image is updated to minimize the mean-squared error between the Gram matrix of the style image and the Gram matrix of the target image. 

For more details please refer original paper [[https://arxiv.org/abs/1508.06576]]

