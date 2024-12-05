import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import os

# Check for GPU availability
device = torch.device("cpu")
torch.set_default_device(device)

# Image loader and settings
imsize = 128  # Use smaller size if no GPU
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def load_and_resize_images(style_path, content_path):
    """
    Load the style and content images without altering the content image's resolution.
    Args:
        style_path (str): Path to the style image.
        content_path (str): Path to the content image.
    Returns:
        style_tensor, content_tensor: Loaded tensors for images.
    """
    style_image = Image.open(style_path)
    content_image = Image.open(content_path)

    # Convert images to tensors
    content_tensor = loader(content_image).unsqueeze(0).to(device, torch.float)

    # Resize the style image to match the content image dimensions if necessary
    style_image = style_image.resize(content_image.size, Image.LANCZOS)
    style_tensor = loader(style_image).unsqueeze(0).to(device, torch.float)

    return style_tensor, content_tensor



def imshow(tensor, title=None):
    """
    Display an image tensor using matplotlib.
    Args:
        tensor (Tensor): Image tensor to display.
        title (str, optional): Title of the image.
    """
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # Clone to prevent altering the tensor
    image = image.squeeze(0)      # Remove batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause for plot updates


class ContentLoss(nn.Module):
    """
    Loss class for content preservation during style transfer.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """
    Compute Gram matrix for style representation.
    """
    a, b, c, d = input.size()  # Batch size, feature maps, height, width
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # Compute gram product
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """
    Loss class for style matching during style transfer.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """
    Normalizes input using mean and std for pretrained models.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    """
    Build the style transfer model and losses.
    """
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    return optim.LBFGS([input_img.requires_grad_()])


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """
    Perform style transfer using the VGG19 network.
    """
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss = {style_score:.4f}, Content Loss = {content_score:.4f}")
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def on_top_of(style_path, content_path, output_path, start_x, start_y, width, height, num_steps=300):
    """
    Perform style transfer on a specific region of the content image and blend it with the original.
    Save the final output image.
    """
    # Load images
    style_img, content_img = load_and_resize_images(style_path, content_path)

    input_img = content_img.clone()

    # Region of Interest from the content image
    content_roi = content_img[:, :, start_y:start_y + height, start_x:start_x + width]

    # Resize the style image to match Region of Interest
    style_roi = transforms.Resize((content_roi.shape[2], content_roi.shape[3]))(style_img)


    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Run style transfer on the ROI
    output_roi = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_roi, style_roi, content_roi.clone(), num_steps=num_steps
    )

    # Replace the Region of Interest in the original content image with the stylized ROI
    input_img[:, :, start_y:start_y + height, start_x:start_x + width] = output_roi

    # To tensor to an image and save
    unloader = transforms.ToPILImage()
    output_image = unloader(input_img.squeeze(0).cpu())
    output_image.save(output_path)
    print(f"Stylized image saved to {output_path}")


def style_transfer(style_path, content_path, output_path, num_steps=300):
    """
    Perform style transfer on uploaded images and save the output.
    """
    style_img, content_img = load_and_resize_images(style_path, content_path)
    input_img = content_img.clone()

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img, num_steps=num_steps
    )

    unloader = transforms.ToPILImage()
    output_image = unloader(output.squeeze(0).cpu())
    output_image.save(output_path)
    print(f"Stylized image saved to {output_path}")


'''
if __name__ == "__main__":
    style_path = "/content/360_F_836750955_LVolMx61cAuIFVBbY2ttnR9msbCvkWFv.jpg"
    content_path = "/content/pexels-ekamelev-1096298.jpg"
    output_path = "./output.jpg"
    print(Select an option for editing
          1.- Whole Image transformation
          2.- Segment transformation)
    option = input('Type option:').strip()
    if option == '1':
        style_transfer(style_path, content_path, output_path)
    elif option == '2':
        start_x,start_y,width,height = map(int,input('type start_x, start_y, width and height in that order').strip().split())
        on_top_of(style_path, content_path, output_path, start_x, start_y, width, height, num_steps=300)
    else:
        print('Wrong input')
        '''