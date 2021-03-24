import re
import torch
import torchvision
from captum.attr import LayerActivation, LayerGradientXActivation
from clip import load
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

import copy

from classes import Layer

# hello world!
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = load("RN50", jit=False)


def map_microscope_layer(model: torch.nn.Module, layer: Layer) -> torch.nn.Module:
    microscope_layer_mappings = {
        (1, 0, 1): model.visual.layer1[0].conv1,
        (1, 0, 3): model.visual.layer1[0].conv2,
        (1, 0, 5): model.visual.layer1[0].conv3,
        (1, 0, 8): model.visual.layer1[0].relu,
        (1, 1, 1): model.visual.layer1[1].conv1,
        (1, 1, 3): model.visual.layer1[1].conv2,
        (1, 1, 5): model.visual.layer1[1].conv3,
        (1, 1, 6): model.visual.layer1[1].relu,
        (1, 2, 1): model.visual.layer1[2].conv1,
        (1, 2, 3): model.visual.layer1[2].conv2,
        (1, 2, 5): model.visual.layer1[2].conv3,
        (1, 2, 6): model.visual.layer1[2].relu,

        (2, 0, 1): model.visual.layer2[0].conv1,
        (2, 0, 3): model.visual.layer2[0].conv2,
        (2, 0, 5): model.visual.layer2[0].conv3,
        (2, 0, 7): list(list(model.visual.layer2[0].children())[8].children())[1],
        (2, 0, 8): model.visual.layer2[0].relu,
        (2, 1, 1): model.visual.layer2[1].conv1,
        (2, 1, 3): model.visual.layer2[1].conv2,
        (2, 1, 5): model.visual.layer2[1].conv3,
        (2, 1, 6): model.visual.layer2[1].relu,
        (2, 2, 1): model.visual.layer2[2].conv1,
        (2, 2, 3): model.visual.layer2[2].conv2,
        (2, 2, 5): model.visual.layer2[2].conv3,
        (2, 2, 6): model.visual.layer2[2].relu,
        (2, 3, 1): model.visual.layer2[3].conv1,
        (2, 3, 3): model.visual.layer2[3].conv2,
        (2, 3, 5): model.visual.layer2[3].conv3,
        (2, 3, 6): model.visual.layer2[3].relu,

        (3, 0, 1): model.visual.layer3[0].conv1,
        (3, 0, 3): model.visual.layer3[0].conv2,
        (3, 0, 5): model.visual.layer3[0].conv3,
        (3, 0, 7): list(list(model.visual.layer3[0].children())[8].children())[1],
        (3, 0, 8): model.visual.layer3[0].relu,
        (3, 1, 1): model.visual.layer3[1].conv1,
        (3, 1, 3): model.visual.layer3[1].conv2,
        (3, 1, 5): model.visual.layer3[1].conv3,
        (3, 1, 6): model.visual.layer3[1].relu,
        (3, 2, 1): model.visual.layer3[2].conv1,
        (3, 2, 3): model.visual.layer3[2].conv2,
        (3, 2, 5): model.visual.layer3[2].conv3,
        (3, 2, 6): model.visual.layer3[2].relu,
        (3, 3, 1): model.visual.layer3[3].conv1,
        (3, 3, 3): model.visual.layer3[3].conv2,
        (3, 3, 5): model.visual.layer3[3].conv3,
        (3, 3, 6): model.visual.layer3[3].relu,
        (3, 4, 1): model.visual.layer3[4].conv1,
        (3, 4, 3): model.visual.layer3[4].conv2,
        (3, 4, 5): model.visual.layer3[4].conv3,
        (3, 4, 6): model.visual.layer3[4].relu,
        (3, 5, 1): model.visual.layer3[5].conv1,
        (3, 5, 3): model.visual.layer3[5].conv2,
        (3, 5, 5): model.visual.layer3[5].conv3,
        (3, 5, 6): model.visual.layer3[5].relu,

        (4, 0, 1): model.visual.layer4[0].conv1,
        (4, 0, 3): model.visual.layer4[0].conv2,
        (4, 0, 4): model.visual.layer4[0].avgpool, # HACK(kevin): This isn't on microscope, but used in weights_and_acts
        (4, 0, 5): model.visual.layer4[0].conv3,
        (4, 0, 7): list(list(model.visual.layer4[0].children())[8].children())[1],
        (4, 0, 8): model.visual.layer4[0].relu,
        (4, 1, 1): model.visual.layer4[1].conv1,
        (4, 1, 3): model.visual.layer4[1].conv2,
        (4, 1, 5): model.visual.layer4[1].conv3,
        (4, 1, 6): model.visual.layer4[1].relu,
        (4, 2, 1): model.visual.layer4[2].conv1,
        (4, 2, 3): model.visual.layer4[2].conv2,
        (4, 2, 5): model.visual.layer4[2].conv3,
        (4, 2, 6): model.visual.layer4[2].relu
    }

    return microscope_layer_mappings[tuple(layer)]


def activations(layer, img):
    image_input = transform(img).unsqueeze(0).to(device)

    layer_act = LayerActivation(model.visual, layer)
    return layer_act.attribute(image_input)

# Get layer for block, bottleneck, index
# MUST USE MICRSCOPE INDEXING SYSTEM
# E.g. 2/2/add_3 -> block 2, bottleneck 2, index 3


def get_layer(block, bottleneck, index):
    return map_microscope_layer(model, (block, bottleneck, index))

# Get activation for neuron in layer
# (x, y) spatial position, z channel index


def act_for_neuron(layer, image, x, y, z):
    return activations(layer, image)[0][z][x][y].item()

def act_for_units(layer1, layer2, image, z):
    acts1 = activations(layer1, image)[0][z]
    mean1 = torch.mean(acts1).item()
    mean2 = 0
    if layer2 is not None:
        acts2 = activations(layer2, image)[0][z]
        mean2 = torch.mean(acts2).item()
    return mean1 + mean2

def act_for_unit(layer, channel_num, image):
    acts = activations(get_layer(layer[0], layer[1], layer[2]), image)[0][channel_num]
    mean = torch.mean(acts).item()
    return mean

def mean_channel_acts(layer, image) -> List[float]:
    acts = activations(layer, image)[0]
    return [torch.mean(act).item() for act in acts]


def highest_channel_for_image(layer, image) -> Tuple[int, float]:
    """
    :returns: index of maximally activating channel, maximum activation
    """
    mean_acts = mean_channel_acts(layer, image)
    return np.argmax(mean_acts), np.amax(mean_acts)


def top_n_highest_channels(n, layer, image) -> Dict[int, float]:
    mean_acts = mean_channel_acts(layer, image)

    unsorted_dict = dict()
    for idx, act in enumerate(mean_acts):
        unsorted_dict[idx] = act

    sorted_dict = dict(sorted(unsorted_dict.items(),
                              key=lambda item: item[1], reverse=True))
    return {k: sorted_dict[k] for k in list(sorted_dict)[:n]}


def top_n_channels_ga(n, layer, image) -> Dict[int, float]:
    image_input = transform(image).unsqueeze(0).to(device)

    layer_ga = LayerGradientXActivation(model.visual, layer)
    attribute = layer_ga.attribute(image_input, 0)[0]

    channel_attributions = [torch.mean(ga).item() for ga in attribute]

    unsorted_dict = dict()
    for idx, ga in enumerate(channel_attributions):
        unsorted_dict[idx] = ga

    sorted_dict = dict(sorted(unsorted_dict.items(),
                              key=lambda item: item[1], reverse=True))
    return {k: sorted_dict[k] for k in list(sorted_dict)[:n]}


# def highest_n_channels_for_image(layer, image, n) -> Dict[int, float]:
# 	mean_acts = mean_channel_acts(layer, image)
# 	return dict()

# for a layer


def highest_neuron_for_image(layer, image):

    acts = activations(layer, image)[0]
    max_act = torch.max(acts[0]).item()

    max_index = 0
    for idx, act in enumerate(acts):
        mx = torch.max(act).item()
        if mx > max_act:
            max_act = mx
            max_index = idx
    return max_index, max_act

# Whole network


def highest_unit(image):
    max_layer_name = ""
    max_index = 0
    max_act = 0
    for name, layer in model.visual.named_modules():
        if isinstance(layer, torch.nn.ReLU) | isinstance(layer, torch.nn.Conv2d):
            highest_channel, highest = highest_channel_for_image(layer, image)
            if highest > max_act:
                max_act = highest
                max_index = highest_channel
                max_layer_name = name
    return max_layer_name, max_index, max_act


def channel_difference_engine(layer, image: Image, baseline: Image) -> Dict[int, float]:
    """
    Computes the difference in activation distribution between two images.
    """

    baseline_acts = mean_channel_acts(layer, baseline)
    img_acts = mean_channel_acts(layer, image)

    diff = np.subtract(img_acts, baseline_acts)
    unsorted_dict = dict()
    for idx, act in enumerate(diff):
        unsorted_dict[idx] = act

    return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))


def channel_intersection(layer, img1, img2) -> Dict[int, float]:
    acts1 = mean_channel_acts(layer, img1)
    acts2 = mean_channel_acts(layer, img2)

    mins = np.minimum(acts1, acts2)

    unsorted_dict = dict()
    for idx, act in enumerate(mins):
        unsorted_dict[idx] = act

    return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))

#Return a list of activations for a unit on a number of random imagenet images
#Also returns the median and MAD of the distribution of image activations
def acts_for_images(layer, channel, numImages):
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    imagenet_data = torchvision.datasets.ImageNet(
        '../../seri/datasets/imagenet', transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=16)
    acts = []
    median = 0
    MAD = 0
    for i in tqdm(range(numImages)):
        images, labels = next(iter(data_loader))
        image = images[0]
        image = transforms.ToPILImage()(image)
        activation = act_for_units(layer, image, channel)
        activation = activation.cpu().numpy().item()
        acts.append(activation)
        median += activation / numImages
    #Mean Absolute Deviation is the 'b' parameter from the "Laplace Distribution" wiki page
    #It is how much of change in activation leads to a factor of e change in probability
    for a in acts:
        MAD += (abs(median - a)) / numImages
    return (acts, median, MAD)


class Submodel(torch.nn.Module):
    def hook(self, module, input, output):
        print("HOOKED: output size", output.shape)
        self.output = output

    def __init__(self, model, ending_layer: Layer):
        super(Submodel, self).__init__()
        target: torch.nn.Module = map_microscope_layer(model, ending_layer)

        self.model = model.visual

        target.register_forward_hook(lambda *args: self.hook(*args))

    def forward(self, x):
        self.model(x)
        return self.output


def create_submodel_visual(ending_layer: Layer):
    target = map_microscope_layer(model, ending_layer)
    should_replace = False
    for name, module in model.named_modules():
        if should_replace:
            indexable_name = re.sub(r"\.([0-9])", r'[\1]', name)
            exec(f"model.{indexable_name} = torch.nn.Identity()")
        if target == module:
            should_replace = True

    return model
