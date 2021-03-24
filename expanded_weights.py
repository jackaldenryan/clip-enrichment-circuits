# NOTE: You cannot use this with autoreload! If editing this file, please
# restart your Jupyter kernel afterward.
import captum.optim as optimviz
import torch
from clip import load
import pandas as pd

from typing import Dict, Tuple

from classes import Layer, NeuronId
from activations_tool import map_microscope_layer

m, transform = load("RN50", jit=False)

model = m.visual.eval()


def remove_padding(tensor: torch.Tensor) -> torch.Tensor:
    i = 0
    while tensor[0][0][i].sum().item() == 0:
        i += 1
    p = torch.nn.ConstantPad2d(-i, 0)

    return p(tensor)


def get_expanded_weights(lower_layer: Layer, upper_layer: Layer, input: torch.Tensor = torch.zeros((1, 3, 224, 224), requires_grad=False)):
    """
    Usage:

    get_expanded_weights(
        expanded_weights.model.layer4[0].conv3, expanded_weights.model.layer4[2].conv3)

    Gets the expanded weights of dimensions (upper output channels, lower output channels, height, width)
    """
    for _, module in model.named_modules():
        if hasattr(module, 'relu'):
            module.relu = torch.nn.Identity()

    upper_layer = map_microscope_layer(m, upper_layer)
    lower_layer = map_microscope_layer(m, lower_layer)

    weightmap = optimviz.circuits.extract_expanded_weights(
        model, lower_layer, upper_layer, model_input=input.to(device="cuda"))

    weightmap = remove_padding(weightmap)

    # We have to move to the CPU for some reason, and convert to doubles to have more supported torch operations
    weightmap = weightmap.to("cpu").to(torch.double)

    return weightmap


def top_connected_units(weights, lower_layer: Layer, upper_layer: Layer, upper_unit: int) -> pd.DataFrame:
    """
    :returns: a Pandas DataFrame with columns [lower_unit_num, mean, magnitude, variance], sorted by magnitude by default in descending order

    Example with clickable links:
    weightmap = top_connected_units((4, 1, 3), (4,1,5), unit)

    def make_clickable(val):
        # target _blank to open new window
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)

    weightmap.style.format({'url': make_clickable})
    """
    rows = []
    for lower_unit_num in range(weights.shape[1]):
        weightmap = weights[upper_unit, lower_unit_num, :, :]  # nxn
        mean = weightmap.mean(0).mean(0).item()
        magnitude = torch.linalg.norm(weightmap).item()
        mad = 0
        L1 = 0
        for v in weightmap:
            for w in v:
                mad += abs(mean - w)
                L1 += abs(w)
        mad = mad / len(weightmap)
        crispness = 0
        if L1 > 0:
            crispness = magnitude / L1
        nid = NeuronId(lower_layer, lower_unit_num)
        rows.append(
            [nid, mean, magnitude, mad, L1, crispness, nid.url()])

    df = pd.DataFrame(
        rows, columns=["lower_unit_num", "mean", "magnitude", "mad", "L1", "crispness", "url"])

    df = df.sort_values(by=["magnitude"], ascending=False)

    return df

def top_contributed_units(weights, lower_layer: Layer, upper_layer: Layer, lower_unit: int) -> pd.DataFrame:
    """
    :returns: a Pandas DataFrame with columns [upper_unit_num, mean, magnitude, variance], sorted by magnitude by default in descending order

    Example with clickable links:
    weightmap = top_connected_units((4, 1, 3), (4,1,5), unit)

    def make_clickable(val):
        # target _blank to open new window
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)

    weightmap.style.format({'url': make_clickable})
    """
    rows = []
    for upper_unit_num in range(weights.shape[1]): #is this correct?
        weightmap = weights[upper_unit_num, lower_unit, :, :]  # nxn
        mean = weightmap.mean(0).mean(0).item()
        magnitude = torch.linalg.norm(weightmap).item()
        mad = 0
        L1 = 0
        for v in weightmap:
            for w in v:
                mad += abs(mean - w)
                L1 += abs(w)
        mad = mad / len(weightmap)
        crispness = 0
        if L1 > 0:
            crispness = magnitude / L1
        nid = NeuronId(upper_layer, upper_unit_num)
        rows.append(
            [nid, mean, magnitude, mad, L1, crispness, nid.url()])

    df = pd.DataFrame(
        rows, columns=["upper_unit_num", "mean", "magnitude", "mad", "L1", "crispness", "url"])

    df = df.sort_values(by=["magnitude"], ascending=False)

    return df
