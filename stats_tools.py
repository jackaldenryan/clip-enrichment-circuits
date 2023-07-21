import torch
import torchvision
import numpy as np
import captum.optim as optimviz  # From the optim-wip branch of captum
from expanded_weights import (
    model,
    get_expanded_weights,
    top_connected_units,
    top_contributed_units
)
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Callable, Dict, List, Optional, Tuple
from classes import NeuronId
from IPython.display import display, Markdown, HTML
from activations_tool import (
    acts_for_images,
    act_for_units,
    map_microscope_layer,
    model
)
import pickle
import pandas as pd
from tqdm import tqdm
import scipy
plt.style.use('ggplot')


def show(x: torch.Tensor, figsize: Optional[Tuple[int, int]] = None, scale: float = 255.0):
    """Displays heatmap of expanded weights between two units. Copied from captum.optim(?).

    Side effects: Displays a heatmap of expanded weights.

    :param x: The heatmap to display. Expected to have 3 or 4 dimensions.
    :type x: torch.Tensor
    :param figsize: The size of the figure for the heatmap, defaults to None
                    which results in matplotlib's default figure size.
    :type figsize: Optional[Tuple[int, int]]
    :param scale: The scale factor to apply to the heatmap values, defaults to 255.0.
    :type scale: float, optional

    :return: This function does not return anything.
    :rtype: None
    """
    assert x.dim() == 3 or x.dim() == 4
    x = x[0] if x.dim() == 4 else x
    x = x.cpu().permute(1, 2, 0) * scale
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(x.numpy().astype(np.uint8))
    plt.axis("off")
    plt.show()


def contributions_stats(
    lower_layer: Tuple,
    upper_layer: Tuple,
    unit: int,
    backwardContrib: bool = True,
    numContributions: int = 3,
    sortBy: str = "mean",
    heatMaps: bool = True,
    contribMap: bool = True,
    meansHist: bool = True,
    magsHist: bool = True
):
    """Displays info on how a unit influences / is influenced by a layer's units.

    The influence is measured using "expanded weights", described here under 
    "Dealing with Indirect Interactions": 
    https://distill.pub/2020/circuits/visualizing-weights/. 
    Units are channels, connected by 2-dimensional expanded weights "filters", 
    just as neurons are connected by single weights in an FC layer.

    Side effects: This function prints and displays OpenAI microscope URLs for top 
    and bottom influenced/influential units in a layer with respect to the specified
    unit, sorted by a chosen statistic computed over the expanded weights. It also
    displays heatmaps of expanded weights, the mean influence of the unit for each 
    unit in the specified layer, histograms of means and magnitudes of expanded 
    weights between the unit and layer, and prints summary statistics about those histograms.

    :param lower_layer: The lower layer in the model.
    :type lower_layer: Tuple
    :param upper_layer: The upper layer in the model.
    :type upper_layer: Tuple
    :param unit: The index of the unit in the layer.
    :type unit: int
    :param backwardContrib: Whether to calculate backward contributions, defaults to True.
    :type backwardContrib: bool, optional
    :param numContributions: The # of top & bottom influenced/influential units, defaults to 3.
    :type numContributions: int, optional
    :param sortBy: The statistic by which to sort the units, defaults to "mean".
    :type sortBy: str, optional
    :param heatMaps: Whether to display heatmaps of weights, defaults to True.
    :type heatMaps: bool, optional
    :param contribMap: Whether to display the mean influence of the unit for each 
                       unit in the specified layer, defaults to True.
    :type contribMap: bool, optional
    :param meansHist: Whether to display histograms of means of expanded weights
                      between the unit and layer, defaults to True.
    :type meansHist: bool, optional
    :param magsHist: Whether to display histograms of magnitudes of expanded
                     weights between the unit and layer, defaults to True.
    :type magsHist: bool, optional

    :returns: A list of summary statistics about the histograms.
    :rtype: List
    """
    result = []
    weights = get_expanded_weights(lower_layer, upper_layer)

    # The below computes top_units, a Pandas df of influence stats computed on
    # expanded weights with columns [lower_unit_num, mean, magnitude, mad, L1, crispness, url]
    # sorted by magnitude by default in descending order.
    if backwardContrib:
        top_units = top_connected_units(
            weights, lower_layer, upper_layer, unit)
    else:
        top_units = top_contributed_units(
            weights, lower_layer, upper_layer, unit)

        # So that upper_layer is always the layer that the unit is in
        x = lower_layer
        lower_layer = upper_layer
        upper_layer = x

    # Print top contribution urls and heatmaps
    top = top_units.sort_values(by=[sortBy], ascending=False)[
        sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(top.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th top unit: " + "<a href=\"" +
                url+"\">" + str_contrib_unit + "</a>"))
        if heatMaps:
            if backwardContrib:
                heatmap = optimviz.weights_to_heatmap_2d(
                    weights[unit, contrib_unit, ...] / weights[unit, ...].max())
            else:
                heatmap = optimviz.weights_to_heatmap_2d(
                    weights[contrib_unit, unit, ...] / weights[:, unit, ...].max())
            show(heatmap)

    # Print bottom contribution urls and heatmaps
    top = top_units.sort_values(by=[sortBy], ascending=True)[
        sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(top.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th bottom unit: " + "<a href=\"" +
                url+"\">" + str_contrib_unit + "</a>"))
        if heatMaps:
            if backwardContrib:
                heatmap = optimviz.weights_to_heatmap_2d(
                    weights[unit, contrib_unit, ...] / weights[unit, ...].max())
            else:
                heatmap = optimviz.weights_to_heatmap_2d(
                    weights[contrib_unit, unit, ...] / weights[:, unit, ...].max())
            show(heatmap)

    # Print contribution heatmap
    if contribMap:
        meaned_weights = weights.mean(2).mean(2)
        if backwardContrib:
            mapWidth = 2**math.floor(
                math.log(len(meaned_weights[unit]), 2) / 2)
            mapHeight = len(meaned_weights[unit]) / mapWidth
            heatmap = meaned_weights[unit].reshape(
                (int(mapWidth), int(mapHeight)))
        else:
            mapWidth = 2**math.floor(
                math.log(len(meaned_weights[:, unit]), 2) / 2)
            mapHeight = len(meaned_weights[:, unit]) / mapWidth
            heatmap = meaned_weights[:, unit].reshape(
                (int(mapWidth), int(mapHeight)))
        ax = sns.heatmap(heatmap, linewidth=0.5, center=0)
        plt.figure(figsize=(8, 6))
        plt.show()

    # Fill result with stats
    positiveSum = 0
    positiveCount = 0
    negativeSum = 0
    negativeCount = 0
    for x in top_units["mean"]:
        if x >= 0:
            positiveSum += x
            positiveCount += 1
        else:
            negativeSum += x
            negativeCount += 1
    result.append(NeuronId(upper_layer, unit))
    result.append(np.mean(abs(top_units["mean"])))
    result.append(positiveSum/positiveCount)
    result.append(negativeSum/negativeCount)
    result.append(np.mean(top_units["magnitude"]))

    # Print histogram of means of expanded weights between unit and layer
    if meansHist:
        plot1 = plt.figure("mean:" + str(lower_layer) +
                           str(upper_layer) + str(unit))
        plt.hist(top_units["mean"], bins=200)
        ax1 = plt.axes()
        ax1.set_facecolor("white")
        plt.savefig('fig1.png', dpi=300)
        print("Mean of absolute value of means:" + str(result[1]))
        print("Mean of positive means:" + str(result[2]))
        print("Mean of negative means:" + str(result[3]))

    # Print histogram of magnitudes of expanded weights between unit and layer
    if magsHist:
        plot2 = plt.figure("magnitude:" + str(lower_layer) +
                           str(upper_layer) + str(unit))
        plt.hist(top_units["magnitude"], bins=200)
        ax2 = plt.axes()
        ax2.set_facecolor("white")
        print("Mean of magnitudes:" + str(result[4]))
    return result


def imageActDistribution(layer: Tuple, unit: int, numImages: int, image=None, filename=None):
    """Output info to see if an image causes high unit activation relative to random images. 

    Side effects: Prints median and MAD for the set of activations of the unit over
    the random ImageNet images, prints the activation for the input image and the # of MADs
    away that activation is from the median activation for the ImageNet images, 
    plots the histogram of activations for ImageNet images with activation for 
    input image overlaid, and saves the activations to a file.

    :param layer: The layer in the model containing the specified unit.
    :type layer: Tuple
    :param unit: The index of the unit in the layer.
    :type unit: int
    :param numImages: The number of random ImageNet images to use for the distribution.
    :type numImages: int
    :param image: An optional image tensor. If provided, the function will 
                  also compute the activation of the specified unit for this 
                  image and print its relation to the computed baseline 
                  distribution in terms of Median Absolute Deviation (MAD).
    :type image: Optional[torch.Tensor]
    :param filename: An optional filename. If provided, the function will save
                     the generated activations to a file with this name.
    :type filename: Optional[str]

    :return: This function does not return anything.
    :rtype: None
    """
    layer = map_microscope_layer(model, layer)

    # This function returns a list of activations of the unit over numImages
    # random ImageNet images as well as the median and MAD of the activations
    acts, median, MAD = acts_for_images(layer, unit, numImages)
    print("Median: " + str(median) + ", MAD: " + str(MAD))

    # Compute activation for image, print activation and # of MADs from median for random images
    if image is not None:
        act = act_for_units(layer, image, unit).item()
        numMADs = (act - median) / MAD
        print("Activation for image: " + str(act))
        print("Number of MADs: " + str(numMADs))

    # Plot histogram(s) of activations
    plot1 = plt.figure("Activations")
    numBins = int(math.floor(numImages / 2))
    plt.hist(acts, bins=numBins)
    if image is not None:
        plt.hist([act], bins=10*numBins)
    plt.legend(loc='upper right')

    # Save activations to filename
    if filename is not None:
        with open(filename, 'wb') as handle:
            pickle.dump(acts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def versatility_stats(
    lower_layer: Tuple,
    upper_layer: Tuple,
    numUnits: int,
    sortBy: str = "magMeans",
    numContributions: int = 3,
    meanAbsValMeansHist: bool = True,
    meanPositiveMeansHist: bool = True,
    meanNegativeMeansHist: bool = True,
    magMeansHist: bool = True,
    meanMagsHist: bool = True,
    magMagsHist: bool = True,
    meanMeansHist: bool = True
):
    """Prints highly influenced units, and displays distributions of influencedness.

    Influencedness (or versatility/influentiality, when computing influence in the other direction) 
    is measured for a unit with respect to a prior layer, and is most straighforwardly measured by
    taking the mean of the absolute value of all the contribution strengths between the unit and the
    layer. This function also allows for measuring "positive influencedness" and
    "negative influencedness" as well as other possible measures, using sortBy. For each measure, a 
    histogram of that measure's value over all upper units is displayed. Contribution strengths are 
    measured as either the mean or magnitude of expanded weights between two units.

    Side effects: Shows progress bar for looping over upper units, prints urls for 
    most and least influenced units (depends on value of sortBy), displays histograms 
    of the different statistics (ranging over all upper units) with each stat being a summary of
    the distribution of contributions between the unit and a lower layer, and prints the mean (over 
    upper units) of the distribution of the mean (over lower units) of means of expanded weights.

    :param lower_layer: The lower layer in the model.
    :type lower_layer: Tuple
    :param upper_layer: The upper layer in the model.
    :type upper_layer: Tuple
    :param numUnits: The # of upper units to consider.
    :type numUnits: int
    :param sortBy: The measure of influentiality to use. Default is "magMeans".
    :type sortBy: str, optional
    :param numContributions: The # of most and least influential units to print. Default is 3.
    :type numContributions: int, optional
    :param meanAbsValMeansHist: Flag indicating whether to generate histogram for the meansof the
                                absolute value of the means of expanded weights. Default is True.
    :type meanAbsValMeansHist: bool, optional
    :param meanPositiveMeansHist: Flag indicating whether to generate histogram for the means of
                                  the positive means of expanded weights. Default is True.
    :type meanPositiveMeansHist: bool, optional
    :param meanNegativeMeansHist: Flag indicating whether to generate histogram for the means of
                                  the negative means of expanded weights. Default is True.
    :type meanNegativeMeansHist: bool, optional
    :param magMeansHist: Flag indicating whether to generate histogram for the magnitudes of
                         the means of expanded weights. Default is True.
    :type magMeansHist: bool, optional
    :param meanMagsHist: Flag indicating whether to generate histogram for the means of
                         the magnitudes of expanded weights. Default is True.
    :type meanMagsHist: bool, optional
    :param magMagsHist: Flag indicating whether to generate histogram for the magnitudes
                        of the magnitudes of expanded weights. Default is True.
    :type magMagsHist: bool, optional
    :param meanMeansHist: Flag indicating whether to generate histogram for the
                          means of the means of expanded weights. Default is True.
    :type meanMeansHist: bool, optional

    :return: A DataFrame where each row, corresponding to each upper unit, contains 
             stats on the distribution of "contribution strengths" to the upper
             unit from the given lower layer. Contribution strengths are measured
             as either the mean or magnitude of expanded weights between two units.
    :rtype: pandas.DataFrame
    """
    numberUpperUnits = numUnits
    stats = []
    weights = get_expanded_weights(lower_layer, upper_layer)

    # Loop over upper layer units, computing statistics on the distribution of
    # contribution strengths between the unit and a lower layer, adding the stats
    # to the stats object
    for u in tqdm(range(numberUpperUnits)):
        result = []

        # top_units is a Pandas df of influence stats computed on expanded weights
        # with columns [lower_unit_num, mean, magnitude, mad, L1, crispness, url]
        # sorted by magnitude by default in descending order.
        top_units = top_connected_units(weights, lower_layer, upper_layer, u)
        positiveSum = 0
        positiveCount = 0
        negativeSum = 0
        negativeCount = 0
        for x in top_units["mean"]:
            if x >= 0:
                positiveSum += x
                positiveCount += 1
            else:
                negativeSum += x
                negativeCount += 1
        result.append(NeuronId(upper_layer, u))
        result.append(np.mean(np.abs(top_units["mean"])))
        result.append(positiveSum/positiveCount)
        result.append(-negativeSum/negativeCount)
        result.append(np.mean(top_units["magnitude"]))
        result.append(np.linalg.norm(top_units["mean"]))
        result.append(np.linalg.norm(top_units["magnitude"]))
        result.append(np.mean(top_units["mean"]))
        stats.append(result)

    # Each row of this object, corresponding to each upper unit, contains stats on the dist. of
    # "contribution strengths" to the upper unit from the given lower layer. Contribution strengths
    # are measured as either the mean or magnitude of expanded weights between two units.
    versatilityStats = pd.DataFrame(stats, columns=[
                                    "upper_unit_num", "meanAbsValMeans",
                                    "meanPositiveMeans", "meanNegativeMeans",
                                    "meanMags", "magMeans", "magMags", "meanMeans"])

    # Print urls for most influenced/influential units. For instance, if sortBy
    # is "meanPositiveMeans" then this will sort upper units in order of
    # "most positively influenced by lower units." If top_units is computed using
    # top_contributed_units, then these would be "most influential" or "most versatile."
    top = versatilityStats.sort_values(by=[sortBy], ascending=False)[
        sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(top.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th top unit: " + "<a href=\"" +
                url+"\">" + str_contrib_unit + "</a>"))

    # Print urls for least influenced/influential units. See previous comment for details.
    bottom = versatilityStats.sort_values(by=[sortBy], ascending=True)[
        sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(bottom.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th bottom unit: " + "<a href=\"" +
                url+"\">" + str_contrib_unit + "</a>"))

    # Display histograms of the different statistics (ranging over all upper units) with each stat
    # being a summary of the distribution of contributions between the unit and a lower layer
    if meanAbsValMeansHist:
        plot1 = plt.figure("meanAbsValMeans:" +
                           str(lower_layer) + str(upper_layer))
        plot1.suptitle(
            "Histogram of mean of absolute value of forward mean-contributions", fontsize=12)
        plt.hist(versatilityStats["meanAbsValMeans"], bins=200)
    if meanPositiveMeansHist:
        plot2 = plt.figure("meanPositiveMeans:" +
                           str(lower_layer) + str(upper_layer))
        plot2.suptitle(
            "Histogram of mean of positive forward mean-contributions", fontsize=12)
        plt.hist(versatilityStats["meanPositiveMeans"], bins=200)
    if meanNegativeMeansHist:
        plot3 = plt.figure("meanNegativeMeans:" +
                           str(lower_layer) + str(upper_layer))
        plot3.suptitle(
            "Histogram of mean of negative forward mean-contributions", fontsize=12)
        plt.hist(versatilityStats["meanNegativeMeans"], bins=200)
    if magMeansHist:
        plot4 = plt.figure("magMeans:" + str(lower_layer) + str(upper_layer))
        plot4.suptitle(
            "Histogram of magnitude of forward mean-contributions", fontsize=12)
        plt.hist(versatilityStats["magMeans"], bins=200)
    if meanMagsHist:
        plot5 = plt.figure("meanMags:" + str(lower_layer) + str(upper_layer))
        plot5.suptitle(
            "Histogram of mean of forward magnitude-contributions", fontsize=12)
        plt.hist(versatilityStats["meanMags"], bins=200)
    if magMagsHist:
        plot6 = plt.figure("magMags:" + str(lower_layer) + str(upper_layer))
        plot6.suptitle(
            "Histogram of magnitude of forward magnitude-contributions", fontsize=12)
        plt.hist(versatilityStats["magMags"], bins=200)
    if meanMeansHist:

        # Print mean (over upper units) of the distribution of the mean
        # (over lower units) of means of expanded weights
        print("Mean of meanMeansHist:" +
              str(np.mean(versatilityStats["meanMeans"])))
        plot7 = plt.figure("meanMeans:" + str(lower_layer) + str(upper_layer))
        plot7.suptitle(
            "Histogram of mean of forward mean-contributions", fontsize=12)
        plt.hist(versatilityStats["meanMeans"], bins=200)
    return versatilityStats
