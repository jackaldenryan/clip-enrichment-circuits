import torch
import torchvision
import numpy as np
import captum.optim as optimviz
from expanded_weights import model, get_expanded_weights, top_connected_units, top_contributed_units
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Callable, Dict, List, Optional, Tuple
from classes import NeuronId
from IPython.display import display, Markdown, HTML
from activations_tool import acts_for_images, act_for_units, map_microscope_layer, model
import pickle
import pandas as pd
from tqdm import tqdm
import scipy
plt.style.use('ggplot')

def show(x: torch.Tensor, figsize: Optional[Tuple[int, int]] = None, scale: float = 255.0) -> None:
    assert x.dim() == 3 or x.dim() == 4
    x = x[0] if x.dim() == 4 else x
    x = x.cpu().permute(1, 2, 0) * scale
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(x.numpy().astype(np.uint8))
    plt.axis("off")
    plt.show()


#A catch-all function that lets you specify what contributions data you want to see.
#If you want to see some new data, edit this function and make an input bool for the data
def contributions_stats(lower_layer: Tuple, upper_layer: Tuple, unit: int, backwardContrib: bool=True, numContributions: int=3, sortBy: str="mean", heatMaps: bool=True, contribMap: bool=True, meansHist: bool=True, magsHist: bool=True):
    #Result will hold the avg reverse contributions (for returning the top units for this later)
    result = []
    weights = get_expanded_weights(lower_layer, upper_layer)
    if backwardContrib:
        top_units = top_connected_units(weights, lower_layer, upper_layer, unit)
    else:
        top_units = top_contributed_units(weights, lower_layer, upper_layer, unit)
        x = lower_layer
        lower_layer = upper_layer
        upper_layer = x
      
    #Print top contributions and heatmaps
    top = top_units.sort_values(by=[sortBy], ascending=False)[sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(top.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th top unit: " + "<a href=\""+url+"\">" + str_contrib_unit + "</a>"))
        if heatMaps:
            if backwardContrib:
                heatmap = optimviz.weights_to_heatmap_2d(weights[unit, contrib_unit, ...] / weights[unit, ...].max())
            else:
                heatmap = optimviz.weights_to_heatmap_2d(weights[contrib_unit, unit, ...] / weights[:, unit, ...].max())
            show(heatmap)

    #Print bottom contributions and heatmaps
    top = top_units.sort_values(by=[sortBy], ascending=True)[sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(top.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th bottom unit: " + "<a href=\""+url+"\">" + str_contrib_unit + "</a>"))
        if heatMaps:
            if backwardContrib:
                heatmap = optimviz.weights_to_heatmap_2d(weights[unit, contrib_unit, ...] / weights[unit, ...].max())
            else:
                heatmap = optimviz.weights_to_heatmap_2d(weights[contrib_unit, unit, ...] / weights[:, unit, ...].max())
            show(heatmap)

    #Print contribution heatmap
    if contribMap:
        meaned_weights = weights.mean(2).mean(2)
        if backwardContrib:
            mapWidth = 2**math.floor(math.log(len(meaned_weights[unit]), 2) / 2)
            mapHeight = len(meaned_weights[unit]) / mapWidth
            heatmap = meaned_weights[unit].reshape((int(mapWidth), int(mapHeight)))
        else:
            mapWidth = 2**math.floor(math.log(len(meaned_weights[:, unit]), 2) / 2)
            mapHeight = len(meaned_weights[:, unit]) / mapWidth
            heatmap = meaned_weights[:, unit].reshape((int(mapWidth), int(mapHeight)))
        ax = sns.heatmap(heatmap, linewidth=0.5, center=0)
        plt.figure(figsize=(8, 6))
        plt.show()
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
    #Print histogram of means
    if meansHist:
        plot1 = plt.figure("mean:" + str(lower_layer) + str(upper_layer) + str(unit))
        plt.hist(top_units["mean"], bins=200);
        ax1 = plt.axes()
        ax1.set_facecolor("white")
        plt.savefig('fig1.png', dpi = 300)
        print("Mean of absolute value of means:" + str(result[1]))
        print("Mean of positive means:" + str(result[2]))
        print("Mean of negative means:" + str(result[3]))
        

    #Print histogram of magnitudes
    if magsHist:
        plot2 = plt.figure("magnitude:" + str(lower_layer) + str(upper_layer) + str(unit))
        plt.hist(top_units["magnitude"], bins=200);
        ax2 = plt.axes()
        ax2.set_facecolor("white")
        print("Mean of magnitudes:" + str(result[4]))
    return result

#For a specific unit, plots distribution of activations for random imagenet images and gives comparison of
#the activation of some input image to this base-line distribution
def imageActDistribution(layer: Tuple, unit: int, numImages: int, image=None, filename=None):
    layer = map_microscope_layer(model, layer)
    acts, median, MAD = acts_for_images(layer, unit, numImages)
    print("Median: " + str(median) + ", MAD: " + str(MAD))

    if image is not None:
        act = act_for_units(layer, image, unit).item()
        numMADs = (act - median) / MAD
        print("Activation for image: " + str(act))
        print("Number of MADs: " + str(numMADs))

    plot1 = plt.figure("Activations")
    numBins = int(math.floor(numImages / 2))
    plt.hist(acts, bins=numBins);
    if image is not None:
        plt.hist([act], bins=10*numBins);
    plt.legend(loc='upper right')

    #Save activations to filename    
    if filename is not None:
        with open(filename, 'wb') as handle:
            pickle.dump(acts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def versatility_stats(lower_layer: Tuple, upper_layer: Tuple, numUnits: int, sortBy: str="magMean", numContributions: int=3, meanAbsValMeanHist: bool=True, positiveMeanHist: bool=True, negativeMeanHist: bool=True, magMeanHist: bool=True, meanMagHist: bool=True, magMagHist: bool=True, meanMeanHist: bool=True):
    numberUpperUnits = numUnits
    stats = []
    weights = get_expanded_weights(lower_layer, upper_layer)
    for u in tqdm(range(numberUpperUnits)):
        result = []
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
    versatilityStats = pd.DataFrame(stats, columns=["upper_unit_num", "meanAbsValMean", "positiveMean", "negativeMean", "meanMag", "magMean", "magMag", "meanMean"])

    top = versatilityStats.sort_values(by=[sortBy], ascending=False)[sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(top.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th top unit: " + "<a href=\""+url+"\">" + str_contrib_unit + "</a>"))
        
    bottom = versatilityStats.sort_values(by=[sortBy], ascending=True)[sortBy].head(n=numContributions)
    for i in range(numContributions):
        contrib_unit = int(list(bottom.index)[i])
        neuron = NeuronId(lower_layer, contrib_unit)
        url = neuron.url()
        str_contrib_unit = str(contrib_unit)
        display(HTML(str(i) + "th bottom unit: " + "<a href=\""+url+"\">" + str_contrib_unit + "</a>"))
    if meanAbsValMeanHist:
        plot1 = plt.figure("meanAbsValMean:" + str(lower_layer) + str(upper_layer))
        plot1.suptitle("Histogram of mean of absolute value of forward mean-contributions", fontsize=12)
        plt.hist(versatilityStats["meanAbsValMean"], bins=200);
    if positiveMeanHist:
        plot2 = plt.figure("positiveMean:" + str(lower_layer) + str(upper_layer))
        plot2.suptitle("Histogram of mean of positive forward mean-contributions", fontsize = 12)
        plt.hist(versatilityStats["positiveMean"], bins=200);
    if negativeMeanHist:
        plot3 = plt.figure("negativeMean:" + str(lower_layer) + str(upper_layer))
        plot3.suptitle("Histogram of mean of negative forward mean-contributions", fontsize = 12)
        plt.hist(versatilityStats["negativeMean"], bins=200);
    if magMeanHist:
        plot4 = plt.figure("magMean:" + str(lower_layer) + str(upper_layer))
        plot4.suptitle("Histogram of magnitude of forward mean-contributions", fontsize = 12)
        plt.hist(versatilityStats["magMean"], bins=200);
    if meanMagHist:
        plot5 = plt.figure("meanMag:" + str(lower_layer) + str(upper_layer))
        plot5.suptitle("Histogram of mean of forward magnitude-contributions", fontsize = 12)
        plt.hist(versatilityStats["meanMag"], bins=200);
    if magMagHist:
        plot6 = plt.figure("magMag:" + str(lower_layer) + str(upper_layer))
        plot6.suptitle("Histogram of magnitude of forward magnitude-contributions", fontsize = 12)
        plt.hist(versatilityStats["magMag"], bins=200);
    if meanMeanHist:
        print("Mean of meanMeanHist:" + str(np.mean(versatilityStats["meanMean"])))
        plot7 = plt.figure("meanMean:" + str(lower_layer) + str(upper_layer))
        plot7.suptitle("Histogram of mean of forward mean-contributions", fontsize = 12)
        plt.hist(versatilityStats["meanMean"], bins=200);
    return versatilityStats
        
    
