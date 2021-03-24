import math
import os
import numpy as np
from activations_tool import act_for_units
from torchvision import transforms

#Given a dataloader, unit, and desired number of images, produces list of act, image tuples
#representing the activations for each image from the dataloader for the unit in CLIP RN50
#The model is designated in activations_tool
def acts_and_imgs(dataloader, layer1, layer2, channel, numImages):
    result = []
    for i in range(numImages):
        images, labels = next(iter(dataloader))
        image = images[0]
        image = transforms.ToPILImage()(image)
        activation = act_for_units(layer1, layer2, image, channel)
        activation = activation.cpu().numpy()
        result.append((activation, image))
    return result

#Takes a list of act, image tuples and returns a dictionary from buckets (in standard deviations from mean)
#to lists of images that activate within that standard deviation bucket
#stdActLower and stdBucketWidth are in std units
#stdBucketWidth should divide stdActLower unless you want 0 in the middle of a bucket
def sort_acts_and_imgs(actsAndImgs, stdBucketWidth, 
                            stdActLower, numBuckets, minExamples, maxExamples):
    sortedImages = {}
    acts = [a for (a, i) in actsAndImgs]
    std = np.std(acts)
    mean = np.mean(acts)
    for i in range(numBuckets):
        bucketStd = stdActLower + i*stdBucketWidth + (stdBucketWidth / 2)
        sortedImages.update({bucketStd:[]})
    for (a, i) in actsAndImgs:
        #Calculate lower activation (in std) of bucket an act belongs in by calculating which bucket index act belongs to
        #(mean + stdActLower*std) is actLower in raw activation units
        index = math.floor((a - (mean + stdActLower*std)) / (stdBucketWidth*std))
        bucketStd = stdActLower + index*stdBucketWidth + (stdBucketWidth / 2)
        if bucketStd not in sortedImages.keys():
            continue
        images = sortedImages[bucketStd]
        images.append(i)
        sortedImages.update({bucketStd:images})
    return sortedImages

# sortedImages should then be unsorted and given to someone who only knows what neuron these images were selected for.
# That person then sorts the images into their own subjective categories based on how close to the neuron's feature
# each image is.

#This function takes in two dictionaries: sortedImages (activation levels mapped to lists of images that activate at that level)
#and humanSort (partition of the same sorted images into subjective labelled categories, represented as dictionary from labels to lists of images)
#The output of this function is a dictionary that maps activation, label to a list of images, which can be used to estimate
#the probability of a subjective category given an activation label
def act_distribution(sortedImages, humanSort):
    distribution = {} #dictionary from tuple(activation, label) to list (of images)
    for a in sortedImages:
        for s in sortedImages[a]:
            for l in humanSort.keys():
                if s in humanSort[l]:
                    if (a, l) not in distribution.keys():
                        distribution.update({(a, l):[]})
                    list1 = distribution[(a, l)]
                    list1.append(s)
                    distribution.update({(a, l):list1})
                    break
    return distribution



    '''
Deprecated but kept just in case


def get_images_at_activations(actsAndImgs, dataLoader, layer, channel, stdBucketWidth, 
                            stdActLower, numBuckets, minExamples, maxExamples):
    
    sortedImages = {} #dictionary from floats (lower activation of a bucket) to lists (of images)
    for i in range(numBuckets):
        sortedImages.update({activationLower + i*bucketWidth:[]})
    while minImageCounts < minExamples:
        
        #Get image and activation
        images, labels = next(iter(dataLoader))
        image = images[0]
        image = transforms.ToPILImage()(image)
        activation = act_for_unit(layer, image, channel)
        activation.cpu().numpy().item()
        bucketActivation = (math.floor(((activation - activationLower) / bucketWidth))) * bucketWidth + activationLower
    
        #If activation is out of bounds, continue to next image
        if bucketActivation not in sortedImages.keys():
            print("out of bounds")
            continue
            
        #If we still need more images in this bucket, add this image to dict
        if len(sortedImages[bucketActivation]) < maxExamples:
            list1 = sortedImages[bucketActivation]
            list1.append(image)
            sortedImages.update({bucketActivation:list1})
        minImageCounts = len(min(sortedImages.values(), key = lambda k: len(k)))
        print(minImageCounts)
        print(activation)
        print(bucketActivation)
        print(activationLower)
        print(activation - activationLower)
    return sortedImages
    
    
'''