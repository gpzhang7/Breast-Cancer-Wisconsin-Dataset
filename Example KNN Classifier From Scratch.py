import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
    # use this to avoid using a lower K value than we have groups
from math import sqrt
from collections import Counter
    # use this to get the most popular votes
style.use('fivethirtyeight')

# Tutorial can be found at the following link: https://pythonprogramming.net/programming-k-nearest-neighbors-machine-learning-tutorial/?completed=/euclidean-distance-machine-learning-tutorial/

# create some data:
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
    # this is a dictionary with k & r as the classes, and the numbers are the datapoints attributed with that class
new_features = [5,7]

# make a quick scatterplot:
"""for i in dataset:
    for ii in dataset[ii]:
        plt.scatter(ii[0],ii[1],s=100,color=i)"""
    # this graph can be consolidated into the one line below:
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)

plt.show()
    # we're going to attempt to classify the blue dot (the new_features data point)

# create a function to classify the data:

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less that total voting groups!')
            # if this is trye, you will attempt to use the KNN fn to vote where the nearest neighbors selected is less than or = to the number of groups that can vote
                # allowing for this could still give us a tie

# the main downfall of KNN is that you have to compare the data in question to all of the points from the dataset before you can know what the closest k number of pionts are
# therefore, KNN performs slower and slower the more data that you have

# so how do you find the k number of closest points?

    distances = []
# create a list
    for group in data:
# which contains another list
        for features in data[group]:
# which contains the distance           
            euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
# followed by the class, per point in out dataset
            distances.append([euclidean_distance,group])


# now we want to sort that list of distances and classes, and then take the first k elements, taking the index of 1, which is the class
    votes = [i[1] for i in sorted(distances)[:k]]
# so those are the 3 votes, now we need to find the majority vote (using Counter from the collections Python module)
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_features)
print (result)

# graphing this:
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()
# the result is the [5,7] (new_features) data point turning red
