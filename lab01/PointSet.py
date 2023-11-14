from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.min_split_points = 1
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        label_0 = sum(self.labels == False)
        label_1 = sum(self.labels == True)
        gini = 1 - (label_0/(label_0 + label_1))**2 - (label_1/(label_0 + label_1))**2
        return gini
        # raise NotImplementedError('Please implement this function for Question 1')

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        ID_best_gini_gain = -1
        best_gini_gain = 0
        self.best_split_type = None
        
        gini = self.get_gini()
        for j in range(self.features.shape[1]):
            if self.types[j] == FeaturesTypes.BOOLEAN:
                features_0 = []
                features_1 = []
                labels_0 = []
                labels_1 = []
                for i in range(self.features.shape[0]):
                    if self.features[i][j] == 0:
                        features_0.append(self.features[i])
                        labels_0.append(self.labels[i])
                    else:
                        features_1.append(self.features[i])
                        labels_1.append(self.labels[i])
                
                if len(labels_0) < self.min_split_points or len(labels_1) < self.min_split_points:
                    continue
                    
                pointset_0 = PointSet(features_0, labels_0, self.types)
                pointset_1 = PointSet(features_1, labels_1, self.types)
                gini_0 = pointset_0.get_gini()
                gini_1 = pointset_1.get_gini()
                gini_gain = gini - (gini_0*len(labels_0) + gini_1*len(labels_1))/len(self.labels)
                
                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    ID_best_gini_gain = j
                    self.best_split_type = FeaturesTypes.BOOLEAN
                    self.best_split = None
                
            elif self.types[j] == FeaturesTypes.CLASSES:
                features_dict = {}
                for i in range(self.features.shape[0]):
                    if self.features[i][j] not in features_dict:
                        features_dict[self.features[i][j]] = 0
                    features_dict[self.features[i][j]] += 1
                
                for k in features_dict.keys():
                    features_0 = []
                    features_1 = []
                    labels_0 = []
                    labels_1 = []
                    for i in range(self.features.shape[0]):
                        if self.features[i][j] == k:
                            features_0.append(self.features[i])
                            labels_0.append(self.labels[i])
                        else:
                            features_1.append(self.features[i])
                            labels_1.append(self.labels[i])
                    
                    if len(labels_0) < self.min_split_points or len(labels_1) < self.min_split_points:
                        continue
                    
                    pointset_0 = PointSet(features_0, labels_0, self.types)
                    pointset_1 = PointSet(features_1, labels_1, self.types)
                    gini_0 = pointset_0.get_gini()
                    gini_1 = pointset_1.get_gini()
                    gini_gain = gini - (gini_0*len(labels_0) + gini_1*len(labels_1))/len(self.labels)
                    
                    if gini_gain > best_gini_gain:
                        best_gini_gain = gini_gain
                        ID_best_gini_gain = j
                        self.best_split_type = FeaturesTypes.CLASSES
                        self.best_split = k
                
            elif self.types[j] == FeaturesTypes.REAL:
                # we use an efficient way to deal with the calculation of threshold
                # we use a pointer to point to the position of the current threshold after sorting
                # we will not calculate the whole list of features_0 and features_1
                # instead, at each update of the threshold, we only need to update the number of points whose label is 0 or 1
                # the sorting complexity is O(nlogn), but for finding the threshold it takes only O(n) which is the complexity of traversing the list
                # so the total complexity is O(nlogn)
                feature_label = np.column_stack((self.features, self.labels))
                sorted_indices = np.argsort(feature_label[:, j])
                sorted_features = feature_label[sorted_indices]
                
                feature_0_label_0 = 0
                feature_0_label_1 = 0
                feature_1_label_0 = np.sum(sorted_features[:, -1] == False)
                feature_1_label_1 = np.sum(sorted_features[:, -1] == True)
                
                last_threshold_index = 0
                for threshold_index in range(sorted_features.shape[0] - 1):
                    if sorted_features[threshold_index][j] == sorted_features[threshold_index+1][j]:
                        continue
                    
                    for i in range(last_threshold_index, threshold_index+1):
                        if sorted_features[i][-1] == False:
                            feature_0_label_0 += 1
                            feature_1_label_0 -= 1
                        else:
                            feature_0_label_1 += 1
                            feature_1_label_1 -= 1
                    
                    last_threshold_index = threshold_index + 1
                    
                    if feature_0_label_0 + feature_0_label_1 < self.min_split_points or feature_1_label_0 + feature_1_label_1 < self.min_split_points:
                        continue
                    
                    gini_0 = 1 - (feature_0_label_0/(feature_0_label_0 + feature_0_label_1))**2 - (feature_0_label_1/(feature_0_label_0 + feature_0_label_1))**2
                    gini_1 = 1 - (feature_1_label_0/(feature_1_label_0 + feature_1_label_1))**2 - (feature_1_label_1/(feature_1_label_0 + feature_1_label_1))**2
                    gini_gain = gini - (gini_0*(feature_0_label_0 + feature_0_label_1) + gini_1*(feature_1_label_0 + feature_1_label_1))/len(self.labels)
                    
                    if gini_gain > best_gini_gain:
                        best_gini_gain = gini_gain
                        ID_best_gini_gain = j
                        self.best_split_type = FeaturesTypes.REAL
                        self.best_split = (sorted_features[threshold_index][j] + sorted_features[threshold_index+1][j])/2
                    
        if ID_best_gini_gain == -1:
            return None, None
            
        return ID_best_gini_gain, best_gini_gain
        
        # raise NotImplementedError('Please implement this function for Question 2')

    def get_best_threshold (self) -> float:
        if self.best_split_type==None:
            raise Exception("Bad call to get_best_threshold")
        
        return self.best_split
    
    def add_min_split_points (self, min_split_points):
        self.min_split_points = min_split_points
