from typing import List

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        ID : int
            The ID of the feature along which the tree splits
        decision : bool
            The decision of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1,
                 height: int = 0):
        """
        Parameters
        ----------
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            labels : List[bool]
                The labels of the training points.
            types : List[FeaturesTypes]
                The types of the features.
            h : int
                The maximum height of the tree.
            min_split_points : int
                The minimum number of points required to split a node.
            height : int
                The height of the tree.
        """
        
        self.points = PointSet(features, labels, types)
        self.points.add_min_split_points(min_split_points)
        ID_best_gini_gain = self.points.get_best_gain()[0]
        self.height = height
        self.types = types
        
        if ID_best_gini_gain != None and height < h:
            
            if types[ID_best_gini_gain] == FeaturesTypes.BOOLEAN:
                features_0 = []
                features_1 = []
                labels_0 = []
                labels_1 = []
                for i in range(len(features)):
                    if features[i][ID_best_gini_gain] == 0:
                        features_0.append(features[i])
                        labels_0.append(labels[i])
                    else:
                        features_1.append(features[i])
                        labels_1.append(labels[i])
                
                self.ID = ID_best_gini_gain
                self.left_node = Tree(features_0, labels_0, types, h, min_split_points, height + 1)
                self.right_node = Tree(features_1, labels_1, types, h, min_split_points, height + 1)
                
            elif types[ID_best_gini_gain] == FeaturesTypes.CLASSES:
                features_0 = []
                features_1 = []
                labels_0 = []
                labels_1 = []
                for i in range(len(features)):
                    if features[i][ID_best_gini_gain] == self.points.get_best_threshold():
                        features_0.append(features[i])
                        labels_0.append(labels[i])
                    else:
                        features_1.append(features[i])
                        labels_1.append(labels[i])
                
                self.ID = ID_best_gini_gain
                self.left_node = Tree(features_0, labels_0, types, h, min_split_points, height + 1)
                self.right_node = Tree(features_1, labels_1, types, h, min_split_points, height + 1)
                    
            else:
                features_0 = []
                features_1 = []
                labels_0 = []
                labels_1 = []
                for i in range(len(features)):
                    if features[i][ID_best_gini_gain] < self.points.get_best_threshold():
                        features_0.append(features[i])
                        labels_0.append(labels[i])
                    else:
                        features_1.append(features[i])
                        labels_1.append(labels[i])
                
                self.ID = ID_best_gini_gain
                self.left_node = Tree(features_0, labels_0, types, h, min_split_points, height + 1)
                self.right_node = Tree(features_1, labels_1, types, h, min_split_points, height + 1)
            
        else:
            self.ID = None
            cnt = 0
            for label in labels:
                if label == True:
                    cnt += 1
                
            if cnt >= len(labels) - cnt:
                self.decision = True
            else:
                self.decision = False

        # raise NotImplementedError('Implement this method for Question 4')

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        if self.ID == None:
            return self.decision
        if self.types[self.ID] == FeaturesTypes.BOOLEAN:
            if features[self.ID] == 0:
                return self.left_node.decide(features)
            else:
                return self.right_node.decide(features)
        elif self.types[self.ID] == FeaturesTypes.CLASSES:
            if features[self.ID] == self.points.get_best_threshold():
                return self.left_node.decide(features)
            else:
                return self.right_node.decide(features)
        else:
            if features[self.ID] < self.points.get_best_threshold():
                return self.left_node.decide(features)
            else:
                return self.right_node.decide(features)
        
        # raise NotImplementedError('Implement this method for Question 4')

