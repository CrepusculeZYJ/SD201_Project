from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """

    TP = 0
    FP = 0
    FN = 0
    for i in range(len(expected_results)):
        if expected_results[i] == True and actual_results[i] == True:
            TP += 1
        elif expected_results[i] == False and actual_results[i] == True:
            FP += 1
        elif expected_results[i] == True and actual_results[i] == False:
            FN += 1

    if TP == 0:
        return 0, 0
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    return precision, recall
    
    # raise NotImplementedError('Implement this method for Question 3')

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    
    precision, recall = precision_recall(expected_results, actual_results)
    if precision == 0 and recall == 0:
        return 0
    F1_score = 2*precision*recall/(precision+recall)
    return F1_score
    # raise NotImplementedError('Implement this method for Question 3')
