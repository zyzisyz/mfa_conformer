from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

def compute_eer(labels, scores):
    """sklearn style compute eer
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold


def compute_minDCF(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
    """MinDCF
    Computes the minimum of the detection cost function.  The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

