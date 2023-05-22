import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import sys, os

# gt: ground truth
# pr: predicted
# ob: observed


def recall(gt, pr, ob):
    tp = np.sum((ob == 0) & (gt == 1) & (pr == 1))
    den = np.sum((gt == 1) & (ob == 0))
    # print("tp:", tp, "rden: ", den)
    return tp / den


def PRC(gt, pr, ob, out_path):
    """
    saves precision recall curve and returns recall, precision, fscore, and best thershold
    """

    groundtruth = np.array(gt[(ob == 0)]).flatten()
    predicted = np.array(pr[(ob == 0)]).flatten()
    precision, recall, thresholds = metrics.precision_recall_curve(
        groundtruth, predicted
    )
    fmeasure = 2 * recall * precision / (recall + precision)

    # setting nan elements to zero so that `index` would not be empty
    fmeasure[np.isnan(fmeasure)] = 0
    index = (np.where(fmeasure == np.amax(fmeasure)))[0]
    print(
        "Max Value, precision: %.2f%%,recall: %.2f%%, f-score: %0.2f%%"
        % (precision[index] * 100, recall[index] * 100, fmeasure[index] * 100)
    )
    # Save the raw precision and recall results to a pickle since we might want
    # to analyse them later.

    if out_path:
        out_file = os.path.join(out_path, "precision_recall.pkl")
        with open(out_file, "wb") as out:
            pickle.dump(
                {"precision": precision, "recall": recall, "thresholds": thresholds},
                out,
            )

        # Save precision and recalls in a N * 2 np array
        np.savetxt(
            os.path.join(out_path, "pr.txt"),
            np.array([precision, recall]).T,
            delimiter=",",
            fmt="%1.8f",
        )

        # Create the precision-recall curve.
        out_file = os.path.join(out_path, "precision_recall.png")
        plt.clf()
        plt.plot(recall, precision, label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(out_file)

    rec = recall[index]
    prec = precision[index]
    fscore = fmeasure[index]
    return rec, prec, fscore, thresholds[index]


def AUC(gt, pr, ob):
    groundtruth = np.array(gt[(ob == 0)]).flatten()
    predicted = np.array(pr[(ob == 0)]).flatten()
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, predicted)
    AUC = metrics.auc(fpr, tpr)
    return AUC


def MCC(gt, pr, ob):
    y_true = np.array(gt[(ob == 0)]).flatten()
    y_pred = np.array(pr[(ob == 0)]).flatten()
    return metrics.matthews_corrcoef(y_true, y_pred)


def precision(gt, pr, ob):
    tp = np.sum((ob == 0) & (gt == 1) & (pr == 1))
    den = np.sum((pr == 1) & (ob == 0))
    # print("pden:", den)
    return tp / den


def f_score(gt, pr, ob):
    rec = recall(gt, pr, ob)
    prec = precision(gt, pr, ob)
    return 2 * rec * prec / (rec + prec)


def accuracy(gt, pr, ob):
    nom = np.sum((gt == 0) & (pr == 0)) + np.sum((gt == 1) & (ob == 0) & (pr == 1))
    den = np.sum(ob == 0)
    return nom / den


def fnorm(X):
    return np.sum(X**2)


def snr(gt, pr, ob):
    groundtruth = np.array(gt[(ob == 0)]).flatten()
    predicted = np.array(pr[(ob == 0)]).flatten()
    value = fnorm(groundtruth) / fnorm(groundtruth - predicted)
    return 10 * np.log10(value)


def sre(gt, pr, ob):
    groundtruth = np.array(gt[(ob == 0)]).flatten()
    predicted = np.array(pr[(ob == 0)]).flatten()
    value = fnorm(groundtruth) / fnorm(groundtruth - predicted)
    return value


def print_metrics(gt, probs, ob, out_path=None):
    """print metrics and return best G"""

    with open('test.npy', 'wb') as f:
        np.save(f, gt)
        np.save(f, probs)
        np.save(f, ob)

    rec, prec, f, tr = PRC(gt, probs, ob, out_path)
    auc = AUC(gt, probs, ob)

    print("best threshold: %.2f" % tr)

 
    # Compute output graph based on best threshold
    pr = probs.copy()
    pr[pr < tr] = 0
    pr[pr >= tr] = 1

    rec = recall(gt, pr, ob)
    prec = precision(gt, pr, ob)

    acc = accuracy(gt, pr, ob)
    mcc = MCC(gt, pr, ob)

    print("Treshold: %.2f%%" % tr)
    print(
        "auc: %.2f%%,acc: %.2f%%,precision: %.2f%%, recall: %0.2f%%, f-score: %0.2f%%, mcc: %0.2f%%"
        % (auc * 100, acc * 100, prec * 100, rec * 100, f * 100, mcc * 100)
    )
    print("SNR: %.4f" % snr(gt, pr, ob))
    print("SRE: %.4f" % sre(gt, pr, ob))

    print(f'{acc}\t{prec}\t{rec}\t{2 * prec * rec / (prec + rec)}')

    return pr
