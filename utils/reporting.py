
#metrics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from IPython.display import display

def report(y, ds, n_classes):

    y_pred = np.argmax(y, 1)
    y_target_one_hot = ds.test.labels
    y_true = np.argmax(y_target_one_hot, 1)

    y_pred_max_idx = np.argmax(y, 1)
    y_pred_one_hot = np.eye(n_classes)[y_pred_max_idx]

    print("")
    print("Train Class Distribution")
    train_class_sum = np.sum(ds.Y[ds.train_idxs], axis=0)
    df_train_class_sum = pd.DataFrame([train_class_sum], index=['distribution'], columns=np.arange(n_classes).astype(str))
    display(df_train_class_sum)

    print("")
    print("Test Class Distribution")
    test_class_sum = np.sum(ds.Y[ds.test_idxs], axis=0)
    df_test_class_sum = pd.DataFrame([test_class_sum], index=['distribution'], columns=np.arange(n_classes).astype(str))
    display(df_test_class_sum)

    # Stats overall
    prec_all = precision_score(y_true, y_pred, average='macro')
    recall_all = recall_score(y_true, y_pred, average='macro')
    f1_all = f1_score(y_true, y_pred, average='macro')

    print("")
    print("Train Results")
    stats_all = []
    stats_all.append(prec_all)
    stats_all.append(recall_all)
    stats_all.append(f1_all)
    df_stats_all = pd.DataFrame(stats_all, index=['prec ovall', 'recall overall ', 'f1 overall'], columns=['value'])
    display(df_stats_all.round(3))

    # Stats by class
    prec = precision_score(y_target_one_hot,  y_pred_one_hot, average=None)
    recall = recall_score(y_target_one_hot,  y_pred_one_hot, average=None)
    f1 = f1_score(y_target_one_hot,  y_pred_one_hot, average=None)

    print("")
    print("Test Results by Class")
    stats=[]
    stats.append(prec)
    stats.append(recall)
    stats.append(f1)
    df_stats = pd.DataFrame(stats, index=['prec', 'recall', 'f1'], columns=np.arange(n_classes))
    display(df_stats.round(3))
    return