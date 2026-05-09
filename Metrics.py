import numpy as np

def accuracy_score2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    

    labels = np.unique(np.concatenate((y_true, y_pred)))
        
    n_classes = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1
            
    return cm


def f1_score2(y_true, y_pred, average='macro'):
    cm = confusion_matrix2(y_true, y_pred)
    
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    eps = 1e-9
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    if average == 'macro':
        return np.mean(f1)
    else:
        return f1


def classification_report2(y_true, y_pred, digits=2):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix2(y_true, y_pred)
    
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1) 
    
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    report = "--- Class Metrics ---\n"
    for i, label in enumerate(labels):
        display_label = int(label) if float(label).is_integer() else label
        report += f"Class {display_label} -> Precision: {precision[i]:.{digits}f} | Recall: {recall[i]:.{digits}f} | F1: {f1[i]:.{digits}f} | Support: {support[i]}\n"
        
    
    accuracy = accuracy_score2(y_true, y_pred)
    macro_prec = np.mean(precision)
    macro_rec = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    report += "\n--- Overall Metrics ---\n"
    report += f"Accuracy:  {accuracy:.{digits}f}\n"
    report += f"Macro Avg: Precision: {macro_prec:.{digits}f} | Recall: {macro_rec:.{digits}f} | F1: {macro_f1:.{digits}f}\n"
    
    return report