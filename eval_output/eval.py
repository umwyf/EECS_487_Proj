import numpy as np

def classify(logit: float):
    if logit<0.5:
        return 0  # female
    else:
        return 1  # male

def map2np(obj):
    return np.array(list(obj))

with open("label.txt", 'r') as fp_label:
    all_labels = fp_label.read().split('\n')
    all_labels = [float(i) for i in all_labels if i!=""]
    all_labels = np.array(all_labels)

with open("pred.txt", 'r') as fp_pred:
    all_preds = fp_pred.read().split('\n')
    all_preds = [float(i) for i in all_preds if i!=""]
    all_preds = np.array(all_preds)
    
assert len(all_labels) == len(all_preds)

# accuracy
total_count = len(all_preds)
print(total_count)
pred_class = map2np(map(classify, all_preds))
label_class = map2np(map(classify, all_labels))
total_correct = np.sum(pred_class==label_class)
print(total_correct)
print(f"binary accuracy: {total_correct/total_count}")

# L1/L2 distance
dist_arr = np.abs(all_labels-all_preds)
l1 = np.sum(dist_arr)/total_count
mse = np.sum(dist_arr*dist_arr)/total_count
print(f"l1 distance: {l1}")
print(f"mse: {mse}")

# F1
TP = np.sum(np.logical_and(label_class==1, pred_class==1))
TN = np.sum(np.logical_and(label_class==0, pred_class==0))
FP = np.sum(np.logical_and(label_class==1, pred_class==0))
FN = np.sum(np.logical_and(label_class==0, pred_class==1))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2*precision*recall/(precision+recall)

print(f"F1: {F1}")
