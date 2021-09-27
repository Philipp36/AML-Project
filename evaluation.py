import os
import torch
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

from dataLoader import createDataset_300W_LP


class BaselineClass(torch.nn.Module):
    def __init__(self, c):
        super(BaselineClass, self).__init__()
        self.c = c

    def forward(self, x):
        label_pred = torch.zeros(1, 4)
        label_pred[0, self.c] = 1
        return torch.zeros(size=(BATCH_SIZE, 448, 448)), label_pred.repeat(BATCH_SIZE, 1)


def plot_confusion_matrix(df_confusion, labels, path, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion[0]))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks[::-1], labels[::-1])
    #plt.tight_layout()
    #plt.ylabel(df_confusion.index.name)
    #plt.xlabel(df_confusion.columns.name)
    plt.savefig(os.path.join(path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    MODEL_PATH = 'resnet-18-train-all-heatmaps_epoch9.pth'
    MODEL_NAME = ''
    TEST_PATH = './siim-covid19-detection/test'
    TRAIN_PATH = './siim-covid19-detection/train'
    METADATA_IMAGE = './siim-covid19-detection/train_image_level.csv'
    METADATA_STUDY = './siim-covid19-detection/train_study_level.csv'
    LOG_FOLDER = './logs/'
    BATCH_SIZE = 16

    labels = ["negative", "typical", "indeterminate", "atypical"]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_dataloader, _ = createDataset_300W_LP(dataset_train_path=TRAIN_PATH, dataset_test_path=TEST_PATH,
                                                  metadata_image_path=METADATA_IMAGE, metadata_study_path=METADATA_STUDY,
                                                  test_batch=BATCH_SIZE, train_batch=BATCH_SIZE,
                                                  conf_data_loading_test={'num_workers': 16})
    model = torch.load(MODEL_PATH, map_location=dev)
    #model = BaselineClass(1)
    model.to(dev)
    class_preds = []
    truths = []
    heatLoss = torch.nn.MSELoss()
    labelLoss = torch.nn.CrossEntropyLoss()
    heat_losses = []
    label_losses = []

    with torch.no_grad():
        model.eval()
        for batch_ndx, (image, truth, heat) in enumerate(tqdm(test_dataloader)):
            image = image.to(dev)
            heat_pred, label_pred = model(image)
            label_pred, heat_pred = label_pred.cpu(), heat_pred.cpu()
            label_losses.append(labelLoss(label_pred, truth).item())
            heat_losses.append(heatLoss(heat_pred, heat).item())
            label_prob = softmax(label_pred, dim=1)
            class_pred = torch.argmax(label_prob, dim=1)

            class_preds += class_pred.tolist()
            truths += truth.tolist()

class_report = classification_report(truths, class_preds, target_names=labels, output_dict=True)
print(classification_report(truths, class_preds, target_names=labels))
print('label_loss:', np.mean(label_losses))
print('heat_loss:', np.mean(heat_losses))
truths = np.array(truths)
class_preds = np.array(class_preds)
for c in range(4):
    tn = class_preds[truths != c] != c
    fp = class_preds[truths != c] == c
    spec = sum(tn) / (sum(fp) + sum(tn))
    print(f'specificty of {c}: {spec}')

cm = confusion_matrix(truths, class_preds)
cm_norm = confusion_matrix(truths, class_preds, normalize='true')

plot_confusion_matrix(cm, labels, LOG_FOLDER)
