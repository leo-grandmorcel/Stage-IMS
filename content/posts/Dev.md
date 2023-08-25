---
title: "Développement Réseau de neurones"
date: 2023-06-19
---
## U-Net
Le but de ce projet était de développer un réseau de neurones pour la segmentation d'images de vignes. Pour ce faire, j'ai utilisé un réseau de neurones U-Net, qui est un réseau de neurones convolutifs, très utilisé pour la segmentation d'images médicales, notamment pour la segmentation de cellules. Mais il peut être performant pour la segmenation d'autres objets comme les feuilles de vignes ou des symptômes de maladie.

## Architecture du réseau de neurones U-Net
![U-Net architecture](/images/U-Net.png) 
Il est composé d'un encodeur et d'un décodeur. L'encodeur est utilisé pour extraire les caractéristiques de l'image, tandis que le décodeur est utilisé pour reconstruire l'image segmentée à partir des caractéristiques extraites par l'encodeur. Le réseau de neurones U-Net utilisé a été pré-entrainé sur le jeu de données ImageNet, qui est un jeu de données d'images de 1000 classes différentes. Il a ensuite été entrainé sur un jeu de données de 200 images de vignes, qui ont été annoté manuellement.

Le modèle est disponible via la librairie PyTorch, et peut être utilisé comme suit :
```python
import segmentation_models_pytorch as smp
model = smp.Unet(
    "resnet50",
    encoder_weights="imagenet",
    classes=2,
    activation=None,
    encoder_depth=5,
    decoder_channels=[256, 128, 64, 32, 16],
)
```

## Entraînement
Afin d'entraîner le modèle, j'ai développé un script Python qui permet de charger les images et les annotations, de les pré-charger dans la mémoire si possible, puis de les passer au modèle pour l'entrainement. Le script permet également de sauvegarder le modèle à chaque époque, ainsi que les métriques d'entrainement, afin de pouvoir reprendre l'entrainement à tout moment. Le script permet également d'avoir un retour d'informations sur les performances actuelles du modèle.
Pendant mon stage j'ai eu le temps d'ajouter des fonctionnalités au script, comme un parseur d'argument pour pouvoir changer les paramètres d'entrainement.

### Parser
```python
from argparse import ArgumentParser
from datetime import date


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser(description="Vineyard segmentation model")
    parser.add_argument("-n", "--name", type=str, required=True, help="Model name")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "-bs", "--batch_size", dest="bs", type=int, default=8, help="Batch size"
    )
    parser.add_argument(
        "-p",
        "--pré-load",
        dest="pre",
        type=bool,
        default=False,
        help="Preload dataset in memory",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Directory of the dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"Training_{date.today()}",
        help="Output path",
    )
    parser.add_argument(
        "-c", "--classes", type=int, default=2, help="Number of classes"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        dest="lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        dest="wd",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from existing model",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, help="Model path for resume training"
    )
    parser.add_argument("-s", "--save", type=int, help="Save model every n epochs")
    return parser.parse_args()
```


### Classe Dataset
```python
import os
from dataclasses import dataclass
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class VineyardDataset(Dataset):
    """Dataset class for vineyard images and masks."""

    path: str
    images: list
    masks: list = None
    loaded: bool = False
    transform: A.Compose = None

    def __init__(
        self,
        path: str,
        images: list,
        masks: list = None,
        loaded: bool = False,
        transform: A.Compose = None,
    ):
        """Initialize the dataset, and load the images and masks if loaded is True"""
        self.path = path
        self.loaded = loaded
        self.transform = transform
        if self.loaded:
            self.images = [
                cv2.imread(os.path.join(path, f"{filename}.jpg")) for filename in images
            ]
            self.masks = [
                cv2.imread(
                    os.path.join(path, f"{filename}_bin.png"), cv2.IMREAD_GRAYSCALE
                )
                for filename in images
            ]
        else:
            self.images = images
            self.masks = masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        if not self.loaded:
            img = cv2.imread(os.path.join(self.path, f"{self.images[idx]}.jpg"))
            mask = cv2.imread(
                os.path.join(self.path, f"{self.images[idx]}_bin.png"),
                cv2.IMREAD_GRAYSCALE,
            )
        else:
            img = self.images[idx]
            mask = self.masks[idx]
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

    def get_mask(self, idx: int) -> np.ndarray:
        """Returns the flatten mask of the image"""
        if self.loaded:
            return np.array(self.masks[idx]).flatten()
        else:
            return np.array(
                Image.open(os.path.join(self.path, f"{self.images[idx]}_bin.png"))
            ).flatten()
```

### Classe Trainer
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from dataclasses import dataclass
import os


@dataclass
class Trainer:
    """Class to train a model"""

    device: torch.device
    nbr_classes: int
    epochs: int
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    optimizer: torch.optim
    scheduler: torch.optim.lr_scheduler
    model_name: str = "unet"
    save_every: int = 0

    def fit(self):
        """Fits the model"""
        torch.cuda.empty_cache()
        self.metrics = []
        self.model.to(self.device)
        fit_time = time.time()
        for e in range(1, self.epochs + 1):
            since = time.time()
            train_iou, train_precision, train_recall, train_accuracy = self.train()
            val_iou, val_precision, val_recall, val_accuracy = self.validate()
            mean_iou = np.transpose(
                [np.mean(train_iou, axis=0), np.mean(val_iou, axis=0)]
            )
            mean_accuracy = np.transpose(
                [np.mean(train_accuracy, axis=0), np.mean(val_accuracy, axis=0)]
            )
            mean_precision = np.transpose(
                [np.mean(train_precision, axis=0), np.mean(val_precision, axis=0)]
            )
            mean_recall = np.transpose(
                [np.mean(train_recall, axis=0), np.mean(val_recall, axis=0)]
            )
            self.metrics.append([mean_iou, mean_accuracy, mean_precision, mean_recall])
            print(
                f"Epoch: {str(e).zfill(len(str(self.epochs)))}/{self.epochs}",
                f"Time: {(time.time()-since)/60:.2f}m",
                f"MIoU: {np.mean(mean_iou):.3f}",
                f"MAccuracy: {np.mean(mean_accuracy):.3f}",
                f"MPrecision: {np.mean(mean_precision):.3f}",
                f"MRecall: {np.mean(mean_recall):.3f}\n",
            )
            if self.save_every and e % self.save_every == 0:
                self.save(f"{self.model_name}_{e}")
        print(f"Total time: {(time.time()- fit_time)/60:.2f}m")
        self.metrics = np.array(self.metrics)
        self.metrics = pd.DataFrame(
            self.metrics.reshape(self.metrics.shape[0], -1),
            columns=pd.MultiIndex.from_product(
                [
                    ["IoU", "Accuracy", "Precision", "Recall"],
                    [f"Class {i}" for i in range(self.nbr_classes)],
                    ["Train", "Val"],
                ]
            ),
        )

    def train(
        self,
    ) -> list:
        """Train the model over the training set"""
        self.model.train()
        train_iou = []
        train_precision = []
        train_recall = []
        train_accuracy = []
        print("Training...")
        for images, masks in tqdm(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            cfmatrix = calculateconfmatrix(outputs, masks, self.nbr_classes)
            train_iou.append(iou(cfmatrix))
            train_precision.append(precision(cfmatrix))
            train_recall.append(recall(cfmatrix))
            train_accuracy.append(accuracy(cfmatrix))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        return train_iou, train_precision, train_recall, train_accuracy

    def validate(self) -> list:
        """Validates the model over the validation set"""
        self.model.eval()
        val_iou = []
        val_precision = []
        val_recall = []
        val_accuracy = []
        print("Validating...")
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                cfmatrix = calculateconfmatrix(outputs, masks, self.nbr_classes)
                val_iou.append(iou(cfmatrix))
                val_precision.append(precision(cfmatrix))
                val_recall.append(recall(cfmatrix))
                val_accuracy.append(accuracy(cfmatrix))
        return val_iou, val_precision, val_recall, val_accuracy

    def plot_metrics(self):
        """Plots the metrics"""
        fig, ax = plt.subplots(2, 2, figsize=(50, 50))
        ax[0, 0].plot(self.metrics["IoU"])
        ax[0, 0].xaxis.get_major_locator().set_params(integer=True)
        ax[0, 0].grid()
        ax[0, 0].legend(self.metrics["IoU"].columns)
        ax[0, 0].set_xlabel("Epoch", fontsize=15)
        ax[0, 0].set_ylabel("IoU Per Class", fontsize=15)
        ax[0, 0].set_title("IoU")
        ax[0, 1].plot(self.metrics["Accuracy"])
        ax[0, 1].xaxis.get_major_locator().set_params(integer=True)
        ax[0, 1].grid()
        ax[0, 1].legend(self.metrics["Accuracy"].columns)
        ax[0, 1].set_xlabel("Epoch", fontsize=15)
        ax[0, 1].set_ylabel("Accuracy Per Class", fontsize=15)
        ax[0, 1].set_title("Accuracy")
        ax[1, 0].plot(self.metrics["Precision"])
        ax[1, 0].xaxis.get_major_locator().set_params(integer=True)
        ax[1, 0].grid()
        ax[1, 0].legend(self.metrics["Precision"].columns)
        ax[1, 0].set_xlabel("Epoch", fontsize=15)
        ax[1, 0].set_ylabel("Precision Per Class", fontsize=15)
        ax[1, 0].set_title("Precision")
        ax[1, 1].plot(self.metrics["Recall"])
        ax[1, 1].xaxis.get_major_locator().set_params(integer=True)
        ax[1, 1].grid()
        ax[1, 1].legend(self.metrics["Recall"].columns)
        ax[1, 1].set_xlabel("Epoch", fontsize=15)
        ax[1, 1].set_ylabel("Recall Per Class", fontsize=15)
        ax[1, 1].set_title("Recall")
        plt.show()

    def save(self, path: str = "Models", name: str = None):
        """Saves the model"""
        os.makedirs(os.path.join(path), exist_ok=True)
        torch.save(
            self.model, os.path.join(path, f"{name if name else self.model_name}.pt")
        )


def calculateconfmatrix(
    pred_masks: torch.Tensor, true_masks: torch.Tensor, nbr_classes: int
) -> list:
    """Calculates the confusion matrix for the given masks"""
    res = []
    confmat = ConfusionMatrix(task="multiclass", num_classes=nbr_classes)
    pred_masks = pred_masks.to(torch.device("cpu"))
    true_masks = true_masks.to(torch.device("cpu"))
    result = confmat(pred_masks, true_masks)
    for index, row in enumerate(result):
        TP = row[index]
        FN = row.sum() - TP
        FP = result[:, index].sum() - TP
        TN = result.sum() - TP - FN - FP
        res.append([[TP.item(), FN.item()], [FP.item(), TN.item()]])
    return res


def recall(matrix: list) -> list:
    """Calculates the recall for each class in the matrix"""
    res = []
    for clas in matrix:
        TP = clas[0][0]
        FN = clas[0][1]
        if TP + FN == 0:
            res.append(0)
        else:
            res.append(TP / (TP + FN))
    return res


def precision(matrix: list) -> list:
    """Calculates the precision for each class in the matrix"""
    res = []
    for clas in matrix:
        TP = clas[0][0]
        FP = clas[1][0]
        if TP + FP == 0:
            res.append(0)
        else:
            res.append(TP / (TP + FP))
    return res


def iou(matrix: list) -> list:
    """Calculates the IoU for each class in the matrix"""
    res = []
    for clas in matrix:
        TP = clas[0][0]
        FP = clas[1][0]
        FN = clas[0][1]
        if TP + FP + FN == 0:
            res.append(0)
        else:
            res.append(TP / (TP + FP + FN))
    return res


def accuracy(matrix: list) -> list:
    """Calculates the accuracy for each class in the matrix"""
    res = []
    for clas in matrix:
        TP = clas[0][0]
        FP = clas[1][0]
        FN = clas[0][1]
        TN = clas[1][1]
        if TP + FP + FN + TN == 0:
            res.append(0)
        else:
            res.append((TP + TN) / (TP + FP + FN + TN))
    return res


def lai(pred) -> float:
    """Calculates the leaf area index for the given mask"""
    return float(torch.mean(pred.float()))

```

### Fonction Binarisation
Afin d'utiliser le modèle de segmentation, il faut binariser les images. Pour cela, nous avons utilisé la fonction suivante :

```python
def binarisation255(mask: np.ndarray) -> np.ndarray:
    """Binarise the mask from RGB to Binary values"""
    mask = np.copy(mask)
    a, b, _ = np.shape(mask)
    bin_mask = np.zeros((a, b), dtype=np.uint8)
    background = np.where(np.sum(mask[:, :, :], axis=-1) >= 740)
    leaves = np.where(mask[:, :, 2] >= 240)
    tronc = np.where(mask[:, :, 0] >= 240)
    branch = np.where(mask[:, :, 1] >= 240)
    bin_mask[leaves] = 1
    bin_mask[tronc] = 2
    bin_mask[branch] = 2
    bin_mask[background] = 0
    return bin_mask
```

### Fonction principale
Enfin voici la fonction principale qui permet de lancer l'entraînement du modèle de segmentation :

```python
import torch
import cv2
import os
import segmentation_models_pytorch as smp
import albumentations as A
import torch.nn as nn
import numpy as np
import pandas as pd
from trainer import Trainer, calculateconfmatrix, iou, precision, recall, accuracy, lai
from tqdm import tqdm
from unet import UNet
from dataset import VineyardDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from personnal_parser import parse_args


def predict_image_mask(
    device: torch.device, model: nn.Module, image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Predicts the mask for the given image"""
    model.to(device)
    model.eval()
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def calculateclassweights(train_set: VineyardDataset) -> torch.Tensor:
    """Calculates the class weights for the given dataset"""
    labels = [train_set.get_mask(i) for i in range(len(train_set))]
    labels = np.concatenate(labels)
    class_counts = np.bincount(labels)
    total_samples = np.sum(class_counts)
    return torch.tensor(
        total_samples / (len(class_counts) * class_counts), dtype=torch.float
    )


def display_segmentation(image, prediction, alpha: int = 0.5) -> Image:
    """Displays the segmentation over the image"""
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    image = image.permute(1, 2, 0)
    image = image.numpy().astype("uint8")
    image = Image.fromarray(image)
    prediction = prediction.numpy().astype("uint8")
    prediction = Image.fromarray(prediction)
    prediction.putpalette(colors)
    prediction = prediction.convert("RGB")
    return Image.blend(image, prediction, alpha)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    dataset = [
        filename.split(".")[0]
        for filename in os.listdir(args.directory)
        if filename.endswith(".jpg")
    ]
    dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=19)
    X_train, X_val = train_test_split(dataset, test_size=0.4, random_state=19)
    print(f"Total Images : {len(dataset)}")
    print(f"Train Size   : {len(X_train)}")
    print(f"Val Size     : {len(X_val)}")
    print(f"Test Size    : {len(test_dataset)}")
    print(f"N Classes    : {args.classes}")
    print(f"Model Name   : {args.name}")
    print(f"Batch Size   : {args.bs}")
    print(f"Epochs       : {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    augmentation = A.Compose(
        [
            A.RandomCrop(512, 512),
            A.CLAHE(),
            A.RandomGamma(),
            A.HorizontalFlip(),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        ]
    )

    train_set = VineyardDataset(
        path=f"{args.directory}", images=X_train, loaded=True, transform=augmentation
    )
    val_set = VineyardDataset(
        path=f"{args.directory}", images=X_val, loaded=True, transform=augmentation
    )
    test_set = VineyardDataset(
        path=f"{args.directory}",
        images=test_dataset,
        loaded=False,
        transform=A.Compose(
            [
                A.Resize(2048, 2048, interpolation=cv2.INTER_LANCZOS4),
            ]
        ),
    )
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=True)
    if args.resume and args.model is not None:
        model = torch.load(args.model)
    else:
        model = smp.Unet(
            "resnet50",
            encoder_weights="imagenet",
            classes=args.classes,
            activation=None,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
        )
        # model = UNet(in_channels=3, out_channels=args.classes, n_blocks=5, dim=2)
    if args.epochs > 0:
        if args.classes > 2:
            class_weights = calculateclassweights(train_set).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
        )
        trainer = Trainer(
            device=device,
            nbr_classes=args.classes,
            epochs=args.epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=sched,
            model_name=args.name,
            save_every=args.save,
        )
        trainer.fit()
        trainer.save()
        trainer.metrics.to_csv(
            os.path.join(args.output, f"Train_{args.name}.csv"), encoding="utf-8"
        )
        trainer.plot_metrics()
    test_metrics = []
    os.makedirs(os.path.join(args.output, "Predictions", args.name), exist_ok=True)
    print("Testing...")
    for index, (image, mask) in tqdm(enumerate(test_set)):
        pred_mask = predict_image_mask(device, model, image, mask)
        cfmatrix = calculateconfmatrix(pred_mask, mask, args.classes)
        test_metrics.append(
            [
                iou(cfmatrix),
                precision(cfmatrix),
                recall(cfmatrix),
                accuracy(cfmatrix),
                [0, lai(pred_mask)],
                [0, lai(mask)],
            ]
        )
        output = display_segmentation(image, pred_mask)
        output.save(os.path.join(args.output, "Predictions", args.name, f"{index}.png"))

    test_metrics = np.array(test_metrics)
    test_metrics = pd.DataFrame(
        test_metrics.reshape(test_metrics.shape[0], -1),
        columns=pd.MultiIndex.from_product(
            [
                ["IoU", "Precision", "Recall", "Accuracy", "Predicted LAI", "True LAI"],
                [f"Class {i}" for i in range(args.classes)],
            ]
        ),
    )
    test_metrics.to_csv(
        os.path.join(args.output, f"Test_{args.name}.csv"), encoding="utf-8"
    )
```

Un exemple d'utilisation de celle-ci :
```bash
python train.py --name "Modele_1" --epochs 300 --batch_size 8 --pré-load --directory "Dataset" -o "Result" --classes 2 --learning_rate 1e-4 --save 20
```