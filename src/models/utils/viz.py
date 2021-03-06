import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pathlib import Path


def comp_corr_pred(cm: np.ndarray) -> float:
    """Returns the ratio of true positives over the total.
    Estimates how many correct predictions the model did

    Args:
        cm (np.ndarray): confusion matrix of n-size (where n
        is the number of labels)

    Returns:
        float: Number of correct predictions (between 0 and 1)
    """
    m_size, _ = cm.shape
    return (cm * np.identity(m_size)).sum() / cm.sum()


def iou(cm: np.ndarray) -> float:
    return cm[-1, -1] / (cm.sum() - cm[0, 0])


def plot_loss(results: dict, fig_folder: Path) -> None:
    """Shows the evolution of the training and validation loss
    on a line plot.
    The results are saved in a .png file.

    Args:
        results (dict): a dictionary containing the results
        fig_folder (Path): the folder where vizualisations are saved
    """
    fig_folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results["train"]["loss"], label="train")
    ax.plot(results["val"]["loss"], label="val")
    ax.set_title("Loss evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss evolution")
    ax.legend()
    fig.savefig(fig_folder / "loss_evolution.png")


def plot_last_cm(results: dict, fig_folder: Path) -> None:
    """Plots the last confusion matrix on a heatmap.
    Results are displayed in percent (%).
    The results are saved in a .png file.

    Args:
        results (dict): a dictionary containing the results
        fig_folder (Path): the folder where vizualisations are saved
    """
    fig_folder.mkdir(exist_ok=True)
    cm_epoch = results["val"]["conf_matrix"]
    last_cm = cm_epoch[-1]
    last_cm /= last_cm.sum()

    fig, ax = plt.subplots(figsize=(22, 14))
    # sns.heatmap(100 * last_cm, vmin=0, vmax=100,
    # center=50, annot=True, ax=ax, fmt='.1f')
    sns.heatmap(100 * last_cm, annot=True, ax=ax, fmt='.1f')
    ax.set_title(f"Last confusion matrix (epoch {len(cm_epoch)})")
    fig.savefig(fig_folder / f"last_cm_epoch_{len(cm_epoch)}.png")


def plot_correct_preds(results: dict, fig_folder: Path) -> None:
    """Shows the evolution of the correct predictions made on a
    line plot.
    The results are saved in a .png file.

    Args:
        results (dict): a dictionary containing the results
        fig_folder (Path): the folder where vizualisations are saved
    """
    fig_folder.mkdir(exist_ok=True)
    cm_epoch = results["val"]["conf_matrix"]
    corr_pred = np.array([comp_corr_pred(cm) for cm in cm_epoch])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(corr_pred)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correct preds")
    ax.set_title("N° of correct preds")
    fig.savefig(fig_folder / "corr_preds.png")


def plot_accuracy(results: dict, fig_folder: Path) -> None:
    """Plots evolution of accuracy during training.

    Args:
        results (dict): a dictionary containing the results
        fig_folder (Path): the folder where vizualisations are saved
    """
    fig_folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results["train"]["accuracy"], label="train")
    ax.plot(results["val"]["accuracy"], label="val")
    ax.set_title("Accuracy evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(fig_folder / "accuracy_evolution.png")


def plot_iou(results: dict, fig_folder: Path) -> None:
    """Shows the evolution of the intersection over union ratio.
    The results are saved in a .png file.

    Args:
        results (dict): a dictionary containing the results
        fig_folder (Path): the folder where vizualisations are saved
    """
    fig_folder.mkdir(exist_ok=True)
    cm_epoch = results["val"]["conf_matrix"]
    corr_pred = np.array([iou(cm) for cm in cm_epoch])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(corr_pred)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Evolution of IoU")
    fig.savefig(fig_folder / "iou.png")


def generate_masks(ds, model, fig_folder, slot_number, n_samples):
    fig_folder.mkdir(exist_ok=True)
    train_mask_folder = fig_folder / 'train_masks'
    train_mask_folder.mkdir(exist_ok=True)
    
    random_idx = np.random.choice(len(ds), size=n_samples)
    imgs = []
    masks = []
    
    for i in random_idx:
        (img, mask_img), _ = ds[i]
        imgs.append(torch.from_numpy(img))
        masks.append(torch.from_numpy(mask_img))
    
    imgs, masks = torch.stack(imgs), torch.stack(masks)
    
    with torch.no_grad():
        pred = model(imgs)
        
    n_rows = pred.shape[0]
    n_cols = 4

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 18))
    fig.suptitle('Image, mask, probabilities and predictions', fontsize=12)

    for i_row, axrow in enumerate(axs):
        img_rgb = imgs[i_row, [4, 2, 1], :650, :650].numpy()
        img_rgb = np.transpose(img_rgb, axes=(1, 2, 0))
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
        
        axrow[0].imshow(img_rgb)
        axrow[1].imshow(masks[i_row, :650, :650].numpy())
        axrow[2].imshow(pred[i_row, 1, :650, :650].numpy())
        axrow[3].imshow(pred.argmax(dim=1)[i_row, :650, :650].numpy())
    
    fig.savefig(fig_folder / f"unet_mask_results_{slot_number}.png")
    
    