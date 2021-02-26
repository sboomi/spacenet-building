import click
import logging
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path
from functools import partial
from utils.dataset import load_tif, load_stats, split_dataset, get_tif_dims, \
    compute_mean_std, MulPanSharpenDataset
from utils.model import train
from utils.viz import plot_loss, plot_last_cm, plot_correct_preds, \
    plot_accuracy, plot_iou, generate_masks
from utils.segtransforms import SemSegCompose


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('train_model')


@click.command()
@click.argument("image_folder", type=click.Path(exists=True))
@click.argument("model_folder", type=click.Path(exists=True))
@click.argument("building_folder", type=click.Path(exists=True))
@click.argument("figure_folder", type=click.Path(exists=True))
@click.argument("info_csv", type=click.Path(exists=True))
@click.argument("stats_json_file", type=click.Path())
@click.argument("train_results", type=click.Path())
@click.argument("batch_size", type=int)
@click.argument("ideal_batch_size", type=int)
@click.argument("epochs", type=int)
@click.argument("device_name", type=str, default="cuda:0")
def main(image_folder,
         model_folder,
         building_folder,
         figure_folder,
         info_csv,
         stats_json_file,
         train_results,
         batch_size,
         ideal_batch_size,
         epochs,
         device_name):
        
    # Configure folders
    image_folder = Path(image_folder)
    model_folder = Path(model_folder)
    building_folder = Path(building_folder)
    stats_json_file = Path(stats_json_file)
    figure_folder = Path(figure_folder)
    
    # Seed program
    torch.manual_seed(0)
    
    # retrieve information on buildings
    df = pd.read_csv(info_csv)
    
    # Get information on a TIF
    n_ch, h, w = get_tif_dims(list(image_folder.iterdir())[0])
    
    # Get means and stds
    if stats_json_file.exists():
        mean_channels, std_channels = load_stats(stats_json_file)
    else:
        stats = compute_mean_std(image_folder, n_ch)
        mean_channels = stats['mean']
        std_channels = stats["std"]
        
        with open(stats_json_file, 'w') as file:
            json.dump(stats, file)
    
    # New dims for image
    pad_h, pad_w = 8 - h % 8, 8 - w % 8
    
    # Prepare padding
    if pad_h % 2==0:
        top_p = bottom_p = pad_h//2
    else:
        top_p, bottom_p = pad_h//2, pad_h-(pad_h//2)
        
    if pad_w % 2==0:
        left_p = right_p = pad_w//2
    else:
        left_p, right_p = pad_w//2, pad_w-(pad_w//2)
    
    img_pad = (left_p, top_p, right_p, bottom_p)
    
    # Compose transforms
    transforms = SemSegCompose(img_pad, mean_channels, std_channels, 360)
    
    # Prepare loading function
    # load_tif_with_mask = partial(
    #     load_tif,
    #     df=df,
    #     mean_vec=mean_channels,
    #     std_vec=std_channels,
    #     building_folder=building_folder,
    #     padding=(pad_h, pad_w))
    
    # Make dataset
    ds = MulPanSharpenDataset(image_folder,
                              building_folder,
                              df,
                              transforms=transforms)
    # ds = datasets.DatasetFolder(root=image_folder,
    #                             loader=load_tif_with_mask,
    #                             extensions=('.tif',))

    logger.info(f"N° of images: {len(ds)}")
    logger.info(f"Type of img: {ds.label}")
    
    train_ds, val_ds = split_dataset(ds, train_size=0.8)
    
    logger.info(f"Train set size: {len(train_ds)}")
    logger.info(f"Val set size: {len(val_ds)}")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    logger.info(f"N° of iterations per batch (train): {len(train_dl)}")
    logger.info(f"N° of iterations per batch (val): {len(val_dl)}")
    
    # get model
    logger.info("Getting U-Net model")
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=n_ch, out_channels=2,
                           init_features=32, pretrained=False)

    device = torch.device(device_name)
    logger.info(f"Mounting model on {device}")
    model = model.to(device)
    
    # Define criterion 
    logger.info("Defining cross-entropy loss with 11/89 ratio")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([.11, .89]))
    criterion = criterion.to(device)
    
    # Define optimizer
    logger.info("Defining Adam optimizer")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    results = train(model, train_dl, val_dl, criterion,
                    optimizer, epochs, device, 2,
                    model_folder, ideal_batch_size // batch_size)
    
    # Saves model metrics as pickle file
    with open(train_results, "wb") as f:
        pickle.dump(results, f)
        
    # Saves model
    torch.save(model.state_dict(), model_folder / 'unet_model')
    logger.info(f"Saved model at {model_folder / 'unet_model'}")

    logging.info(f"Metrics evaluation. Check {figure_folder} for results.")

    # Plot metrics
    plot_loss(results, figure_folder)
    logger.info("Loss curve created.")

    plot_last_cm(results, figure_folder)
    logger.info("Last confusion matrix created")

    plot_correct_preds(results, figure_folder)
    logger.info("Evolution of correct predictions created;")

    plot_accuracy(results, figure_folder)
    logger.info("Accuracy plot created.")
    
    plot_iou(results, figure_folder)
    logger.info("IoU evolution plot created.")
    
    for i in range(5):
        generate_masks(ds, model, figure_folder, i, 5)
    logger.info("Created model results.")


if __name__ == "__main__":
    main()