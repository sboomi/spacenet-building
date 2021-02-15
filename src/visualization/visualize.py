import logging
import click
import json
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from visualization.utils import prepare_img
from tqdm import tqdm


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('vizualize_results')


@click.command()
@click.argument("figure_folder", type=click.Path(exists=True))
@click.argument("test_img_folder", type=click.Path(exists=True))
@click.argument("model_name", type=click.Path(exists=True))
@click.argument("stats_json", type=click.File('rb'))
@click.argument("n_ch", type=int)
@click.argument("test_name", type=str)
def main(figure_folder,
         test_img_folder,
         model_name,
         stats_json,
         n_ch,
         test_name):
    
    logger.info(f"{test_name.upper()}")
    logger.info("Begin inference...")
    
    logger.info(f"Test files in {test_img_folder}")
    test_img_folder = Path(test_img_folder)
    n_test_img = len(list(test_img_folder.iterdir()))
    logger.info(f"{n_test_img} files detected")
    
    # Load model for inference
    logger.info(f"Loading U-Net model ({n_ch} channels)")
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=n_ch, out_channels=2, init_features=32, 
                           pretrained=False)

    model_path = Path(model_name)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info(f"Loaded from {model_name}")
    
    
    # Prepare data
    figure_folder = Path(figure_folder)
    test_result_folder = figure_folder / 'test_result'
    test_result_folder.mkdir(exist_ok=True)
    logger.info(f"Results will be saved at {test_result_folder}")
    
    stats = json.load(stats_json)
    
    mean_ch = stats['mean']
    std_ch = stats['std']
    logger.info(f"Mean from dataset: {mean_ch}")
    logger.info(f"Std from dataset: {std_ch}")
    
    # Iterate on image number
    logger.info(f"Generating {n_test_img} image results")
    for i, fn in tqdm(enumerate(test_img_folder.iterdir()),
                      desc="Generating masks on dataset",
                      total=n_test_img):
        logger.info(f"Image N°{i}/{n_test_img}")
        
        with rasterio.open(fn) as ds:
            n_ch = ds.count
            w = ds.width
            h = ds.height
                
            pad_h, pad_w = 8 - h % 8, 8 - w % 8
            
            test_img = prepare_img(fn,
                                   mean_ch,
                                   std_ch,
                                   padding=(pad_h, pad_w),
                                   as_torch=True)
            
            # Begin figure
            fig, ax = plt.subplots(1, 3, figsize=(18, 12))
            fig.suptitle(f"{test_name}\n")
            
            img_rgb = test_img[[4, 2, 1], :650, :650].numpy()
            img_rgb = np.transpose(img_rgb, axes=(1, 2, 0))
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() 
                                                   - img_rgb.min())
            
            with torch.no_grad():
                pred_mask = model(test_img.unsqueeze(0))
                
            pred_mask = pred_mask.squeeze(0)
            
            ax[0].imshow(img_rgb)
            ax[1].imshow(pred_mask[1, :h, :w])
            ax[2].imshow(pred_mask.argmax(dim=0)[:h, :w].numpy())
            
            fig.savefig(test_result_folder / f"{test_name}_{i}.png")
            logger.info(f"{test_name}_{i}.png saved!")


if __name__ == "__main__":
    main()