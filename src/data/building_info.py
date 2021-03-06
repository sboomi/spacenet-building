import pandas as pd
import logging
import click
from pathlib import Path

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('building_info')


def generate_csv(csv_folder, csv_res):
    dfs = [pd.read_csv(csv_file) for csv_file in csv_folder.iterdir()]
    result = pd.concat(dfs)
    result.to_csv(csv_res, index=None)
    return result


def count_images(df):
    cities = ["Paris", "Shanghai", "Khartoum", "Vegas"]
    total_samples = df['ImageId'].unique().size
    for city in cities:
        city_samples = df[df['ImageId'].str.contains(city)]['ImageId'].unique().size
    logger.info(f"{city}: {city_samples}/{total_samples}")


def count_files_per_city(image_folder):
    cities = ["Paris", "Shanghai", "Khartoum", "Vegas"]
    count_dict = {city: 0 for city in cities}
    
    for filename in image_folder.iterdir():
        for city in cities:
            if city in filename.stem:
                count_dict[city] += 1
    
    for city, count in count_dict.items():
        logger.info(f"{city}: {count} images")
        

@click.command()
@click.argument("image_folder", type=click.Path(exists=True))
@click.argument("csv_folder", type=click.Path(exists=True))
@click.argument("csv_res", type=click.Path())
def main(image_folder, csv_folder, csv_res):
    csv_res = Path(csv_res)
    csv_folder = Path(csv_folder)
    image_folder = Path(image_folder)
    
    if not csv_res.exists():
        df = generate_csv(csv_folder, csv_res)
    else:
        df = pd.read_csv(csv_res)

    count_images(df)
    count_files_per_city(image_folder)    


if __name__ == "__main__":
    main()