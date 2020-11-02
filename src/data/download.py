"""CLI to download the raw dataset."""
import io
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import click
from loguru import logger
import requests

logger.add(
    Path('/tmp') / f"turbofan_create_dataset_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('url')
@click.argument('folder', type=click.Path())
def download_file(url: str, folder: Path):
    """
    Download the zip archive containing the dataset.

    Parameters
    ----------
    url : str
        URL to the Turbofan dataset.
    folder : Path
        Folder to store the downloaded dataset.
    """
    logger.info(f'Downloading from {url} to {folder}')
    Path(folder).mkdir(exist_ok=True)
    try:
        response = requests.get(url)
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {errh}')
    except requests.exceptions.ConnectionError as errc:
        logger.error(f'Error connecting: {errc}')
    except requests.exceptions.Timeout as errt:
        logger.error(f'Timeout error: {errt}')
    except requests.exceptions.RequestException as err:
        logger.error(f'Something else happened: {err}')
    else:
        with ZipFile(io.BytesIO(response.content)) as zip_obj:
            zip_obj.extractall(folder)


if __name__ == '__main__':
    download_file()  # pylint: disable=no-value-for-parameter
