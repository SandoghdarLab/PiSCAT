from piscat.InputOutput.read_write_data import download_url
import os


if __name__ == '__main__':
    current_path = os.path.abspath(os.path.join('.'))
    download_url(url='https://owncloud.gwdg.de/index.php/s/AsPgd3U5YSXahBe/download', save_path=current_path)
