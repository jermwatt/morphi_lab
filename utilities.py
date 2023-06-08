# pull some images from the web
import urllib.request

def download_file(url, output_path):
    urllib.request.urlretrieve(url, output_path)