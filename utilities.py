import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

def download_file(url, output_path):
    urllib.request.urlretrieve(url, output_path)
    
def show_img(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  
    plt.show()






