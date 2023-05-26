import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from PIL import Image
from io import BytesIO


def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def image_plot_detections(image, detections):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch
    for detection in detections:
        x = detection['box']['xmin']
        y = detection['box']['ymin']
        width = detection['box']['xmax'] - detection['box']['xmin']
        height = detection['box']['ymax'] - detection['box']['ymin']
        label = detection['label']

        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(x, y, label, color='lime', verticalalignment='top', backgroundcolor='black', fontsize=8)
    plt.axis('off')
    plt.show()
