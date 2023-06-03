import cv2
import os
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))


class AV:
    def __init__(self, img_path):
        self.img_path = img_path 
        self.read_img_path(img_path)
    
    def read_img_path(self, img_path):
        self.img = cv2.imread(img_path)

    def read_img(self, img):
        self.img = img

    def show_result(self):
        image_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis('off')  
        plt.show()
