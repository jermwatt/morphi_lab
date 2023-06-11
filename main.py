from segmenter import segment_image, label_lookup_dict
from diffuser import diffuse_segmented_img
from utilities import show_all_results


def main(img_path,
         labels=List,
         prompt='an ape, smiling, high resolution, holding something',
         seed=None,
         negative_prompt=None,
         num_inference_steps=100,
         verbose=False):
  
    # check for required arguments
    if img_path is None:
        print('FAILURE: img_path')
  
    # segment the donut out of the test image
    img_path = "/content/test_donut.png"
    labels = ['person']
    
    # star
    img, mask, seg = segment_image(img_path,
                                   labels=labels)

    # diffuse the masked segmentation 
    diffused_img = diffuse_segmented_img(img,
                                         mask,
                                         prompt,
                                         seed=3433)

    # show results
    show_all_results(seg.orig_img,
                     mask,
                     diffused_img)
