from src.deepsegment.modules.reorder_files import reorder_files
import os

def main(root_dir):
    image_dir = os.path.join(root_dir, "images")
    mask_dir = os.path.join(root_dir, "masks")
    
    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)
    masks = reorder_files(masks, images)

    return images, masks

if __name__ == '__main__':
    root_dir = "/Users/chrisviets/Documents/deepsegment/images/training_set"
    print(main(root_dir))