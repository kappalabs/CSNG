

def correct_images(images_path: str, correct_images_path: str):
    # load all png images in a directory
    import os
    from os import listdir
    from os.path import isfile, join
    from PIL import Image
    import numpy as np

    # create correct_images_path if it doesn't exist
    if not os.path.exists(correct_images_path):
        os.makedirs(correct_images_path)

    # load all png images in a directory and all subdirectories of images_path
    images_pil = []
    for dirpath, dirnames, filenames in os.walk(images_path):
        for filename in filenames:
            if filename.endswith('.png'):
                filepath = os.path.join(dirpath, filename)
                image = Image.open(filepath)
                images_pil.append((image, dirpath, filename))

    # for each image, multiply its value by 255/100 and save it to correct_images_path
    for image_pil, dirpath, image_file in images_pil:
        new_dirpath = dirpath.replace(images_path, correct_images_path)
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)

        image = np.array(image_pil)
        image = image / 100 * 255
        # clip to 0-1
        image = np.clip(image, 0, 255)
        image = Image.fromarray(image.astype(np.uint8))
        image.save(join(new_dirpath, image_file))


def main():
    correct_images("images", "correct_images")


if __name__ == '__main__':
    main()
