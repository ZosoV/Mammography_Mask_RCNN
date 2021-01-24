import argparse
import numpy as np
import pandas as pd
import os
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		description='Transform the data in the correct format to be processed')

    parser.add_argument('--dataset_dir', required=True,
                        metavar="/path/to/dataset",
                        help="Path to dataset")
                    
    parser.add_argument('--csv', required=True,
                        metavar="/path/to/annotation.csv",
                        help="csv file with annotation data")   

    args = parser.parse_args()

    output_dir = os.path.join(args.dataset_dir,"data")

    # create important directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the csv
    data = pd.read_csv(args.csv)

    print(data.head())
    print("Reading: ",args.csv)

    rows, _ = data.shape
    print("Records: ",rows)

    base_names = []

    for row_id in range(rows):
        # Get the paths and sample name
        image_name = data.loc[row_id]['images']
        mask_name = data.loc[row_id]['masks']
        base_name = image_name.split("/")[0]

        # Reading the real image and its mask
        # img_0 = cv2.imread(join(DATA_DIR,image_name))
        # print("Base image shape: ", img_0.shape)

        mask_0 = cv2.imread(join(DATA_DIR,mask_name))
        print("Mask shape:{} Max: {} Min: {}".format(mask_0.shape,
                                                    np.amax(mask_0),
                                                    np.amin(mask_0)))

        # Moving the original image
        src = os.path.join(args.dataset_dir,image_name)
        dst = os.path.join(output_dir,base_name + ".png")

        shutil.copy2(src,dst)

        # Creating the .npz from image mask
        mask_0 = mask_0.clip(max = 1)

        # Compress to npz file
        compress_path = os.path.join(output_dir,base_name + ".npz")
        np.savez_compressed(compress_path, arr_0=mask)

        # Append basenames
        base_names.append(base_name)

        # Storing base_names in a .txt file
        with open(os.path.join(args.dataset_dir,'base_names.txt'), 'a') as file:
            for b_name in base_names:
                file.write(b_name +"\n")
