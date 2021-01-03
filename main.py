# Import external libraries
from tensorflow.keras.applications.vgg16    import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing         import image
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# Import internal libraries
from analysis import analysis
from utils import process_image, rescale, get_args

def main():
    # Get arguments
    current_folder          =   os.path.abspath(os.path.normpath(os.path.join(__file__, os.path.pardir)))
    in_folder, out_folder   =   get_args()
    source                  =   os.path.join(current_folder, in_folder)
    target                  =   os.path.join(current_folder, out_folder)

    # Define the target model
    modelname               =   "vgg"
    model                   =   VGG16(weights="imagenet")
    preprocessor            =   preprocess_input
    decoder                 =   decode_predictions
    target_layer            =   "block5_conv3"
    n_classes               =   1000

    # Get all image filenames
    filenames               =   glob.glob(source)

    # Define processing images folder name
    if not os.path.isdir(target):   os.mkdir(target)

    errorlog                =   []

    print("Analysing all provided images.")

    # Loop over all files
    for filename in tqdm(filenames):
        # Extract filename without extension
        name    =   os.path.basename(filename).split(".")[0]

        # Load the image
        img     =   image.load_img(filename, target_size=(224,224))

        # Preprocess the image
        _,ground_truth = process_image(img, preprocess_input)

        # Apply explanation methods
        try:
            backprop, gradcam, squadcam, data = analysis(img, model, preprocessor, decoder, target_layer, n_classes)
        except ValueError:
            errorlog.append("ValueError: Image {} -  Expected image size (224, 224). Got {}".format(filename, img.size))

        # Create folder for output if none exists
        if not os.path.isdir(os.path.join(target, name)):
            os.mkdir(os.path.join(target,name))
        
        # Save explanations and ground truth
        plt.imsave("{}/{}/ground_truth.jpg".format(target, name), rescale(ground_truth))
        plt.imsave("{}/{}/backprop_{}_{}.jpg".format(target, name, modelname, data[1]), backprop)
        plt.imsave("{}/{}/gradcam_{}_{}.jpg".format(target, name, modelname, data[1]), gradcam)
        plt.imsave("{}/{}/squadcam_{}_{}.jpg".format(target, name, modelname, data[1]), squadcam)
    if errorlog:
        print("Errors:\n{}".format("\n".join(errorlog)))

    print("Analysis completed")
    
    
if __name__ == "__main__":
    main()