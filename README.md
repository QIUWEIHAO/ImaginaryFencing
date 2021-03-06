# Overview

This project contains the Deep Neural Network models to train fencing game poses and generate new poses and images.

## Folder Structure:

```java
- Scripts
	- Datasets   // scripts for constructing datasets
	- Models     // scripts of model definition, loss function, and helper functions
- Notebooks    // notebook for training and inferencing the model (interaction generator
- Results      // results generated by the scripts, including constructed datasets, trained models, and image translation input outpu
	- Datasets
	- Models
	- PSP_translation
```

## To construct the datasets from scratch:


1. Download the dataset zip file, which contains the following folders:
    1. video: the video clips
    2. openpose_results: the JSON file for each video clip produced by running each video with OpenPose
    3. marker_crop: the score indicator patch for determining the game results, i.e. which side won the game
2. Modify and run two scripts in `ImaginaryFencing/Datasets/` 
    1. extract video information such as winning results, scoring moment, two players’ poses, etc.
        
        ```bash
        python extract_videoinfo_poses.py
        ```
        
    2. generate image crops of the winners and losers, which can be used to train the StyleGAN2-ada model, and then used for training the pSp model
        
        ```bash
        python crop_pose_image.py
        ```
        

## Run the Notebook


The notebook `Notebooks/train_interaction_generator.ipynb` trains an interaction generator by having a neural network to predict the winners poses based on the losers poses and the distance between them. The poses were aligned to their bounding box centers and distances computed with subtracting two bounding box centers.

Besides model training, the notebook also generates previews of aggregated poses at new distances, which are “imagined” by the model. The individual imagined pose was rendered into photographs by my pre-trained pSp model. The rendered photograph were then used to composite the final image in Photoshop.
