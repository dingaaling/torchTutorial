#! /bin/bash

# Copy Imagenet
echo "Setting up FineTuning data for CS5304 CNN Assignment 3"
cd ~/data
echo "Working in `pwd`"

# Clean slate to start
/bin/rm -rf tiny-imagenet*

tinyimgnet="tiny-imagenet-200.zip"
curl -O http://cs231n.stanford.edu/$tinyimgnet
unzip -q $tinyimgnet

# Create dataset directories
mkdir -p tiny-imagenet-5/train
mkdir -p tiny-imagenet-5/val/images

# Get the 5 classes
LABELS=$(cat tiny-imagenet-200/val/val_annotations.txt | grep n09 | cut -f2 | sort | uniq)

# Copy training data
echo Copying training data
for label in ${LABELS[@]}; do
    echo $label
    cp -r tiny-imagenet-200/train/$label tiny-imagenet-5/train/$label
done

# Copy validation data
echo Copying validation data
for label in ${LABELS[@]}; do
    echo $label
    IMAGES=$(cat tiny-imagenet-200/val/val_annotations.txt | grep $label | cut -f1)
    for image in ${IMAGES[@]}; do
        cp -r tiny-imagenet-200/val/images/$image tiny-imagenet-5/val/images/$image
    done
done

cat tiny-imagenet-200/val/val_annotations.txt | grep n09 > tiny-imagenet-5/val/val_annotations.txt

# Go back to original directory
cd -
