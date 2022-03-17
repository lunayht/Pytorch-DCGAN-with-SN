#!/bin/sh
# Install required packages
pip install -r requirements.txt --user

# Create folders
mkdir -p checkpoint celeba img/fake img/real

# Download and Extract Datasets
if [! -f ./celeba.zip] ;
then 
    wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
    unzip celeba.zip -d celeba
fi

# Download and Extract FID and Inception Scripts
wget -N --no-check-certificate 'https://docs.google.com/uc?export=download&id=1InzR1qylS3Air4IvpS9CoamqJ0r9bqQg' -O inception.py
wget -N --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AtTxnuasIaSTTmI9MI7k8ugY8KJ1cw3Y' -O fid_score.py