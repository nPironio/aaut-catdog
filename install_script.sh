#!/bin/bash

pip install --upgrade pip

# production
pip install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html --user
pip install -r requirements.txt