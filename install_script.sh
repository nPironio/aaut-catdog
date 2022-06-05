#!/bin/bash

pip install --upgrade pip

# install pytorch

# old ver
# pip install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# new ver
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# other libs
pip install -r requirements.txt

# install local package
cd src/catdog/
python setup.py develop --user

