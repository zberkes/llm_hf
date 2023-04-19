# Prepare:

source: https://pub.towardsai.net/run-very-large-language-models-on-your-computer-390dd33838bb


````shell
conda create -n device_map python=3.10
conda activate device_map

conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge transformers
conda install -c conda-forge accelerate
````
