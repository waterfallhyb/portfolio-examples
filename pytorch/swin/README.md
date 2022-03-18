## Environment Setup

First, Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```


## Datasets

Download the datasets:
* ImageNet dataset (available at [http://www.image-net.org/](http://www.image-net.org/))

The ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes, can be downloaded from [the ImageNet website](http://www.image-net.org/download). It is approximately 150GB for the training and validation sets. Please note you need to register and request permission to download this dataset on the Imagenet website. You cannot download the dataset until ImageNet confirms your registration and sends you a confirmation email. If you do not get the confirmation email within a couple of days, contact [ImageNet support](support@imagenet.org) to see why your registration has not been confirmed. Once your registration is confirmed, go to the download site. The dataset is available for non-commercial use only. Full terms and conditions and more information are available on the [ImageNet download](http://www.image-net.org/download)

Please place or symlink the ImageNet data in `./data/imagenet1k`.
The imagenet1k dataset folder contains `train` and `validation` folders, in which there are the 1000 folders of different classes of images.
```console
imagenet1k
|-- train [1000 entries exceeds filelimit, not opening dir]
`-- validation [1000 entries exceeds filelimit, not opening dir]
```
Then modify the DATA.DATA_PATH parameter in config we use into ImageNet's path

## Qucik start
ImageNet1k training(default is fp32.32):
```console
bash train_swin.sh
```

## Run the application

Setup your environment as explained above. You can run SWIN on ImageNet1k datasets. Train SWIN on IPU. 
We can train with fp32.32,fp16.16 and fp16.16.(The first 16 or 32 means that input is half or float, and the second means that weight is half or float)
We can use the PRECISION parameter in config to modify the required precision

Acc as follows: 

| model | input size | precision | machine |     acc     |
|-------|------------|-----------|---------|-------------|
| tiny  |   224      |    16.16  |  pod16  |    80.9%    |
| tiny  |   224      |    16.32  |  pod16  |    81.29%   |
| tiny  |   224      |    32.32  |  pod16  |    81.21%   |
| tiny  |   224      |     mix   |  v100   |  81.3%(SOTA)|
| base  |   224      |    32.32  |  pod16  |    82.9%    |
| base  |   224      |     mix   |  v100   |  83.5%(SOTA)|
| base  |   384      |    32.32  |  pod16  |    83.8%    |
| base  |   384      |     mix   |  v100   |  84.5%(SOTA)|

Once the training finishes, you can validate accuracy:
```console
python3 validate.py --cfg ./configs/swin_**.yaml --checkpoint ./output/*/*/.../ckpt_**.pth
```

## Licensing
This application is licensed under Apache License 2.0. Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under Apache License, Version 2.0  (<sup>*</sup> means additional license information stated following this list):
* train_swin.py
* validate.py
* options.py
* models/gelu.py
* models/build.py
* dataset/build_ipu.py  
* README.md
* swin_test.py
* utils.py

The following files include code derived from this [repo](https://github.com/microsoft/Swin-Transformer) which uses Microsoft and MIT license:
* model/swin_transformer.py
* dataset/cached_image_folder.py
* dataset/samplers.py
* dataset/zipreader.py
* config.py
* lr_scheduler.py
* optimizer.py
* train_swin.sh

External packages:
- `transformers` is licenced under Apache License, Version 2.0
- `pytest` is licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License