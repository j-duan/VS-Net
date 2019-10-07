# VS-Net: Variable splitting network for accelerated parallel MRI reconstruction 


The code in this repository implements VS-Net, a model-driven neural network for accelerated parallel MRI reconstruction.

# Overview
The files in this repository are organized into 3 directories:
* [code](code) : contains base functions for segmentation, co-registration, mesh generation, and motion tracking:
  * code entrance - [code/DMACS.py](code/DMACS.py)
  * deep learning segmentation with the pre-trained model - [code/deepseg.py](code/deepseg.py)
  * co-registration to fit a high-resolution model - [code/p1&2processing.py](demo/p1&2processing.py)
  * fitting meshes to high-resolution model - [code/meshfitting.py](code/meshfitting.py)
  * useful image processing functions used in the pipeline - [code/image_utils.py](code/image_utils.py)
  * downsample mesh resolution while remain its geometry - [code/decimation.py](code/decimation.py)
* [model](model) : contains a tensorflow model pre-trained on ~400 manual annotations on PH patients
* [data](data) : data download address, which contains three sample datasets (4D NIfTI) on which functions from the `code` directory can be run. You should download the data and place them into this folder.

To run the code in the [code](code) directory, we provide a [Docker](https://www.docker.com) image with all the necessary dependencies pre-compiled. 



## 3. Citation
If you find this software useful for your project or research. Please give some credits to authors who developed it by citing some of the following papers. We really appreciate that. 

[1] Duan J, Schlemper J, Qin C, Ouyang C, Bai W, Biffi C, Bello G, Statton B, O'Regan DP, Rueckert D. VS-Net: Variable splitting network for accelerated parallel MRI reconstruction. arXiv preprint arXiv:1907.10033. *[MICCAI]* (2019). 

[2] Bello GA, Dawes TJW, Duan J, Biffi C, de Marvao A, Howard LSGE, Gibbs JSR, Wilkins MR, Cook SA, Rueckert D, O'Regan DP. Deep learning cardiac motion analysis for human survival prediction. *[Nature Machine Intelligence](https://doi.org/10.1038/s42256-019-0019-2)* 1, 95–104 (2019).
