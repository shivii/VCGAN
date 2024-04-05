# VCGAN

VCGAN performs unpaired Image-to-Image translation for the restoration of the image of the text. Rather  than  having  the  target  image  precisely  match the ground truth, we encourage the similar-  
ity of underlying semantic structural distributions.  We achieve this by a new similarity norm Top-k  
Variable loss  T V(k)

## Prerequisite
* Linux or macOs
* python3
* CPU or Nvidia GPU + CUDA CuDNN

## Getting Started
### Installation
Clone this repo
`git clone https://github.com/shivii/VCGAN.git`
`cd VCGAN`
### Datasets
Download the datasets for training and testing :
#### Training : 
IAM Strikethrough Database:
https://zenodo.org/records/4767095
#### Testing: 
Clone the following repository
```git clone https://github.com/shivii/Real-Strike-off-dataset.git```
### Train the model
```python train.py --dataroot ./dataset/IAM/```
## Citation
If you use this code for you research please cite our papers.
```
Nigam, S., Behera, A.P., Verma, S. and Nagabhushan, P., 2022. Deformity removal from handwritten text documents using variable cycle gan.
```
