# Interactive Object Segmentation in 3D Point Clouds



<div align="center">
<a href="https://theodorakontogianni.github.io/">Theodora Kontogianni</a>, <a href="https://www.gfz-potsdam.de/staff/ekin.celikkan/sec14">Ekin Celikkan</a>, <a href="https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html">Siyu Tang</a>, <a href="https://igp.ethz.ch/personen/person-detail.html?persid=143986">Konrad Schindler</a>

ETH Zurich


<!-- ![teaser](./inter3dpng) -->
<img src="./inter3d.png" width=80% height=80%>

</div>

This repository provides code, data and pretrained models for:

[[Paper](https://arxiv.org/abs/2204.07183)]

### Training code now available!!!
### Toy Data for Inference

Toy dataset can be found on: /InterObject3D/Minkowski/training/mini_dataset/

### ICRA 2023 Paper Data

You can download the datasets used in the paper here: :https://drive.google.com/drive/folders/1B1uTY8Y8FeCyEfwlfZfivhcAFOOfUtPC?usp=sharing

### Pre-trained weights
You can download the pretrained weights in:
https://omnomnom.vision.rwth-aachen.de/data/3d_inter_obj_seg/scannet_official/weights/


### Installation

Preferably install a python virtual env (conda) using the requirements file in the repository or use it as a guideline since the ME engine needs to be installed seperately.
The code is based on the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine), and the [documentation page](https://nvidia.github.io/MinkowskiEngine/overview.html#installation) contains useful instructions on how to install the Minkowski Engine.




### Run pre-trained model

Run for a single instance:
```
python run_inter3d.py --verbal=True --instance_counter_id=1 --number_of_instances=1 --cubeedge=0.05 --pretraining_weights='/media/kontogianni/Samsung_T5/intobjseg/datasets/scannet_official/weights/exp_14_limited_classes/weights_exp14_14.pth' --dataset='scannet'  --visual=True --save_results_file=True --results_file_name=results_scannet_mini.txt
```
Run for all 5 instances in the toy dataset:
```
python run_inter3d.py --verbal=True --instance_counter_id=0 --number_of_instances=5 --cubeedge=0.05 --pretraining_weights='/media/kontogianni/Samsung_T5/intobjseg/datasets/scannet_official/weights/exp_14_limited_classes/weights_exp14_14.pth' --dataset='scannet'  --visual=True --save_results_file=True --results_file_name=results_scannet_mini.txt
```

### Evaluate pre-trained model 

#### ScanNet-val

Results from our evaluation on ScanNet-val are in the InterObject3D/Minkowski/training/results folder. 
If you run the evaluation script please follow a similar format for the results. Then:

Go to the following directory

```
cd InterObject3D/Minkowski/training/
```
If needed adjust the results paths and run:

```
python evaluation/compute_noc.py
```

### Train model

#### Prepare training data
We trained our model on ScanNet-train (excluding some classes if needed for some experiments). However our setup requires adjusting it for binary (foreground/background) segmentation so every 3D scene for example with 20 objects is split into 20 scenes each one of them with a single object instance as foreground.

Go to the following directory

```
cd InterObject3D/Minkowski/datasetgen/
```
If needed adjust the results paths and run for all scenes:

```
python main_scannet.py --name=<scene name>
```
This will result in a folder containing with the adjusted input data (scropped scenes around the object, binary gt, clicks for training)
```
└── results/
    └── crops5x5
      ├──scene0000_00
      |   	├──scene0000_00_crop_0
      |   	├──scene0000_00_crop_1
      |   	├──.....
      ├──scene0000_01
      ├──.....
    └── scans5x5
      ├──scene0000_00
      |   	├──scene0000_00_crop_0
      |   	├──scene0000_00_crop_1
      |   	├──.....
      ├──scene0000_01
      ├──.....
```

**Similar setups can be used for other training datasets**
#### Run training

Store a list of your training scenes + object ids in a numpy array in the examples folder:

'examples/dataset_train.npy'
```
array([['scene0191_00', '0'],
       ['scene0191_00', '1'],
       ['scene0191_00', '2'],
       ...,
       ['scene0567_01', '18'],
       ['scene0567_01', '19'],
       ['scene0567_01', '20']], dtype='<U32')

```

Go to the following directory

```
cd InterObject3D/Minkowski/training/
```
If needed adjust the results paths and run:

```
python train.py
```
## License
Copyright (c) 2021 Theodora Kontogianni, ETH Zurich

By using this code you agree to the terms in the LICENSE.

Moreover, you agree to cite the `Interactive Object Segmentation in 3D Point Clouds` paper in 
any documents that report on research using this software or the manuscript.


<details>
  <summary> Show LICENSE (click to expand) </summary>
Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.
For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

</details>



## Citation
```
@article{kontogianni2022interObj3d,  
  author = {Kontogianni, Theodora and Celikkan, Ekin and Tang, Siyu and Schindler, Konrad},
  title = {{Interactive Object Segmentation in 3D Point Clouds}},
  journal = {ICRA},
  year = {2023},
  }
```

## Aknowledgements
We would like to thank the authors of ME for providing their codebase.

* [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)


