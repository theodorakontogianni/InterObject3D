# Interactive Object Segmentation in 3D Point Clouds



<div align="center">
<a href="https://theodorakontogianni.github.io/">Theodora Kontogianni</a>, <a href="https://www.gfz-potsdam.de/staff/ekin.celikkan/sec14">Ekin Celikkan</a>, <a href="https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html">Siyu Tang</a>, <a href="https://igp.ethz.ch/personen/person-detail.html?persid=143986">Konrad Schindler</a>

ETH Zurich


<!-- ![teaser](./inter3dpng) -->
<img src="./inter3d.png" width=80% height=80%>

</div>

This repository provides code, data and pretrained models for:

[[Paper](https://arxiv.org/abs/2204.07183)]

## Code
Inference code with pre-trained model is available - training code is coming soon

### Toy Data

Toy dataset can be found on: /InterObject3D/Minkowski/training/mini_dataset/


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


