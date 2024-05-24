# M2PNet: Multi-Scale-and-Multi-Direction-Perception-Network-for-Tubular-Structure-Segmentation
## Abstract
Accurate segmentation of tubular structure images, such as roads and blood vessels, is critical for downstream tasks in many fields. Existing methods mainly focus on exploiting the topological structure of individual tubular shapes, often ignoring valuable prior knowledge embedded within the image context. In particular, the diverse size range of these tubular structures and their intricate branching configurations exhibit the typical multi-scale and multi-directional features. Drawing inspiration from these, we propose a contextual information-aware multi-scale and multi-direction perception network (M $^2$ PNet) for tubular structure segmentation, which aims to augment the depth perception of tubular structure image features through improved image feature extraction, fusion, refinement, and enhancement. First, a large kernel snake convolution is designed to dynamically prioritize tubular structures while ensuring an expansive receptive field to capture contextual information. Then, we propose a multi-directional feature fusion strategy that leverages multi-scale hybrid attention mechanisms to enhance the integration of features from diverse directions. Finally, a multi-scale feature refinement unit and a branching feature enhancement module are proposed to amplify the depth of feature comprehension and utilization. Experimental results on four representative datasets show that the proposed M$^2$PNet outperforms several state-of-the-art methods in tubular structure segmentation tasks.


## Usage
### Install
```bash
git clone https://github.com/xwf12345678/M2PNet.git
```
### Training for DRIVE dataset
```bash
cd Drive
python Main.py
```
### Training for other datasets
Coming Soon
