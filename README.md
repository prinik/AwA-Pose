
AwA Pose Dataset:
============

Download Images
------------
Download images from https://cvml.ist.ac.at/AwA2/

Download Annotations
------------
```bash
git clone https://github.com/prinik/AwA-Pose.git
```
Read pickle annotation file
------------
```bash
import pickle
with open('antelope_1234.pickle', 'rb') as f:
    x = pickle.load(f)

```    
arXiv Paper link
------------
https://arxiv.org/pdf/2108.13958.pdf
------------

Annotation
------------
<p align="center">
<img src="Images/sample.png" width="380" height='430'>
</p>

------------

Prediction Result
------------
<p align="center">
<img src="Images/pred_hr.png" width="380" height='430'>
</p>