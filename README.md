AwA Pose Dataset:
============

Download Images
------------
Download images from https://cvml.ist.ac.at/AwA2/

Download Annotations
------------
```bash
git clone git+https://www.github.com/prinik/awa-pose
```
Read pickle annotation file
------------
```bash
import pickle
with open('antelope_1234.pickle', 'rb') as f:
    x = pickle.load(f)
```    
