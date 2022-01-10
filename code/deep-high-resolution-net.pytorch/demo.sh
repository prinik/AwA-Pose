#/bin/bash
#CUDA_VISIBLE_DEVICES=0 python tools/test.py --cfg experiments/awa/w48_384x288_1.yaml
CUDA_VISIBLE_DEVICES=0 python tools/demo.py --cfg experiments/awa/w48_384x288_sup_5.yaml --imFile 'Saint-Aignan_(Loir-et-Cher)._Okapi.jpg'