# Cross on Cross Attention: Deep Fusion Transformer for Image Captioning

## Environment setup
Clone the repository and create the `dft` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate dft
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.8 is required to run our code. 


## Data preparation
To run the code, annotations and visual features for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://pan.baidu.com/s/17ik-2OZGFaQ5-AzCCWkL9w) (Extraction code: ska0) and extract it.

To reproduce our result, please download the features files [COCO2014_RN50x4_GLOBAL.hdf5, COCO2014_VinVL.hdf5](https://pan.baidu.com/s/17ik-2OZGFaQ5-AzCCWkL9w) (Extraction code: ska0), in which features of each image are stored under the `<image_id>_features` key. `<image_id>` is the id of each COCO image, without leading zeros (e.g. the `<image_id>` for `COCO_val2014_000000037209.jpg` is `37209`). VinVL region feature dimension is (N, 2048), N is the number of region features; CLIP grid feature dimension is (M, 2560), M is the number of grid features.


## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [meshed_memory_transformer.pth](https://ailb-web.ing.unimore.it/publicfiles/drive/meshed-memory-transformer/meshed_memory_transformer.pth) and place it in the code folder.


## Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--output` | Output path|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 20) |
| `--workers` | Number of workers (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--N_enc` | Number of encoder layers|
| `--N_dec` | Number of decoder layers|
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--use_rl` | Whether to turn on reinforcement learning|
| `--clip_path` | CLIP grid feature path|
| `--vinvl_path` | VinVL region feature path|
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

For example, to train our model with the parameters used in our experiments, use
```
python train.py --exp_name dft --batch_size 20 --clip_path /path/to/clip_gird_features --clip_path /path/to/vinvl_region_features --annotation_folder /path/to/annotations
```

#### References
[1] Cornia M, Stefanini M, Baraldi L, et al. Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.  
[2] Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning.  
[3] Zhang P, Li X, Hu X, et al. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

#### Acknowledgements
Thank Cornia _et.al_ for their open source code ([meshed-memory-transformer
](https://github.com/aimagelab/meshed-memory-transformer)), on which our implements are based.  
Thanks to Zhang et al. for the powerful region features ([VinVL](https://github.com/pzzhang/VinVL)).
