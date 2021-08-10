# Small repo for running some pytorch ssl models

## Installation

```
# module load anaconda3
conda env create -f environment.yml
conda activate ssl_runs

# instal apex for mixed precision
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
cd ../

```


## Pretraining models
```
DATADIR='/location/of/dataset'
OUTDIR='/location/where/to/save'

### run DINO

cd dino
sbatch scripts/resnet_ep800.sh $DATADIR $OUTDIR
sbatch scripts/vit_small16_ep800.sh $DATADIR $OUTDIR

# the run below requires 172 GPUs. this might take a while to queue 
sbatch scripts/vit_base8_ep300.sh $DATADIR $OUTDIR

cd ../

### SwAV/DC

cd swav
sbatch scripts/deepclusterv2_800ep_pretrain.sh $DATADIR $OUTDIR
sbatch scripts/swav_800ep_pretrain.sh $DATADIR $OUTDIR
sbatch scripts/swav_100ep_pretrain.sh $DATADIR $OUTDIR

sbatch scripts/swav_RN50w4_400ep_pretrain.sh $DATADIR $OUTDIR
cd ../

### run MoCo

cd moco
sbatch scripts/moco_800ep.sh $DATADIR $OUTDIR

```