#SBATCH --nodes=22
#SBATCH --gpus=176
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --job-name=dino_vit_base8_ep300
#SBATCH --constraint=volta32gb
#SBATCH --time=70:00:00
#SBATCH --mem=450G

module load anaconda3
source activate ssl_runs

DATADIR=$1
OUTDIR=$2
OUTDIR+='/dino_vit_base8_ep300'
mkdir -p ${OUTDIR}

python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
         --world_size 176 --ngpus 8 --nodes 2 --optimizer adamw --use_bn_in_head false \
         --arch vit_base --patch_size 8  --weight_decay 0.04 \
         --drop_path_rate 0.1 \
         --warmup_teacher_temp_epochs 50 \
         --clip_grad 3.0 --batch_size_per_gpu 6 --epochs 300 --freeze_last_layer 3 \
         --lr 0.0005 --warmup_epochs 10 --min_lr 2e-06 \
         --weight_decay_end 0.4 --global_crops_scale 0.25 1 \
         --out_dim 65536 \
         --use_fp16 false \
         --warmup_teacher_temp 0.03 \
         --teacher_temp 0.07 \
         --norm_last_layer true \
         --local_crops_scale 0.05 0.25 --local_crops_number 10 \
         --data_path $DATADIR \
         --output_dir $OUTDIR