#SBATCH --nodes=2
#SBATCH --gpus=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --job-name=dino_vit_small16_800ep
#SBATCH --constraint=volta32gb
#SBATCH --time=70:00:00
#SBATCH --mem=450G

module load anaconda3
source activate ssl_runs

DATADIR=$1
OUTDIR=$2
OUTDIR+='/dino_vit_small16_800ep'
mkdir -p ${OUTDIR}


python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
         --world_size 16 --ngpus 8 --nodes 2 --optimizer adamw --use_bn_in_head false \
         --arch vit_small --patch_size 16  --weight_decay 0.04 \
         --drop_path_rate 0.1 \
         --warmup_teacher_temp_epochs 30 \
         --clip_grad 0 --batch_size_per_gpu 64 --epochs 800 --freeze_last_layer 1 \
         --lr 0.0005 --warmup_epochs 10 --min_lr 1e-5 \
         --weight_decay_end 0.4 --global_crops_scale 0.25 1 \
         --out_dim 65536 \
         --use_fp16 false \
         --warmup_teacher_temp 0.04 \
         --teacher_temp 0.07 \
         --norm_last_layer false \
         --local_crops_scale 0.05 0.25 --local_crops_number 10 \
         --data_path $DATADIR \
         --output_dir $OUTDIR