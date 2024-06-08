result_dir="result/AVQA/"

exp_name='avqa_crema_aff'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port 29503 train.py \
--cfg-path lavis/projects/crema/train/music_avqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','frame','flow'] \
model.modalities='rgb_audio_flow' \
run.batch_size_train=24 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1

exp_name='avqa_crema_affdn'
ckpt='crema_initial.pth'
python -m torch.distributed.run --nproc_per_node=$1 --master_port 29513 train.py \
--cfg-path lavis/projects/crema_v2/train/music_avqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','frame','flow','depth','norm'] \
model.modalities='rgb_audio_flow_depth_norm' \
run.batch_size_train=24 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=200 \
run.warmup_steps=1000 \
run.accum_grad_iters=1

exp_name='avqa_crema_aff_early_exit'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port 29503 train.py \
--cfg-path lavis/projects/crema/train/music_avqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq-gradsel-es' \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','frame','flow'] \
model.modalities='rgb_audio_flow' \
run.batch_size_train=24 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 \
model.es_temperature 1.4
