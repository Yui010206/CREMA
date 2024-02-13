result_dir="result/AVQA/"

exp_name='avqa_video+audio'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port 29503 train.py \
--cfg-path lavis/projects/crema/train/music_avqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','video'] \
model.modalities='rgb_audio_flow' \
run.batch_size_train=24 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1

exp_name='avqa_video+audio+flow'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port 29503 train.py \
--cfg-path lavis/projects/crema/train/music_avqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','frame','flow'] \
model.modalities='rgb_audio_flow' \
run.batch_size_train=24 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1


exp_name='avqa_espresso'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port 29503 train.py \
--cfg-path lavis/projects/crema/train/music_avqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso' \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','frame','flow'] \
model.modalities='rgb_audio_flow' \
run.batch_size_train=24 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1