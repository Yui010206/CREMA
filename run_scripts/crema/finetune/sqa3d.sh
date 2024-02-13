result_dir="result/SQA3D/"

exp_name='sqa3d_video+3d'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port 29506 train.py \
--cfg-path lavis/projects/crema/train/sqa3d.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.downstream_task='oeqa' \
model.modalities='rgb_pc' \
datasets.sqa3d.data_type=['pc','video'] \
run.batch_size_train=16 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1

exp_name='sqa3d_video+3d+depth'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port 29506 train.py \
--cfg-path lavis/projects/crema/train/sqa3d.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth' \
datasets.sqa3d.data_type=['pc','frame','depth'] \
run.batch_size_train=16 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1

exp_name='sqa3d_espresso'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port 29506 train.py \
--cfg-path lavis/projects/crema/train/sqa3d.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso' \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth' \
datasets.sqa3d.data_type=['pc','frame','depth'] \
run.batch_size_train=16 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1