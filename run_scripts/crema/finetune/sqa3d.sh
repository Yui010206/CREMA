result_dir="result/SQA3D/"

exp_name='sqa3d_crema_pfdn'
ckpt='crema_initial.pth'
python -m torch.distributed.run --nproc_per_node=$1 --master_port 29503 train.py \
--cfg-path lavis/projects/crema_v2/train/sqa3d.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth_norm' \
datasets.sqa3d.data_type=['pc','frame','depth','norm'] \
run.batch_size_train=8 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1


exp_name='sqa3d_crema_pfdn_early_exit'
python -m torch.distributed.run --nproc_per_node=$1 --master_port 29503 train.py \
--cfg-path lavis/projects/crema_v2/train/sqa3d.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq-gradsel-es' \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth_norm' \
datasets.sqa3d.data_type=['pc','frame','depth','norm'] \
run.batch_size_train=8 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 \
model.es_temperature 1.8