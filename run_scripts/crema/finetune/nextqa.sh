result_dir="result/NeXTQA/"

exp_name='nextqa_crema_vtndf'
ckpt='crema_initial.pth'

CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29203 train.py \
--cfg-path lavis/projects/crema/train/nextqa-exp3-bs2.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_norm_depth_flow' \
datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=1e-4 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 