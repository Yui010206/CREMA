result_dir="result/SQA3D/"

# to load fine-tuned model: set model.mmqa_ckpt='path_to_ft_model'

exp_name='sqa3d_zs_video+3d'
ckpt='crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/sqa3d_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=24 \
datasets.sqa3d.data_type=['pc','video'] \
model.downstream_task='oeqa' \
model.modalities='pc_rgb'

exp_name='sqa3d_zs_video'
ckpt='crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/sqa3d_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=24 \
datasets.sqa3d.data_type=['video'] \
model.downstream_task='oeqa' \
model.modalities='rgb-skip'

exp_name='sqa3d_zs_3d'
ckpt='crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/sqa3d_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=24 \
datasets.sqa3d.data_type=['pc'] \
model.downstream_task='oeqa' \
model.modalities='pc'