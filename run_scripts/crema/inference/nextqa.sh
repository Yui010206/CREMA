result_dir="result/SQA3D/"

# to load fine-tuned model: set model.mmqa_ckpt='path_to_ft_model'

exp_name='nextqa_vtnd'
crema_ckpt='crema_pretrained.pth'
finetuned_ckpt='' # path to finetuning output checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${crema_ckpt} \
model.mmqa_ckpt=${finetuned_ckpt} \
model.task='espresso-concat-seq' \
model.frame_num=4 \
run.batch_size_eval=8 \
model.modalities='rgb_norm_depth' \
datasets.nextqa.modality_type=['rgb','norm','depth'] \
model.downstream_task='oeqa'