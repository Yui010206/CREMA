result_dir="result/AVQA/"

exp_name='avqa_zs_audio+video'
ckpt='crema_ckpt.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/music_avqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=32 \
datasets.musicavqa_mm_instruct.data_type=['audio','video'] \
model.downstream_task='oeqa' \
model.modalities='rgb_audio'
