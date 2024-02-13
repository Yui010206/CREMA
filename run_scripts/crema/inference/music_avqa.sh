result_dir="/nas-hdd/shoubin/result/debug/"

exp_name='test_avqa_video'
ckpt='crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/music_avqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=32 \
datasets.musicavqa_mm_instruct.data_type=['video'] \
model.downstream_task='oeqa' \
model.modalities='rgb-skip'

exp_name='test_avqa_video+audio'
ckpt='crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/music_avqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=24 \
datasets.musicavqa_mm_instruct.data_type=['audio','video'] \
model.downstream_task='oeqa' \
model.modalities='audio_rgb-skip'

exp_name='test_avqa_audio'
ckpt='crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/music_avqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
run.batch_size_eval=32 \
datasets.musicavqa_mm_instruct.data_type=['audio'] \
model.downstream_task='oeqa' \
model.modalities='audio'
