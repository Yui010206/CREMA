import os
import json
import subprocess
from tqdm import tqdm 

def extract_frames_ffmpeg(video_path, output_dir, frame_rate):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    # 构建ffmpeg命令
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={frame_rate}',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    # 运行ffmpeg命令
    subprocess.run(cmd)

data = json.load(open('touchqa_train.json')) + json.load(open('touchqa_val.json')) 

for d in tqdm(data):
    vid = d['video_id']
    qid = d['question_id']
    # 使用示例
    video_path = 'clips/{}.mp4'.format(str(qid))  #
    output_dir = 'video_frames/{}'.format(str(qid))  
    frame_rate = 3  # 
    extract_frames_ffmpeg(video_path, output_dir, frame_rate)

    video_path = 'touch_clips/{}_touch.mp4'.format(str(qid))  # 
    output_dir = 'touch_frames/{}'.format(str(qid))  # 
    frame_rate = 3  # 
    extract_frames_ffmpeg(video_path, output_dir, frame_rate)
