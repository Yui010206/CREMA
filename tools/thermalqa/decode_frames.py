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


data = json.load(open('thermalqa_train.json')) + json.load(open('thermalqa_val.json')) 

for d in tqdm(data):
    vid = d['video_id']
    qid = d['question_id']
    # 使用示例
    video_path = 'clips/{}.mp4'.format(str(qid))  # 替换为你的视频文件路径
    output_dir = 'video_frames/{}'.format(str(qid))  # 替换为你希望保存帧的目录
    frame_rate = 3  # 每秒提取3帧
    extract_frames_ffmpeg(video_path, output_dir, frame_rate)

    video_path = 'thermal_clips/{}_touch.mp4'.format(str(qid))  # 替换为你的视频文件路径
    output_dir = 'thermal_frames/{}'.format(str(qid))  # 替换为你希望保存帧的目录
    frame_rate = 3  # 每秒提取3帧
    extract_frames_ffmpeg(video_path, output_dir, frame_rate)