import subprocess
import cv2

import json
from tqdm import tqdm

data = json.load(open('touchqa_train.json')) + json.load(open('touchqa_val.json')) 
base_video_path = 'dataset/'
save_touch_path = 'touch_clips/'
save_video_path = 'clips/'

for d in tqdm(data):
    vid = d['video_id']
    qid = d['question_id']
    start_frame, end_frame = int(d['start']), int(d['end'])
    # 视频文件路径
    video_path = 'dataset/{}/video.mp4'.format(vid)
    # print(video_path)
    touch_path = 'dataset/{}/gelsight.mp4'.format(vid)
    video_output = 'clips/{}.mp4'.format(str(qid))
    touch_output = 'touch_clips/{}_touch.mp4'.format(str(qid))
    
    # 获取视频的帧率
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 将帧数转换为时间（秒）
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time

    # 使用 FFmpeg 剪辑视频
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # 直接复制视频流，不重新编码
        video_output
    ]

    subprocess.run(ffmpeg_command)

    ffmpeg_command = [
        'ffmpeg',
        '-i', touch_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # 直接复制视频流，不重新编码
        touch_output
    ]

    subprocess.run(ffmpeg_command)
