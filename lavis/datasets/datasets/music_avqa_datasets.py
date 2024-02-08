"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import copy
import os
import random
import json
import ast
import re
from PIL import Image
from lavis.datasets.datasets.base_dataset import BaseDataset

class MusicAVQADataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'], kwargs['modalities'])

        self.modalities = kwargs['modalities']

        for modality in self.modalities:
            if 'image' in modality:
                setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
                continue
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())

        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])

        try:
            self.annotation = [ann for ann in self.annotation if ann['video_id'] in self.sample_ids]
        except:
            self.sample_ids = set.intersection(*[set(self.get_existing_audio_pt_annotations()) for modality in self.modalities])
            self.annotation = [ann for ann in self.annotation if ann['id'] in self.sample_ids]
        
    def get_existing_audio_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.audio_root)]
    
    def get_existing_frame_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.frame_root)]
    
    def get_existing_flow_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.flow_root)]

    # def get_existing_audio_pt_annotations(self):
    #     return [f.split('.')[0] for f in os.listdir('/nas-ssd2/shoubin/datasets/audios')]

    # def get_pt_audio_path(self, ann):
    #     return os.path.join('/nas-ssd2/shoubin/datasets/audios', f'{str(ann["id"])}.wav')
    
    def get_existing_video_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.video_root)]
    
    def get_audio_path(self, ann):
        # return os.path.join(self.audio_root, f'{ann["video_id"]}.flac')
        return os.path.join(self.audio_root, f'{ann["video_id"]}.mp4')
    
    def get_video_path(self, ann):
        return os.path.join(self.video_root, f'{ann["video_id"]}.mp4')
    
    def get_flow_path(self, ann):
        return os.path.join(self.flow_root, f'{ann["video_id"]}/')
    
    def get_frame_path(self, ann):
        return os.path.join(self.frame_root, f'{ann["video_id"]}/')

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])

        for modality in self.modalities:
            if 'id' not in ann:
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                if type(ann[f"{modality}_path"]) == list:
                    ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])

            if 'image' in modality:
                ann['rgb'] = self.vis_processor(Image.open(ann[f"images_path"]))
            else:
                if modality == 'video':
                    rgb, indices, fps = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"])
                    ann['rgb'] = rgb.to(torch.float32)
                
                if modality == 'frame':
                    indices, clip = None, None
                    ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                    frms, indices = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices)
                    rgb = frms.permute(1, 0, 2, 3)
                    assert len(rgb) == getattr(self, f"{modality}_processor").n_frms
                    ann['rgb'] = rgb
                
                if modality == 'flow':
                    assert indices is not None
                    ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                    flow, _ = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices, type='depth')
                    ann['flow'] = flow
                    
                if modality == 'audio':
                    if 'id' in ann: # aduio pt data
                        ann[f"{modality}_path"] = self.get_pt_audio_path(ann)
                    ann[modality] = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"]).to(torch.float32)

        if 'id' not in ann:
            ann["sample_id"] = ann["video_id"]
            if len(ann['templ_values']) != 0:
                question = ann['question_content']
                templ_values = ast.literal_eval(ann['templ_values'])
                matches = re.findall(r'<(.*?)>', question)
                for k, v in zip(matches, templ_values):
                    question = question.replace('<'+k+'>', v)
            else:
                question = ann['question_content']
            ann['qa_input'] =  'Question: ' + self.text_processor(question) + ' Based on the frames and audio information, answer the question using a single word or phase.'
            ann["question_id"] = ann['question_id']
            answers = ann['anser']
            if '_' in answers:
                answers = answers.replace('_', ' ')
            ann['answers'] = answers #ann['anser']

        else:
            ann["sample_id"] = ann["id"]
            ann['answers'] = ann['answer']
            ann["question_id"] = ann['id']
            question = ann['question']
            ann['qa_input'] =  'Question: ' + self.text_processor(question) + ' Based on the audio information, answer the question using a single word or phase.'
            
        return ann
    
class MusicAVQAInstructDataset(MusicAVQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['answer'] = data["answers"] # needed to use gqa task
            data['qa_output'] = data["answers"]
                
        return data
