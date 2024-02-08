"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np
import copy

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from lavis.datasets.datasets.base_dataset import BaseDataset


def get_qclass(question):
    lques = question
    if 'What' in lques:
        return 'What'
    if 'How' in lques:
        return 'How'
    if 'Can' in lques:
        return 'Can'
    if 'Is' in lques:
        return 'Is'
    if 'Which' in lques:
        return 'Which'
    return 'Other'

class ThreeDVQADataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'], kwargs['modalities'])

        self.modalities = kwargs['modalities']

        self.scene_ids = {}
        n = 0
        new_annotation = []
        for ann in self.annotation:
            try:
                img_id = ann["scene_id"]
                if img_id not in self.scene_ids.keys():
                    self.scene_ids[img_id] = n
                    n += 1
                new_annotation.append(ann)
            except:
                pass
        self.annotation = new_annotation
        

        for modality in self.modalities:
            if 'pc' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                self.pc_feat_root = self.pc_root + '/voxelized_features_sam_nonzero_preprocess/'
                self.voxel_root = self.pc_root + '/voxelized_voxels_sam_nonzero_preprocess/'
                self.annotation = [
                ann for ann in self.annotation if os.path.exists(os.path.join(self.pc_feat_root, str(ann["scene_id"]) + ".pt"))
            ]
            if 'video' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
                setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
            
            if 'frame' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])

            if 'depth' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
    
    def get_existing_video_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.video_root)]
    
    def get_video_path(self, ann):
        return os.path.join(self.video_root, f'{ann["scene_id"]}.mp4')
    
    def get_frame_path(self, ann):
        return os.path.join(self.frame_root, f'{ann["scene_id"]}/')
    
    def get_depth_path(self, ann):
        return os.path.join(self.depth_root, f'{ann["scene_id"]}/')

    def __getitem__(self, index):

        ann = copy.deepcopy(self.annotation[index])
        if 'question_id' in ann.keys(): # 3dqa data
            qa_input = self.text_processor(ann['situation']) + '. Question: ' + self.text_processor(ann["question"]) + ' Based on the frames and 3D Model information, answer the question using a single word or phase.'
            qtype = get_qclass(ann['question'])
            question_id = ann['question_id']
            answer = ann["answers"][0]
            question_id = qtype + '_' + str(question_id)
        else: # pre-training data
            qa_input = 'Question: ' + self.text_processor(ann["question"]) + ' Based on 3D Model information, answer the question.'
            answer = ann["answers"][0]
            answer = self.text_processor(answer)
            question_id = str(ann["scene_id"])

        scene_id = str(ann["scene_id"])

        out = {
            "qa_input": qa_input,
            "qa_output": answer,
            "scene_id": self.scene_ids[ann["scene_id"]],
            "question_id": question_id,
        }

        for modality in self.modalities:
            
            if modality == 'pc':
                pc_feat = torch.load(os.path.join(self.pc_feat_root, f"{scene_id}.pt"), map_location="cpu")
                if isinstance(pc_feat, np.ndarray):
                    pc_feat = torch.tensor(pc_feat).float()
                pc = np.load(os.path.join(self.voxel_root, f"{scene_id}.npy"))
                pc = torch.tensor(pc).float().cpu()
                if pc_feat.shape[0] > 5000:
                    idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
                    pc_feat = pc_feat[idxes]
                    pc = pc[idxes]
                else:
                    pc_feat = torch.cat([pc_feat, torch.zeros(5000 - pc_feat.shape[0], 1408)], dim=0)
                    pc = torch.cat([pc, torch.zeros(5000 - pc.shape[0], 3)], dim=0)
                out["pc_feat"] = pc_feat
                out['pc'] = pc
            
            if modality == 'video':
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                rgb, indices, fps = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"])
                out['rgb'] = rgb.to(torch.float32)
            
            if modality == 'frame':
                indices = None
                clip = None
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                frms, indices = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices)
                rgb = frms.permute(1, 0, 2, 3)
                assert len(rgb) == getattr(self, f"{modality}_processor").n_frms
                out['rgb'] = rgb

            if modality == 'depth':
                assert indices is not None
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                depth, _ = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices, type='depth')
                out['depth'] = depth

        return out

    def __len__(self):
        return len(self.annotation)


class ThreeDVQAEvalDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'], kwargs['modalities'])

        self.modalities = kwargs['modalities']

        self.scene_ids = {}
        n = 0
        new_annotation = []
        for ann in self.annotation:
            try:
                img_id = ann["scene_id"]
                if img_id not in self.scene_ids.keys():
                    self.scene_ids[img_id] = n
                    n += 1
                new_annotation.append(ann)
            except:
                pass
        self.annotation = new_annotation
        
        for modality in self.modalities:
            if 'pc' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                self.pc_feat_root = self.pc_root + '/voxelized_features_sam_nonzero_preprocess/'
                self.voxel_root = self.pc_root + '/voxelized_voxels_sam_nonzero_preprocess/'
                # pc_root = '/nas-ssd2/shoubin/datasets/scannet_feat/'
                # self.pc_feat_root = pc_root + '/voxelized_features_sam_nonzero_preprocess/'
                # self.voxel_root = pc_root + '/voxelized_voxels_sam_nonzero_preprocess/'
                self.annotation = [
                    ann for ann in self.annotation if os.path.exists(os.path.join(self.pc_feat_root, str(ann["scene_id"]) + ".pt"))
                ]
            if 'video' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
                setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())

            if 'frame' in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])

            if 'depth' in modality:
                # todo
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
    
    def get_existing_video_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.video_root)]

    def get_video_path(self, ann):
        return os.path.join(self.video_root, f'{ann["scene_id"]}.mp4')
    
    def get_frame_path(self, ann):
        return os.path.join(self.frame_root, f'{ann["scene_id"]}/')
    
    def get_depth_path(self, ann):
        return os.path.join(self.depth_root, f'{ann["scene_id"]}/')

    def __getitem__(self, index):

        ann = copy.deepcopy(self.annotation[index])
        if 'question_id' in ann.keys(): # 3dqa data
            qa_input = self.text_processor(ann['situation']) + '. Question: ' + self.text_processor(ann["question"]) + ' Based on the frames and 3D Model information, answer the question using a single word or phase.'
            qtype = get_qclass(ann['question'])
            question_id = ann['question_id']
            answer = ann["answers"][0]
            question_id = qtype + '_' + str(question_id)
        else: # pre-training data
            # ann['qa_input'] =  'Question: ' + self.text_processor(question) + ' Based on the frames information, answer the question using a single word or phase.'
            qa_input = 'Question: ' + self.text_processor(ann["question"]) + ' Based on 3D Model information, answer the question using a single word or phase.'
            answer = ann["answers"][0]
            answer = self.text_processor(answer)
            question_id = str(ann["scene_id"])

        scene_id = str(ann["scene_id"])
        

        out = {
            "qa_input": qa_input,
            "qa_output": answer,
            "scene_id": self.scene_ids[ann["scene_id"]],
            "question_id": question_id,
        }

        for modality in self.modalities:
            
            if modality == 'pc':
                pc_feat = torch.load(os.path.join(self.pc_feat_root, f"{scene_id}.pt"), map_location="cpu")  # [N, 1408]
                if isinstance(pc_feat, np.ndarray):
                    pc_feat = torch.tensor(pc_feat).float()
                pc = np.load(os.path.join(self.voxel_root, f"{scene_id}.npy"))
                pc = torch.tensor(pc).float().cpu()
                # sample 10000 points: [N, 1408] -> [10000, 1408]
                if pc_feat.shape[0] > 5000:
                    idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
                    pc_feat = pc_feat[idxes]
                    pc = pc[idxes]
                else:
                    pc_feat = torch.cat([pc_feat, torch.zeros(5000 - pc_feat.shape[0], 1408)], dim=0)
                    pc = torch.cat([pc, torch.zeros(5000 - pc.shape[0], 3)], dim=0)

                out["pc_feat"] = pc_feat
                out['pc'] = pc
            if modality == 'video':
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                rgb, indices, fps = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"])
                out['rgb'] = rgb.to(torch.float32)
            
            if modality == 'frame':
                indices = None
                clip = None
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                frms, indices = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices)
                rgb = frms.permute(1, 0, 2, 3)
                assert len(rgb) == getattr(self, f"{modality}_processor").n_frms
                out['rgb'] = rgb

            if modality == 'depth':
                assert indices is not None
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                depth, _ = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices, type='depth')
                out['depth'] = depth

        return out

    def __len__(self):
        return len(self.annotation)