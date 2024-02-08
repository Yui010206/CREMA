"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import torch
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
import random

class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )

ANS_MAPPING = {0:'A',1:'B',2:'C',3:'D',4:'E'}

# NextQA
class MCVideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, modality_type):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, modality_type)

    def _load_auxiliary_mappings(self):
        pass
    
    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        
        result, flow_flag = None, False
        out = {}

        while result is None:
            ann = self.annotation[index]
            qid = ann['qid'] 
            q = ann['question']

            if 'start' in ann:
                start, end = float(ann['start']), float(ann['end'])
                clip = [start, end]
            else:
                clip = None  

            # for QA
            prompt = 'Question: ' + q
            for j in range(ann['num_option']):
                a = ann['a{}'.format(j)]
                prompt += ' Option {}: '.format(ANS_MAPPING[j])
                prompt += a
            qa_prompt = prompt + ' Considering the information presented in the frame, select the correct answer from the options.'
            # loc_prompt = 'Question: ' + q +  ' ' + hints + ' Does the information within the frame provide the necessary details to accurately answer the given question?'                
            answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
            duration = 1

            # print(self.modality_type)
            indices = None
            
            if 'rgb' in self.modality_type:
                vpath = os.path.join(self.vis_root, str(ann['video']))
                frms, indices = self.vis_processor(vpath, clip_proposal=clip, indices=indices)
                rgb = frms.permute(1, 0, 2, 3)
                # print(indices)
                assert len(rgb) == self.vis_processor.n_frms
                out['rgb'] = rgb
            
            if 'depth' in self.modality_type:
                depth_root = self.vis_root[:-1] + '_depth/'
                # if 'nas-hdd' in depth_root:
                #     depth_root = depth_root.replace('nas-hdd','nas-ssd2')
                depth_path = os.path.join(depth_root, str(ann['video']))
                depth, indices_ = self.vis_processor(depth_path, clip_proposal=clip, indices=indices, type='depth')
                if indices is not None:
                    assert indices == indices_
                    indices = indices_
                    
                out['depth'] = depth
                
            if 'flow' in self.modality_type:
                flow_root = self.vis_root[:-1] + '_flow/'
                # if 'nas-hdd' in flow_root:
                #     flow_root = flow_root.replace('nas-hdd','nas-ssd2')
                flow_path = os.path.join(flow_root, str(ann['video']))
                try: 
                    flow, indices_ = self.vis_processor(flow_path, clip_proposal=clip, indices=indices, type='flow')
                    if indices is not None:
                        assert indices == indices_
                        indices = indices_
                    out['flow'] = flow
                    result = True
                    flow_flag = False
                except Exception as e:
                    print(f"Error while read flow file idx")
                    print("video is: {}".format(ann['video']))
                    index = random.randint(0, len(self.annotation) - 1)
                    flow_flag = True 
                                    
            if 'norm' in self.modality_type:
                norm_root = self.vis_root[:-1] + '_norm/'
                norm_path = os.path.join(norm_root, str(ann['video']))
                norm, indices_ = self.vis_processor(norm_path, clip_proposal=clip, indices=indices, type='norm')
                if indices is not None:
                    assert indices == indices_
                    indices = indices_
                out['norm'] = norm

            if not flow_flag:
                result = True
            
            out['qa_input'] = qa_prompt
            out['qa_output'] = answers
            out['question_id'] = qid
            out['duration'] = duration
            
        return out