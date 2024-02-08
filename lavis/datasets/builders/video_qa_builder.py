"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset

from lavis.datasets.datasets.rgbd_vqa_datasets import MCVideoQADataset
from lavis.datasets.datasets.music_avqa_datasets import MusicAVQAInstructDataset, MusicAVQADataset

from lavis.datasets.datasets.threedvqa_datasets import ThreeDVQADataset, ThreeDVQAEvalDataset

class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def build(self):
        datasets = super().build()

        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)

        for split in datasets:
            datasets[split]._build_class_labels(ans2label)

        return datasets
    
class MCVideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MCVideoQADataset
    eval_dataset_cls = MCVideoQADataset

    def build(self):
        datasets = super().build()

        for split in datasets:
            datasets[split]._load_auxiliary_mappings()

        return datasets

@registry.register_builder("msrvtt_qa")
class MSRVTTQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa.yaml",
    }


@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa.yaml",
    }

# multi-choice videoqa
# to do update it 
@registry.register_builder("nextqa")
class NextQA3DBuilder(MCVideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nextqa/defaults_qa.yaml",
    }

# @registry.register_builder("perception_test")
# class NextQA3DBuilder(MCVideoQA3DBuilder):
#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/perception/defaults_qa.yaml",
#     }
    
# open-ended QA

@registry.register_builder("musicavqa_mm")
class MusicAVQABuilder(MultiModalDatasetBuilder):
    train_dataset_cls = MusicAVQADataset
    eval_dataset_cls = MusicAVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/music_avqa/defaults_mm_qa.yaml"}

@registry.register_builder("musicavqa_mm_instruct")
class MusicAVQAInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = MusicAVQAInstructDataset
    eval_dataset_cls = MusicAVQAInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/music_avqa/defaults_mm_qa_instruct.yaml"}


@registry.register_builder("sqa3d")
class ThreeDVQABuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ThreeDVQADataset
    eval_dataset_cls = ThreeDVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sqa3d/defaults.yaml"}