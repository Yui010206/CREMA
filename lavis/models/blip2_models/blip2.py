"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from transformers import BertTokenizer

import math
from torch import Tensor
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parameter import Parameter

class _LoRALayer_multimodal(nn.Module):
    def __init__(self, w, w_a, w_b, dropout, modulars):
        super().__init__()
        self.w = w

        # multimodal lora
        if 'rgb' in modulars:
            self.w_a_rgb = w_a['rgb']
            self.w_b_rgb = w_b['rgb']
            self.dropout_rgb = dropout['rgb']
        if 'depth' in modulars:
            self.w_a_depth = w_a['depth']
            self.w_b_depth = w_b['depth']
            self.dropout_depth = dropout['depth']
        if 'flow' in modulars:
            self.w_a_flow = w_a['flow']
            self.w_b_flow = w_b['flow']
            self.dropout_flow = dropout['flow']
        if 'norm' in modulars:
            self.w_a_norm = w_a['norm']
            self.w_b_norm = w_b['norm']
            self.dropout_norm = dropout['norm']
        if 'audio' in modulars:
            self.w_a_audio = w_a['audio']
            self.w_b_audio = w_b['audio']
            self.dropout_audio  = dropout['audio']
        if 'pc' in modulars:
            self.w_a_pc = w_a['pc']
            self.w_b_pc = w_b['pc']
            self.dropout_pc = dropout['pc']

    def forward(self, x, modular='rgb'):
        w_b = getattr(self, f'w_b_{modular}')
        w_a = getattr(self, f'w_a_{modular}')
        dropout = getattr(self, f'dropout_{modular}')
        
        if 'skip' in modular:
            x = self.w(x)
        else:
            x = self.w(x) + w_b(w_a(dropout(x)))

        return x

# multitask Qformer
class LoRA_Multimodal_QFormer(nn.Module):
    """Applies low-rank adaptation to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, qformer, modulars,
                 r: int, 
                 lora_layer=None,
                 cross_attention_freq=2,
                 lora_dropout=0.1):
        super(LoRA_Multimodal_QFormer, self).__init__()

        
        assert r > 0
        base_vit_dim = qformer.bert.encoder.layer[0].attention.self.query.out_features
        dim = base_vit_dim

        self.modulars = modulars
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(qformer.bert.encoder.layer)))
            
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # self.w_As_cross = [] 
        # self.w_Bs_cross = []

        # lets freeze first / yui: nothing changes here for test
        for param in qformer.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        # yui: yeah, let's do the second surgery for the multimodal adapter here ;-)
        
        for t_layer_i, blk in enumerate(qformer.bert.encoder.layer):
                # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            # replace self-attention
            w_q_linear = blk.attention.self.query
            w_v_linear = blk.attention.self.value
            w_a_linear_q, w_a_linear_v = {}, {}
            w_b_linear_q, w_b_linear_v = {}, {}
            dropout_q, dropout_v = {}, {}
            
            for m in self.modulars:
                # if m == 'rgb':
                #     continue
                w_a_linear_q[m] = nn.Linear(dim, r, bias=False)
                w_a_linear_v[m] = nn.Linear(dim, r, bias=False)
                w_b_linear_q[m] = nn.Linear(r, dim, bias=False)
                w_b_linear_v[m] = nn.Linear(r, dim, bias=False)
                if lora_dropout > 0.0:
                    dropout_q[m] = nn.Dropout(p=lora_dropout)
                    dropout_v[m] = nn.Dropout(p=lora_dropout)
                else:
                    dropout_q[m] = nn.Identity()
                    dropout_v[m] = nn.Identity()
                                
                self.w_As.append(w_a_linear_q[m])
                self.w_Bs.append(w_b_linear_q[m])
                self.w_As.append(w_a_linear_v[m])
                self.w_Bs.append(w_b_linear_v[m])
                
            blk.attention.self.query = _LoRALayer_multimodal(w_q_linear, w_a_linear_q, w_b_linear_q, dropout_q, modulars)
            blk.attention.self.value = _LoRALayer_multimodal(w_v_linear, w_a_linear_v, w_b_linear_v, dropout_v, modulars)

        self.reset_parameters()
        self.lora_qformer = qformer

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

        # for w_A in self.w_As_cross:
        #         nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        # for w_B in self.w_Bs_cross:
        #     nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_qformer(x)

class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )                 
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    @classmethod
    def init_Multimodal_Qformer(cls, num_query_token, vision_width, 
            modulars, r=64, lora_layer=None, lora_dropout=0.1):

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        ) 

        LoRA_Multimodal_QFormer(Qformer, modulars, 
            r=r,
            lora_layer=lora_layer,
            cross_attention_freq=encoder_config.cross_attention_freq,
            lora_dropout=lora_dropout
            )
        
        return Qformer, encoder_config
    
    @classmethod
    def init_ln(cls, num_features, load_ln_path=False, load_ln_type=""):
        ln = LayerNorm(num_features)
        if load_ln_path and load_ln_type:
            url_or_filename=load_ln_path
            logging.info(f"Loading pretrained layer norm weights from {url_or_filename} of type {load_ln_type}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            if load_ln_type:
                load_ln_type = f"{load_ln_type}_ln" if "vision" not in load_ln_type else "ln_vision"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if load_ln_type in k:
                    loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            ln.load_state_dict(loaded_state_dict, strict=False)
        
        return ln

    @classmethod
    def init_audio_encoder(self, 
            model_name, cached_audio, load_ln_path=False, load_ln_type=""):
        assert model_name in [
            'beats'
        ], "audio model must be in [beats]"

        # load_ln_path = kwargs['load_ln_path']
        # del kwargs['load_ln_path']
        # load_ln_type=kwargs['load_ln_type']
        # del kwargs['load_ln_type']
        kwargs = {}
        if "beats" in model_name:
            from lavis.models.beats_encoder import BeatsEncoder
            if cached_audio:
                audio_encoder = lambda x: x
                ln_audio = self.init_ln(768, load_ln_path=load_ln_path, load_ln_type=load_ln_type)
            else:
                audio_encoder = BeatsEncoder(**kwargs)

        if not cached_audio:
            ln_audio = self.init_ln(audio_encoder.num_features, load_ln_path=load_ln_path, load_ln_type=load_ln_type)
        self.audio_enc_name = model_name

        return audio_encoder, ln_audio
    
    @classmethod
    def init_TemporalQFormer(cls, num_of_frame):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.query_length = num_of_frame
        Qformer = BertLMHeadModel.from_pretrained(
        "bert-base-uncased", config=encoder_config
        )                 
        query_tokens = nn.Parameter(
            torch.zeros(1, num_of_frame, 1, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_vision_encoder_sevila(
        cls, img_size, drop_path_rate, use_grad_checkpoint, precision, in_chans=3
    ):
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision, in_chans
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        ln_vision2 = LayerNorm(visual_encoder.num_features) 
        return visual_encoder, ln_vision, ln_vision2

    @classmethod
    def init_vision_encoder_only(
        cls, img_size, drop_path_rate, use_grad_checkpoint, precision, in_chans=3
    ):
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision, in_chans
        )
        return visual_encoder

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        #print('state_dict',state_dict.keys())
        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
    
    
    def load_lora(self, lora_ckpt):
        if lora_ckpt=='':
            return
        checkpoint = torch.load(lora_ckpt, map_location="cpu")
        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % lora_ckpt)
        

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
