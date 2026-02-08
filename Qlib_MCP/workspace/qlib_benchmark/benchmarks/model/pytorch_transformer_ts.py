# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 8192,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        use_mixed_pooling=True,  # 是否使用混合池化（Attention + Mean + EMA）
        ema_decay_lambda=0.9,    # EMA 池化的衰减系数
        use_learnable_pe=True,  # 是否使用可学习位置编码
        **kwargs,
    ):
        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        pool_info = "with mixed pooling (Attention+Mean+EMA)" if use_mixed_pooling else "with last-step pooling"
        pe_info = "learnable" if use_learnable_pe else "sinusoidal"
        self.logger.info("Transformer Model (SOTA Enhanced):" 
                        "\nbatch_size : {}" 
                        "\ndevice : {}" 
                        "\npooling : {}" 
                        "\npositional_encoding : {}"
                        "\ninput_norm : LayerNorm".format(
            self.batch_size, self.device, pool_info, pe_info))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(
            d_feat, d_model, nhead, num_layers, dropout, self.device,
            use_mixed_pooling=use_mixed_pooling,
            ema_decay_lambda=ema_decay_lambda,
            use_learnable_pe=use_learnable_pe
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.model.train()

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())  # .float()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class PositionalEncoding(nn.Module):
    """固定正弦位置编码"""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


# class LearnablePositionalEncoding(nn.Module):
#     """可学习的位置编码"""
#     def __init__(self, d_model, max_len=1000):
#         super(LearnablePositionalEncoding, self).__init__()
#         self.pe = nn.Parameter(torch.zeros(max_len, d_model))
#         nn.init.normal_(self.pe, mean=0.0, std=0.02)

#     def forward(self, x):
#         # [T, N, F]
#         return x + self.pe[: x.size(0), :]

class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        nn.init.normal_(pe, mean=0.0, std=0.02)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None,
                 use_mixed_pooling=True, ema_decay_lambda=0.9, use_learnable_pe=False):
        super(Transformer, self).__init__()
        # 改进 1: LayerNorm + Linear（而不是直接 Linear）
        self.input_norm = nn.LayerNorm(d_feat)
        self.feature_layer = nn.Linear(d_feat, d_model)
        
        # 改进 2: 可学习位置编码（可选）
        if use_learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(d_model)
        else:
            self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat
        
        # 混合池化相关参数
        self.use_mixed_pooling = use_mixed_pooling
        if use_mixed_pooling:
            self.ema_decay_lambda = ema_decay_lambda
            # Attention 池化的 query
            self.attn_query = nn.Parameter(torch.empty(d_model))
            self.attn_scale = 1.0 / math.sqrt(d_model)
            # 混合池化的权重（可学习）
            self.pool_mixer = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 32),
                nn.SiLU(),
                nn.Linear(32, 3),  # 3种池化：attention, mean, ema
            )
            # 初始化
            nn.init.normal_(self.attn_query, mean=0.0, std=0.02)

    def _ema_pool_fp32(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        有限窗口 EMA 池化（确定性，显式归一化）
        
        h_seq: (N, T, H) 或 (T, N, H)
        returns: (N, H)
        
        有限窗口归一化：w_t ∝ (1-λ) λ^{T-t}, normalized over finite T
        使用 fp32 计算提高数值稳定性
        """
        # 处理维度：确保是 (N, T, H)
        # 只有当第一个维度小于第二个维度时，才可能是 (T, N, H) 格式需要转置
        if h_seq.dim() == 3 and h_seq.size(0) < h_seq.size(1):  # 可能是 (T, N, H) 且 T < N
            h_seq = h_seq.transpose(0, 1)  # -> (N, T, H)
        
        N, T, H = h_seq.shape
        lam = float(self.ema_decay_lambda)
        
        # 有限窗口权重计算（fp32 提高数值稳定性）
        t_idx = torch.arange(T, device=h_seq.device, dtype=torch.float32)  # 0..T-1
        expn = (T - 1) - t_idx  # T-1..0 (最近的时间步权重更大)
        w_raw = (1.0 - lam) * (lam ** expn)  # (T,)
        w = w_raw / w_raw.sum()  # 归一化：确保权重总和为 1
        
        # fp32 计算提高数值稳定性
        h32 = h_seq.float()
        c32 = (h32 * w.view(1, T, 1)).sum(dim=1)  # (N, H)
        return c32.to(dtype=h_seq.dtype)
    
    def _attention_pool(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        Attention 池化
        
        h_seq: (N, T, H) 或 (T, N, H)
        returns: (N, H)
        """
        # 处理维度
        # 只有当第一个维度小于第二个维度时，才可能是 (T, N, H) 格式需要转置
        if h_seq.dim() == 3 and h_seq.size(0) < h_seq.size(1):  # 可能是 (T, N, H) 且 T < N
            h_seq = h_seq.transpose(0, 1)  # -> (N, T, H)
        
        N, T, H = h_seq.shape
        
        # 计算 attention scores
        scores = torch.einsum("nth,h->nt", h_seq, self.attn_query) * self.attn_scale
        
        # Softmax
        a = F.softmax(scores, dim=1)  # (N, T)
        
        # Weighted sum
        c_attn = torch.einsum("nt,nth->nh", a, h_seq)  # (N, H)
        return c_attn

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        # 改进 1: 先应用 LayerNorm，再 Linear
        src = self.input_norm(src)  # [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # 改进：混合池化（而不是只取最后一步）
        if self.use_mixed_pooling:
            # 转换为 (N, T, H) 格式
            output_nt = output.transpose(0, 1)  # [N, T, d_model]
            
            # 三种池化方式
            c_attn = self._attention_pool(output_nt)  # [N, d_model]
            c_mean = output_nt.mean(dim=1)  # [N, d_model]
            c_ema = self._ema_pool_fp32(output_nt)  # [N, d_model]
            
            # 混合权重（基于 mean 池化的结果）
            mixer_in = F.layer_norm(c_mean, (c_mean.size(-1),))
            g_logits = self.pool_mixer(mixer_in)  # [N, 3]
            g = F.softmax(g_logits, dim=-1)  # [N, 3]
            
            # 混合
            c = g[:, 0:1] * c_attn + g[:, 1:2] * c_mean + g[:, 2:3] * c_ema  # [N, d_model]
            
            # 输出
            output = self.decoder_layer(c)  # [N, 1]
        else:
            # 原始方式：只取最后一步
            output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()