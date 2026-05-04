import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.BSTCDSR import BSTCDSR
import pdb
import numpy as np


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class CDSRTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if opt["model"] == "BSTCDSR":
            self.model = BSTCDSR(opt)
        else:
            print("please select a valid model")
            exit(0)

        self.mi_loss = 0
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = 0.75
        self.contrastive_topk = opt["top_k"]
        self.contrastive_tau = opt.get("contrastive_tau", 0.1)
        self.contrastive_weight = 1e-3

        if opt['cuda']:
            self.model.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.optimizer.zero_grad()

    def get_dot_score(self, A_embedding, B_embedding):
        output = (A_embedding * B_embedding).sum(dim=-1)
        return output

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            x_ground = inputs[8]
            y_ground = inputs[9]
            negative_sample_x = inputs[10]
            negative_sample_y = inputs[11]
            time = inputs[12]
            x_time = inputs[13]
            y_time = inputs[14]
            x_interval_time = inputs[15]
            y_interval_time = inputs[16]
            interval_time = inputs[17]
            m_semantic = inputs[18]
            x_semantic = inputs[19]
            y_semantic = inputs[20]
            x_ground_mask_share = inputs[21]
            y_ground_mask_share = inputs[22]
            share_x_mask = inputs[23]
            share_y_mask = inputs[24]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            x_ground = inputs[8]
            y_ground = inputs[9]
            negative_sample_x = inputs[10]
            negative_sample_y = inputs[11]
            time = inputs[12]
            x_time = inputs[13]
            y_time = inputs[14]
            x_interval_time = inputs[15]
            y_interval_time = inputs[16]
            interval_time = inputs[17]
            m_semantic = inputs[18]
            x_semantic = inputs[19]
            y_semantic = inputs[20]
            x_ground_mask_share = inputs[21]
            y_ground_mask_share = inputs[22]
            share_x_mask = inputs[23]
            share_y_mask = inputs[24]
        return seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, x_ground, y_ground, negative_sample_x, negative_sample_y, time, x_time, y_time, x_interval_time, y_interval_time, interval_time, m_semantic, x_semantic, y_semantic, x_ground_mask_share, y_ground_mask_share, share_x_mask, share_y_mask

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            x_ground = inputs[6]
            y_ground = inputs[7]
            x_ground_mask = inputs[8]
            y_ground_mask = inputs[9]
            time = inputs[10]
            x_time = inputs[11]
            y_time = inputs[12]
            x_interval_time = inputs[13]
            y_interval_time = inputs[14]
            interval_time = inputs[15]
            m_semantic = inputs[16]
            x_semantic = inputs[17]
            y_semantic = inputs[18]
            m_small_s = inputs[19]
            x_small_s = inputs[20]
            y_small_s = inputs[21]
            m_big_s = inputs[22]
            x_big_s = inputs[23]
            y_big_s = inputs[24]
            share_x_mask = inputs[25]
            share_y_mask = inputs[26]
            x_ground_mask_share = inputs[27]
            y_ground_mask_share = inputs[28]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            x_ground = inputs[6]
            y_ground = inputs[7]
            x_ground_mask = inputs[8]
            y_ground_mask = inputs[9]
            time = inputs[10]
            x_time = inputs[11]
            y_time = inputs[12]
            x_interval_time = inputs[13]
            y_interval_time = inputs[14]
            interval_time = inputs[15]
            m_semantic = inputs[16]
            x_semantic = inputs[17]
            y_semantic = inputs[18]
            m_small_s = inputs[19]
            x_small_s = inputs[20]
            y_small_s = inputs[21]
            m_big_s = inputs[22]
            x_big_s = inputs[23]
            y_big_s = inputs[24]
            share_x_mask = inputs[25]
            share_y_mask = inputs[26]
            x_ground_mask_share = inputs[27]
            y_ground_mask_share = inputs[28]
        return seq, x_seq, y_seq, position, x_position, y_position, x_ground, y_ground, x_ground_mask, y_ground_mask, time, x_time, y_time, x_interval_time, y_interval_time, interval_time, m_semantic, x_semantic, y_semantic, m_small_s, x_small_s, y_small_s, m_big_s, x_big_s, y_big_s, share_x_mask, share_y_mask, x_ground_mask_share, y_ground_mask_share

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def train(self, global_step, train_batch):
        self.model.train()
        train_loss = 0
        for batch in train_batch:
            global_step += 1
            loss = self.train_batch(batch)
            train_loss += loss
        return global_step, train_loss

    def _fuse_semantic(self, seqs_fea, x_seqs_fea, y_seqs_fea, m_sem_seq, x_sem_seq, y_sem_seq, seq, x_seq, y_seq):
        seqs_fea = seqs_fea + m_sem_seq
        x_seqs_fea = x_seqs_fea + x_sem_seq
        y_seqs_fea = y_seqs_fea + y_sem_seq
        return seqs_fea, x_seqs_fea, y_seqs_fea

    def info_nce_with_small_and_big(
            self,
            h_orig: torch.Tensor,  # [B, D]
            h_small: torch.Tensor,  # [B, D]
            h_big: torch.Tensor,  # [B, D]
            tau: float = 0.1,
    ) -> torch.Tensor:
        device = h_orig.device
        B, D = h_orig.shape

        h_orig = F.normalize(h_orig, dim=-1)
        h_small = F.normalize(h_small, dim=-1)
        h_big = F.normalize(h_big, dim=-1)

        sim_small = (h_orig * h_small).sum(dim=-1)  # [B]
        sim_big = (h_orig * h_big).sum(dim=-1) # [B]

        sim_matrix = (h_orig @ h_orig.t())
        mask = torch.eye(B, device=device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        k = min(10, B - 1)
        topk_vals, _ = torch.topk(sim_matrix, k=k, dim=1)
        sim_neg = topk_vals.mean(dim=1)

        diff_1 = (sim_small - sim_big) / tau
        loss_order_1 = -F.logsigmoid(diff_1).mean()

        diff_2 = (sim_big - sim_neg) / tau
        loss_order_2 = -F.logsigmoid(diff_2).mean()

        return loss_order_1 + loss_order_2

    def train_batch(self, batch):
        seq, x_seq, y_seq, position, x_position, y_position, x_ground, y_ground, x_ground_mask, y_ground_mask, time, x_time, y_time, x_interval_time, y_interval_time, interval_time, m_semantic, x_semantic, y_semantic, m_small_s, x_small_s, y_small_s, m_big_s, x_big_s, y_big_s, share_x_mask, share_y_mask, x_ground_mask_share, y_ground_mask_share = self.unpack_batch(
            batch)
        seqs_fea, x_seqs_fea, y_seqs_fea, u_loss, s_loss = self.model(seq, x_seq, y_seq, position, x_position, y_position, time, x_time,
                                                      y_time, x_interval_time, y_interval_time, interval_time)

        B, L, D = seqs_fea.size()
        m_sem_proj = self.model.adapter(m_semantic)  # [B, D]
        x_sem_proj = self.model.adapter_x(x_semantic)
        y_sem_proj = self.model.adapter_y(y_semantic)

        m_sem_seq = m_sem_proj.unsqueeze(1).expand(-1, L, -1)  # [B,L,D]
        x_sem_seq = x_sem_proj.unsqueeze(1).expand(-1, L, -1)
        y_sem_seq = y_sem_proj.unsqueeze(1).expand(-1, L, -1)

        fused_fea, x_fused_fea, y_fused_fea = self._fuse_semantic(
            seqs_fea, x_seqs_fea, y_seqs_fea, m_sem_seq, x_sem_seq, y_sem_seq, seq, x_seq, y_seq
        )

        m_small_proj = self.model.adapter(m_small_s)  # [B, D]
        x_small_proj = self.model.adapter_x(x_small_s)
        y_small_proj = self.model.adapter_y(y_small_s)

        m_big_proj = self.model.adapter(m_big_s)  # [B, D]
        x_big_proj = self.model.adapter_x(x_big_s)
        y_big_proj = self.model.adapter_y(y_big_s)

        tau = 0.5
        m_loss_cf = self.info_nce_with_small_and_big(
            h_orig=m_sem_proj,
            h_small=m_small_proj,
            h_big=m_big_proj,
            tau=tau,
        )
        a_loss_cf = self.info_nce_with_small_and_big(
            h_orig=x_sem_proj,
            h_small=x_small_proj,
            h_big=x_big_proj,
            tau=tau,
        )
        b_loss_cf = self.info_nce_with_small_and_big(
            h_orig=y_sem_proj,
            h_small=y_small_proj,
            h_big=y_big_proj,
            tau=tau,
        )
        loss_cf = m_loss_cf + a_loss_cf + b_loss_cf

        domain_seq = torch.zeros_like(seq)  # [B,L]
        domain_seq[share_x_mask > 0] = 1
        domain_seq[share_y_mask > 0] = 2
        z_time = self.model.tpe(domain_seq, interval_time, position)  # [B, 1]
        x_domain_seq = torch.zeros_like(seq)
        x_domain_seq[x_position > 0] = 1
        x_z_time = self.model.tpe(x_domain_seq, x_interval_time, x_position)  # [B, 1]
        y_domain_seq = torch.zeros_like(seq)
        y_domain_seq[y_position > 0] = 2
        y_z_time = self.model.tpe(y_domain_seq, y_interval_time, y_position)  # [B, 1]
        w_t_a = self.model.time_a(torch.cat([x_z_time, z_time], dim=-1))
        w_t_b = self.model.time_b(torch.cat([y_z_time, z_time], dim=-1))

        x_mask = x_ground_mask_share / x_ground_mask_share.sum(dim=1, keepdim=True)
        real_x_fea = (fused_fea * x_mask.unsqueeze(-1)).sum(dim=1)
        y_mask = y_ground_mask_share / y_ground_mask_share.sum(dim=1, keepdim=True)
        real_y_fea = (fused_fea * y_mask.unsqueeze(-1)).sum(dim=1)
        r_x_fea = real_x_fea
        r_y_fea = real_y_fea

        w_r_a = self.model.relation_a(torch.cat([r_x_fea, r_y_fea], dim=-1))
        w_r_b = self.model.relation_b(torch.cat([r_y_fea, r_x_fea], dim=-1))

        w_A = self.model.weight_x(torch.cat([w_t_a, w_r_a], dim=-1))
        w_B = self.model.weight_y(torch.cat([w_t_b, w_r_b], dim=-1))

        used = 10
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]

        fused_fea = real_x_fea.unsqueeze(1).expand(-1, used, -1)
        mix_fea = w_A.view(-1, 1, 1) * x_fused_fea[:, -used:] + (1 - w_A.view(-1, 1, 1)) * fused_fea[:, -used:]
        specific_x_result = self.model.lin_X(mix_fea)
        # specific_x_result = self.model.lin_X(fused_fea[:,-used:] + x_fused_fea[:, -used:])  # b * seq * X_num
        specific_x_pad_result = self.model.lin_PAD(x_fused_fea[:, -used:])  # b * seq * 1
        specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)

        fused_fea = real_y_fea.unsqueeze(1).expand(-1, used, -1)
        mix_fea = w_B.view(-1, 1, 1) * y_fused_fea[:, -used:] + (1 - w_B.view(-1, 1, 1)) * fused_fea[:, -used:]
        specific_y_result = self.model.lin_Y(mix_fea)
        # specific_y_result = self.model.lin_Y(fused_fea[:,-used:] + y_fused_fea[:, -used:])  # b * seq * Y_num
        specific_y_pad_result = self.model.lin_PAD(y_fused_fea[:, -used:])  # b * seq * 1
        specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

        x_loss = self.CS_criterion(
            specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            x_ground.reshape(-1))  # b * seq
        y_loss = self.CS_criterion(
            specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            y_ground.reshape(-1))  # b * seq

        x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
        y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
        main_loss = x_loss + y_loss

        loss = main_loss + self.opt["ode_loss_weight"] * (u_loss + s_loss) + self.opt["sem_loss_weight"] * loss_cf
        self.mi_loss += main_loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def test_batch(self, batch):
        seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, x_ground, y_ground, negative_sample_x, negative_sample_y, time, x_time, y_time, x_interval_time, y_interval_time, interval_time, m_semantic, x_semantic, y_semantic, x_ground_mask_share, y_ground_mask_share, share_x_mask, share_y_mask = self.unpack_batch_predict(
            batch)
        seqs_fea, x_seqs_fea, y_seqs_fea, _, _ = self.model(seq, x_seq, y_seq, position, x_position, y_position, time, x_time,
                                                      y_time, x_interval_time, y_interval_time, interval_time)

        B, L, D = seqs_fea.size()
        m_sem_proj = self.model.adapter(m_semantic)  # [B, D]
        x_sem_proj = self.model.adapter_x(x_semantic)
        y_sem_proj = self.model.adapter_y(y_semantic)

        m_sem_seq = m_sem_proj.unsqueeze(1).expand(-1, L, -1)  # [B,L,D]
        x_sem_seq = x_sem_proj.unsqueeze(1).expand(-1, L, -1)
        y_sem_seq = y_sem_proj.unsqueeze(1).expand(-1, L, -1)

        seqs_fea, x_seqs_fea, y_seqs_fea = self._fuse_semantic(
            seqs_fea, x_seqs_fea, y_seqs_fea, m_sem_seq, x_sem_seq, y_sem_seq, seq, x_seq, y_seq
        )

        domain_seq = torch.zeros_like(seq)  # [B,L]
        domain_seq[share_x_mask > 0] = 1
        domain_seq[share_y_mask > 0] = 2
        z_time = self.model.tpe(domain_seq, interval_time, position)  # [B, 1]
        x_domain_seq = torch.zeros_like(seq)
        x_domain_seq[x_position > 0] = 1
        x_z_time = self.model.tpe(x_domain_seq, x_interval_time, x_position)  # [B, 1]
        y_domain_seq = torch.zeros_like(seq)
        y_domain_seq[y_position > 0] = 2
        y_z_time = self.model.tpe(y_domain_seq, y_interval_time, y_position)  # [B, 1]
        w_t_a = self.model.time_a(torch.cat([x_z_time, z_time], dim=-1))
        w_t_b = self.model.time_b(torch.cat([y_z_time, z_time], dim=-1))

        x_mask = x_ground_mask_share / x_ground_mask_share.sum(dim=1, keepdim=True)
        real_x_fea = (seqs_fea * x_mask.unsqueeze(-1)).sum(dim=1)
        y_mask = y_ground_mask_share / y_ground_mask_share.sum(dim=1, keepdim=True)
        real_y_fea = (seqs_fea * y_mask.unsqueeze(-1)).sum(dim=1)
        r_x_fea = real_x_fea
        r_y_fea = real_y_fea

        w_r_a = self.model.relation_a(torch.cat([r_x_fea, r_y_fea], dim=-1))
        w_r_b = self.model.relation_b(torch.cat([r_y_fea, r_x_fea], dim=-1))

        w_A = self.model.weight_x(torch.cat([w_t_a, w_r_a], dim=-1))
        w_B = self.model.weight_y(torch.cat([w_t_b, w_r_b], dim=-1))

        X_pred = []
        Y_pred = []
        for id, fea in enumerate(seqs_fea):
            # share_fea = seqs_fea[id, -1]
            share_fea = real_x_fea[id]
            specific_fea = x_seqs_fea[id, X_last[id]]
            final_fea = specific_fea * w_A[id].item() + share_fea * (1 - w_A[id].item())
            X_score = self.model.lin_X(final_fea).squeeze(0)
            # X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0)
            cur = X_score[x_ground[id]]
            score_larger = (X_score[negative_sample_x[id]] > (cur + 0.00001)).data.cpu().numpy()
            true_item_rank = np.sum(score_larger) + 1
            X_pred.append(true_item_rank)

        for id, fea in enumerate(seqs_fea):  # b * s * f
            # share_fea = seqs_fea[id, -1]
            share_fea = real_y_fea[id]
            specific_fea = y_seqs_fea[id, Y_last[id]]
            final_fea = specific_fea * w_B[id].item() + share_fea * (1 - w_B[id].item())
            Y_score = self.model.lin_Y(final_fea).squeeze(0)
            # Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
            cur = Y_score[y_ground[id]]
            score_larger = (Y_score[negative_sample_y[id]] > (cur + 0.00001)).data.cpu().numpy()
            true_item_rank = np.sum(score_larger) + 1
            Y_pred.append(true_item_rank)

        return X_pred, Y_pred

