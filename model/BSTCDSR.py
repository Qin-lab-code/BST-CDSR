import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GCNLayer  # 保留原有依赖，防止其他地方引用


class Discriminator(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        for m in self.modules():
            self._weights_init(m)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        score = self.f_k(node, S)
        if s_bias is not None:
            score += s_bias
        return score


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs: [B, L, D]
        x = inputs.transpose(-1, -2)  # [B, D, L]
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = x.transpose(-1, -2)  # [B, L, D]
        x = x + inputs
        return x



class ATTENTION(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.emb_dropout = nn.Dropout(p=opt["dropout"])
        self.pos_emb = nn.Embedding(opt["maxlen"], opt["hidden_units"], padding_idx=0)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(opt["hidden_units"], eps=1e-8)

        for _ in range(opt["num_blocks"]):
            self.attention_layernorms.append(
                nn.LayerNorm(opt["hidden_units"], eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(
                    opt["hidden_units"], opt["num_heads"], opt["dropout"]
                )
            )
            self.forward_layernorms.append(
                nn.LayerNorm(opt["hidden_units"], eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(opt["hidden_units"], opt["dropout"])
            )

    def forward(self, seqs_data, seqs, position):
        device = seqs.device

        seqs = seqs + self.pos_emb(position)
        seqs = self.emb_dropout(seqs)

        pad_mask = (seqs_data == self.opt["itemnum"] - 1).to(device)  # [B, L]
        seqs = seqs * (~pad_mask).unsqueeze(-1)

        L = seqs.shape[1]
        attn_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

        for i in range(len(self.attention_layers)):
            # self-attention
            x = seqs.transpose(0, 1)  # [L, B, D]
            Q = self.attention_layernorms[i](x)
            attn_out, _ = self.attention_layers[i](Q, x, x, attn_mask=attn_mask)
            x = Q + attn_out
            x = x.transpose(0, 1)  # [B, L, D]

            # FFN
            x = self.forward_layernorms[i](x)
            x = self.forward_layers[i](x)
            x = x * (~pad_mask).unsqueeze(-1)
            seqs = x

        log_feats = self.last_layernorm(seqs)
        return log_feats


class ContinuousGRU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        in_dim = hidden_dim + 1

        self.lin_z = nn.Linear(in_dim, hidden_dim)
        self.lin_r = nn.Linear(in_dim, hidden_dim)
        self.lin_h = nn.Linear(in_dim, hidden_dim)

    def forward(self, h, dt_norm):
        inp = torch.cat([h, dt_norm], dim=-1)  # [B, D+1]

        z = torch.sigmoid(self.lin_z(inp))     # [B, D]
        r = torch.sigmoid(self.lin_r(inp))     # [B, D]

        h_tilde_inp = torch.cat([r * h, dt_norm], dim=-1)  # [B, D+1]

        u_hat = torch.tanh(self.lin_h(h_tilde_inp))        # [B, D]

        dh = (1.0 - z) * (u_hat - h)
        return dh


class LongShortODEEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.hidden_units = opt["hidden_units"]
        self.pad_idx = opt["itemnum"] - 1  # padding id

        self.cont_ode_u = ContinuousGRU(self.hidden_units)
        self.cont_ode_s = ContinuousGRU(self.hidden_units)

        self.gru_u = nn.GRUCell(self.hidden_units, self.hidden_units)
        self.gru_s = nn.GRUCell(self.hidden_units, self.hidden_units)

        self.alpha_u = nn.Parameter(torch.ones(self.hidden_units))
        self.alpha_s = nn.Parameter(torch.ones(self.hidden_units))

        self.fuse_gate = nn.Linear(self.hidden_units * 3, self.hidden_units)

        max_interval = float(opt.get("max_interval", 1.0))
        self.max_interval = max_interval if max_interval > 0 else 1.0

    def forward(self, seq_ids, item_emb, interval_time, time_emb):
        B, L, D = item_emb.size()
        device = item_emb.device

        u = torch.zeros(B, D, device=device)
        s = torch.zeros(B, D, device=device)

        outputs = []

        u_hist = []
        s_hist = []

        for t in range(L):
            mask_t = (seq_ids[:, t] != self.pad_idx).float().unsqueeze(-1)  # [B,1]
            mask_bool = mask_t.bool()

            dt_raw = interval_time[:, t].float()  # [B]
            dt_norm = torch.log2(dt_raw + 2.0)  # [B]
            dt_norm = dt_norm / self.max_interval
            dt_norm = dt_norm.unsqueeze(-1)  # [B,1]
            inp = item_emb[:, t, :]  # [B,D]

            if t == 0:
                base = item_emb[:, 0, :]  # [B,D]
                u = base * mask_t
                s = base * mask_t
            else:
                du = self.cont_ode_u(u, dt_norm)  # [B,D]
                ds = self.cont_ode_s(s, dt_norm)  # [B,D]

                u = u + dt_norm * du
                s = s + dt_norm * ds

                u_candidate = self.gru_u(inp, u)
                s_candidate = self.gru_s(inp, s)

                u = torch.where(mask_bool, u_candidate, u)
                s = torch.where(mask_bool, s_candidate, s)

            u_hist.append(u.unsqueeze(1))  # [B,1,D]
            s_hist.append(s.unsqueeze(1))  # [B,1,D]

            t = time_emb[:, t, :]
            fuse_inp = torch.cat([u, s, t], dim=-1)  # [B,3D]
            g = torch.sigmoid(self.fuse_gate(fuse_inp))  # [B,D]
            h = g * s + (1.0 - g) * u  # [B,D]

            h = h * mask_t

            outputs.append(h.unsqueeze(1))  # [B,1,D]

        outputs = torch.cat(outputs, dim=1)  # [B,L,D]
        u_hist = torch.cat(u_hist, dim=1)  # [B,L,D]
        s_hist = torch.cat(s_hist, dim=1)  # [B,L,D]

        valid_mask = (seq_ids != self.pad_idx).float()  # [B,L]
        K = min(6, L - 1)

        if L > 1:
            u_t = u_hist[:, 1:, :]  # [B,L-1,D]
            u_prev = u_hist[:, :-1, :]  # [B,L-1,D]
            step_mask = valid_mask[:, 1:] * valid_mask[:, :-1]  # [B,L-1]
            step_mask = step_mask.unsqueeze(-1)  # [B,L-1,1]

            dt_raw_seq = interval_time[:, 1:].float()  # [B, L-1]  （和 u_t 对齐）
            dt_w = 1 - torch.log2(dt_raw_seq + 2.0) / self.max_interval
            w = dt_w.unsqueeze(-1)  # [B, L-1, 1]

            u_t_tail = u_t[:, -K:, :]  # [B,K_u,D]
            u_prev_tail = u_prev[:, -K:, :]  # [B,K_u,D]
            step_mask_tail = step_mask[:, -K:, :]  # [B,K_u,1]
            w_tail = w[:, -K:, :]

            u_diff_sq = ((u_t_tail - u_prev_tail) ** 2) * step_mask_tail  # [B,K_u,D]
            u_diff_sq = u_diff_sq * w_tail

            denom_u = step_mask_tail.sum() * D + 1e-8
            u_smooth_loss = u_diff_sq.sum() / denom_u
        else:
            u_smooth_loss = torch.tensor(0.0, device=device)

        s_tail = s_hist[:, -K:, :]  # [B,K,D]
        h_tail = item_emb[:, -K:, :]  # [B,K,D]
        valid_tail = valid_mask[:, -K:]  # [B,K,1]

        loss_sum = 0.0

        for t in range(K):
            v = valid_tail[:, t].bool()  # [B]
            if v.sum() <= 1:
                continue

            s = F.normalize(s_tail[:, t, :], dim=-1)  # [B,D]
            h = F.normalize(h_tail[:, t, :], dim=-1)  # [B,D]

            s = s[v]  # [Bv, D]
            h = h[v]  # [Bv, D]

            logits = (s @ h.t()) / 0.1  # [Bv, Bv]
            labels = torch.arange(logits.size(0), device=logits.device)
            loss_sum = loss_sum + F.cross_entropy(logits, labels)

        s_align_loss = loss_sum

        return outputs, u_smooth_loss, s_align_loss


class Adapter(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.semantic_dim = opt.get("semantic_dim", 0)
        self.semantic_adapter = nn.Sequential(
            nn.Linear(self.semantic_dim, opt["hidden_units"]),
            nn.ReLU(),
            nn.Dropout(opt.get("dropout", 0.2)),
            nn.Linear(opt["hidden_units"], opt["hidden_units"])
        )

    def forward(self, x):
        x = self.semantic_adapter(x)
        return x


class UserTemporalPatternEncoder(nn.Module):
    def __init__(self, opt, d_model=64):
        super().__init__()
        self.opt = opt
        self.domain_emb = nn.Embedding(3, d_model, padding_idx=0)

        self.interval_time_emb = nn.Embedding(
            int(opt["interval_scale"] * opt["max_interval"]) + 2,
            d_model,
            padding_idx=0
        )

        num_heads = opt.get("time_num_heads", 4)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=opt.get("dropout", 0.2),
            batch_first=True,
        )

        self.max_len = opt["maxlen"]
        self.pos_emb = nn.Embedding(self.max_len, d_model, padding_idx=0)

        self.attn_layernorm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ffn_layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(opt.get("dropout", 0.2))

    def forward(self, domain_seq, dt_seq, pos):
        d_emb = self.domain_emb(domain_seq)  # [B, L, D]

        interval = torch.log2(dt_seq.float() + 2)
        interval_index = torch.floor(self.opt["interval_scale"] * interval).long()
        dt_emb = self.interval_time_emb(interval_index)
        x = d_emb + dt_emb  # [B, L, D]

        pos_emb = self.pos_emb(pos)
        x = x + pos_emb

        pad_mask = (domain_seq == 0)
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=pad_mask
        )  # [B, L, D]
        x = self.attn_layernorm(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)  # [B, L, D]
        x = self.ffn_layernorm(x + self.dropout(ffn_out))

        u = x[:, -1]
        return u


class Weight(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        x = self.mlp(x)
        return torch.sigmoid(x)

class BSTCDSR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.item_emb_X = nn.Embedding(
            opt["itemnum"], opt["hidden_units"], padding_idx=opt["itemnum"] - 1
        )
        self.item_emb_Y = nn.Embedding(
            opt["itemnum"], opt["hidden_units"], padding_idx=opt["itemnum"] - 1
        )
        self.item_emb = nn.Embedding(
            opt["itemnum"], opt["hidden_units"], padding_idx=opt["itemnum"] - 1
        )

        self.time_emb = nn.Embedding(
            opt["time_num"], opt["hidden_units"], padding_idx=0
        )

        print("max interval:", int(opt["interval_scale"] * opt["max_interval"]))

        self.lin_X = nn.Linear(opt["hidden_units"], opt["source_item_num"])
        self.lin_Y = nn.Linear(opt["hidden_units"], opt["target_item_num"])
        self.lin_PAD = nn.Linear(opt["hidden_units"], 1)

        self.encoder = ATTENTION(opt)
        self.encoder_X = ATTENTION(opt)
        self.encoder_Y = ATTENTION(opt)

        self.ode_encoder_global = LongShortODEEncoder(opt)
        self.ode_encoder_x = LongShortODEEncoder(opt)
        self.ode_encoder_y = LongShortODEEncoder(opt)

        self.adapter = Adapter(opt)
        self.adapter_x = Adapter(opt)
        self.adapter_y = Adapter(opt)

        self.tpe = UserTemporalPatternEncoder(opt, opt["hidden_units"])

        hidden = 64
        self.time_a = nn.Linear(opt["hidden_units"] * 2, hidden)
        self.time_b = nn.Linear(opt["hidden_units"] * 2, hidden)
        self.relation_a = nn.Linear(opt["hidden_units"] * 2, hidden)
        self.relation_b = nn.Linear(opt["hidden_units"] * 2, hidden)

        self.weight_x = Weight(hidden)
        self.weight_y = Weight(hidden)

    def forward(
            self,
            o_seqs,
            x_seqs,
            y_seqs,
            position,
            x_position,
            y_position,
            time,
            x_time,
            y_time,
            x_interval_time,
            y_interval_time,
            interval_time,
    ):
        seqs = self.item_emb(o_seqs)  # [B,L,D]
        seqs = seqs * (self.item_emb.embedding_dim ** 0.5)
        seqs = seqs + self.time_emb(time)
        seqs_fea = self.encoder(o_seqs, seqs, position)

        x_emb = self.item_emb_X(x_seqs)
        x_emb = x_emb * (self.item_emb.embedding_dim ** 0.5)
        x_emb = x_emb + self.time_emb(x_time)
        x_seqs_fea = self.encoder_X(x_seqs, x_emb, x_position)

        y_emb = self.item_emb_Y(y_seqs)
        y_emb = y_emb * (self.item_emb.embedding_dim ** 0.5)
        y_emb = y_emb + self.time_emb(y_time)
        y_seqs_fea = self.encoder_Y(y_seqs, y_emb, y_position)

        seqs_fea, u_loss_m, s_loss_m = self.ode_encoder_global(
            seq_ids=o_seqs,
            item_emb=seqs_fea,
            interval_time=interval_time,
            time_emb=self.time_emb(time)
        )

        x_seqs_fea, u_loss_x, s_loss_x = self.ode_encoder_x(
            seq_ids=x_seqs,
            item_emb=x_seqs_fea,
            interval_time=x_interval_time,
            time_emb=self.time_emb(x_time)
        )

        y_seqs_fea, u_loss_y, s_loss_y = self.ode_encoder_y(
            seq_ids=y_seqs,
            item_emb=y_seqs_fea,
            interval_time=y_interval_time,
            time_emb=self.time_emb(y_time)
        )

        u_loss = (u_loss_m + u_loss_x + u_loss_y) / 3
        s_loss = (s_loss_m + s_loss_x + s_loss_y) / 3

        return seqs_fea, x_seqs_fea, y_seqs_fea, u_loss, s_loss
