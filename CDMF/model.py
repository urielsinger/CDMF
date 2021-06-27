from itertools import product

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class CDMF(nn.Module):
    def __init__(self, n_items, n_users=None, n_features=64, emb_dim=128, tau=0.01, per_user=False):
        """
        Implementation of the CDMF model
        Args:
            n_items: number of items
            n_users: number of users - used only if per_user==True
            emb_dim: item embedding size
            n_features: number of features per item interaction
            tau: used for activation function: max(x,tau)
            per_user: bool - True if learn different behavior per user
        """
        super(CDMF, self).__init__()
        self.n_features = n_features
        self.per_user = per_user

        self.item_embedding = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)

        if per_user:
            self.w = nn.Embedding(n_users + 1, n_features, padding_idx=0)
        else:
            self.w = nn.Parameter(torch.Tensor(n_features))
            nn.init.normal_(self.w)

        self.h = nn.Threshold(tau, tau)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, users, items, R_ui, mask=None):
        """
        Args:
            users: A tensor of size [num_seq] representing the user_id of each sequence in R_ui
            items: A tensor of size [num_seq] representing the item_id of each sequence in R_ui
            R_ui: A tensor of size [num_seq X seq_len X d] where for each sequence i we have all occurrences
                  of users[i] with items[i] (maximum of seq_len), where each occurrence is represented by
                  d features
            mask: optional - for padding sequences
        Returns:
            r: A tensor of size [num_seq] where r[i] represents the score of interaction between users[i] and items[i]
        """
        if mask is None:
            mask = torch.ones(R_ui.size()[:-1], device=self.device).bool()

        if self.per_user:
            w = self.w(users).unsqueeze(1)
        else:
            w = self.w.view(*list([1] * (R_ui.dim() - 1) + [-1]))
        Z = (w * R_ui).sum(-1)
        W = self.h(Z)
        W[~mask] = 0

        W_alpha = (W ** self.alpha).sum(dim=-1) ** (1 / self.alpha)
        R_beta = mask.float().sum(dim=-1) ** self.beta
        w_tag = W_alpha * R_beta
        w = w_tag ** self.gamma

        q = self.item_embedding(items)
        p = torch.zeros_like(q)
        for user in users.unique():
            user_mask = users == user
            p[user_mask] = (w[user_mask].unsqueeze(-1) * q[user_mask]).sum(0) / w[user_mask].sum(0)

        r = (p * q).sum(-1)

        return r


class CDMFModule(pl.LightningModule):
    def __init__(self, n_items, n_users=None, n_features=64, emb_dim=128, tau=0.01, per_user=False, learning_rate=1e-3, weight_decay=0):
        """
        pytorch_lightning module handling the CDMF model
        Args:
            n_items: number of items
            n_users: number of users - used only if per_user==True
            emb_dim: item embedding size
            n_features: number of features per item interaction
            tau: used for activation function: max(x,tau)
            per_user: bool - True if learn different behavior per user
        """
        super(CDMFModule, self).__init__()
        self.model = CDMF(n_items=n_items, n_users=n_users, n_features=n_features, emb_dim=emb_dim, tau=tau, per_user=per_user)
        self.criterion = nn.BCEWithLogitsLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def BPR_loss(self, logits, labels, group=None):
        pair_indices = np.array(list(product(range(len(labels)), repeat=2)))
        pairs_labels = torch.stack([labels[pair_indices[:, i]] for i in range(pair_indices.shape[1])]).T
        pairs_logits = torch.stack([logits[pair_indices[:, i]] for i in range(pair_indices.shape[1])]).T
        if group is not None:
            pairs_group = torch.stack([group[pair_indices[:, i]] for i in range(pair_indices.shape[1])]).T
            same_group_mask = pairs_group[:, 0] == pairs_group[:, 1]
            pairs_labels = pairs_labels[same_group_mask]
            pairs_logits = pairs_logits[same_group_mask]

        diff_labels = pairs_labels[:, 0] - pairs_labels[:, 1]
        diff_logits = pairs_logits[:, 0] - pairs_logits[:, 1]

        mask = (diff_labels > 0) & (~torch.isinf(diff_labels))

        new_logits = diff_logits[mask]
        new_labels = torch.ones_like(new_logits)
        loss = self.criterion(new_logits, new_labels)

        return loss

    def forward(self, users, items, R_ui, mask=None):
        return self.model(users, items, R_ui, mask)

    def step(self, batch, name):
        users, items, R_ui, labels, mask = batch['user'], batch['item'], batch['features'], batch['labels'], batch['mask']
        logits = self(users, items, R_ui, mask=mask)
        loss = self.BPR_loss(logits, labels, group=users)
        self.log(f'{name}/loss', loss)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='train')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='val')

    def test_step(self, batch: dict, batch_idx: int):
        return self.step(batch, name='test')

    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return opt
