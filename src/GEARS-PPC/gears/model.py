import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SGConv

class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]
        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)

class GEARS_Model(torch.nn.Module):
    """
    GEARS model with residual-gated score scaling on perturbation pathway.
    """
    def __init__(self, args):
        super(GEARS_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
        
        # Residual gate allows both attenuation and amplification around 1.0.
        self.score_gate_w = nn.Linear(1, hidden_size)
        self.score_gate_alpha = 0.5
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size, hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes, hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
            
    # def forward(self, data):
    #     """
    #     Forward pass of the model
    #     """
    #     x, pert_idx = data.x, data.pert_idx
    #     pert_scores = data.pert_score if hasattr(data, 'pert_score') else None
        
    #     if self.no_perturb:
    #         out = x.reshape(-1,1)
    #         out = torch.split(torch.flatten(out), self.num_genes)           
    #         return torch.stack(out)
    #     else:
    #         num_graphs = len(data.batch.unique())

    #         ## get base gene embeddings
    #         emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
    #         emb = self.bn_emb(emb)
    #         base_emb = self.emb_trans(emb)        

    #         pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
    #         for idx, layer in enumerate(self.layers_emb_pos):
    #             pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
    #             if idx < len(self.layers_emb_pos) - 1:
    #                 pos_emb = pos_emb.relu()

    #         base_emb = base_emb + 0.2 * pos_emb
    #         base_emb = self.emb_trans_v2(base_emb)
            
    #         ## get perturbation index and embeddings
    #         pert_index = []
    #         for idx, i in enumerate(pert_idx):
    #             for j in i:
    #                 if j != -1:
    #                     pert_index.append([idx, j])
    #         pert_index = torch.tensor(pert_index).T

    #         pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

    #         ## augment global perturbation embedding with GNN
    #         for idx, layer in enumerate(self.sim_layers):
    #             pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
    #             if idx < self.num_layers - 1:
    #                 pert_global_emb = pert_global_emb.relu()

    #         # Score gate is only used on the perturbation pathway.
    #         score_gate_vec = None
    #         if pert_scores is not None:
    #             raw_gate = self.score_gate_w(pert_scores.to(self.args['device']))
    #             score_gate_vec = 1.0 + self.score_gate_alpha * torch.tanh(raw_gate)
            
    #         base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

    #         if pert_index.shape[0] != 0:
    #             # pert_track 的 Key 必须是 Batch Index (样本在batch中的位置 0~31)
    #             pert_track = {}
                
    #             # pert_index[0] 是样本索引 (Sample Index / Batch Index)
    #             # pert_index[1] 是扰动ID (Perturbation ID)
                
    #             for i, sample_idx_tensor in enumerate(pert_index[0]):
    #                 sample_idx = sample_idx_tensor.item() # 必须取 item() 转为整数
    #                 pert_id = pert_index[1][i].item()     # 具体的扰动 ID (如 1509)
                    
    #                 # 1. 获取当前扰动的向量
    #                 current_pert_vec = pert_global_emb[pert_id]
                    
    #                 # Apply residual gating using the matched sample score.
    #                 if score_gate_vec is not None:
    #                     current_pert_vec = current_pert_vec * score_gate_vec[sample_idx]

    #                 # 3. 累加到字典中 (KEY 必须是 sample_idx)
    #                 if sample_idx in pert_track:
    #                     pert_track[sample_idx] = pert_track[sample_idx] + current_pert_vec
    #                 else:
    #                     pert_track[sample_idx] = current_pert_vec

    #             if len(list(pert_track.values())) > 0:
    #                 # 将字典的值堆叠起来进行融合
    #                 if len(list(pert_track.values())) == 1:
    #                     # 处理 batch size = 1 的特殊情况
    #                     emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
    #                 else:
    #                     emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

    #                 # 将融合后的扰动向量加回 base_emb
    #                 # key 是 sample_idx (0~31)，所以 base_emb[key] 是合法的
    #                 for idx, key in enumerate(pert_track.keys()):
    #                     base_emb[key] = base_emb[key] + emb_total[idx]

    #         base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
    #         base_emb = self.bn_pert_base(base_emb)

    #         ## apply the first MLP
    #         base_emb = self.transform(base_emb)        
    #         out = self.recovery_w(base_emb)
    #         out = out.reshape(num_graphs, self.num_genes, -1)
    #         out = out.unsqueeze(-1) * self.indv_w1
    #         w = torch.sum(out, axis = 2)
    #         out = w + self.indv_b1

    #         # Cross gene
    #         cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
    #         cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

    #         cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
    #         cross_gene_out = torch.cat([out, cross_gene_embed], 2)

    #         cross_gene_out = cross_gene_out * self.indv_w2
    #         cross_gene_out = torch.sum(cross_gene_out, axis=2)
    #         out = cross_gene_out + self.indv_b2        
    #         out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
    #         out = torch.split(torch.flatten(out), self.num_genes)

    #         ## uncertainty head
    #         if self.uncertainty:
    #             out_logvar = self.uncertainty_w(base_emb)
    #             out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
    #             return torch.stack(out), torch.stack(out_logvar)
            
    #         return torch.stack(out)

    def forward(self, data):
        """
        Forward pass of the model.

        Expected fields:
        - data.x
        - data.pert_idx
        - optional data.pert_score with shape [num_graphs, 1] or [num_graphs]
        """
        x, pert_idx = data.x, data.pert_idx

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)

        device = self.args["device"]
        num_graphs = len(data.batch.unique())

        # ---------- score ----------
        pert_scores = getattr(data, "pert_score", None)
        if pert_scores is None:
            # default: full-strength perturbation
            pert_scores = torch.ones((num_graphs, 1), dtype=torch.float32, device=device)
        else:
            pert_scores = pert_scores.to(device).float()
            if pert_scores.dim() == 1:
                pert_scores = pert_scores.unsqueeze(-1)
            if pert_scores.size(0) != num_graphs:
                raise ValueError(
                    f"pert_score batch size mismatch: got {pert_scores.size(0)}, expected {num_graphs}"
                )

        # ---------- base gene embeddings ----------
        gene_ids = torch.arange(self.num_genes, device=device).repeat(num_graphs)
        emb = self.gene_emb(gene_ids)
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)

        pos_emb = self.emb_pos(gene_ids)
        for idx, layer in enumerate(self.layers_emb_pos):
            pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                pos_emb = pos_emb.relu()

        base_emb = base_emb + 0.2 * pos_emb
        base_emb = self.emb_trans_v2(base_emb)

        # ---------- perturbation indices ----------
        pert_index = []
        for sample_idx, perts in enumerate(pert_idx):
            for j in perts:
                if j != -1:
                    pert_index.append([sample_idx, j])

        if len(pert_index) > 0:
            pert_index = torch.tensor(pert_index, device=device).T
        else:
            pert_index = torch.empty((2, 0), dtype=torch.long, device=device)

        pert_global_emb = self.pert_emb(torch.arange(self.num_perts, device=device))

        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        # ---------- score gate ----------
        raw_gate = self.score_gate_w(pert_scores)
        score_gate_vec = 1.0 + self.score_gate_alpha * torch.tanh(raw_gate)

        # ---------- fuse perturbation into each sample ----------
        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

        if pert_index.shape[1] != 0:
            pert_track = {}
            for i, sample_idx_tensor in enumerate(pert_index[0]):
                sample_idx = sample_idx_tensor.item()
                pert_id = pert_index[1][i].item()

                current_pert_vec = pert_global_emb[pert_id]
                current_pert_vec = current_pert_vec * score_gate_vec[sample_idx]

                if sample_idx in pert_track:
                    pert_track[sample_idx] = pert_track[sample_idx] + current_pert_vec
                else:
                    pert_track[sample_idx] = current_pert_vec

            if len(pert_track) > 0:
                stacked = torch.stack(list(pert_track.values()))
                if stacked.shape[0] == 1:
                    stacked = torch.cat([stacked, stacked], dim=0)
                emb_total = self.pert_fuse(stacked)

                for idx, key in enumerate(pert_track.keys()):
                    base_emb[key] = base_emb[key] + emb_total[idx]

        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        base_emb = self.bn_pert_base(base_emb)

        # ---------- decoder ----------
        base_emb = self.transform(base_emb)
        out = self.recovery_w(base_emb)
        out = out.reshape(num_graphs, self.num_genes, -1)
        out = out.unsqueeze(-1) * self.indv_w1
        w = torch.sum(out, axis=2)
        out = w + self.indv_b1

        cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)
        cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])

        cross_gene_out = torch.cat([out, cross_gene_embed], 2)
        cross_gene_out = cross_gene_out * self.indv_w2
        cross_gene_out = torch.sum(cross_gene_out, axis=2)
        out = cross_gene_out + self.indv_b2

        out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
        out = torch.split(torch.flatten(out), self.num_genes)

        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar)

        return torch.stack(out)