"""
EGNN architecture from Schneuing et al. 2023 & Satorras et al. 2022
"""

from torch import nn
import torch.nn.functional as F
import torch
import math

from torch.nn.utils.parametrizations import orthogonal

from ReQFlow.so3_utils import rotmat_to_rotvec, rotvec_to_rotmat


def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Args:
        d6: Batch of 6D vectors [Batch, 6]
    Returns:
        Batch of rotation matrices [Batch, 3, 3]
    """
    # 1. Split into two 3D vectors
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]
    
    # 2. Normalize the first vector
    x = F.normalize(x_raw, dim=-1, eps=1e-5)
    
    # 3. Orthogonalize the second vector (Gram-Schmidt)
    # y = y_raw - (x · y_raw) * x
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-5)

    # 4. Compute the third vector (Cross product)
    y = torch.cross(z, x, dim=-1)
    
    # 5. Stack to form the 3x3 matrix
    # Reshape vectors to [B, 3, 1] and concatenate
    rot = torch.stack([x, y, z], dim=-1)
    
    return rot


def norm_angle(s):

    # s needs to be change (B, 7, 2)

    norm_denom = torch.sqrt(
        torch.clamp(
            torch.sum(s ** 2, dim=-1, keepdim=True),
            min=1e-2,
        )
    )
    s = s / norm_denom
    return s


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 reflection_equiv=True, variational=True, ba=False):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        self.ba = ba
        input_edge = hidden_nf * 2 + edges_in_d
        input_rot = hidden_nf * 2 + 1 + edges_in_d 
        self.variational = variational
        self.angle_dim = 14
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False))

        nn.init.zeros_(self.coord_mlp[-1].weight) 
        
        self.cross_product_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        ) if not self.reflection_equiv else None
        if self.cross_product_mlp is not None:
            nn.init.zeros_(self.cross_product_mlp[-1].weight)
        
        self.rot_mlp = nn.Sequential(
            nn.Linear(input_rot, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3, bias=False)
        )
        nn.init.zeros_(self.rot_mlp[-1].weight) 

        self.angle_mlp = nn.Sequential(
            nn.Linear(input_edge + self.angle_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.angle_dim, bias=False)
        )

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if self.ba:
            ba_input_dim = input_edge + 18
            self.ba1_mlp = torch.nn.Sequential(
                torch.nn.Linear(ba_input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
            )
            self.ba2_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_nf + 64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.ba and (prefix + 'ba1_mlp.0.weight') in state_dict:
            checkpoint_weight = state_dict[prefix + 'ba1_mlp.0.weight']
            checkpoint_input_dim = checkpoint_weight.shape[1]
            current_input_dim = self.ba1_mlp[0].in_features
            if checkpoint_input_dim != current_input_dim:
                self.ba1_mlp = torch.nn.Sequential(
                    torch.nn.Linear(checkpoint_input_dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                ).to(checkpoint_weight.device)
        super(EquivariantUpdate, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def message(self, h, x, rot, a, edge_index, coord_diff, coord_cross,
                              edge_attr, edge_mask, update_coords_mask, mol_dim):

        row, col = edge_index

        rot_diff = torch.bmm(rot[row].transpose(-2, -1), rot[col])
        rot_diff = rotmat_to_rotvec(rot_diff)
        dist_sq = torch.sum(rot_diff**2, dim=-1, keepdim=True)

        a_row = a[row].view(-1, 7, 2) 
        a_col = a[col].view(-1, 7, 2) 
        sin_diff = a_row[..., 0] * a_col[..., 1] - a_row[..., 1] * a_col[..., 0]
        cos_diff = a_row[..., 1] * a_col[..., 1] + a_row[..., 0] * a_col[..., 0]
        angle_diff = torch.stack([sin_diff, cos_diff], dim=-1).view(-1,self.angle_dim) 

        m = torch.cat([h[row], h[col], edge_attr, dist_sq, angle_diff], dim=1)

        return m

    def coord_model(self, h, x, m, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None, mol_dim=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)

        if self.tanh:
            l = torch.tanh(self.coord_mlp(input_tensor))
            trans = coord_diff * l * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        agg = torch.clamp(agg, min=-20.0, max=20.0)

        x = x + agg

        return x
    
    def rot_model(self, h, rot, m, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None,
                    mol_dim=None):

        row, col = edge_index
        rot_diff = torch.bmm(rot[row].transpose(-2, -1), rot[col])
        rot_diff = rotmat_to_rotvec(rot_diff)
        dist_sq = torch.sum(rot_diff**2, dim=-1, keepdim=True)
        input_tensor = torch.cat([h[row], h[col], edge_attr, dist_sq], dim=-1)
        rot_matrix = self.rot_mlp(input_tensor)

        if self.tanh:
            rot_matrix = torch.tanh(rot_matrix) * self.coords_range
        
        trans = rot_diff * rot_matrix

        if edge_mask is not None:
            trans = trans * edge_mask   

        agg = unsorted_segment_sum(trans, row, num_segments=rot.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        
        

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        agg = rotvec_to_rotmat(agg)
        rot = torch.einsum("ijk, ikn -> ijn", rot, agg)
        
        rot_peptide = rot[:mol_dim]
        d6_x = rot_peptide[:, :, 0]
        d6_y = rot_peptide[:, :, 1]
        d6 = torch.cat([d6_x, d6_y], dim=-1)
        rot_peptide = rotation_6d_to_matrix(d6)
        rot = torch.cat([rot_peptide, rot[mol_dim:]], dim=0)

        return rot
    
    def angle_model(self, h, a, m, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None, angle_mask=None, mol_dim=None):
        row, col = edge_index
        
        a_row = a[row].view(-1, 7, 2) 
        a_col = a[col].view(-1, 7, 2) 
        sin_diff = a_row[..., 0] * a_col[..., 1] - a_row[..., 1] * a_col[..., 0]
        cos_diff = a_row[..., 1] * a_col[..., 1] + a_row[..., 0] * a_col[..., 0]
        angle_diff = torch.stack([sin_diff, cos_diff], dim=-1).view(-1,self.angle_dim) 

        input_tensor = torch.cat([h[row], h[col], edge_attr, angle_diff], dim=1)

        if self.tanh:
            trans = angle_diff * torch.tanh(self.angle_mlp(input_tensor))
        else:
            trans = angle_diff * self.angle_mlp(input_tensor)
        
        if edge_mask is not None:
            trans = trans * edge_mask
        
        agg = unsorted_segment_sum(trans, row, num_segments=a.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        a_reshaped = a.view(-1, 7, 2)
        agg_reshaped = agg.view(-1, 7, 2)
        sin_a, cos_a = a_reshaped[..., 0], a_reshaped[..., 1]
        sin_agg, cos_agg = agg_reshaped[..., 0], agg_reshaped[..., 1]
        new_sin = sin_a * cos_agg + cos_a * sin_agg
        new_cos = cos_a * cos_agg - sin_a * sin_agg
        a_reshaped = torch.stack([new_sin, new_cos], dim=-1)
        a = a_reshaped.view(-1, self.angle_dim)

        if angle_mask is not None:
            a_mol = a[:mol_dim].reshape(-1, 7, 2)
            a_mol = a_mol * angle_mask.view(-1, 7).unsqueeze(-1)
            a = torch.cat([a_mol.view(-1, self.angle_dim), a[mol_dim:]], dim=0)

        return a

    def ba_model(self, h, a, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None, angle_mask=None, mol_dim=None, mask=None, rot=None):

        row, col = edge_index
        
        rot_diff = torch.bmm(rot[row].transpose(-2, -1), rot[col])
        rot_diff = rotmat_to_rotvec(rot_diff)
        dist_sq = torch.sum(rot_diff**2, dim=-1, keepdim=True)

        a_row = a[row].view(-1, 7, 2) 
        a_col = a[col].view(-1, 7, 2) 
        sin_diff = a_row[..., 0] * a_col[..., 1] - a_row[..., 1] * a_col[..., 0]
        cos_diff = a_row[..., 1] * a_col[..., 1] + a_row[..., 0] * a_col[..., 0]
        angle_diff = torch.stack([sin_diff, cos_diff], dim=-1).view(-1,self.angle_dim) 

        edge_inputs = torch.cat([
            h[row], h[col], edge_attr,
            coord_diff, dist_sq, angle_diff
        ], dim=1)
        edge_features = self.ba1_mlp(edge_inputs)

        if edge_mask is not None:
            edge_features = edge_features * edge_mask

        agg = unsorted_segment_sum(edge_features, row, num_segments=h.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        combined = torch.cat([h, agg], dim=1) 
        ba = self.ba2_mlp(combined)  

        return ba


    def forward(self, h, rot, x, angles, edge_index, coord_diff, coord_cross,
                edge_attr=None, node_mask=None, edge_mask=None,
                update_coords_mask=None, mol_dim=None, angle_mask=None, mask=None):

        angles_orig = angles
        rot_orig = rot
        coord_diff_orig = coord_diff

        m = self.message(h, x, rot, angles, edge_index, coord_diff, coord_cross,
                              edge_attr, edge_mask,
                              update_coords_mask=update_coords_mask, mol_dim=mol_dim)

        x = self.coord_model(h, x, m, edge_index, coord_diff, coord_cross,
                              edge_attr, edge_mask,
                              update_coords_mask=update_coords_mask, mol_dim=mol_dim)
        rot = self.rot_model(h, rot, m, edge_index, coord_diff, coord_cross,
                            edge_attr, edge_mask, update_coords_mask=update_coords_mask,
                            mol_dim=mol_dim)
        a = self.angle_model(h, angles, m, edge_index, coord_diff, coord_cross,
                            edge_attr, edge_mask, update_coords_mask=update_coords_mask,
                            angle_mask=angle_mask, mol_dim=mol_dim)
        if self.ba:
            ba = self.ba_model(h, angles_orig, edge_index, coord_diff_orig, coord_cross,
                                edge_attr, edge_mask, update_coords_mask=update_coords_mask,
                                angle_mask=angle_mask, mol_dim=mol_dim, mask=mask, rot=rot_orig)
        else:
            ba = None

        if node_mask is not None:
            x = x * node_mask
            rot = rot * node_mask
            a = a * node_mask

        return rot, x, a, ba


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', reflection_equiv=True, variational=True, ba=False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv
        self.variational = variational
        self.ba = ba

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,
                                                       reflection_equiv=self.reflection_equiv,
                                                       variational=self.variational,
                                                       ba=self.ba))
        self.to(self.device)

    def forward(self, h, rot, x, angles, edge_index, node_mask=None, edge_mask=None,
                edge_attr=None, update_coords_mask=None, batch_mask=None, mol_dim=None,
                angle_mask=None, mask=None):

        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, batch_mask,
                                      self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                               node_mask=node_mask, edge_mask=edge_mask)
        rot, x, a, ba = self._modules["gcl_equiv"](h, rot, x, angles, edge_index, coord_diff, coord_cross, edge_attr,
                                           node_mask, edge_mask, update_coords_mask=update_coords_mask, mol_dim=mol_dim,
                                           angle_mask=angle_mask, mask=mask)

        if node_mask is not None:
            h = h * node_mask
        return h, rot, x, a, ba


class EGNN_all_atom(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', reflection_equiv=True, edge_sin_attr=False, all_atom=False, variational=True, ba=False):
        super(EGNN_all_atom, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv
        # self.edge_sin_attr = edge_sin_attr
        self.all_atom = all_atom
        self.variational = variational
        self.ba = ba

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
        
        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               reflection_equiv=self.reflection_equiv,
                                                               variational=self.variational,
                                                               ba=self.ba))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, update_coords_mask=None,
                batch_mask=None, edge_attr=None, rot=None, angles=None, mol_dim=None, angle_mask=None, mask=None):
        
        rot = rot.reshape(-1, 3, 3)

        # Edit Emiel: Remove velocity as input
        edge_feat, _ = coord2diff(x, edge_index)

        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
            
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)

        h = self.embedding(h)

        for i in range(0, self.n_layers):
            h, rot, x, a, ba = self._modules["e_block_%d" % i](
                h, rot, x, angles, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask,
                batch_mask=batch_mask, mol_dim=mol_dim, angle_mask=angle_mask, mask=mask)

        # TODO: For adding confidence heads
        h_last_layer = h

        # Important, the bias of the last linear might be non-zero
        h_out = self.embedding_out(h)

        if node_mask is not None:
            h_out = h_out * node_mask

        rot = rot.view(-1, 9)

        return h_out, x, h_last_layer, rot, a, ba


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-5)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-5)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def coord2cross(x, edge_index, batch_mask, norm_constant=1):

    mean = unsorted_segment_sum(x, batch_mask,
                                num_segments=batch_mask.max() + 1,
                                normalization_factor=None,
                                aggregation_method='mean')
    row, col = edge_index
    cross = torch.cross(x[row]-mean[batch_mask[row]],
                        x[col]-mean[batch_mask[col]], dim=1)
    norm = torch.sqrt(torch.sum(cross**2, dim=1, keepdim=True) + 1e-5)
    cross = cross / (norm + norm_constant)
    if torch.any(norm < 1e-6):
        print("Warning: Zero cross product detected in coord2cross. Check for collinear points.")
    return cross


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments,) + data.shape[1:]
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    
    if len(data.shape) == 2:
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    else:
        segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, data.size(1), data.size(2))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
