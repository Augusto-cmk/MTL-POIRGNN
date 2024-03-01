import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from spektral.layers.convolutional import GCNConv # Utilizar o pythorch geometric

class CamadaNeural(nn.Module):
    def __init__(self, weights, use_entropy, n_classes, activation='softmax'):
        super(CamadaNeural, self).__init__()
        self.n_components = len(weights)
        self.components_weights = nn.Parameter(torch.Tensor(weights))
        self.n_classes = n_classes
        self.activation = activation
        self.use_entropy_flag = use_entropy
        self.a_variables = nn.Parameter(torch.ones(int(sum(use_entropy))), requires_grad=True)

    def forward(self, inputs):
        components = inputs

        entropies = []

        for i in range(self.n_components):
            component_out = components[i]
            entropy = 1 / torch.mean(F.categorical.entropy(component_out))
            entropies.append(entropy)

        out_sum = None
        for i in range(self.n_components):
            entropy = entropies[i]
            if self.use_entropy_flag[i] == 1.:
                out = self.formula(i, components[i], entropy, self.j.item())
                self.j.data.add_(1)
            else:
                out = self.components_weights[i] * components[i]

            if out_sum is None:
                out_sum = out
            else:
                out_sum += out

        return out_sum

    def formula(self, component_weight_index, component_out, entropy, use_entropy_component_index):
        out = (self.components_weights[component_weight_index] + entropy) * self.a_variables[use_entropy_component_index] * component_out
        return out

class MFA_RNN(nn.Module):
    def __init__(self):
        super(MFA_RNN, self).__init__()
        self.model_name = "GRUenhaced original 10mil"

    def forward(self,location_input_dim,time_input_dim,num_users, location_category_input, temporal_input, distance_input, duration_input, user_id_input,
                categories_distance_matrix, adjancency_matrix,
                categories_durations_matrix,
                poi_category_probabilities):

        gru_units = 30
        emb_category = nn.Embedding(location_input_dim, 7)
        emb_time = nn.Embedding(time_input_dim, 3)
        emb_id = nn.Embedding(num_users, 2)
        emb_distance = nn.Embedding(51, 3)
        emb_duration = nn.Embedding(49, 3)

        spatial_embedding = emb_category(location_category_input)
        temporal_embedding = emb_time(temporal_input)
        id_embedding = emb_id(user_id_input)
        distance_embbeding = emb_distance(distance_input)
        duration_embbeding = emb_duration(duration_input)

        spatial_flatten = spatial_embedding.view(spatial_embedding.size(0), -1)

        distance_duration = 0.1 * torch.mul(distance_embbeding, duration_embbeding)

        l_p = torch.cat([spatial_embedding, temporal_embedding, distance_embbeding, duration_embbeding, distance_duration], dim=2)

        y_cup = torch.cat([id_embedding, l_p], dim=2).view(l_p.size(0), -1)

        srnn = nn.GRU(l_p.size(2), gru_units, batch_first=True)
        srnn_out, _ = srnn(l_p)
        srnn_out = F.dropout(srnn_out, p=0.5, training=self.training)

        att = nn.MultiheadAttention(srnn_out.size(2), 4)(srnn_out, srnn_out, srnn_out)

        distance_duration_matrix = torch.mul(categories_distance_matrix, categories_durations_matrix)

        distance_matrix = categories_distance_matrix
        x_distances = GCNConv(22)(distance_matrix, adjancency_matrix)
        x_distances = F.dropout(x_distances, p=0.5, training=self.training)
        x_distances = GCNConv(10)(x_distances, adjancency_matrix)
        x_distances = F.dropout(x_distances, p=0.5, training=self.training)
        x_distances = x_distances.view(x_distances.size(0), -1)

        durations_matrix = categories_durations_matrix
        x_durations = GCNConv(22)(durations_matrix, adjancency_matrix)
        x_durations = GCNConv(10)(x_durations, adjancency_matrix)
        x_durations = F.dropout(x_durations, p=0.3, training=self.training)
        x_durations = x_durations.view(x_durations.size(0), -1)

        distance_duration_matrix = GCNConv(22)(distance_duration_matrix, adjancency_matrix)
        distance_duration_matrix = GCNConv(10)(distance_duration_matrix, adjancency_matrix)
        distance_duration_matrix = F.dropout(distance_duration_matrix, p=0.3, training=self.training)
        distance_duration_matrix = distance_duration_matrix.view(distance_duration_matrix.size(0), -1)

        srnn_out = srnn_out.view(srnn_out.size(0), -1)
        y_sup = torch.cat([srnn_out, att, x_distances], dim=1)
        y_sup = F.dropout(y_sup, p=0.3, training=self.training)
        y_sup = nn.Linear(y_sup.size(1), location_input_dim)(y_sup)
        y_cup = F.dropout(y_cup, p=0.5, training=self.training)
        y_cup = nn.Linear(y_cup.size(1), location_input_dim)(y_cup)
        spatial_flatten = nn.Linear(spatial_flatten.size(1), location_input_dim)(spatial_flatten)

        gnn = torch.cat([x_durations, distance_duration_matrix], dim=1)
        gnn = F.dropout(gnn, p=0.3, training=self.training)
        gnn = nn.Linear(gnn.size(1), location_input_dim)(gnn)

        pc = nn.Linear(14, location_input_dim)(poi_category_probabilities)
        pc = F.dropout(pc, p=0.5, training=self.training)
        pc = pc.view(pc.size(0), -1)
        pc = nn.Linear(pc.size(1), location_input_dim)(pc)

        y_up = CamadaNeural([1., 0.5, -0.2, 8.], [0., 1., 0., 0.], location_input_dim)([y_cup, y_sup, spatial_flatten, gnn])

        return y_up

