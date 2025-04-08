import torch
import math
from torch.nn import Sequential, ReLU
from torch_geometric.seed import seed_everything
from torch_geometric.nn import Linear
from models.layers.geodlinear_layer import GeodLinear
from models.layers.feature_transform_layer import FeatureTransformLayer
from models.layers.dgcnn_layer import DGCNNLayer
from models.layers.gatv2_layer import GATv2Layer

class DisplacerNet(torch.nn.Module):

    def __init__(self, k, mesh, mode='gat', out_channels_list=[256, 256, 256, 256], in_feature_mask=None):
        super(DisplacerNet, self).__init__()
        self.mode = mode

        self.no_graph_layers = len(out_channels_list)

        # Setting seed.
        seed_everything(42)

        # Feature transform layer, if requested.
        if in_feature_mask is not None:
            self.feature_transf = FeatureTransformLayer(mask=in_feature_mask, out_channels=256)

        # First layer.
        if self.mode == 'dgcnn':
            self.layer_1 = DGCNNLayer(out_channels_list[0], k, aggr='mean')
        elif self.mode == 'gat':
            self.layer_1 = GATv2Layer(out_channels_list[0], k)
        else:
            raise ValueError('Non valid layer mode inserted.')

        # Following layers.
        for layer in range(len(out_channels_list) - 1):
            if self.mode == 'dgcnn':
                setattr(self, 'layer_' + str(layer + 2), DGCNNLayer(out_channels_list[layer + 1], k, aggr='mean'))
            elif self.mode == 'gat':
                setattr(self, 'layer_' + str(layer + 2), GATv2Layer(out_channels_list[layer + 1], k))

        # Shared mlp.
        # Recall that raw_geodesic_distance_firm contains the geodesic distances of vertices (i.e min geodetic distance) from firm vertices;
        # The initial scaling value is chosen so that a mesh-max-edge geodesic distance from fixed boundary is scaled 0.9, assumung that the shape
        # is entirely flat. In general, a scale factor based on the cosine of maximum normal incident angle is used.
        # boundary_incidence_scale_factor := max((1 + cos)/2, eps), where:
        #   *   cos := cos(n_v, z-axis) with vertex v chosen s.t. the normal n_v yields the maximum angle with z-axis
        #   *   since -1 < cos < 1, (1 + cos)/2 belongs to the range [0,1]
        #   *   eps is used to prevent boundary_incidence_scale_factor = 0             
        eps = 1e-2
        boundary_incidence_scale_factor = max((1 + mesh.max_z_angle_incident_cosine)/2, eps)
        self.mlp = Sequential(  Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 64),
                                ReLU(),
                                GeodLinear(-1, 3, mesh.raw_geodesic_distance_firm, -math.log(1-0.9*boundary_incidence_scale_factor)/torch.max(mesh.edge_lengths))  )
        
        # Splitting parameter groups.
        if hasattr(self, 'feature_transf'):
            base_params_list = [self.feature_transf]
        else:
            base_params_list = []
        for layer in range(len(out_channels_list)):
            base_params_list.append(getattr(self, 'layer_' + str(layer + 1)))
        self.params = torch.nn.ModuleDict({
            'base': torch.nn.ModuleList(base_params_list),
            'shared_mlp': torch.nn.ModuleList([self.mlp])})

    def forward(self, x):
        # Transforming input features, if requested.
        if hasattr(self, 'feature_transf'):
            x = self.feature_transf(x)

        # List of layer outputs.
        out_list = [x]

        # Applying consecutive layers.
        for layer in range(1, self.no_graph_layers + 1):
            current_layer = getattr(self, 'layer_' + str(layer))
            out_list.append(current_layer(out_list[-1]))

        # Chaining all layer outputs.
        out = torch.cat(out_list, dim=1)

        # Processing dgcnn_out via shared mlp.
        return self.mlp(out)

    @staticmethod
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -1e-4, 1e-4)
            m.bias.data.fill_(0.)


class MultiDisplacerNet(torch.nn.Module):
    def __init__(self, k, mesh, in_feature_mask, out_channels_list=[256, 512, 512, 256], out_transf_channels=256):
        super(MultiDisplacerNet, self).__init__()

        self.no_graph_layers = len(out_channels_list)
        self.out_transf_channels = out_transf_channels
        self.no_batches = len(in_feature_mask)

        # Setting seed.
        seed_everything(42)

        # Feature transform layer
        self.feature_transf = FeatureTransformLayer(mask=in_feature_mask, out_channels=out_transf_channels)

        # First layer.
        self.layer_1 = GATv2Layer(out_channels_list[0], k)

        # Following layers.
        for layer in range(len(out_channels_list) - 1):
            setattr(self, 'layer_' + str(layer + 2), GATv2Layer(out_channels_list[layer + 1], k))

        # Shared mlp.
        # Recall that input_features[:, 8] contains the geodesic distances of vertices (i.e min geodetic distance) from firm vertices;
        # The initial scaling value is chosen so that a mesh-medium-edge geodesic distance from fixed boundary is scaled 0.5.
        self.mlp = Sequential(  Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 64),
                                ReLU(),
                                GeodLinear(-1, 3, mesh.input_features[:, 8], math.atanh(0.5)/torch.mean(mesh.edge_lengths))  )

    def forward(self, x):
        # Batch vector.
        no_vertices = len(x)
        batch_vector = torch.kron(torch.arange(self.no_batches), torch.ones(no_vertices)).long().to(next(self.parameters()).device)
        
        # Transforming input features to reach graph dimension.
        x = self.feature_transf(x)
        x = torch.cat(torch.split(x, self.out_transf_channels, dim=1), dim=0)

        # List of layer outputs.
        out_list = [x]

        # Applying first layer.
        out_list.append(self.layer_1(out_list[0], batch_vector))

        # Applying following layers.
        for layer in range(2, self.no_graph_layers + 1):
            current_layer = getattr(self, 'layer_' + str(layer))
            net_input = torch.cat(out_list[-2: ], dim=1)
            out_list.append(current_layer(net_input, batch_vector))

        # Building model output.
        out = torch.cat(torch.split(out_list[-1], no_vertices, dim=0), dim=1)
        return self.mlp(out)


class MultiMaxDisplacerNet(torch.nn.Module):
    def __init__(self, k, mesh, in_feature_mask, out_channels_list=[256, 512, 512, 512], out_transf_channels=256):
        super(MultiMaxDisplacerNet, self).__init__()

        self.no_graph_layers = len(out_channels_list)
        self.out_transf_channels = out_transf_channels
        self.no_batches = len(in_feature_mask)

        # Setting seed.
        seed_everything(42)

        # Feature transform layer
        self.feature_transf = FeatureTransformLayer(mask=in_feature_mask, out_channels=out_transf_channels, activation='sigmoid')

        # First layer.
        self.layer_1 = GATv2Layer(out_channels_list[0], k)

        # Following layers.
        for layer in range(len(out_channels_list) - 1):
            setattr(self, 'layer_' + str(layer + 2), GATv2Layer(out_channels_list[layer + 1], k))

        # Shared mlp.
        # Recall that input_features[:, 8] contains the geodesic distances of vertices (i.e min geodetic distance) from firm vertices;
        # The initial scaling value is chosen so that a mesh-medium-edge geodesic distance from fixed boundary is scaled 0.9.
        self.mlp = Sequential(  Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 64),
                                ReLU(),
                                GeodLinear(-1, 3, mesh.input_features[:, 8], math.atanh(0.9)/torch.mean(mesh.edge_lengths))  )

    def forward(self, x):
        # Batch vector.
        no_vertices = len(x)
        batch_vector = torch.kron(torch.arange(self.no_batches), torch.ones(no_vertices)).long().to(next(self.parameters()).device)
        
        # Transforming input features to reach graph dimension.
        x = self.feature_transf(x)
        x = torch.cat(torch.split(x, self.out_transf_channels, dim=1), dim=0)

        # List of layer outputs.
        out_list = [x]

        # Applying consecutive layers.
        for layer in range(1, self.no_graph_layers + 1):
            current_layer = getattr(self, 'layer_' + str(layer))
            out_list.append(current_layer(out_list[-1], batch_vector))

        # Building model output.
        out = torch.max(torch.stack(torch.split(torch.cat(out_list, dim=1), no_vertices, dim=0), dim=0), dim=0).values
        return self.mlp(out)

class MultiMeanDisplacerNet(torch.nn.Module):
    def __init__(self, k, mesh, in_feature_mask, out_channels_list=[256, 512, 512, 512], out_transf_channels=256):
        super(MultiMeanDisplacerNet, self).__init__()

        self.no_graph_layers = len(out_channels_list)
        self.out_transf_channels = out_transf_channels
        self.no_batches = len(in_feature_mask)

        # Setting seed.
        seed_everything(42)

        # Feature transform layer
        self.feature_transf = FeatureTransformLayer(mask=in_feature_mask, out_channels=out_transf_channels)

        # First layer.
        self.layer_1 = GATv2Layer(out_channels_list[0], k)

        # Following layers.
        for layer in range(len(out_channels_list) - 1):
            setattr(self, 'layer_' + str(layer + 2), GATv2Layer(out_channels_list[layer + 1], k))

        # Shared mlp.
        # Recall that input_features[:, 8] contains the geodesic distances of vertices (i.e min geodetic distance) from firm vertices;
        # The initial scaling value is chosen so that a mesh-medium-edge geodesic distance from fixed boundary is scaled 0.5.
        self.mlp = Sequential(  Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 64),
                                ReLU(),
                                GeodLinear(-1, 3, mesh.input_features[:, 8], math.atanh(0.5)/torch.mean(mesh.edge_lengths))  )

    def forward(self, x):
        # Batch vector.
        no_vertices = len(x)
        batch_vector = torch.kron(torch.arange(self.no_batches), torch.ones(no_vertices)).long().to(next(self.parameters()).device)
        
        # List of layer outputs.
        out_list = []

        # Transforming input features.
        x = self.feature_transf(x)

        # Changing input tensor to separate groups of features.
        x = torch.cat(torch.split(x, self.out_transf_channels, dim=1), dim=0)

        # Applying first layer and averaging over feature groups.
        y = self.layer_1(x, batch_vector)
        y = torch.mean(torch.stack(torch.split(y, no_vertices, dim=0), dim=0), dim=0)
        out_list.append(y)

        # Applying consecutive layers.
        for layer in range(2, self.no_graph_layers + 1):
            current_layer = getattr(self, 'layer_' + str(layer))
            out_list.append(current_layer(out_list[-1]))

        # Building model output.
        out = torch.max(torch.stack(torch.split(torch.cat(out_list, dim=1), no_vertices, dim=0), dim=0), dim=0).values
        return self.mlp(out)
