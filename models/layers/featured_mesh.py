import torch
from models.layers.mesh import Mesh
from torch.nn.functional import normalize
from utils.utils import extract_apss_principal_curvatures, extract_geodesic_distances

# Class that extends Mesh for admitting vertex input feature vectors
class FeaturedMesh(Mesh):

    def __init__(self, file, vertices=None, faces=None, device='cpu'):
        super(FeaturedMesh, self).__init__(file, vertices, faces, device)
        self.file = file

    def compute_mesh_input_features(self):
        feature_list = []
        feature_mask = []

        # input_features[:, 0:4]: geometric-invariant vertex and normal coordinates.
        # **Vertex coordinates** The original mesh is translated so that the output mesh (coords. (x',y',z')) lies on the z=0 plane and
        # the barycenter on the z axis (translation of (-x_B, -y_B, -z_min) where (x_B, y_B, z_B) are the barycenter 
        # coordinates and z_min is the minimum z-component of vertices). 
        # Invariant coordinates are (x',y',z') -> (sqrt(x'**2 + y'**2), z')
        barycenter = torch.mean(self.vertices, dim=0)
        z_min = torch.min(self.vertices[:, 2])
        scaled_vertices = self.vertices - torch.cat([barycenter[0:2], z_min.unsqueeze(0)], dim=0)
        planar_component = torch.sqrt(scaled_vertices[:, 0]**2 + scaled_vertices[:, 1]**2).unsqueeze(1)
        height_component = scaled_vertices[:, 2].unsqueeze(1)
        # **Normal coordinates** (x_n, y_n, z_n) -> (sqrt(x_n**2 + y_n**2), z_n)
        self.compute_vertex_normals()
        planar_normal_component = torch.sqrt(self.vertex_normals[:, 0]**2 + self.vertex_normals[:, 1]**2).unsqueeze(1)
        height_normal_component = self.vertex_normals[:, 2].unsqueeze(1)
        feature_list.append(planar_component)
        feature_list.append(height_component)
        feature_list.append(planar_normal_component)
        feature_list.append(height_normal_component)
        feature_mask.append([0, 1, 2, 3])        

        # input_features[:, 4:6]: principal curvatures.
        k1, k2 = extract_apss_principal_curvatures(self.file)
        k1 = torch.from_numpy(k1).to(self.device)
        k2 = torch.from_numpy(k2).to(self.device)
        feature_list.append(k1)
        feature_list.append(k2)
        feature_mask.append([4, 5])

        # input_features[:, 6]: geodesic distance (i.e min geodetic distance) from firm vertices;
        # input_features[:, 7]: geodesic centrality (i.e mean of geodetic distances) from firm vertices;
        # input_features[:, 8]: geodesic distance (i.e min geodetic distance) from mesh boundary;
        # input_features[:, 9]: geodesic centrality (i.e mean of geodetic distances) from mesh boundary.
        raw_geodesic_distance_firm, geodesic_distance_firm, geodesic_centrality_firm, geodesic_distance_bound, geodesic_centrality_bound = extract_geodesic_distances(self.file)
        raw_geodesic_distance_firm = torch.from_numpy(raw_geodesic_distance_firm).to(self.device)
        geodesic_distance_firm = torch.from_numpy(geodesic_distance_firm).to(self.device).unsqueeze(1)
        geodesic_centrality_firm = torch.from_numpy(geodesic_centrality_firm).to(self.device).unsqueeze(1)
        geodesic_distance_bound = torch.from_numpy(geodesic_distance_bound).to(self.device).unsqueeze(1)
        geodesic_centrality_bound = torch.from_numpy(geodesic_centrality_bound).to(self.device).unsqueeze(1)
        feature_list.append(geodesic_distance_firm)
        feature_list.append(geodesic_centrality_firm)
        feature_list.append(geodesic_distance_bound)
        feature_list.append(geodesic_centrality_bound)
        feature_mask.append([6, 7, 8, 9])

        self.input_features = torch.cat(feature_list, dim=1)
        self.feature_mask = feature_mask
        self.raw_geodesic_distance_firm = raw_geodesic_distance_firm

        # Computing cosine of maximum incident angle.
        z_axis = torch.tensor([[0., 0., 1.]], device=self.device)
        self.max_z_angle_incident_cosine = torch.min(torch.cosine_similarity(self.vertex_normals[self.vertex_is_red], z_axis, dim=1))

    def compute_vertex_normals(self):
        ############################################################################################################
        # Computing vertex normals by weighting normals from incident faces.
        # Some details: vec.scatter_add(0, idx, src) with vec, idx, src 1d tensors, add at vec positions specified
        # by idx corresponding src values.
        vertex_normals = torch.zeros(self.vertices.shape[0], 3, device=self.device)

        vertex_normals[:, 0].scatter_add_(0, self.faces.flatten(), torch.stack([self.face_normals[:, 0]] * 3, dim=1).flatten())
        vertex_normals[:, 1].scatter_add_(0, self.faces.flatten(), torch.stack([self.face_normals[:, 1]] * 3, dim=1).flatten())
        vertex_normals[:, 2].scatter_add_(0, self.faces.flatten(), torch.stack([self.face_normals[:, 2]] * 3, dim=1).flatten())

        # Applying final l2-normalization.
        self.vertex_normals = normalize(vertex_normals, p=2, dim=1)
        