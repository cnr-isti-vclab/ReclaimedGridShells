import torch
import random
import numpy as np
from torch.nn.functional import cosine_similarity

def sample_points_from_mesh(mesh, num_samples, num_boundary_and_crease_samples, reproducible, seed, device):
    # Setting randomization seed
    if reproducible:
        random.seed(seed)
        np.random.seed(seed)

    # MESH INNER SAMPLES
    # Multinomial face sampling according to face areas.
    if reproducible:
        sampling_probas = torch.nn.functional.softmax(mesh.face_areas, dim=0)
        sample_face_idx = torch.from_numpy(np.random.choice(np.arange(len(mesh.face_areas)), num_samples, p=sampling_probas.detach().cpu().numpy())).to(device)
    else:
        areas_pruned = mesh.face_areas * (mesh.face_areas > 0)
        sample_face_idx = areas_pruned.view(1, -1).multinomial(num_samples, replacement=True).flatten()

    # Selecting vertices of sampled faces.
    face_verts = mesh.vertices[mesh.faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    a = v0[sample_face_idx]
    b = v1[sample_face_idx]
    c = v2[sample_face_idx]

    # Randomly generating barycentric coordinates.
    w0, w1, w2 = generate_random_barycenter_coords(num_samples, 1, device, mode='face')
    mesh_inner_samples = a * w0 + b * w1 + c * w2

    # CREASE SAMPLES
    # Selecting crease edges.
    cosines = cosine_similarity(mesh.face_normals[mesh.faces_per_edge[:, 0]], mesh.face_normals[mesh.faces_per_edge[:, 1]], dim=1)
    edge_is_crease = (torch.abs(cosines) < 0.95)
    crease_edges = mesh.edge_pos[edge_is_crease]

    # Multinomial edge sampling according to edge lengths.
    if len(crease_edges) != 0:
        if reproducible:
            sampling_probas = torch.nn.functional.softmax(mesh.edge_lengths[crease_edges], dim=0)
            sample_edge_idx = torch.from_numpy(np.random.choice(np.arange(len(mesh.edges[crease_edges, :])), num_boundary_and_crease_samples, p=sampling_probas.detach().cpu().numpy())).to(device)
        else:
            crease_edge_lengths_pruned = mesh.edge_lengths[crease_edges] * (mesh.edge_lengths[crease_edges] > 0)
            sample_edge_idx = crease_edge_lengths_pruned.view(1, -1).multinomial(num_boundary_and_crease_samples, replacement=True).flatten()

        # Selecting vertices of sampled edges.
        edge_verts = mesh.vertices[mesh.edges[crease_edges, :]]
        v0, v1 = edge_verts[:, 0], edge_verts[:, 1]
        a = v0[sample_edge_idx]
        b = v1[sample_edge_idx]
        
        # Randomly generating barycentric coordinates.
        w0, w1 = generate_random_barycenter_coords(num_boundary_and_crease_samples, 1, device, mode='edge')
        crease_samples = a * w0 + b * w1
    else:
        crease_samples = torch.zeros(0, 3, device=device)
    
    # BOUNDARY MESH SAMPLES
    # Selecting free boundary edges
    # edge_is_not_fixed = torch.logical_or(torch.logical_not(mesh.vertex_is_red)[mesh.edges][:, 0], torch.logical_not(mesh.vertex_is_red)[mesh.edges][:, 1])
    # edge_is_on_boundary = torch.logical_and(mesh.vertex_is_on_boundary[mesh.edges][:, 0], mesh.vertex_is_on_boundary[mesh.edges][:, 1])
    # free_boundary_edges = torch.arange(len(mesh.edges), device=device)[torch.logical_and(edge_is_not_fixed, edge_is_on_boundary)]

    # Multinomial edge sampling according to edge lengths.
    if len(mesh.free_boundary_edges) != 0:
        if reproducible:
            sampling_probas = torch.nn.functional.softmax(mesh.edge_lengths[mesh.free_boundary_edges], dim=0)
            sample_edge_idx = torch.from_numpy(np.random.choice(np.arange(len(mesh.edges[mesh.free_boundary_edges, :])), num_boundary_and_crease_samples, p=sampling_probas.detach().cpu().numpy())).to(device)
        else:
            free_boundary_edge_lengths_pruned = mesh.edge_lengths[mesh.free_boundary_edges] * (mesh.edge_lengths[mesh.free_boundary_edges] > 0)
            sample_edge_idx = free_boundary_edge_lengths_pruned.view(1, -1).multinomial(num_boundary_and_crease_samples, replacement=True).flatten()

        # Selecting vertices of sampled edges.
        edge_verts = mesh.vertices[mesh.edges[mesh.free_boundary_edges, :]]
        v0, v1 = edge_verts[:, 0], edge_verts[:, 1]
        a = v0[sample_edge_idx]
        b = v1[sample_edge_idx]
        
        # Randomly generating barycentric coordinates.
        w0, w1 = generate_random_barycenter_coords(num_boundary_and_crease_samples, 1, device, mode='edge')
        free_boundary_samples = a * w0 + b * w1
    else:
        free_boundary_samples = torch.zeros(0, 3, device=device)

    return torch.cat([mesh_inner_samples, crease_samples, free_boundary_samples], dim=0)

def generate_random_barycenter_coords(size1, size2, device, mode):
    if mode == 'face':
        uv = torch.rand(2, size1, size2, device=device)
        u, v = uv[0], uv[1]
        u_sqrt = u.sqrt()
        w0 = 1.0 - u_sqrt
        w1 = u_sqrt * (1.0 - v)
        w2 = u_sqrt * v
        return w0, w1, w2
    elif mode == 'edge':
        u = torch.rand(size1, size2, device=device)
        w0 = u
        w1 = 1.0 - u
        return w0, w1
