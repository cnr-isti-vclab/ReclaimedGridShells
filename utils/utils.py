import numpy as np
import matplotlib.pyplot as plt
import pymeshlab
import torch
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

def load_mesh(path):
    # Creating pymeshlab MeshSet, loading mesh from file and selecting it.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()

    # Selecting vertex and face matrices.
    vertices = np.float32(mesh.vertex_matrix())
    faces = mesh.face_matrix()

    # Deducing vertex constraintness from red or blue coloring.
    colors = mesh.vertex_color_matrix()[:, :3]
    vertex_is_red = np.where((colors[:,0] == 1) & (colors[:,1] == 0) & (colors[:,2] == 0), True, False)

    # Getting boundary vertices mask.
    ms.compute_selection_from_mesh_border()
    vertex_is_on_boundary = mesh.vertex_selection_array()

    # Computing bounding box diagonal length.
    bounding_box_diagonal = mesh.bounding_box().diagonal()
   
    return vertices, faces, vertex_is_red, vertex_is_on_boundary, bounding_box_diagonal

def save_mesh(mesh, filename, v_quality=np.array([], dtype=np.float64)):
    # Changing torch.tensors to np.arrays.
    vertices = np.float64(mesh.vertices.detach().cpu().numpy())
    faces = mesh.faces.detach().cpu().numpy()
    if type(v_quality) is torch.Tensor:
        v_quality = np.float64(v_quality.detach().cpu().numpy())

    # Creating vertex_color_matrix from vertex_is_red.
    colors = np.zeros((vertices.shape[0], 4))
    for idx, red_vertex in enumerate(mesh.vertex_is_red):
        if red_vertex:
            colors[idx, :] = np.array([1., 0., 0., 1.])
        else:
            colors[idx, :] = np.array([0.75294118, 0.75294118, 0.75294118, 1])

    # Creating pymeshlab MeshSet and adding mesh.
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_color_matrix=colors, v_scalar_array=v_quality)
    ms.add_mesh(mesh, set_as_current=True)

    # Saving mesh on filename.
    ms.save_current_mesh(filename, save_vertex_color=True)

def extract_apss_principal_curvatures(path, filterscales=[8.], nsmooth=16):
    # Creating pymeshlab MeshSet, loading mesh from file and selecting it.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()

    # Getting first principal curvature.
    k1 = []
    for scale in filterscales:
        ms.compute_curvature_and_color_apss_per_vertex(filterscale=scale, curvaturetype='K1')
        for _ in range(nsmooth):
            ms.apply_scalar_smoothing_per_vertex()
        k1.append(np.float32(mesh.vertex_scalar_array()))
    k1 = np.stack(k1, axis=1)
    
    # Getting second principal curvature.
    k2 = []
    for scale in filterscales:
        ms.compute_curvature_and_color_apss_per_vertex(filterscale=scale, curvaturetype='K2')
        for _ in range(nsmooth):
            ms.apply_scalar_smoothing_per_vertex()
        k2.append(np.float32(mesh.vertex_scalar_array()))
    k2 = np.stack(k2, axis=1)

    return k1, k2

def extract_geodesic_distances(path, nsmooth=16):
    # Creating pymeshlab MeshSet, loading mesh from file and selecting it.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()

    # Selecting firm vertices via compute_selection_by_condition_per_vertex filter.
    ms.compute_selection_by_condition_per_vertex(condselect='((r == 255) && (g == 0) && (b == 0)) || ((r == 0) && (g == 0) && (b == 255))')

    # Getting geodesic distance of mesh vertices from firm ones.
    ms.compute_scalar_by_geodesic_distance_from_selection_per_vertex(maxdistance=pymeshlab.PercentageValue(100.))
    raw_from_firm_geodesic_distance = np.float32(mesh.vertex_scalar_array())
    for _ in range(nsmooth):
        ms.apply_scalar_smoothing_per_vertex()
    from_firm_geodesic_distance = np.float32(mesh.vertex_scalar_array())

    # Getting geodesic centrality of mesh vertices from firm ones.
    ms_new = pymeshlab.MeshSet()
    v = mesh.vertex_matrix()
    f = mesh.face_matrix()
    from_firm_geodesic_centrality = np.zeros(len(v))
    for vertex in v[mesh.vertex_selection_array()]:
        ms.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(startpoint=vertex, maxdistance=pymeshlab.PercentageValue(100.))
        from_firm_geodesic_centrality += mesh.vertex_scalar_array()
    from_firm_geodesic_centrality /= len(v[mesh.vertex_selection_array()])
    mesh_new = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f, v_scalar_array=from_firm_geodesic_centrality)
    ms_new.add_mesh(mesh_new, set_as_current=True)
    for _ in range(nsmooth):
        ms_new.apply_scalar_smoothing_per_vertex()
    from_firm_geodesic_centrality = np.float32(mesh_new.vertex_scalar_array())

    # Selecting red vertices via compute_selection_from_mesh_border filter.
    ms.compute_selection_from_mesh_border()

    # Getting geodesic distance of mesh vertices from boundary ones.
    ms.compute_scalar_by_geodesic_distance_from_selection_per_vertex(maxdistance=pymeshlab.PercentageValue(100.))
    for _ in range(nsmooth):
        ms.apply_scalar_smoothing_per_vertex()
    from_bound_geodesic_distance = np.float32(mesh.vertex_scalar_array())

    # Getting geodesic centrality of mesh vertices from boundary ones.
    ms_new = pymeshlab.MeshSet()
    v = mesh.vertex_matrix()
    f = mesh.face_matrix()
    from_bound_geodesic_centrality = np.zeros(len(v))
    for vertex in v[mesh.vertex_selection_array()]:
        ms.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(startpoint=vertex, maxdistance=pymeshlab.PercentageValue(100.))
        from_bound_geodesic_centrality += mesh.vertex_scalar_array()
    from_bound_geodesic_centrality /= len(v[mesh.vertex_selection_array()])
    mesh_new = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f, v_scalar_array=from_bound_geodesic_centrality)
    ms_new.add_mesh(mesh_new, set_as_current=True)
    for _ in range(nsmooth):
        ms_new.apply_scalar_smoothing_per_vertex()
    from_bound_geodesic_centrality = np.float32(mesh_new.vertex_scalar_array())

    return raw_from_firm_geodesic_distance, from_firm_geodesic_distance, from_firm_geodesic_centrality, from_bound_geodesic_distance, from_bound_geodesic_centrality

def isotrophic_remesh(mesh, filename, target_length):
    # Changing torch.tensors to np.arrays.
    vertices = np.float64(mesh.vertices.detach().cpu().numpy())
    faces = mesh.faces.detach().cpu().numpy()

    # Creating vertex_color_matrix from vertex_is_red.
    colors = np.zeros((vertices.shape[0], 4))
    for idx, red_vertex in enumerate(mesh.vertex_is_red):
        if red_vertex:
            colors[idx, :] = np.array([1., 0., 0., 1.])
        else:
            colors[idx, :] = np.array([0.75294118, 0.75294118, 0.75294118, 1])

    # Creating pymeshlab MeshSet, adding mesh.
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_color_matrix=colors)
    ms.add_mesh(mesh, set_as_current=True)
    
    # Getting target_lenght percentage.
    mesh.update_bounding_box()
    bb_diagonal = mesh.bounding_box().diagonal()
    target_length_perc = pymeshlab.PercentageValue(target_length * 100 / bb_diagonal)

    # Applying isotrophic remeshing.
    ms.remeshing_isotropic_explicit_remeshing(iterations=20, adaptive=True, targetlen=target_length_perc)

    # Saving mesh on filename
    ms.save_current_mesh(filename)

def save_cloud(points, filename, color=None):
    # Changing torch.tensors to np.arrays.
    points = np.float64(points.detach().cpu().numpy())
    if color is not None and type(color) is torch.Tensor:
        color = np.float64(color.detach().cpu().numpy())

    # Creating pymeshlab MeshSet and adding cloud "mesh".
    ms = pymeshlab.MeshSet()
    if color is None:
        mesh = pymeshlab.Mesh(vertex_matrix=points)
    else:
        mesh = pymeshlab.Mesh(vertex_matrix=points, v_color_matrix=color)
    ms.add_mesh(mesh, set_as_current=True)

    # Saving mesh on filename.
    ms.save_current_mesh(filename)

def map_to_color_space(tensor, bytes=True, vmin=None, vmax=None, cmap='jet', fraction_start=0., fraction_end=1.):
    # Initializing a ScalarMappable object.
    if fraction_start == 0 and fraction_end == 1:
        mappable = ScalarMappable(cmap=cmap)
    else:
        start_cmap = plt.get_cmap(cmap)
        new_cmap = ListedColormap(start_cmap(np.linspace(fraction_start, fraction_end, 128)))
        mappable = ScalarMappable(cmap=new_cmap)
    
    # Setting scale limits if requested.
    if (vmin is not None and vmax is not None):
        mappable.set_clim(vmin=vmin, vmax=vmax)

    # Mapping tensor to color space (bytes=True refers to values in 0-255 range instead of 0-1).
    return mappable.to_rgba(tensor, bytes=bytes)


def edge_connectivity(face_matrix):
    # Initializing edge lists.
    edge_list = []
    edge_per_face = []

    for face in face_matrix:
        current_face_list = []

        for i in range(3):
            edge = sorted([face[i], face[(i + 1) % 3]])
            if edge not in edge_list:
                current_face_list.append(len(edge_list))
                edge_list.append(edge)
            else:
                current_face_list.append(edge_list.index(edge))
                
        edge_per_face.append(current_face_list)

    return np.array(edge_list), np.array(edge_per_face)

def export_vector(v, filename, format='%f'):
    # Changing torch.tensors to np.arrays.
    if type(v) is not np.ndarray:
        v = np.float64(v.detach().cpu().numpy())
    if v.shape == ():
        v = np.expand_dims(v, axis=0)
    np.savetxt(filename, v, delimiter=',', fmt=format)
