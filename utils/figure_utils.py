from pathlib import Path
import bpy
import os
import pymeshlab
import platform
import igl
import numpy as np
import torch
from scipy.ndimage import median
from optim.structural_calculus import StructuralCalculus
from PIL import Image
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.scale import FuncScale
from utils.utils import map_to_color_space, export_vector

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# First stock: uniform
stock_u = [0, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2, 2.5, 3]
stock_capacities_u = [float('inf'), 400, 400, 400, 400, 400, 400, 400, 400, 400]
stock_name_u = 'stock_uniform'

# Second stock: non-uniform 1
stock_nu1 = [0, 1.59, 1.93, 1.95, 1.99, 2.22, 2.77, 2.80, 2.94, 2.98, 3]
stock_capacities_nu1 = [float('inf'), 540, 54, 54, 180, 270, 252, 180, 117, 27, 90]
stock_name_nu1 = 'stock_nonuniform1'

# Third stock: non-uniform 2
stock_nu2 = [0, 1.59, 1.93, 1.95, 1.99, 2.22, 2.77, 2.80, 2.94, 2.98, 3]
stock_capacities_nu2 = [float('inf'), 200, 20, 20, 60, 110, 100, 65, 45, 5, 40]
stock_name_nu2 = 'stock_nonuniform2'

def global_max(l):
    return max([np.max(sublist) for sublist in l])

def global_min(l):
    return min([np.min(sublist) for sublist in l])

# *** Auxiliary functions for plotting ***
# Custom transformation function
def custom_transform(x):
    # Adjust the scale here according to your preference
    return (x + 10) * (x > 0) + x * (x == 0)

def get_freqs_and_cols(basepath, case):
    cm_filepath = os.path.join(basepath, 'color_map_' + case + '.csv')
    freqs_filepath = os.path.join(basepath, 'batch_matching_frequencies_' + case + '.csv')
    colss = np.loadtxt(cm_filepath, delimiter=',')
    cols = [(colss[i, 0]/255, colss[i, 1]/255, colss[i, 2]/255) for i in range(1, len(colss))]
    cols.append((1-(32/255), 1-(32/255), 1-(32/255)))
    freqs = np.loadtxt(freqs_filepath, delimiter=',')
    freqs_new = np.zeros(len(freqs))
    freqs_new[:-1] = freqs[1:]
    freqs_new[-1] = freqs[0]
    return freqs_new, cols

def get_freqs_start_end(basepath, case):
    start_freqs_filepath = os.path.join(basepath, 'batch_matching_frequencies_start_' + case + '.csv')
    end_freqs_filepath = os.path.join(basepath, 'batch_matching_frequencies_' + case + '.csv')
    start_freqs = np.loadtxt(start_freqs_filepath, delimiter=',')
    end_freqs = np.loadtxt(end_freqs_filepath, delimiter=',')
    start_freqs_new = np.zeros(len(start_freqs))
    start_freqs_new[:-1] = start_freqs[1:]
    start_freqs_new[-1] = start_freqs[0]
    end_freqs_new = np.zeros(len(end_freqs))
    end_freqs_new[:-1] = end_freqs[1:]
    end_freqs_new[-1] = end_freqs[0]
    return start_freqs_new, end_freqs_new

def compute_edge_lengths(case):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(case)
    ms.compute_selection_by_condition_per_vertex(condselect = '(r == 255) && (g == 0) && (b == 0)')
    v, f = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
    v_is_red = ms.current_mesh().vertex_selection_array()
    e = igl.edges(f)
    reusable_e = e[np.logical_not(np.logical_and(v_is_red[e][:, 0], v_is_red[e][:, 1]))]
    edge_directions = v[reusable_e][:, 1] - v[reusable_e][:, 0]
    reusable_edge_lengths = np.linalg.norm(edge_directions, axis=1)
    return reusable_edge_lengths

# *** Auxiliary function for remeshing ***
def isotropic_remesh(input_mesh, target_lengths, save_prefix, mesh_name):
    case_labels = []
    for target_length in target_lengths:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_mesh)
        ms.meshing_isotropic_explicit_remeshing(iterations=30, targetlen=pymeshlab.PureValue(target_length), reprojectflag=False)
        case_name = f"{mesh_name}_{target_length:.2f}"
        output_mesh = case_name + ".ply"
        case_labels.append(case_name)
        ms.save_current_mesh(os.path.join(save_prefix, output_mesh))
        print(f"Remeshing with target length {target_length:.1f} completed. Output saved as {output_mesh}")
    return case_labels

def single_mesh_loading(input_mesh, save_prefix, mesh_name):
    case_labels = [f"{mesh_name}"]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh)
    output_mesh = case_labels[0] + ".ply"
    ms.save_current_mesh(os.path.join(save_prefix, output_mesh))
    return case_labels


# *** AUXILIARY FUNCTIONS FOR RENDERING ***

# config
gpu_device = 'CUDA'

def set_cycles_device(device_type):
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = device_type
    bpy.context.preferences.addons['cycles'].preferences.refresh_devices()
    
    # print available devices
    print('Available devices:')
    for dev in bpy.context.preferences.addons['cycles'].preferences.devices:
        print(dev.name)

# import model (assumes everything is ok)
def import_model(filepath, material_name=None, scale_model=True, smooth=False):
    bpy.ops.wm.ply_import(filepath=filepath)
    if smooth:
        bpy.ops.object.shade_smooth()
    if scale_model:
        bpy.context.active_object.scale = (0.1, 0.1, 0.1)
    if material_name:
        material = bpy.data.materials.get(material_name)
        bpy.context.active_object.data.materials.append(material)
    return bpy.context.active_object.name

def render(output_path, camera_name=None):
    # set active camera
    if camera_name:
        bpy.context.scene.camera = bpy.data.objects[camera_name]
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

# This is the main function
def auto_render(blend_file_path, models_dir, model_name, output_dir=None):
    # print bpy version
    print('using bpy version', bpy.app.version_string)

    # check if directory exists
    if not os.path.exists(models_dir):
        print('MODELS DIRECTORY NOT FOUND:', models_dir)
        return

    output_dir = os.path.join(os.getcwd(), output_dir)

    # set output directory
    if output_dir is None:
        output_dir = models_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # set rendering device
    set_cycles_device(gpu_device)

    # check blend file
    if not os.path.exists(blend_file_path):
        print('BLEND FILE NOT FOUND:', blend_file_path)
        return
    # open blender file
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # START
    start_file = os.path.join(models_dir, model_name + '_0.ply')
    start_edges_file = os.path.join(models_dir, model_name + '_edges.ply')
    # check if file exists
    ok = os.path.exists(start_file) and os.path.exists(start_edges_file)
    if not ok:
        print('START FILES NOT FOUND:', start_file, 'or', start_edges_file)
    else:
        # import ply file into scene
        mesh_id = import_model(start_file, 'glass', True, False)
        edges_id = import_model(start_edges_file, 'Materialmetal', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_start.png'), 'Camera')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])
        bpy.data.objects.remove(bpy.data.objects[edges_id])

    # END
    end_file = os.path.join(models_dir, 'model_' + model_name + '.ply')
    end_edges_file = os.path.join(models_dir, 'model_' + model_name + '_edges.ply')
    # check if file exists
    ok = os.path.exists(end_file) and os.path.exists(end_edges_file)
    if not ok:
        print('END FILES NOT FOUND:', end_file, 'or', end_edges_file)
    else:
        # import ply file into scene
        mesh_id = import_model(end_file, 'glass', True, False)
        edges_id = import_model(end_edges_file, 'Materialmetal', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_end.png'), 'Camera')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])
        bpy.data.objects.remove(bpy.data.objects[edges_id])

    # ENERGY START
    energy_file = os.path.join(models_dir, 'energy_' + model_name + '.ply')
    # check if file exists
    if not os.path.exists(energy_file):
        print('ENERGY FILE NOT FOUND:', energy_file)
    else:
        # import ply file into scene
        mesh_id = import_model(energy_file, 'plastic', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_energy_start.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])

    # ENERGY END
    energy_file = os.path.join(models_dir, 'energy_end_' + model_name + '.ply')
    # check if file exists
    if not os.path.exists(energy_file):
        print('ENERGY FILE NOT FOUND:', energy_file)
    else:
        # import ply file into scene
        mesh_id = import_model(energy_file, 'plastic', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_energy_end.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])

    # MATCHING COLORS START
    energy_file = os.path.join(models_dir, 'batch_colors_' + model_name + '_w.ply')
    # check if file exists
    if not os.path.exists(energy_file):
        print('ENERGY FILE NOT FOUND:', energy_file)
    else:
        # import ply file into scene
        mesh_id = import_model(energy_file, 'plastic', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_batch_colors_start.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])

    # MATCHING COLORS END
    energy_file = os.path.join(models_dir, 'batch_colors_end_' + model_name + '_w.ply')
    # check if file exists
    if not os.path.exists(energy_file):
        print('ENERGY FILE NOT FOUND:', energy_file)
    else:
        # import ply file into scene
        mesh_id = import_model(energy_file, 'plastic', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_batch_colors_end.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])

    # DEFLECTIONS START
    deflections_file = os.path.join(models_dir, 'deflections_' + model_name + '.ply')
    deflections_edges_file = os.path.join(models_dir, model_name + '_edges.ply')
    # check if file exists
    ok = os.path.exists(deflections_file) and os.path.exists(deflections_edges_file)
    if not ok:
        print('DEFLECTIONS FILES NOT FOUND:', deflections_file, 'or', deflections_edges_file)
    else:
        # import ply file into scene
        mesh_id = import_model(deflections_file, 'plastic', True, False)
        edges_id = import_model(deflections_edges_file, 'plastic_grey', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_deflections_start.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])
        bpy.data.objects.remove(bpy.data.objects[edges_id])


    # DEFLECTIONS END
    deflections_file = os.path.join(models_dir, 'deflections_end_' + model_name + '.ply')
    deflections_edges_file = os.path.join(models_dir, 'model_' + model_name + '_edges.ply')
    # check if file exists
    ok = os.path.exists(deflections_file) and os.path.exists(deflections_edges_file)
    if not ok:
        print('DEFLECTIONS FILES NOT FOUND:', deflections_file, 'or', deflections_edges_file)
    else:
        # import ply file into scene
        mesh_id = import_model(deflections_file, 'plastic', True, False)
        edges_id = import_model(deflections_edges_file, 'plastic_grey', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_deflections_end.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])
        bpy.data.objects.remove(bpy.data.objects[edges_id])

    # DISPLACEMENTS
    displacements_file = os.path.join(models_dir, 'displacements_end_' + model_name + '.ply')
    displacements_edges_file = os.path.join(models_dir, 'model_' + model_name + '_edges.ply')
    # check if file exists
    ok = os.path.exists(displacements_file) and os.path.exists(displacements_edges_file)
    if not ok:
        print('DISPLACEMENTS FILES NOT FOUND:', displacements_file, 'or', displacements_edges_file)
    else:
        # import ply file into scene
        mesh_id = import_model(displacements_file, 'plastic', True, False)
        edges_id = import_model(displacements_edges_file, 'plastic_grey', True, True)

        # render
        render(os.path.join(output_dir, model_name + '_displacements.png'), 'Camera_top')

        # delete imported objects
        bpy.data.objects.remove(bpy.data.objects[mesh_id])
        bpy.data.objects.remove(bpy.data.objects[edges_id])

def render_original(blend_file_path, model_path, output_dir=None):
    # print bpy version
    print('using bpy version', bpy.app.version_string)

    # check if file exists
    if not os.path.exists(model_path):
        print('MODEL FILE NOT FOUND:', model_path)
        return
    
    output_dir = os.path.join(os.getcwd(), output_dir)

    # set output directory
    if output_dir is None:
        output_dir = Path(model_path).parent.resolve()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # check blend file
    if not os.path.exists(blend_file_path):
        print('BLEND FILE NOT FOUND:', blend_file_path)
        return
    
    # set rendering device
    set_cycles_device(gpu_device)

    # open blender file
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # ORIGINAL SMOOTH
    # import ply file into scene
    mesh_id = import_model(model_path, 'plastic_lite', True, True)

    # render
    out_file = Path(model_path).stem + '.png'
    render(os.path.join(output_dir, out_file), 'Camera')

    # delete imported objects
    bpy.data.objects.remove(bpy.data.objects[mesh_id])

def render_wireframe(blend_file_path, model_path, output_dir=None):
    # print bpy version
    print('using bpy version', bpy.app.version_string)

    # check if file exists
    if not os.path.exists(model_path):
        print('MODEL FILE NOT FOUND:', model_path)
        return
    
    output_dir = os.path.join(os.getcwd(), output_dir)

    # set output directory
    if output_dir is None:
        output_dir = Path(model_path).parent.resolve()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # check blend file
    if not os.path.exists(blend_file_path):
        print('BLEND FILE NOT FOUND:', blend_file_path)
        return
    
    # set rendering device
    set_cycles_device(gpu_device)

    # open blender file
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # WIREFRAME + SMOOTH
    # import ply file into scene
    mesh_id = import_model(model_path, 'plastic_lite', True, True)

    # activate freestyle layer
    bpy.context.scene.render.use_freestyle = True
    #set freestyle line thickness
    ls = bpy.context.scene.view_layers['ViewLayer'].freestyle_settings.linesets['LineSet']
    ls.linestyle.thickness = 1.5
    ls.select_silhouette = False
    ls.select_edge_mark = True
    # set round end caps
    ls.linestyle.caps = 'ROUND'

    # bpy.context.scene.view_layers['ViewLayer'].freestyle_settings.linesets['LineSet'].linestyle.thickness = 0.75
    # bpy.context.scene.view_layers['ViewLayer'].freestyle_settings.linesets['LineSet'].select_silhouette = False
    # bpy.context.scene.view_layers['ViewLayer'].freestyle_settings.linesets['LineSet'].select_edge_mark = True

    # render at 200% resolution
    bpy.context.scene.render.resolution_percentage = 200
    

    # mark all edges as freestyle
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # render
    out_file = Path(model_path).stem + '_wireframe.png'
    render(os.path.join(output_dir, out_file), 'Camera')

    # delete imported objects
    bpy.data.objects.remove(bpy.data.objects[mesh_id])

# Usage examples:

# render all images (start, end, energy start, energy end, deflections start, 
# deflections end, displacements)
# auto_render('render.blend', './', 'tent', output_dir='img')

# render original model
# render_original('render.blend', 'tent_original.ply', output_dir='img')
#
#
#
#
#
#
### AUXILIARY FUNCTIONS FOR STRUCTURAL HISTS ###
def extract_mesh_scalar_array(meshpath):
    # Creating pymeshlab MeshSet, loading mesh from file and selecting it.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(meshpath)
    mesh = ms.current_mesh()
    return mesh.vertex_scalar_array()

def make_colorbar(min, max, savename, colormap, extend=False, yticks_format='%.2f'):
    fig = plt.figure(figsize=(1.2,4.5))
    ax = plt.gca()
    fig.subplots_adjust(bottom=0.2, top=0.76, left=0.25)

    fraction = 1

    norm = matplotlib.colors.Normalize(vmin=min, vmax=max)

    if extend:
        ext = 'max'
    else:
        ext = 'neither'

    cb1 = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),
                                    ax=ax, fraction=fraction, shrink=1.6, aspect=20,
                                    pad=0.35, extend=ext, format=FormatStrFormatter(yticks_format))
    cb1.ax.tick_params(labelsize=18)
    ax.axis('off')

    plt.savefig(savename)
    plt.close()

def make_histograms(scalar_array_list, savename_list, vmin, vmax, cmax, no_bins, colormap, left_border_space=0.25, yticks_format='%.2f', alpha_scale=1/30, transparent=False):
    # Getting colormap.
    colormap = cm.get_cmap(colormap)
   
    # Finding max frequency.
    freq_list = []
    for idx in range(len(scalar_array_list)):
        freq, _, _ = plt.hist(scalar_array_list[idx], bins=no_bins, orientation='horizontal')
        freq_list.append(freq)
        plt.close()
    max_freq = np.array(freq_list).max()

    # Building histograms.
    for idx in range(len(scalar_array_list)):
        fig = plt.figure(figsize=(2,4.5))
        ax = plt.gca()
        fig.subplots_adjust(left=left_border_space, right=0.98)

        _, bins, patches = plt.hist(scalar_array_list[idx], bins=no_bins, range=(vmin,vmax), orientation='horizontal')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Setting axis scales and limits.
        alpha = alpha_scale * max_freq
        def forward(v):
            return alpha * v * (v <= 1) + (alpha-1) * (v > 1) + v * (v > 1)
        def backward(v):
            return v/alpha * (v <= alpha) + (1-alpha) * (v > alpha) + v * (v > 1)
        scale_fn = FuncScale(ax, (forward, backward))
        ax.set_xscale(scale_fn)
        ax.tick_params(labelsize=15)
        plt.xlim([0, max_freq])
        plt.ylim([vmin, vmax])
    
        # Scaling values to interval [0,1].
        col = bin_centers - vmin
        col /= (cmax - vmin)

        # Coloring bars according to colormap.
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', colormap(c))

        # Setting y-axis ticks format.
        ax.yaxis.set_major_formatter(FormatStrFormatter(yticks_format))

        # Removing spines e setting y-axis gray and dotted.
        ax.set_xticks(([]))
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_linewidth(1e-3)

        plt.savefig(savename_list[idx], transparent=transparent)
        plt.close()

def structural_hists_task(path, casename, vmin_energy, vmax_energy, cmax_energy, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements, device):
    prefixes = ['deflections', 'displacements', 'energy']
    methods = ['', 'end']
    file_extension = '.ply'

    for prefix in prefixes:
        if prefix == 'deflections':
            filepaths = [os.path.join(path, prefix + '_' + method + '_' + casename + file_extension) if method != '' else os.path.join(path, prefix + '_' + casename + file_extension) for method in methods]
            scalar_arrays = [extract_mesh_scalar_array(filepath) for filepath in filepaths]
        elif prefix == 'displacements':
            filepaths = [os.path.join(path, prefix + '_' + 'end' + '_' + casename + file_extension)]
            scalar_arrays = [extract_mesh_scalar_array(filepath) for filepath in filepaths]
        else:
            scalar_arrays = []
            filepaths = [os.path.join(path, casename + '_0' + file_extension), os.path.join(path, 'model_' + casename + file_extension)]
            for fp in filepaths:
                lc = StructuralCalculus(file=fp, device=device)
                scalar_arrays.append(lc.beam_energy.cpu().detach().numpy())


        if len(scalar_arrays) != 0: 
            # vmin = 0
            # vmax = np.concatenate(scalar_arrays, axis=0).max()
            savename = os.path.join(path, 'colorbar_' + prefix + '_' + casename + '.pdf')
            if prefix == 'deflections':
                cmax = vmax_deflections
                vmin = vmin_deflections
                vmax = vmax_deflections
                extend = False
            elif prefix == 'displacements':
                cmax = vmax_displacements
                vmin = vmin_displacements
                vmax = vmax_displacements
                extend = False
            else:
                # cmax = 10 * np.mean(scalar_arrays[0])
                vmin = vmin_energy
                vmax = vmax_energy
                cmax = cmax_energy
                extend = True

            # Making and saving colorbars.
            make_colorbar(vmin, vmax, savename, 'jet', extend)

            # Making and saving histograms.
            if prefix == 'displacements':
                savename_list = [os.path.join(path, 'hist_' + prefix + '_' + 'end' + '_' + casename  + '.pdf')]   
            else:
                savename_list = [os.path.join(path, 'hist_' + prefix + '_' + method + '_' + casename  + '.pdf') if method != '' else os.path.join(path, 'hist_' + prefix + '_' + casename  + '.pdf') for method in methods]      
            make_histograms(scalar_arrays, savename_list, vmin, vmax, cmax, 150, 'jet', left_border_space=0.35, transparent=False)


# *** FUNCTIONS FOR PLOTTING AND RENDERING FIGURES ***
def draw_curves(output_name):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]

    matching_percentages = []
    waste = []
    new_material = []

    # Loading curve data.
    for case in case_labels:
        matching_percentages.append(np.loadtxt(os.path.join(output_name, case, 'matching_percentages.csv'), delimiter=','))
        waste.append(np.loadtxt(os.path.join(output_name, case, 'wastages.csv'), delimiter=','))
        new_material.append(np.loadtxt(os.path.join(output_name, case, 'new_material.csv'), delimiter=','))

    max_waste = max([np.max(c) for c in waste])
    min_nm = 0
    max_nm = max([np.max(c) for c in new_material])

    # Getting y_min and y_max values for the plots.
    y_max_matching_percentages = 1.1
    y_max_waste = max_waste
    if min_nm == max_nm:
        y_max_new_material = 500.
    else:
        y_max_new_material = max_nm

    for idx, case in enumerate(case_labels):
        fig = plt.figure()
        plt.plot(matching_percentages[idx], color='gray', linewidth=2.4)
        plt.xlabel('Iteration', fontsize=22)
        plt.ylabel('Matching %', fontsize=22)
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.ylim([0, y_max_matching_percentages])
        plt.tight_layout()
        plt.savefig(os.path.join(output_name, case, 'matching_percentages.pdf'))
        plt.close(fig)

        fig = plt.figure()
        plt.plot(waste[idx], color='orange', linewidth=2.4)
        plt.xlabel('Iteration', fontsize=22)
        plt.ylabel('Waste (m)', fontsize=22)
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.ylim([0, y_max_waste])
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_name, case, 'wastages.pdf'))
        plt.close(fig)

        fig = plt.figure()
        plt.plot(new_material[idx], color='blue', linewidth=2.6)
        plt.xlabel('Iteration', fontsize=22)
        plt.ylabel('New material (m)', fontsize=22)
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.ylim([0, y_max_new_material])
        plt.tight_layout()
        plt.savefig(os.path.join(output_name, case, 'new_material.pdf'))
        plt.close(fig)

def draw_histograms(output_name):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]

    # Getting maximum edge number for axis y_lim.
    edge_numbers = []
    for idx, case in enumerate(case_labels):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply'))
        v, f = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
        e = igl.edges(f)
        edge_numbers.append(len(e))
    max_edge_number = max(edge_numbers)

    for idx, case in enumerate(case_labels):
        # Setting stock lengths & categories.
        spl = case.rsplit('_', 2)
        if spl[1] + '_' + spl[2] == 'stock_uniform':
            stock = stock_u
            stock_capacities = stock_capacities_u
            stock_name = stock_name_u
        elif spl[1] + '_' + spl[2] == 'stock_nonuniform1':
            stock = stock_nu1
            stock_capacities = stock_capacities_nu1
            stock_name = stock_name_nu1
        elif spl[1] + '_' + spl[2] == 'stock_nonuniform2':
            stock = stock_nu2
            stock_capacities = stock_capacities_nu2
            stock_name = stock_name_nu2
        categories = [f'{stock[i]:.2f}' for i in range(1, len(stock))] + ['New']

        values, colors = get_freqs_and_cols(os.path.join(output_name, case), case_labels_short[idx])

        fig = plt.figure()

        # Create bar plot
        bars = plt.bar(categories, values, color=colors)

        # Bar midpoint list.
        bar_midpoints = []

        # Add values at the top of each bar
        for bar, value in zip(bars, values):
            bar_midpoints.append(bar.get_x() + bar.get_width() / 2)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 18, str(int(value)), ha='center', color='black', fontsize=18)

        # Rotate x-axis labels by 45 degrees
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        bar_midpoints = [bp - 1 for bp in bar_midpoints]

        # Add a horizontal dashed line starting from the first bar and ending at the last bar
        plt.plot(bar_midpoints, np.array(stock_capacities) + 10, color=(0.6, 0.6, 0.6), linestyle='--', linewidth=2.5, label='Inventory')

        # Transform y-values using custom transformation function
        transformed_values = custom_transform(values)

        # Plot transformed values
        plt.bar(categories, transformed_values, color=colors, edgecolor='black', linewidth=1)

        # Add labels and title
        # plt.title(f'case:{case}')
        plt.legend()
        plt.xlabel('Length (m)', fontsize=20)
        plt.ylabel('Number of elements', fontsize=20)
        # plt.ylim((0, 0.4*max_edge_number))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.ylim((0, 1600))

        # Saving plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(os.path.join(os.path.join(output_name, case), f'{case}_matches.pdf'))
        plt.close(fig)

        start_values, end_values = get_freqs_start_end(os.path.join(output_name, case), case_labels_short[idx])

        fig = plt.figure()

        # Create bar plot
        bars = plt.bar(categories, start_values, color=colors)

        # Bar midpoint list.
        bar_midpoints = []

        # Add values at the top of each bar
        for bar, value in zip(bars, start_values):
            bar_midpoints.append(bar.get_x() + bar.get_width() / 2)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 18, str(int(value)), ha='center', color='black', fontsize=18)

        # Rotate x-axis labels by 45 degrees
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        bar_midpoints = [bp - 1 for bp in bar_midpoints]

        # Add a horizontal dashed line starting from the second bar
        plt.plot(bar_midpoints, np.array(stock_capacities) + 10, color=(0.6, 0.6, 0.6), linestyle='--', linewidth=2.5, label='Inventory')

        # Transform y-values using custom transformation function
        transformed_values = custom_transform(start_values)

        # Plot transformed values
        plt.bar(categories, transformed_values, color=colors, edgecolor='black', linewidth=1)

        # Add labels and title
        # plt.title(f'case:{case}')
        plt.legend()
        plt.xlabel('Length (m)', fontsize=20)
        plt.ylabel('Number of elements', fontsize=20)
        # plt.ylim((0, 0.4*max_edge_number))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.ylim((0, 1600))

        # Saving plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(os.path.join(os.path.join(output_name, case), f'{case}_start_matches.pdf'))
        plt.close(fig)

        cp = stock_capacities[1: ] + [0]
        df = pd.DataFrame({ 
            'Lengths': categories, 
            'Inventory': cp,
            'Input': start_values, 
            'Output': end_values 
        })

        fig = plt.figure()
        df.plot(x='Lengths', y=['Input', 'Output', 'Inventory'], color={'Input': '#1f77b4', 'Output': '#ff7f0e', 'Inventory': (0.8, 0.8, 0.8)}, kind='bar')
        # Rotate x-axis labels by 45 degrees
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Length (m)', fontsize=20)
        # plt.ylim((0, 0.6*max_edge_number))
        plt.legend(fontsize=20)
        plt.ylim((0, 4500))
        plt.tight_layout()

        plt.savefig(os.path.join(os.path.join(os.path.join(output_name, case), f'{case_labels_short[idx]}_start_end_matches.pdf')))
        plt.close(fig)

# Putting information into dictionaries
    for idxx, label in enumerate(case_labels):
        # Setting stock lengths & categories.
        spl = label.rsplit('_', 2)
        if spl[1] + '_' + spl[2] == 'stock_uniform':
            stock = stock_u
            stock_capacities = stock_capacities_u
            stock_name = stock_name_u
        elif spl[1] + '_' + spl[2] == 'stock_nonuniform1':
            stock = stock_nu1
            stock_capacities = stock_capacities_nu1
            stock_name = stock_name_nu1
        elif spl[1] + '_' + spl[2] == 'stock_nonuniform2':
            stock = stock_nu2
            stock_capacities = stock_capacities_nu2
            stock_name = stock_name_nu2

        input_lengths = {}
        output_lengths = {}
        init_class_lengths = {}
        class_lengths = {}
        base_path = os.path.join(output_name, label)
        for case in os.listdir(base_path):
            if '.pdf' not in case:
                if '_0.ply' in case:
                    input_lengths[case[:-6]] = compute_edge_lengths(os.path.join(base_path, case))   
                elif 'model_' in case and 'edges' not in case and '.jpg' not in case and '.png' not in case:
                    output_lengths[case[case.find('model_') + 6: -4]] = compute_edge_lengths(os.path.join(base_path, case))
                elif 'start_' in case and 'class_' in case:
                    casename = case[case.find('class_') + 8 : -4]
                    if casename not in init_class_lengths:
                        init_class_lengths[casename] = {} 
                    class_number = case[case.find('class_') + 6]
                    l_arr = np.loadtxt(os.path.join(os.path.join(os.path.join(output_name, label), case)))
                    if l_arr.shape == ():
                        l_arr = np.expand_dims(l_arr, axis=0)
                    init_class_lengths[casename][class_number] = l_arr
                elif 'class_' in case:
                    casename = case[case.find('class_') + 8 : -4]
                    if casename not in class_lengths:
                        class_lengths[casename] = {} 
                    class_number = case[case.find('class_') + 6]
                    l_arr = np.loadtxt(os.path.join(os.path.join(os.path.join(output_name, label), case)))
                    if l_arr.shape == ():
                        l_arr = np.expand_dims(l_arr, axis=0)
                    class_lengths[casename][class_number] = l_arr

            
        stock_lengths = []
        for idx in range(1, len(stock)):
            stock_lengths += stock_capacities[idx] * [stock[idx]]

        # Saving global lengths histograms
        for case in input_lengths:
            fig = plt.figure()

            min_len = min(min(input_lengths[case]), min(output_lengths[case]))
            max_len = max(max(input_lengths[case]), max(output_lengths[case]))

            # foo = np.random.normal(loc=1, size=100)
            # _, no_bins, _ = plt.hist(foo, bins=50, range=[0,max_len+0.1])
            # plt.clf()
            eps = 5e-4
            no_bins = np.arange(0, 4.05, step=0.05) + eps

            # Create the histogram with transparency (alpha) for overlapping
            plt.hist(stock_lengths, bins=no_bins, color=(0.8, 0.8, 0.8), edgecolor='black', linewidth=0.2, alpha=0.8, label='Inventory')
            plt.hist(input_lengths[case], bins=no_bins, color='salmon', edgecolor='black', alpha=0.5, label='Input')
            plt.hist(output_lengths[case], bins=no_bins, color='lightgreen', edgecolor='black', alpha=0.5, label='Output')

            # Add titles and labels
            # plt.title(f'case: {case}')
            plt.xlabel('Length (m)', fontsize=20)
            plt.ylabel('Frequency (# beams)', fontsize=20)
            # plt.ylim([0, 1.1*max(stock_capacities[1: ])])
            #plt.ylim([0, 0.4*max_edge_number])
            plt.xlim([0, max_len+0.1])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylim([0, 400])
            plt.xticks(fontsize=20)

            # Add a legend
            plt.legend(fontsize=20)

            # Save the plot
            os.makedirs(output_name, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(os.path.join(output_name, label), f'{case_labels_short[idxx]}_lengths.pdf')))
            plt.close(fig)

def find_min_max_deflections_energy(output_name, device):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]

    start_energy_list = []
    end_energy_list = []
    start_deflections_list = []
    end_deflections_list = []
    displacements_list = []
    for idx, case in enumerate(case_labels):
        lc = StructuralCalculus(file=os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply'), device=device)
        start_energy_list.append(lc.beam_energy.cpu().detach().numpy())
        lc = StructuralCalculus(file=os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply'), device=device)
        end_energy_list.append(lc.beam_energy.cpu().detach().numpy())
        ms1 = pymeshlab.MeshSet()
        ms1.load_new_mesh(os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply'))
        start_deflections_list.append(ms1.current_mesh().vertex_scalar_array())
        ms2 = pymeshlab.MeshSet()
        ms2.load_new_mesh(os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply'))
        end_deflections_list.append(ms2.current_mesh().vertex_scalar_array())
        v_difference = ms2.current_mesh().vertex_matrix() - ms1.current_mesh().vertex_matrix()
        displacements_list.append(np.linalg.norm(v_difference, axis=1))

    cmax_energy = 10 * np.mean([np.mean(v) for v in start_energy_list])
    vmin_energy = 0
    vmax_energy = max(global_max(start_energy_list), global_max(end_energy_list))
    vmin_deflections = 0
    vmax_deflections = max(global_max(start_deflections_list), global_max(end_deflections_list))
    vmin_displacements = 0 
    vmax_displacements = global_max(displacements_list)

    for idx, case in enumerate(case_labels):
        export_vector(map_to_color_space(start_energy_list[idx], vmin=0, vmax=cmax_energy), os.path.join(output_name, case, f'[RGBA]energy_{case_labels_short[idx]}_0a.csv'))
        export_vector(map_to_color_space(end_energy_list[idx], vmin=0, vmax=cmax_energy), os.path.join(output_name, case, f'[RGBA]energy_end_{case_labels_short[idx]}_a.csv'))

    return vmin_energy, vmax_energy, cmax_energy, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements, displacements_list, start_energy_list, end_energy_list, start_deflections_list, end_deflections_list

def render_models(output_name, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements, displacements_list):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]

    for idx, case in enumerate(case_labels):
        # Building cylinder-shpere mesh for the input.
        if platform.system() == 'Windows':
            command = "./cpp/draw_shell.exe"
        else:
            command = "./cpp/draw_shell"
        arg1 = os.path.join(output_name, 'remeshing_cases', f'{case_labels_short[idx]}.ply')
        arg2 = os.path.join(output_name, case, f'{case_labels_short[idx]}_edges.ply')
        arg3 = '0.08'
        subprocess.run(args=[command, arg1, arg2, arg3])

        # Building cylinder-shpere mesh for the output. 
        if platform.system() == 'Windows':
            command = "./cpp/draw_shell.exe"
        else:
            command = "./cpp/draw_shell"
        arg1 = os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply')
        arg2 = os.path.join(output_name, case, f'model_{case_labels_short[idx]}_edges.ply')
        arg3 = '0.08'
        subprocess.run(args=[command, arg1, arg2, arg3])

        # Building energy cylinder-shpere mesh for the input.
        if platform.system() == 'Windows':
            command = "./cpp/draw_color_shell.exe"
        else:
            command = "./cpp/draw_color_shell"
        arg1 = os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply')
        arg2 = os.path.join(output_name, case, 'edges.csv')
        arg3 = os.path.join(output_name, case, f'[RGBA]energy_{case_labels_short[idx]}_0a.csv')
        arg4 = os.path.join(output_name, case, f'energy_{case_labels_short[idx]}.ply')
        arg5 = '0.08'
        subprocess.run(args=[command, arg1, arg2, arg3, arg4, arg5])

        # Building energy cylinder-shpere mesh for the output.
        if platform.system() == 'Windows':
            command = "./cpp/draw_color_shell.exe"
        else:
            command = "./cpp/draw_color_shell"
        arg1 = os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply')
        arg2 = os.path.join(output_name, case, 'edges.csv')
        arg3 = os.path.join(output_name, case, f'[RGBA]energy_end_{case_labels_short[idx]}_a.csv')
        arg4 = os.path.join(output_name, case, f'energy_end_{case_labels_short[idx]}.ply')
        arg5 = '0.08'
        subprocess.run(args=[command, arg1, arg2, arg3, arg4, arg5])

        # Building matching-colored cylinder-shpere mesh for the input.
        if platform.system() == 'Windows':
            command = "./cpp/draw_color_shell.exe"
        else:
            command = "./cpp/draw_color_shell"
        arg1 = os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply')
        arg2 = os.path.join(output_name, case, 'edges.csv')
        arg3 = os.path.join(output_name, case, f'[RGBA]batch_colors_{case_labels_short[idx]}_0.csv')
        arg4 = os.path.join(output_name, case, f'batch_colors_{case_labels_short[idx]}.ply')
        arg5 = '0.08'
        subprocess.run(args=[command, arg1, arg2, arg3, arg4, arg5])

        # Building energy cylinder-shpere mesh for the output.
        if platform.system() == 'Windows':
            command = "./cpp/draw_color_shell.exe"
        else:
            command = "./cpp/draw_color_shell"
        arg1 = os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply')
        arg2 = os.path.join(output_name, case, 'edges.csv')
        arg3 = os.path.join(output_name, case, f'[RGBA]batch_colors_end_{case_labels_short[idx]}.csv')
        arg4 = os.path.join(output_name, case, f'batch_colors_end_{case_labels_short[idx]}.ply')
        arg5 = '0.08'
        subprocess.run(args=[command, arg1, arg2, arg3, arg4, arg5])

        # Building colored meshes (input and output) for deflections.
        ms1 = pymeshlab.MeshSet()
        ms1.load_new_mesh(os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply'))
        # defl_start = ms1.current_mesh().vertex_scalar_array()
        ms2 = pymeshlab.MeshSet()
        ms2.load_new_mesh(os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply'))
        # defl_end = ms2.current_mesh().vertex_scalar_array()
        # defl = np.array([defl_start, defl_end])
        # minval = np.min(defl)
        # maxval = np.max(defl)
        ms1.compute_color_from_scalar_per_vertex(minval=vmax_deflections, maxval=vmin_deflections)
        ms1.save_current_mesh(os.path.join(output_name, case, f'deflections_{case_labels_short[idx]}.ply'))
        ms2.compute_color_from_scalar_per_vertex(minval=vmax_deflections, maxval=vmin_deflections)
        ms2.save_current_mesh(os.path.join(output_name, case, f'deflections_end_{case_labels_short[idx]}.ply'))

        # Building colored meshes (output) for displacements.
        # v_difference = ms2.current_mesh().vertex_matrix() - ms1.current_mesh().vertex_matrix()
        # disp_norm = np.linalg.norm(v_difference, axis=1)
        out_mesh = pymeshlab.Mesh(vertex_matrix=ms2.current_mesh().vertex_matrix(), face_matrix=ms2.current_mesh().face_matrix(), v_scalar_array=displacements_list[idx])
        ms2.add_mesh(out_mesh)
        ms2.compute_color_from_scalar_per_vertex(minval=vmax_displacements, maxval=vmin_displacements)
        ms2.save_current_mesh(os.path.join(output_name, case, f'displacements_end_{case_labels_short[idx]}.ply'))

def draw_structural_hists(output_name, device, vmin_energy, vmax_energy, cmax_energy, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]

    for idx, case in enumerate(case_labels):
        structural_hists_task(os.path.join(output_name, case), case_labels_short[idx], vmin_energy, vmax_energy, cmax_energy, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements, device)

def render_all(output_name):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]
    purename = case_labels_short[0].rsplit('_', 1)[0]

    # Rendering orginal smooth shape.
    blend_file_path = os.path.join('blend_files', f'{purename}.blend')
    original_file_path = os.path.join('meshes', f'{purename}.ply')
    render_original(blend_file_path, original_file_path, output_name)

    # Rendering the results.
    for idx, case in enumerate(case_labels):
        auto_render(blend_file_path, os.path.join(output_name, case), case_labels_short[idx], output_dir=os.path.join(output_name, case))

def render_all_wireframe(output_name):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]
    purename = case_labels_short[0].rsplit('_', 1)[0]
    blend_file_path = os.path.join('blend_files', f'{purename}.blend')

    # Rendering the results.
    for idx, case in enumerate(case_labels):
        render_wireframe(blend_file_path, os.path.join(output_name, case, f'{case_labels_short[idx]}_0.ply'), output_dir=os.path.join(output_name, case))
        render_wireframe(blend_file_path, os.path.join(output_name, case, f'model_{case_labels_short[idx]}.ply'), output_dir=os.path.join(output_name, case))

def jpg_convert(output_name):
    for case in os.listdir(output_name):
        if os.path.splitext(case)[1] == '.png':
            im = Image.open(os.path.join(output_name, case))
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(output_name, os.path.splitext(case)[0] + '.jpg'), quality=95, optimize=True)
        case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
        for case in case_labels:
            for el in  os.listdir(os.path.join(output_name, case)):
                if os.path.splitext(el)[1] == '.png':
                    im = Image.open(os.path.join(output_name, case, el))
                    rgb_im = im.convert('RGB')
                    rgb_im.save(os.path.join(output_name, case, os.path.splitext(el)[0] + '.jpg'), quality=95, optimize=True)

def scatter_plot(output_name, device):
    case_labels = [case for case in os.listdir(output_name) if case != 'remeshing_cases' and '.txt' not in case and '.pdf' not in case and '.png' not in case and '.jpg' not in case]
    case_labels_short = [case.rsplit('_', 2)[0] for case in case_labels]

    # Extracting matching percentages, waste, new material, mean strain energy for each case.
    matching_percentages = []
    waste = []
    new_material = []
    mean_strain_energy = []
    min_pos = []
    for idx, case in enumerate(case_labels):
        min_pos.append(np.loadtxt(os.path.join(output_name, case, 'min_pos.csv'), delimiter=',').astype(int))
        matching_percentages.append(np.loadtxt(os.path.join(output_name, case, 'matching_percentages.csv'), delimiter=',')[min_pos[idx]])
        waste.append(np.loadtxt(os.path.join(output_name, case, 'wastages.csv'), delimiter=',')[min_pos[idx]])
        new_material.append(np.loadtxt(os.path.join(output_name, case, 'new_material.csv'), delimiter=',')[min_pos[idx]])
        lc = StructuralCalculus(file=os.path.join(output_name, case, 'model_' + case_labels_short[idx] + '.ply'), device=device)
        mean_strain_energy.append(torch.mean(lc.beam_energy).item())
    new_material = [10 + nm for nm in new_material]

    # Creating a dataframe for the results.
    df = pd.DataFrame({
        'Matching percentage': matching_percentages,
        'Waste': waste,
        'New material': new_material,
        'Mean strain energy': mean_strain_energy,
    }, index=case_labels_short)

    # Scatter plot for the results.
    fig, ax = plt.subplots()
    df.plot.scatter(x='Matching percentage', y='Waste', s='New material', c='Mean strain energy', edgecolors='black', linewidths=0.5, ax=ax)
    for k, v in df.iterrows():
        ax.annotate(k, (v['Matching percentage'], v['Waste']), xytext=(10, -5), textcoords='offset points', family='sans-serif', fontsize=10, color='darkslategrey')
    ax.set_yscale('log')
    plt.savefig(os.path.join(output_name, 'scatter_plot.pdf'))

    # Creating a dataframe for the results.
    target_lengths = [case.rsplit('_', 1)[1] for case in case_labels_short]
    df = pd.DataFrame({
        'Target length': target_lengths,    
        'Matching percentage': matching_percentages,
        'Waste': waste,
        'New material': new_material,
        'Mean strain energy': mean_strain_energy,
    }, index=case_labels_short)

    # Scatter plot for the results.
    fig, ax = plt.subplots()
    df.plot.scatter(x='Target length', y='Waste', s='New material', c='Matching percentage', edgecolors='black', linewidths=0.5, ax=ax)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_name, 'scatter_plot_alt.pdf'))