import os
import bpy
import matplotlib
import pymeshlab
import numpy as np
from scipy.ndimage import median
from optim.beam_recycle_net_optimizer import BeamRecycleNetOptimizer
from options.multistock_recycle_optimizer_options import MultistockRecycleOptimizerOptions
from utils.figure_utils import draw_curves, draw_histograms, find_min_max_deflections_energy, render_models, draw_structural_hists, render_all, render_all_wireframe, jpg_convert, scatter_plot

matplotlib.use('TKAgg')

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

step = 0.1
no_steps = 5

def single_mesh_loading(input_mesh, save_prefix, mesh_name):
    case_labels = [f"{mesh_name}"]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh)
    output_mesh = case_labels[0] + ".ply"
    ms.save_current_mesh(os.path.join(save_prefix, output_mesh))
    return case_labels

#
#
#
#
#
### MAIN FUNCTION ###
if __name__ == '__main__':
    # Parse optimizer options.
    parser = MultistockRecycleOptimizerOptions()
    options = parser.parse()

    # Creating output folder (if it does not exist).
    os.makedirs(options.output_name, exist_ok=True)

    # Selecting mesh case.
    remeshing_cases_path = os.path.join(options.output_name, 'remeshing_cases')
    os.makedirs(remeshing_cases_path)
    input_mesh_path = options.path
    mesh_name = os.path.splitext(os.path.basename(input_mesh_path))[0]
    case_labels = single_mesh_loading(input_mesh_path, remeshing_cases_path, mesh_name)

    # Zipping stock information
    stock_info = list(zip([stock_u, stock_nu1, stock_nu2], [stock_capacities_u, stock_capacities_nu1, stock_capacities_nu2], [stock_name_u, stock_name_nu1, stock_name_nu2]))

    # *** Running the optimizer. ***
    matching_percentages = []
    wastages = []
    for label in case_labels:
        for stock, stock_capacities, stock_name in stock_info:
            # Creating a directory for the current remeshing case.
            current_label_path = os.path.join(options.output_name, label) + '_' + stock_name
            os.makedirs(current_label_path)

            # Running the optimizer for the current remeshing case.
            input_mesh_path = os.path.join(remeshing_cases_path, label + '.ply')
            save_prefix = current_label_path + '/'
            lo = BeamRecycleNetOptimizer(input_mesh_path, options.lr, options.device, options.reproducible, options.seed, stock, stock_capacities, times=options.times)
            matching_percentage, waste = lo.optimize(options.n_iter, options.save, options.save_interval, label, save_prefix)

            # Saving the peformane metrics.
            matching_percentages.append(matching_percentage)
            wastages.append(waste)

    # *** Finding best cases according to the performance. ***
    best_matching_idxs = np.flatnonzero(matching_percentages == np.max(matching_percentages))
    wastages_of_best_matching = np.take(wastages, best_matching_idxs)
    best_wastages_idx = np.flatnonzero(wastages_of_best_matching == np.min(wastages_of_best_matching))
    best_matching_idxs = np.take(best_matching_idxs, best_wastages_idx)
    result_string = f"Best matching cases: {', '.join([case_labels[idx] for idx in best_matching_idxs])} with matching percentages {', '.join([str(matching_percentages[idx]) for idx in best_matching_idxs])} and wastages {', '.join([str(wastages[idx]) for idx in best_matching_idxs])}"
    f = open(os.path.join(options.output_name, 'performance_results.txt'), 'w')
    f.write(result_string)
    f.close()

    # *** Plotting performance curves. ***
    if options.curves:
        draw_curves(options.output_name)

    # *** Plotting the matching results through histograms. ***
    if options.hists:
        draw_histograms(options.output_name)

    if options.hists and options.render_models:
        # Finding min and max deflections and displacements for the color mapping.
        vmin_energy, vmax_energy, cmax_energy, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements, displacements_list, start_energy_list, end_energy_list, start_deflections_list, end_deflections_list = find_min_max_deflections_energy(options.output_name, options.device)

    # ** Building ply models for rendering. **
    if options.render_models:
        render_models(options.output_name, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements, displacements_list)

    if options.struct_hists:
       draw_structural_hists(options.output_name, options.device, vmin_energy, vmax_energy, cmax_energy, vmin_deflections, vmax_deflections, vmin_displacements, vmax_displacements)

    # *** Rendering the results. ***
    if options.render:
        render_all(options.output_name)

    if options.renderw:
        render_all_wireframe(options.output_name)

    if options.jpg:
        jpg_convert(options.output_name)