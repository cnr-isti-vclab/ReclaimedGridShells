import torch
import math
import time
import numpy as np
import matplotlib.cm as cm
from optim.structural_calculus import StructuralCalculus
from models.layers.featured_mesh import FeaturedMesh
from optim.beam_batch_recycle_loss import BeamBatchRecycleLoss
from models.networks import DisplacerNet
from utils.utils import save_mesh, map_to_color_space, export_vector
from utils.mesh_sampling_utils import sample_points_from_mesh
from chamferdist import ChamferDistance
from utils.reproducible_chamfer_distance import repr_chamfer_distance

def angle_penalty(angles, threshold, k=100):
    return torch.sum(torch.exp(-k*(angles[angles < threshold] - threshold[angles < threshold])) - 1)

def area_penalty(areas, threshold_min, threshold_max, k=100):
    return torch.sum(torch.exp(-k*(areas[areas < threshold_min] - threshold_min[areas < threshold_min])) - 1) + torch.sum(torch.exp(k*(areas[areas > threshold_max] - threshold_max[areas > threshold_max])) - 1)
    
def edge_length_penalty(lengths, threshold, k=100):
    return torch.sum(torch.exp(k*(lengths[lengths > threshold] - threshold[lengths > threshold])) - 1)

class BeamRecycleNetOptimizer:

    def __init__(self, file, lr, device, reproducible, seed, stock=None, stock_capacity=None, times=False):
        self.reproducible = reproducible
        self.seed = seed

        # Setting randomization seed (only if we want reproducible results).
        if self.reproducible:
            torch.manual_seed(self.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False

        self.initial_mesh = FeaturedMesh(file=file, device=device)
        self.device = device
        self.lr = lr

        # Computing max and min area threshold.
        self.initial_mesh.make_on_mesh_shared_computations()
        initial_areas = self.initial_mesh.face_areas
        mean_area = torch.mean(initial_areas)
        # min_area, max_area = torch.min(initial_areas), torch.max(initial_areas)
        self.min_area_threshold = torch.where(initial_areas / mean_area < 0.5, initial_areas, 0.5 * mean_area)
        self.max_area_threshold = torch.where(initial_areas / mean_area > 2, initial_areas, 2 * mean_area)

        # Creating colormap.
        unmatched_color = np.array([128, 128, 128, 255])
        colors = cm.jet(np.linspace(0, 1, len(stock)-1), bytes=True)
        colors = np.vstack((unmatched_color, colors))
    
        # Setting beam recycle loss
        self.recycle_loss_fn = BeamBatchRecycleLoss(mesh=self.initial_mesh, device=device, batch=stock, capacities=stock_capacity, cmap=colors)

        # Computing wedge threshold angles.
        self.wedge_threshold_angles = torch.where(self.recycle_loss_fn.get_face_angles(self.initial_mesh) >= math.radians(40), math.radians(40), self.recycle_loss_fn.get_face_angles(self.initial_mesh))

        # Computing edge length thresholds.
        self.edge_length_thresholds = torch.where(self.initial_mesh.edge_lengths > max(self.recycle_loss_fn.beam_batch), self.initial_mesh.edge_lengths, max(self.recycle_loss_fn.beam_batch)-1e-2)

        # Computing numerber of samples according to total surface area (150 points per square meter, totalpoints/4 for bouindary + crease edges)
        self.num_inner_samples = int(150 * torch.sum(self.initial_mesh.face_areas))
        self.num_boundary_samples = int(self.num_inner_samples / 4)

         # Initializing times lists.
        self.times = times
        if self.times:
            self.iteration_times = []
            self.stock_aligner_times = []
            self.structural_optimizer_times = []
            self.ilp_times = []

        # Setting 10 decimal digits tensor display.
        torch.set_printoptions(precision=10)

        self.device = torch.device(device)

        self.make_optimizer()

    def make_optimizer(self):
        # Building parameters matrix.
        self.displacement_matrix = torch.zeros(len(self.initial_mesh.vertices[torch.logical_not(self.initial_mesh.vertex_is_red)]), 3, device=self.device, requires_grad=True)

        # Computing mesh input features.
        self.initial_mesh.compute_mesh_input_features()

        # Building structural learning model.
        self.structural_displacer_net = DisplacerNet(k=16, mesh=self.initial_mesh, in_feature_mask=self.initial_mesh.feature_mask).to(self.device)

        # Building optimizers.
        self.beam_recycle_optimizer = torch.optim.Adam([ self.displacement_matrix ], lr=0.001, eps=1e-3)
        self.structural_optimizer = torch.optim.Adam(self.structural_displacer_net.parameters(), lr=self.lr, eps=1e-3)

        # Building Chamfer Distance function.
        self.chamfer_dist_fn = ChamferDistance()
        self.chamfer_dist_scale_is_set = False
        self.face_area_variance_scale_is_set = False

        # Building structural simulator.
        self.structural_calculus = StructuralCalculus(device=self.device, mesh=self.initial_mesh)
        self.vmax = None

        # Computes edge-face adjacency matrix.
        self.initial_mesh.make_adjacency_matrices()

        # Creating structural reference mesh.
        self.reference_mesh = self.initial_mesh

        # Creating strain energy list.
        self.strain_energy_list = []

        # Computing initial matching matrix.
        
        self.compute_beam_matching_ilp()

    def optimize(self, n_iter, save, save_interval, save_label, save_prefix=''):
        batch_matching_frequencies, _, _, _, _, batch_edge_colors, color_map, per_category_matched_beams_lenghts = self.recycle_loss_fn.get_recycle_metrics(self.initial_mesh)

        # Exporting data.
        export_vector(batch_matching_frequencies, save_prefix + 'batch_matching_frequencies_start_' + save_label + '.csv')
        export_vector(batch_edge_colors, save_prefix + '[RGBA]batch_colors_start_' + save_label + '.csv')
        export_vector(color_map, save_prefix + 'color_map_start_' + save_label + '.csv')
        for idx in range(len(per_category_matched_beams_lenghts)):
            if idx in per_category_matched_beams_lenghts:
                export_vector(per_category_matched_beams_lenghts[idx], save_prefix + 'start_matched_beam_lenghts_on_class_' + str(idx) + '_' + save_label + '.csv')

        if self.times:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
        for current_iteration in range(n_iter):
            if self.times:
                start_time.record()
            if save:
                with torch.no_grad():
                    offset = torch.zeros(self.initial_mesh.vertices.shape, device=self.device)
                    offset[torch.logical_not(self.initial_mesh.vertex_is_red), :] = self.displacement_matrix
                    if current_iteration != 0:
                        displacements = self.structural_displacer_net(self.initial_mesh.input_features)
                        iteration_mesh = self.initial_mesh.update_verts(displacements + offset)
                    else:
                        iteration_mesh = self.initial_mesh.update_verts(offset)
                    self.structural_calculus(iteration_mesh, loss_type='mean_beam_energy')

                # Saving current iteration mesh if requested.
                if current_iteration % save_interval == 0:
                    filename = save_prefix + save_label + '_' + str(current_iteration) + '.ply'
                    quality = torch.norm(self.structural_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    save_mesh(iteration_mesh, filename, v_quality=quality.unsqueeze(1))
                    batch_matching_frequencies, _, min_wastage, max_wastage, _, batch_edge_colors, color_map, per_category_matched_beams_lenghts = self.recycle_loss_fn.get_recycle_metrics(iteration_mesh)
                    export_vector(batch_matching_frequencies, save_prefix + 'batch_matching_frequencies_' + save_label + '_' + str(current_iteration) + '.csv')
                    export_vector(batch_edge_colors, save_prefix + '[RGBA]batch_colors_' + save_label + '_' + str(current_iteration) + '.csv')
                    if self.vmax is None:
                        self.vmax = 10 * torch.mean(self.structural_calculus.beam_energy).item()
                    export_vector(map_to_color_space(self.structural_calculus.beam_energy.detach().cpu(), vmin=0, vmax=self.vmax), save_prefix + '[RGBA]energy_' + save_label + '_' + str(current_iteration) + '.csv')

            # Computing iterations of matching optimizer.
            if current_iteration == 0:
                pass
            else:
                if current_iteration % 20 == 0:
                    self.compute_beam_matching_ilp()
                    self.chamfer_dist_scale_is_set = False
                    self.face_area_variance_scale_is_set = False

            if self.times:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            self.compute_beam_recycle_iterations(2)
            if self.times:
                end.record()
                torch.cuda.synchronize()
                self.stock_aligner_times.append(start.elapsed_time(end))
                start.record()
            self.compute_structural_iterations(1)
            if self.times:
                end.record()
                torch.cuda.synchronize()
                self.structural_optimizer_times.append(start.elapsed_time(end))
            if self.recycle_loss_fn.check_stopping_criteria() == True:
                print(f"Stopping creteria met at iteration {current_iteration}.")
                break
            if self.times:
                end_time.record()
                torch.cuda.synchronize()
                self.iteration_times.append(start_time.elapsed_time(end_time))

        # Redoing matching iterations.
        self.compute_beam_matching_ilp()
        
        batch_matching_frequencies, _, min_wastage, max_wastage, _, batch_edge_colors, color_map, per_category_matched_beams_lenghts = self.recycle_loss_fn.get_recycle_metrics(iteration_mesh)
        wastages = torch.stack([min_wastage, max_wastage], dim=0)

        # Exporting data.
        # self.recycle_loss_fn.save_curves(save_prefix=save_prefix)
        if self.times:
            self.ilp_times = torch.mean(torch.tensor(self.ilp_times)).unsqueeze(0)
            self.iteration_times = torch.mean(torch.tensor(self.iteration_times)).unsqueeze(0)
            self.stock_aligner_times = torch.mean(torch.tensor(self.stock_aligner_times)).unsqueeze(0)
            self.structural_optimizer_times = torch.mean(torch.tensor(self.structural_optimizer_times)).unsqueeze(0)
        self.strain_energy_list = torch.tensor(self.strain_energy_list)
        filename = save_prefix + 'reference_' + save_label + '.ply'
        save_mesh(self.reference_mesh, filename)
        filename = save_prefix + 'model_' + save_label + '.ply'
        save_mesh(iteration_mesh, filename, v_quality=quality.unsqueeze(1))
        np.savetxt(save_prefix + 'matching_percentages.csv', self.recycle_loss_fn.matching_percentages, delimiter=',')
        np.savetxt(save_prefix + 'wastages.csv', self.recycle_loss_fn.wastages, delimiter=',')
        np.savetxt(save_prefix + 'new_material.csv', self.recycle_loss_fn.new_material, delimiter=',')
        if self.times:
            export_vector(self.ilp_times, save_prefix + 'mean_ilp_time_' + save_label + '.csv')
            export_vector(self.iteration_times, save_prefix + 'mean_iteration_time_' + save_label + '.csv')
            export_vector(self.stock_aligner_times, save_prefix + 'stock_aligner_mean_time_' + save_label + '.csv')
            export_vector(self.structural_optimizer_times, save_prefix + 'structural_optimizer_mean_time_' + save_label + '.csv')
        export_vector(iteration_mesh.face_areas, save_prefix + 'face_areas_' + save_label + '.csv')
        export_vector(batch_matching_frequencies, save_prefix + 'batch_matching_frequencies_' + save_label + '.csv')
        export_vector(batch_edge_colors, save_prefix + '[RGBA]batch_colors_end_' + save_label + '.csv')
        export_vector(self.strain_energy_list, save_prefix + 'energy_' + save_label + '.csv')
        export_vector(map_to_color_space(self.structural_calculus.beam_energy.detach().cpu(), vmin=0, vmax=self.vmax), save_prefix + '[RGBA]energy_end_' + save_label + '.csv')
        export_vector(color_map, save_prefix + 'color_map_' + save_label + '.csv')
        export_vector(wastages, save_prefix + 'wastages_' + save_label + '.csv')
        export_vector(self.initial_mesh.edges, save_prefix + 'edges.csv')
        np.savetxt(save_prefix + 'min_pos.csv', np.array([self.recycle_loss_fn.min_pos]), delimiter=',')
        for idx in range(len(per_category_matched_beams_lenghts)):
            if idx in per_category_matched_beams_lenghts:
                export_vector(per_category_matched_beams_lenghts[idx], save_prefix + 'matched_beam_lenghts_on_class_' + str(idx) + '_' + save_label + '.csv')
        # plt.hist(iteration_mesh.face_areas.cpu().numpy(), bins=100)
        # plt.xlim(0.5*self.initial_mesh.face_areas.min().item(), 4*self.initial_mesh.face_areas.max().item())
        # plt.savefig(save_prefix + 'face_areas_histogram_' + save_label + '.pdf')
        return self.recycle_loss_fn.matching_percentages[-1], self.recycle_loss_fn.wastages[-1]

    def compute_beam_matching_ilp(self):
        # Generating displaced mesh (torch.no_grad() since we are optimizing matching weights and not point coords).
        with torch.no_grad():
            offset = torch.zeros(self.reference_mesh.vertices.shape, device=self.device)
            offset[torch.logical_not(self.reference_mesh.vertex_is_red), :] = self.displacement_matrix
            iteration_mesh = self.reference_mesh.update_verts(offset)
            iteration_mesh.make_on_mesh_shared_computations()

        if self.times:
            start = time.time()
        self.recycle_loss_fn.matching_ilp_solve(iteration_mesh)
        if self.times:
            end = time.time()
            self.ilp_times.append(end-start)

    def compute_beam_matching_iterations(self, niter, initial=False): 
        # Generating displaced mesh (torch.no_grad() since we are optimizing matching weights and not point coords).
        with torch.no_grad():
            offset = torch.zeros(self.reference_mesh.vertices.shape, device=self.device)
            offset[torch.logical_not(self.reference_mesh.vertex_is_red), :] = self.displacement_matrix
            iteration_mesh = self.reference_mesh.update_verts(offset)
            iteration_mesh.make_on_mesh_shared_computations()

        # iter_counter = 0
        for iteration in range(niter):
            # Putting grads to None.
            self.beam_matching_optimizer.zero_grad(set_to_none=True)

            # Computing recycle loss.
            loss, choices = self.recycle_loss_fn(iteration_mesh, mode='matching_optimizer')
            print("\t Matching it.", iteration, "\t Beams Recycle:", float(loss))

            # Computing gradients.
            loss.backward()

            # Updating optimizer.
            self.beam_matching_optimizer.step()

            # Saving histogram of matching matrix.
            if (initial == True) and (iteration <= 100 or iteration % 100 == 0):
                self.recycle_loss_fn.save_histograms(iteration_mesh, iteration)

    def compute_beam_recycle_iterations(self, niter):

        if not self.chamfer_dist_scale_is_set:
            self.chamfer_distance_scale = 0
        if not self.face_area_variance_scale_is_set:
            self.face_area_variance_scale = 0

        for iteration in range(niter):
            # Putting grads to None.
            self.beam_recycle_optimizer.zero_grad(set_to_none=True)

            # Generating displaced mesh.
            offset = torch.zeros(self.reference_mesh.vertices.shape, device=self.device)
            offset[torch.logical_not(self.reference_mesh.vertex_is_red), :] = self.displacement_matrix
            iteration_mesh = self.reference_mesh.update_verts(offset)
            iteration_mesh.make_on_mesh_shared_computations()
            
            # Computing point samples from meshes
            reference_mesh_samples = sample_points_from_mesh(self.reference_mesh, self.num_inner_samples, self.num_boundary_samples, self.reproducible, self.seed, device=self.device).unsqueeze(0)
            iteration_mesh_samples = sample_points_from_mesh(iteration_mesh, self.num_inner_samples, self.num_boundary_samples, self.reproducible, self.seed, device=self.device).unsqueeze(0)

            # Computing bidirectional Chamfer Distance
            if self.reproducible:
                chamfer_dist = repr_chamfer_distance(reference_mesh_samples.squeeze(0), iteration_mesh_samples.squeeze(0), simmetrical=True)
            else:
                chamfer_dist = self.chamfer_dist_fn(reference_mesh_samples, iteration_mesh_samples, bidirectional=True)
            print("\t Recycle it.", iteration, "\t Chamfer distance:", float(chamfer_dist))

            # Computing face area penalty.
            face_area_penalty_loss = area_penalty(iteration_mesh.face_areas, self.min_area_threshold, self.max_area_threshold)
            print("\t Recycle it.", iteration, "\t Area penalty:", float(face_area_penalty_loss))

            # Computing angle penalty.
            angles = self.recycle_loss_fn.get_face_angles(iteration_mesh)
            angle_penalty_loss = angle_penalty(angles, threshold=self.wedge_threshold_angles)
            print("\t Recycle it.", iteration, "\t Angle penalty:", float(angle_penalty_loss))

            # Computing edge length penalty.
            lengths = iteration_mesh.edge_lengths
            edge_length_penalty_loss = edge_length_penalty(lengths, threshold=self.edge_length_thresholds)
            print("\t Recycle it.", iteration, "\t Edge length penalty:", float(edge_length_penalty_loss))
            
            # Computing recycle loss.
            recycle_loss = self.recycle_loss_fn(iteration_mesh, mode='structural_optimizer')
            print("\t Recycle it.", iteration, "\t Beams Recycle:", float(recycle_loss))

            # Setting Chamfer Distance scale (if Chamfer Distance is not zero).
            if chamfer_dist != 0 and not self.chamfer_dist_scale_is_set:
                with torch.no_grad():
                    self.chamfer_distance_scale = float(recycle_loss / chamfer_dist)
                self.chamfer_dist_scale_is_set = True

            # Summing loss components.
            loss = recycle_loss + self.chamfer_distance_scale * chamfer_dist + edge_length_penalty_loss + angle_penalty_loss + face_area_penalty_loss # + face_area_variance_penalty_loss + self.face_area_variance_scale * face_area_variance

            # Computing gradients.
            loss.backward()

            # Updating optimizer.
            self.beam_recycle_optimizer.step()
            # self.scheduler.step(loss)

    def compute_structural_iterations(self, niter):
        for iteration in range(niter):
            # Putting grads to None.
            self.structural_optimizer.zero_grad(set_to_none=True)

            # Computing mesh displacements via net model.
            displacements = self.structural_displacer_net(self.initial_mesh.input_features)

            # Computing correction terms (as constants).
            with torch.no_grad():
                offset = torch.zeros(self.reference_mesh.vertices.shape, device=self.device)
                offset[torch.logical_not(self.reference_mesh.vertex_is_red), :] = self.displacement_matrix

            # Generating current iteration displaced mesh.
            iteration_mesh = self.initial_mesh.update_verts(displacements + offset)
            iteration_mesh.make_on_mesh_shared_computations()

            # Computing structural loss.
            loss = self.structural_calculus(iteration_mesh, loss_type='mean_beam_energy')
            print("\t Structural it.", iteration, "\t Structural Loss:", float(loss))
            self.strain_energy_list.append(loss.item())

            # Computing gradients.
            loss.backward()

            # Updating optimizer.
            self.structural_optimizer.step()

            # Deleting grad history in involved tensors.
            self.structural_calculus.clean_attributes()

        # Updating reference mesh.
        with torch.no_grad():
            displacements = self.structural_displacer_net(self.initial_mesh.input_features)
            self.reference_mesh = self.initial_mesh.update_verts(displacements)