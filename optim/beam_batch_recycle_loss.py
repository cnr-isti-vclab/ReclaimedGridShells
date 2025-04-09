import torch
import numpy as np
import gurobipy as gp


# *** FOR TESTING PURPOSES ONLY: default batch and capacities. ***
default_batch = [0, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2, 2.5, 3]
default_batch_capacities = [float('inf'), 400, 400, 400, 400, 400, 400, 400, 400, 400]

default_color_map = [[128, 128, 128, 255],
                     [  0, 204, 102, 255],
                     [255, 128,   0, 255],
                     [255,   0, 127, 255],
                     [186, 125,  85, 255],
                     [149, 142, 128, 255],
                     [  0,   0, 255, 255],
                     [255,   0,   0, 255],
                     [ 0,  255,   0, 255],
                     [ 66,  26,  16, 255],
                     [200,  64,  10, 255],
                     [0,  255, 255, 255],
                     [255, 255,   0, 255],
                     [255,   0, 255, 255],
                     [255, 55, 255, 255]]


class BeamBatchRecycleLoss:
    def matching_lenghts_penalty_function(self, wastage, k=10):
        wastage[:, 0] *= self.unmatched_penalty
        masked_positive_wastage = torch.where(wastage <= 0, float('inf'), wastage)
        masked_positive_wastage[:, 0] = float('inf')
        optimization_mask = wastage - torch.min(masked_positive_wastage, dim=1).values.unsqueeze(1) == 0
        optimization_mask[:, 0] = False
        return wastage * (wastage >= 0) * optimization_mask + (torch.exp(-k*wastage)-1) * (wastage < 0)
   
    def __init__(self, mesh, batch=None, capacities=None, device='cpu', mode='differentiable', cmap=None):
        if batch is None:
            self.beam_batch = torch.tensor(default_batch, device=device)
            self.capacities = torch.tensor(default_batch_capacities, device=device)
        else:
            self.beam_batch = torch.tensor(batch, device=device)
            self.capacities = torch.tensor(capacities, device=device)
        self.device = torch.device(device)

        # Fixed edges (i.e edges between fixed vertices) are not matched.
        self.edge_is_not_fixed = torch.logical_not(mesh.vertex_is_red[mesh.edges][:, 0] * mesh.vertex_is_red[mesh.edges][:, 1])
        if cmap is None:
            self.color_map = torch.tensor(default_color_map).to(self.device)
        else:
            self.color_map = torch.tensor(cmap).to(self.device)
        self.matching_choices = None
       
        # Unmatching weight is 10*max(batch_length)
        self.unmatched_penalty = 10 * torch.max(self.beam_batch)

        # Saving stats lists.
        self.matching_percentages = []
        self.wastages = []
        self.new_material = []

        self.min_waste = float('inf')
        self.min_pos = 0

        # Saving face-to-face per vertex edge matrix. (#wedges, 2, 2)
        l = []
        for idx in range(3):
            l.append(torch.stack([torch.fliplr(mesh.faces[:, [(idx-1)%3, idx]]), mesh.faces[:, [idx, (idx+1)%3]]], dim=1))
        self.per_vertex_edges = torch.cat(l, dim=0)
    
    def get_recycle_metrics(self, query_mesh):
        with torch.no_grad():
            if not hasattr(self, 'matching_choices'):
                self.matching_ilp_solve(query_mesh)
            matching_probas = self.matching_choices
            batch_matching_frequencies = torch.sum(matching_probas, dim=0)

            # Edge wastages (in descendiong order)
            matching_idxs = torch.max(matching_probas, dim=1).indices
            beam_is_matched = (matching_idxs != 0)
            matched_beams_idxs = matching_idxs[matching_idxs.nonzero()].flatten()
            beam_is_matched_on_original_edge_matrix = torch.tensor(len(query_mesh.edge_lengths) * [False], device=self.device)
            counter = 0
            for idx in range(len(query_mesh.edge_lengths)):
                if self.edge_is_not_fixed[idx]:
                    if beam_is_matched[counter]:
                        beam_is_matched_on_original_edge_matrix[idx] = True
                    counter += 1
            sorted_wastages = torch.sort(self.beam_batch[matched_beams_idxs] - query_mesh.edge_lengths[beam_is_matched_on_original_edge_matrix], descending=True)
            min_wastege, max_wastage = None, None
            if sorted_wastages.values.numel() != 0:
                min_wastege, max_wastage = sorted_wastages.values[-1], sorted_wastages.values[0]

            lengths_dict = {}
            matched_edges_lenghts = query_mesh.edge_lengths[self.edge_is_not_fixed]
            for idx in range(len(self.beam_batch)):
                lengths_dict[idx] = matched_edges_lenghts[matching_idxs == idx]

            batch_edge_colors = torch.tensor(len(query_mesh.edges) * [[0, 0, 0, 255]], device=self.device)
            batch_edge_colors[self.edge_is_not_fixed] = self.color_map[matching_idxs].long()

            return batch_matching_frequencies, sorted_wastages.values, min_wastege, max_wastage, matching_idxs, batch_edge_colors, self.color_map, lengths_dict
        
    def get_face_angles(self, query_mesh):
        edge_endpoints = query_mesh.vertices[self.per_vertex_edges]
        edge_vectors = edge_endpoints[:, :, 1, :] - edge_endpoints[:, :, 0, :]
        norms = torch.norm(edge_vectors, dim=2)
        dots = torch.sum(edge_vectors[:, 0] * edge_vectors[:, 1], dim=1)
        return torch.acos(dots/(norms[:, 0] * norms[:, 1]))

    def check_stopping_criteria(self, check_interval=500, var_threshold=1e-5):
        if hasattr(self, 'wastages'):
            if len(self.wastages) > check_interval:
                previous_mm = np.mean(self.wastages[-check_interval-1:-1])
                current_mm = np.mean(self.wastages[-check_interval:])
                print(f'Moving average spread: {abs(1 - previous_mm/current_mm)}')
                if abs(1 - previous_mm/current_mm) < var_threshold:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def matching_ilp_solve(self, query_mesh):
        # Computing wastage matrix.
        wastages = self.beam_batch - query_mesh.edge_lengths[self.edge_is_not_fixed].unsqueeze(1)
        wastages[:, 0] *= -self.unmatched_penalty
        wastages = wastages.detach().cpu().numpy()
        capacities = self.capacities.detach().cpu().numpy()
        negative_wastages_mask = (wastages < 0)

        no_stock_beams = len(self.beam_batch)
        no_beams = len(query_mesh.edge_lengths[self.edge_is_not_fixed])

        # Creating gurobi model.
        try:
            print('Building gurobi model...')
            gurobi_model = gp.Model("matching_ilp")

            # Creating optimization variables.
            self.matching_vars = gurobi_model.addMVar(shape=(no_beams, no_stock_beams), vtype=gp.GRB.BINARY, name='matching_vars')

            # Adding constraints.
            gurobi_model.addConstr(self.matching_vars * negative_wastages_mask == 0, name='positive_wastage_constraints')
            gurobi_model.addConstr(self.matching_vars.sum(axis=1) == 1, name='matching_constraints')
            gurobi_model.addConstr(self.matching_vars[:, 1:].sum(axis=0) - capacities[1:] <= 0, name='capacity_constraints')

            # Adding objective function.
            gurobi_model.setObjective((self.matching_vars * wastages).sum(), gp.GRB.MINIMIZE)
    
            # Solving optimization problem.
            print('Solving optimization problem...')
            gurobi_model.optimize()
    
            status = gurobi_model.status
            if status == gp.GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is unbounded')
                raise Exception()
            elif status == gp.GRB.Status.OPTIMAL:
                print('The optimal objective is %g' % gurobi_model.objVal)
            elif status != gp.GRB.Status.INF_OR_UNBD and status != gp.GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)
                raise Exception()

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError as e:
            print('Encountered an attribute error', e)

        except Exception as e:
            print('Not Optimized!', e)

        # Getting model solution (and casting matching_matrix binary variables to int).
        matching_matrix = np.round(self.matching_vars.X).astype(int)

        # Converting solution to torch tensor.
        self.matching_choices = torch.tensor(matching_matrix, device=self.device)
    
    def __call__(self, query_mesh, mode):
        matching_choices = self.matching_choices
        length_differences = self.beam_batch - query_mesh.edge_lengths[self.edge_is_not_fixed].unsqueeze(1)
        length_differences[:, 0] *= -1

        # Updating stats: matching percentages.
        matching_freq = torch.sum(matching_choices, dim=0)
        self.matching_percentages.append(1 - (matching_freq[0]/torch.sum(matching_freq)).item())

        # Updating stats: wastages and new material.
        matching_idxs = torch.max(matching_choices, dim=1).indices
        beam_is_matched = (matching_idxs != 0)
        matched_beams_idxs = matching_idxs[matching_idxs.nonzero()].flatten()
        beam_is_matched_on_original_edge_matrix = torch.tensor(len(query_mesh.edge_lengths) * [False], device=self.device)
        beam_is_not_matched_on_original_edge_matrix = torch.tensor(len(query_mesh.edge_lengths) * [False], device=self.device)
        counter = 0
        for idx in range(len(query_mesh.edge_lengths)):
            if self.edge_is_not_fixed[idx]:
                if beam_is_matched[counter]:
                    beam_is_matched_on_original_edge_matrix[idx] = True
                else:
                    beam_is_not_matched_on_original_edge_matrix[idx] = True
                counter += 1
        self.wastages.append(torch.sum(self.beam_batch[matched_beams_idxs] - query_mesh.edge_lengths[beam_is_matched_on_original_edge_matrix]).item())
        self.new_material.append(torch.sum(query_mesh.edge_lengths[beam_is_not_matched_on_original_edge_matrix]).item())

        # Computing loss.
        penalty_per_match = self.matching_lenghts_penalty_function(length_differences)
        matching_term = torch.sum(torch.sum(matching_choices * penalty_per_match, dim=1))
        return matching_term