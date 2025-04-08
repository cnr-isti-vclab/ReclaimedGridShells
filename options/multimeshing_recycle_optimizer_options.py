import argparse

class MultimeshingRecycleOptimizerOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for RecycleOptimizer.')
        self.parser.add_argument('--device', dest='device', type=str, default='cpu', help='Device for pytorch')
        self.parser.add_argument('--meshpath', dest='path', type=str, help='Path to starting mesh')
        self.parser.add_argument('--niter', dest='n_iter', type=int, default=100, help='Number of optimization steps')
        self.parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Optimization learning rate')
        self.parser.add_argument('--save', dest='save', action='store_true', default=True, help='True if mesh saving is required')
        self.parser.add_argument('--saveinterval', dest='save_interval', type=int, default=25, help='Mesh saving iterations interval')
        self.parser.add_argument('--reproducible', dest='reproducible', action='store_true', default=False, help='If we want to have reproducible results (and slower computation).')
        self.parser.add_argument('--randseed', dest='seed', type=int, default=42, help='Random seed for reproducibility.')
        self.parser.add_argument('--outputname', dest='output_name', type=str, default='output', help='Label for ouput folder')
        self.parser.add_argument('--stock', dest='stock', type=str, default='uniform', help='Stock name (uniform, nonuniform1, nonuniform2)')
        self.parser.add_argument('--curves', dest='curves', action='store_true', default=False, help='If true, computes performance curves for the results')
        self.parser.add_argument('--hists', dest='hists', action='store_true', default=False, help='If true, computes histograms for the results')
        self.parser.add_argument('--structhists', dest='struct_hists', action='store_true', default=False, help='If true, computes structural histograms for the results')
        self.parser.add_argument('--rendermodels', dest='render_models', action='store_true', default=False, help='If true, generates ply models to be rendered')
        self.parser.add_argument('--render', dest='render', action='store_true', default=False, help='If true, produces renderized images with bpy')
        self.parser.add_argument('--renderw', dest='renderw', action='store_true', default=False, help='If true, produces renderized wireframes with bpy')
        self.parser.add_argument('--scatter', dest='scatter', action='store_true', default=False, help='If true, computes scatter plot for the results')
        self.parser.add_argument('--times', dest='times', action='store_true', default=False, help='If true, computes iteration mean times')
        self.parser.add_argument('--jpg', dest='jpg', action='store_true', default=False, help='If true, comnverts png images to jpg')

    def parse(self):
        return self.parser.parse_args()