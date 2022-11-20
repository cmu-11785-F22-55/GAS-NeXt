from base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str,
                            default='../results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='val',
                            help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=10000,
                            help='how many test images to run')
        parser.add_argument('--n_samples', type=int,
                            default=4, help='#samples, first one encode, others random')
        parser.add_argument('--no_encode', action='store_true',
                            help='do not produce encoded image')
        parser.add_argument('--sync', action='store_true',
                            help='use the same latent code for different input images')
        parser.add_argument('--aspect_ratio', type=float,
                            default=1.0, help='aspect ratio for the results')
        self.isTrain = False
        return parser
