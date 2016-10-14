#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import shlex
import sys
import warnings
import copy
import imp
import ast

nodes = imp.load_source('', 'steps/ctc/nnet2/components.py')
nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for RNN's creation and training",
                                     epilog="See steps/ctc/nnet2_train.sh for example.")

    # Only one of these arguments can be specified, and one of them has to
    # be compulsarily specified
    feat_group = parser.add_mutually_exclusive_group(required = True)
    feat_group.add_argument("--feat-dim", type=int,
                            help="Raw feature dimension, e.g. 13")
    feat_group.add_argument("--feat-dir", type=str,
                            help="Feature directory, from which we derive the feat-dim")

    # only one of these arguments can be specified
    ivector_group = parser.add_mutually_exclusive_group(required = False)
    ivector_group.add_argument("--ivector-dim", type=int,
                                help="iVector dimension, e.g. 100", default=0)
    ivector_group.add_argument("--ivector-dir", type=str,
                                help="iVector dir, which will be used to derive the ivector-dim  ", default=None)

    num_target_group = parser.add_mutually_exclusive_group(required = True)
    num_target_group.add_argument("--num-targets", type=int,
                                  help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    num_target_group.add_argument("--ali-dir", type=str,
                                  help="alignment directory, from which we derive the num-targets")
    num_target_group.add_argument("--tree-dir", type=str,
                                  help="directory with final.mdl, from which we derive the num-targets")

    # CNN options
    parser.add_argument('--cnn.layer', type=str, action='append', dest = "cnn_layer",
                        help="CNN parameters at each CNN layer, e.g. --filt-x-dim=3 --filt-y-dim=8 "
                        "--filt-x-step=1 --filt-y-step=1 --num-filters=256 --pool-x-size=1 --pool-y-size=3 "
                        "--pool-z-size=1 --pool-x-step=1 --pool-y-step=3 --pool-z-step=1, "
                        "when CNN layers are used, no LDA will be added", default = None)
    parser.add_argument("--cnn.bottleneck-dim", type=int, dest = "cnn_bottleneck_dim",
                        help="Output dimension of the linear layer at the CNN output "
                        "for dimension reduction, e.g. 256."
                        "The default zero means this layer is not needed.", default=0)
    parser.add_argument("--cnn.cepstral-lifter", type=float, dest = "cepstral_lifter",
                        help="The factor used for determining the liftering vector in the production of MFCC. "
                        "User has to ensure that it matches the lifter used in MFCC generation, "
                        "e.g. 22.0", default=22.0)

    # General neural network options
    parser.add_argument("--splice-indexes", type=str, required = True,
                        help="Splice indexes at each layer, e.g. '-3,-2,-1,0,1,2,3' "
                        "If CNN layers are used the first set of splice indexes will be used as input "
                        "to the first CNN layer and later splice indexes will be interpreted as indexes "
                        "for the TDNNs.")
    parser.add_argument("--add-lda", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="If \"true\" an LDA matrix computed from the input features "
                        "(spliced according to the first set of splice-indexes) will be used as "
                        "the first Affine layer. This affine layer's parameters are fixed during training. "
                        "If --cnn.layer is specified this option will be forced to \"false\".",
                        default=False, choices = ["false", "true"])
    parser.add_argument("--lda-dim", type=int, default=0,
                        help="dimension of lda transform.")

    parser.add_argument("--include-log-softmax", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="add the final softmax layer ", default=False, choices = ["false", "true"])

    parser.add_argument("--xent-regularize", type=float,
                        help="For chain models, if nonzero, add a separate output for cross-entropy "
                        "regularization (with learning-rate-factor equal to the inverse of this)",
                        default=0.0)
    parser.add_argument("--xent-separate-forward-affine", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="if using --xent-regularize, gives it separate last-but-one weight matrix",
                        default=False, choices = ["false", "true"])
    parser.add_argument("--final-layer-normalize-target", type=float,
                        help="RMS target for final layer (set to <1 if final layer learns too fast",
                        default=1.0)
    parser.add_argument("--subset-dim", type=int, default=0,
                        help="dimension of the subset of units to be sent to the central frame")

    parser.add_argument("--self-repair-scale", type=float,
                        help="A non-zero value activates the self-repair mechanism in the sigmoid and tanh non-linearities of the LSTM", default=None)

    parser.add_argument("--use-presoftmax-prior-scale", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="if true, a presoftmax-prior-scale is added",
                        choices=['true', 'false'], default = True)
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    parser.add_argument("--batch-normalize", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="If \"true\" use batch normalize before nonlinearity(only config ReLU now).",
                        default=False, choices = ["false", "true"])


    parser.add_argument("--objective-type", type=str,
                        help = "the type of objective; i.e. CTC or linear",
                        default="CTC", choices = ["linear", "CTC"])

    parser.add_argument("--affine-type", type=str,
                        help = "the type of active; i.e. AffineComponent or NaturalGradientAffineComponent",
                        default="native", choices = ["native", "natural"])
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="dimension of DNN layers")
    parser.add_argument("--active-type", type=str,
                        help = "the type of active; i.e. ReLU or Sigmoid",
                        default="relu", choices = ["relu", "sigmoid"])

    parser.add_argument("--model.type", type=str, dest = "model_type",
                        help="model type, google|DS2|FT",
                        default="goole", choices = ["google", "DS2", "FT"])
    parser.add_argument("--model.bidirectional", type=str, action=nnet3_train_lib.StrToBoolAction, dest = 'rnn_bidirectional',
                        help="bidirectional.", default=True, choices = ["false", "true"])
    parser.add_argument("--model.rnn-mode", type=int, dest = "rnn_mode",
                        help="CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3", default = 2)
    parser.add_argument("--model.rnn-first", type=int, dest = "rnn_first", action=nnet3_train_lib.StrToBoolAction,
                        help="", default = True, choices = ["false", "true"])
    parser.add_argument("--model.rnn-layers", type=int, dest = "rnn_layers",
                        help="RNN layers number", default = 2)
    parser.add_argument("--model.rnn-max-seq-length", type=int, dest = "rnn_max_seq_length",
                        help="RNN layer max seq length", default = 1000)
    parser.add_argument("--model.cudnn-layers", type=int, dest = "cudnn_layers",
                        help="RNN layers In one CuDNNRecurrentComponent", default = 1)
    parser.add_argument("--model.cell-dim", type=int, dest = "rnn_cell_dim",
                        help="RNN layers number", default = 512)
    parser.add_argument("--model.param-stddev", type=float, dest = "param_stddev",
                        help="RNN params stddev", default = 0.02)
    parser.add_argument("--model.bias-stddev", type=float, dest = "bias_stddev",
                        help="RNN bias stddev", default = 0.2)
    parser.add_argument("--model.clipping-threshold", type=float, dest = "clipping_threshold",
                        help="clipping threshold value", default = 30.0)
    parser.add_argument("--model.norm-based-clipping", type=str, action=nnet3_train_lib.StrToBoolAction,
                        dest = 'norm_based_clipping',
                        help="norm_based_clipping.", default=True, choices = ["false", "true"])
    parser.add_argument("--dropout-proportion", type=float, dest = "dropout_proportion",
                        help="dropout proportion value", default = 0.0)

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.feat_dir is not None:
        args.feat_dim = nnet3_train_lib.GetFeatDim(args.feat_dir)

    if args.ali_dir is not None:
        args.num_targets = nnet3_train_lib.GetNumberOfLeaves(args.ali_dir)
    elif args.tree_dir is not None:
        args.num_targets = chain_lib.GetNumberOfLeaves(args.tree_dir)

    if args.ivector_dir is not None:
        args.ivector_dim = nnet3_train_lib.GetIvectorDim(args.ivector_dir)

    if not args.feat_dim > 0:
        raise Exception("feat-dim has to be postive")

    if not args.num_targets > 0:
        print(args.num_targets)
        raise Exception("num_targets has to be positive")

    if not args.ivector_dim >= 0:
        raise Exception("ivector-dim has to be non-negative")

    if (args.subset_dim < 0):
        raise Exception("--subset-dim has to be non-negative")

    if not args.hidden_dim is None:
        args.hidden_dim = args.hidden_dim

    if args.xent_separate_forward_affine and args.add_final_sigmoid:
        raise Exception("It does not make sense to have --add-final-sigmoid=true when xent-separate-forward-affine is true")

    if args.add_lda and args.cnn_layer is not None:
        args.add_lda = False
        warnings.warn("--add-lda is set to false as CNN layers are used.")

    return args

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.close()

def ParseSpliceString(splice_indexes):
    splice_array = []
    left_context = 0
    right_context = 0
    split1 = splice_indexes.split();  # we already checked the string is nonempty.
    if len(split1) < 1:
        raise Exception("invalid splice-indexes argument, too short: "
                 + splice_indexes)
    try:
        for string in split1:
            split2 = string.split(",")
            if len(split2) < 1:
                raise Exception("invalid splice-indexes argument, too-short element: "
                         + splice_indexes)
            int_list = []
            for int_str in split2:
                int_list.append(int(int_str))
            if not int_list == sorted(int_list):
                raise Exception("elements of splice-indexes must be sorted: "
                         + splice_indexes)
            left_context += -int_list[0]
            right_context += int_list[-1]
            splice_array.append(int_list)
    except ValueError as e:
        raise Exception("invalid splice-indexes argument " + splice_indexes + str(e))
    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

# The function signature of MakeConfigs is changed frequently as it is intended for local use in this script.
def MakeConfigs(config_dir, splice_indexes_string,
                feat_dim, ivector_dim, num_targets, add_lda, lda_dim,
                affine_type, active_type, hidden_dim,
                cudnn_layers,
                dropout_proportion,
                param_stddev, bias_stddev,
                self_repair_scale,
                batch_normalize, objective_type,
                model_type, rnn_bidirectional, rnn_mode, rnn_layers, rnn_max_seq_length, rnn_cell_dim,
                clipping_threshold, norm_based_clipping, rnn_first = True):

    parsed_splice_output = ParseSpliceString(splice_indexes_string.strip())

    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    config_lines = {'components':[]}
    config_files = {}

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')

    prev_layer_output = nodes.AddInputLayer(init_config_lines, feat_dim, splice_indexes[0], ivector_dim)
    if add_lda:
        if lda_dim <= 0:
            lda_dim = prev_layer_output['dimension']
        prev_layer_output = nodes.AddLdaLayer(init_config_lines, lda_dim, config_dir + '/../lda.mat')

    rnn_layer_output = None
    if model_type != 'DS2':
        if model_type == 'FT':
            if active_type.lower() == 'relu':
                prev_layer_output = nodes.AddAffRelNormLayer(init_config_lines, prev_layer_output, hidden_dim,
                                                            affine_type = affine_type,
                                                            self_repair_scale = self_repair_scale,
                                                            batch_normalize = batch_normalize)
            else:
                assert False
            prev_layer_output = nodes.AddClipGradientLayer(init_config_lines, prev_layer_output,
                                        clipping_threshold = clipping_threshold,
                                        norm_based_clipping = norm_based_clipping,
                                        self_repair_scale_clipgradient = None)

        first_rnn_layer = nodes.AddRnnLayer(init_config_lines, prev_layer_output, rnn_cell_dim,
                                    num_layers = cudnn_layers,
                                    max_seq_length = rnn_max_seq_length,
                                    bidirectional = rnn_bidirectional,
                                    rnn_mode = rnn_mode,
                                    clipping_threshold = clipping_threshold,
                                    dropout_proportion = dropout_proportion,
                                    norm_based_clipping = norm_based_clipping,
                                    self_repair_scale_clipgradient = None)
        rnn_layer_output = first_rnn_layer
        nodes.AddAffineLayer(init_config_lines, first_rnn_layer, num_targets)
    else:
        assert False, "Not sppourt DS2, now."

    config_files[config_dir + '/init.config'] = init_config_lines

    # if cnn_layer is not None:
    #     prev_layer_output = AddCnnLayers(config_lines, cnn_layer, cnn_bottleneck_dim, cepstral_lifter, config_dir,
    #                                      feat_dim, splice_indexes[0], ivector_dim)

    left_context = 0
    right_context = 0
    # we moved the first splice layer to before the LDA..
    # so the input to the first affine layer is going to [0] index
    splice_indexes[0] = [0]
    for i in range(0, num_hidden_layers-1):
        # make the intermediate config file for layerwise discriminative training
        local_output_dim = hidden_dim
        first_add_rnn = rnn_first and i < rnn_layers - 1
        last_add_rnn = not rnn_first and i >= num_hidden_layers - rnn_layers
        if model_type == 'google' or first_add_rnn or last_add_rnn:
            # add RNN layer
            assert rnn_layer_output
            rnn_layer_output = nodes.AddRnnLayer(config_lines, rnn_layer_output, rnn_cell_dim,
                                        num_layers = cudnn_layers,
                                        max_seq_length = rnn_max_seq_length,
                                        bidirectional = rnn_bidirectional,
                                        rnn_mode = rnn_mode,
                                        param_stddev = param_stddev, bias_stddev = bias_stddev,
                                        clipping_threshold = clipping_threshold,
                                        dropout_proportion = dropout_proportion,
                                        norm_based_clipping = norm_based_clipping,
                                        self_repair_scale_clipgradient = None)
        elif model_type == "FT":
            assert False, "Error here."
            if active_type.lower() == 'relu':
                prev_layer_output = nodes.AddAffRelNormLayer(config_lines, prev_layer_output, local_output_dim,
                                                            affine_type = affine_type,
                                                            self_repair_scale = self_repair_scale,
                                                            batch_normalize = batch_normalize)
            else:
                assert False
        else:
            assert False, "Error here."
        # elif model_type == "DS2":
        #     prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "Tdnn_{0}".format(i),
        #                                                 prev_layer_output, local_nonlin_output_dim,
        #                                                 self_repair_scale = self_repair_scale,
        #                                                 norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target,
        #                                                 batch_normalize = batch_normalize)

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[]}

    config_lines = {'components':['SoftmaxComponent dim={0}'.format(num_targets)]}
    config_files['{0}/softmax.config'.format(config_dir)] = config_lines

    left_context += int(parsed_splice_output['left_context'])
    right_context += int(parsed_splice_output['right_context'])

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(config_dir + "/vars", "w")
    print('left_context=' + str(left_context), file=f)
    print('right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    print('num_targets=' + str(num_targets), file=f)
    print('add_lda=' + ('true' if add_lda else 'false'), file=f)
    print('objective_type=' + objective_type, file=f)
    f.close()

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])

def Main():
    args = GetArgs()

    MakeConfigs(config_dir = args.config_dir,
                splice_indexes_string = args.splice_indexes,
                feat_dim = args.feat_dim, ivector_dim = args.ivector_dim,
                num_targets = args.num_targets,
                add_lda = args.add_lda, lda_dim = args.lda_dim,
                affine_type = args.affine_type, active_type = args.active_type, hidden_dim = args.hidden_dim,
                dropout_proportion = args.dropout_proportion, cudnn_layers = args.cudnn_layers,
                rnn_max_seq_length = args.rnn_max_seq_length,
                param_stddev = args.param_stddev, bias_stddev = args.bias_stddev,
                self_repair_scale = args.self_repair_scale,
                objective_type = args.objective_type,
                batch_normalize = args.batch_normalize,
                model_type = args.model_type, rnn_bidirectional = args.rnn_bidirectional,
                rnn_mode = args.rnn_mode, rnn_layers = args.rnn_layers,
                rnn_cell_dim = args.rnn_cell_dim,
                clipping_threshold = args.clipping_threshold, norm_based_clipping = args.norm_based_clipping,
                rnn_first = args.rnn_first)

if __name__ == "__main__":
    Main()

