#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
from operator import itemgetter

# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes=[0], ivector_dim=0):
    components = config_lines['components']
    components.append('SpliceComponent input-dim={input_dim} left-context={left_context} right-context={right_context} const-component-dim={ivector_dim}'.format(
        input_dim=feat_dim, left_context=abs(splice_indexes[0]), right_context=abs(splice_indexes[-1]), ivector_dim=ivector_dim))

    output_dim = (abs(int(splice_indexes[0])) + abs(int(splice_indexes[-1])) + 1) * feat_dim + ivector_dim
    return {'dimension': output_dim}

def AddFixedAffineLayer(config_lines, lda_dim, matrix_file):
    components = config_lines['components']
    components.append('FixedAffineComponent matrix={0}'.format(matrix_file))
    return {'dimension': lda_dim}

def AddLdaLayer(config_lines, lda_dim, lda_file):
    return AddFixedAffineLayer(config_lines, lda_dim, lda_file)

def AddAffineLayer(config_lines, input, output_dim, ng_affine_options = "", affine_type = 'native'):
    components = config_lines['components']
    if affine_type == 'natural':
        components.append("NaturalGradientAffineComponent input-dim={0} output-dim={1} {2}".format(input['dimension'], output_dim, ng_affine_options))
    elif affine_type == "native":
        components.append("AffineComponent input-dim={0} output-dim={1} {2}".format(input['dimension'], output_dim, ng_affine_options))

    return {'dimension': output_dim}

def AddAffRelNormLayer(config_lines, input, output_dim, affine_type = 'native', ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0,
        self_repair_scale = None, batch_normalize = False):
    components = config_lines['components']
    AddAffineLayer(config_lines, input, output_dim, affine_type = affine_type, ng_affine_options = ng_affine_options)
    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    # self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    self_repair_string = ""
    if batch_normalize:
        print("Not support BatchNormalize!")
        # components.append("BatchNormalizeComponent dim={1}".format(name, output_dim))
        # components.append("RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
        # return {'dimension': output_dim}
    else:
        components.append("RectifiedLinearComponent dim={0} {1}".format(output_dim, self_repair_string))
        # components.append("NormalizeComponent dim={0} target-rms={1}".format(output_dim, norm_target_rms))

        return {'dimension': output_dim}

def AddClipGradientLayer(config_lines, input,
                 clipping_threshold = 30.0,
                 norm_based_clipping = True, self_repair_scale_clipgradient = None):
    components = config_lines['components']
    output_dim = input['dimension']
    # self_repair_scale_clipgradient is a constant scaling the self-repair vector computed in ClipGradientComponent
    self_repair_clipgradient_string = "self-repair-scale={0:.2f}".format(self_repair_scale_clipgradient) if self_repair_scale_clipgradient is not None else ''
    components.append("ClipGradientComponent dim={0} clipping-threshold={1} norm-based-clipping={2} {3}".format(output_dim,
        clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))

    return { 'dimension':output_dim }

def AddDropoutLayer(config_lines, dim, dropout_proportion):
    assert(dropout_proportion > 0 and dropout_proportion < 1)
    components = config_lines['components']
    components.append('DropoutComponent dim={0} dropout-proportion={1}'.format(dim, dropout_proportion))
    return {'dimension': dim}

def AddRnnLayer(config_lines, input, hidden_dim,
                 bidirectional = True,
                 num_layers  = 1,
                 rnn_mode = 2,
                 max_seq_length = 1000,
                 learning_rate = 0.0001,
                 param_stddev = 0.02,
                 bias_stddev = 0.2,
                 clipping_threshold = 30.0,
                 dropout_proportion = 0,
                 norm_based_clipping = True,
                 self_repair_scale_clipgradient = None):
    components = config_lines['components']

    input_dim = input['dimension']
    output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    direction = "bidirectional={0} ".format("true" if bidirectional else "false" )
    rnn_config = direction + "max-seq-length={0} learning-rate={1} ".format(max_seq_length, learning_rate)
    rnn_config += "rnn-mode={0} num-layers={1} param-stddev={2} bias-stddev={3}".format(rnn_mode, num_layers, param_stddev, bias_stddev)
    components.append("CuDNNRecurrentComponent input-dim={0} output-dim={1} {2} ".format(input_dim, hidden_dim, rnn_config))
    if dropout_proportion > 0:
        AddDropoutLayer(config_lines, output_dim, dropout_proportion)
    AddClipGradientLayer(config_lines, {'dimension': output_dim},
        clipping_threshold = clipping_threshold,
        norm_based_clipping = norm_based_clipping,
        self_repair_scale_clipgradient = self_repair_scale_clipgradient)

    return { 'dimension':output_dim }

