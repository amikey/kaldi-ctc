#!/bin/bash

# Copyright 2016  LingoChamp Feiteng

stage=0
train_stage=-10
use_gpu=true

parallel_opts="-l gpu=1" 
num_threads=1
minibatch_size=16
max_allow_frames=2000
frame_subsampling_factor=1
train_opts=""


mono=true
hidden_dim=1024
model_type="google"
splice_indexes="0 0 0 0 0"

TreeLeaves=2500
adjust_priors=true

num_gpus=3

self_repair_scale=0.00001

suffix=""
egs_dir=""

initial_learning_rate=0.0005
final_learning_rate=0.00001

graph_dir=""
blank_threshold=0.98
decode_sets="dev_clean dev_other test_clean test_other"
decode_iter="final"
decode_suffix=""

nj=20

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e

train_set=train_960
gmm_dir=exp/tri2b
ali_dir=exp/tri2b_960_sp_ali
lang=data/lang_ctc

if [ $stage -le 0 ];then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp

  utils/copy_data_dir.sh data/${train_set}_sp data/${train_set}_sp_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf \
    data/${train_set}_sp_hires || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp_hires || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed, low-resolution data"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --beam 15 --retry-beam 20 \
    data/${train_set}_sp data/lang_nosp $gmm_dir $ali_dir || exit 1
fi

dir=exp/ctc/cudnn_${model_type}$suffix

if $mono;then
  treedir=exp/ctc/tree_mono
else
  treedir=exp/ctc/tree_tri$TreeLeaves
fi

if [ $stage -le 6 ];then
  if [ ! -d data/${train_set}_sp_hires_100k ];then
    utils/subset_data_dir.sh data/${train_set}_sp_hires 100000 data/${train_set}_sp_hires_100k
  fi

  if [ $stage -le 1 ];then
    echo "$0: prepare lang"
    utils/prepare_lang.sh --share-silence-phones true --position-dependent-phones false \
      --num-sil-states 1 --num-nonsil-states 1 \
      data/local/dict_nosp "<UNK>" data/local/lang_tmp_ctc $lang || exit 1;
  fi

  echo "$0: build one state tree"
  # Build a tree using our new topology.
  steps/ctc/build_tree.sh --frame-subsampling-factor 1 --stage -10 --mono $mono \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" $TreeLeaves data/${train_set}_sp_hires_100k $lang $ali_dir $treedir || exit 1;
fi

if [ $stage -le 10 ];then
  steps/ctc/train.sh --stage $train_stage \
    --splice-indexes "$splice_indexes" $train_opts \
    --self-repair-scale $self_repair_scale \
    --model-type $model_type --hidden-dim $hidden_dim \
    --utts-per-eg 5000 --get-egs-stage 0 --io-opts "-tc 10" --egs-dir "$egs_dir" \
    --egs-opts "--num-jobs $nj --feat-type raw --num-utts-subset 1000" --verbose 0 \
    --minibatch-size $minibatch_size --max-allow-frames $max_allow_frames --shuffle-buffer-size 256 \
    --frame-subsampling-factor $frame_subsampling_factor \
    --parallel-opts "$parallel_opts" --num-threads "$num_threads" \
    --num-jobs-nnet $num_gpus \
    --initial-learning-rate $initial_learning_rate --final-learning-rate $final_learning_rate \
    data/${train_set}_sp_hires $lang $treedir $dir || exit 1;
fi

if [ -z $graph_dir ];then
    mono_opt=""
    $mono && mono_opt="--mono"

    for suffix in tgsmall;do
      graph_dir=$dir/graph_nosp_$suffix
      if [[ $stage -le 11 && ! -f $graph_dir/CTC.fst ]]; then
        utils/mkgraph.sh --ctc $mono_opt --self-loop-scale 1.0 --remove-oov \
          data/lang_nosp_test_$suffix $dir $graph_dir || exit 1;
        # romove <UNK> from the graph
        fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/CTC.fst $graph_dir/CTC.fst
      fi
    done
fi

if [ $stage -le 12 ];then
  for decode_set in $decode_sets;do
    if [ ! -f data/${decode_set}_hires/feats.scp ];then
      rm -rf data/${decode_set}_hires
      utils/copy_data_dir.sh data/${decode_set} data/${decode_set}_hires
      steps/make_mfcc.sh --compress false --mfcc-config conf/mfcc_hires.conf --nj 6 \
          data/${decode_set}_hires || exit 1;
      steps/compute_cmvn_stats.sh data/${decode_set}_hires || exit 1;
    fi

    for suffix in tgsmall;do
      graph_dir=$dir/graph_nosp_$suffix
      decode_dir=$dir/decode_${decode_set}${decode_iter:+_$decode_iter}_$suffix$decode_suffix

      if [ ! -d $decode_dir/scoring ];then
        steps/ctc/decode.sh --feat-type raw --blank-threshold $blank_threshold --stage 0 \
          --scoring-opts "--min-lmwt 1 --max-lmwt 10 --decode-mbr false " \
          --acwt 1 --lattice-acoustic-scale 10 --iter $decode_iter \
          --decode-opts "--beam-delta=3 --prune-interval=20" --beam 20 --lattice-beam 10 \
          --verbose 0 --cmd "$decode_cmd" --nj $num_gpus $graph_dir data/${decode_set}_hires \
          $decode_dir
      fi
    done
  done
fi
