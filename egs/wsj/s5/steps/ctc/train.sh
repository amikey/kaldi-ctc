#!/bin/bash

# Copyright  2016   LingoChamp Feiteng
# based on steps/nnet2/train_pnorm_simple2.sh

# Begin configuration section.
cmd=run.pl
num_epochs=5      # Number of epochs of training;
                   # the number of iterations is worked out from this.
initial_learning_rate=0.004
final_learning_rate=0.00004

minibatch_size=64  # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update. 

utts_per_eg=5000   # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet=3    # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.

prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
num_jobs_compute_prior=20 # these are single-threaded, run on GPU.

get_egs_stage=0
online_ivector_dir=

shuffle_buffer_size=128 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.

add_layers_period=6 # by default, add new layers every 6 iterations.

model_type="google"  # google DS2 FT

splice_indexes="0 0 0" # nnet2

lda_opts=
lda_dim=
add_lda=false
lda_alidir=

affine_type="native"
active_type="relu"
bidirectional=true
dropout_proportion=0

max_allow_frames=2000 # 20 seconds
frame_subsampling_factor=1

rnn_mode=2
num_rnn_layers=5
cudnn_layers=1

hidden_dim=1024
rnn_cell_dim=320

clipping_threshold=30 # 30.0
norm_based_clipping=true
param_stddev=0.02
bias_stddev=0.2

self_repair_scale=0.00001
momentum=0.0
max_param_change=20

combine_final_nnet=false

adjust_priors=true
posterior_priors=false
google_prior_const=9

cv_period=10
stage=-4
verbose=0

randprune=4.0 # speeds up LDA.
alpha=4.0 # relates to preconditioning.

# TODO
update_period=4 # relates to online preconditioning: says how often we update the subspace.
num_samples_history=2000 # relates to online preconditioning
precondition_rank_in=40  # relates to online preconditioning
precondition_rank_out=20 # relates to online preconditioning

max_change_per_sample=0.075

num_threads=1
parallel_opts="--num-threads 16 --mem 1G" 

cleanup=true
cleanup_egs=false

egs_dir=
egs_opts=
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.
transform_dir=     # If supplied, overrides alidir
cmvn_opts="--norm-means=true --norm-vars=true"  # will be passed to get_lda.sh and get_egs.sh, if supplied.  
            # only relevant for "raw" features, not lda.
feat_type=  # Can be used to force "raw" features.

nj=6
nnet_type=2

# TODO
align_cmd=              # The cmd that is passed to steps/nnet2/align.sh
align_use_gpu=          # Passed to use_gpu in steps/nnet2/align.sh [yes/no]
realign_epochs=         # List of epochs, the beginning of which realignment is done
num_jobs_align=30       # Number of jobs for realignment
# End configuration section.


echo "$0 $*"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

# set -e

if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of training"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
  echo "  --add-layers-period <#iters|2>                   # Number of iterations between adding hidden layers"
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"--num-threads 16 --mem 1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... "
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|250>                              # Dimension to reduce spliced features to with LDA"
  echo "  --realign-epochs <list-of-epochs|\"\">           # A list of space-separated epoch indices the beginning of which"
  echo "                                                   # realignment is to be done"
  echo "  --align-cmd (utils/run.pl|utils/queue.pl <queue opts>) # passed to align.sh"
  echo "  --align-use-gpu (yes/no)                         # specify is gpu is to be used for realignment"
  echo "  --num-jobs-align <#njobs|30>                     # Number of jobs to perform realignment"
  echo "  --stage <stage|-4>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

if [ ! -z "$realign_epochs" ]; then
  [ -z "$align_cmd" ] && echo "$0: realign_epochs specified but align_cmd not specified" && exit 1
  [ -z "$align_use_gpu" ] && echo "$0: realign_epochs specified but align_use_gpu not specified" && exit 1
fi

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# sdata=$data/split$nj
# utils/split_data.sh $data $nj || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
echo $cmvn_opts >$dir/cmvn_opts
cp $alidir/tree $dir || exit 1;

# Set some variables.
num_leaves=`tree-info $alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1
[ -z $num_leaves ] && echo "\$num_leaves is unset" && exit 1
[ "$num_leaves" -eq "0" ] && echo "\$num_leaves is 0" && exit 1

if [ $stage -le -5 ]; then
  echo "$0: creating neural net configs";
  # LDA config
  [ -d $dir/configs ] && rm -r $dir/configs
  mkdir -p $dir/configs

  lda_config="--add-lda $add_lda "
  [ ! -z "$lda_dim" ] && lda_config="$lda_config --lda-dim $lda_dim "

  steps/ctc/nnet2/make_configs.py \
    --feat-dir $data --splice-indexes "$splice_indexes " \
    --num-targets $[$num_leaves+1] $lda_config \
    --dropout-proportion $dropout_proportion \
    --affine-type $affine_type --active-type $active_type --hidden-dim $hidden_dim \
    --model.type $model_type --model.bidirectional $bidirectional \
    --model.cudnn-layers $cudnn_layers \
    --model.rnn-layers $num_rnn_layers \
    --model.rnn-max-seq-length $max_allow_frames \
    --model.cell-dim $rnn_cell_dim \
    --model.param-stddev $param_stddev --model.bias-stddev $bias_stddev \
    --model.rnn-mode $rnn_mode --model.norm-based-clipping $norm_based_clipping \
    --model.clipping-threshold $clipping_threshold \
    --self-repair-scale $self_repair_scale \
   $dir/configs || exit 1;
fi

. $dir/configs/vars || exit 1;

if ! [ $num_hidden_layers -ge 1 ]; then
  echo "Invalid num-hidden-layers $num_hidden_layers"
  exit 1;
fi

extra_opts=()
[ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
[ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
[ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
extra_opts+=(--transform-dir "$transform_dir")
[ -z "$left_context" ] && left_context=0
[ -z "$right_context" ] && right_context=0
extra_opts+=(--left-context $left_context --right-context $right_context)

if [[ $stage -le -4 && "$add_lda" == "true" ]]; then
  echo "$0: calling get_lda.sh"
  if [ -z $lda_alidir ];then
    echo "$0: lda_alidir is empty!" && exit 1;
  fi
  # Check some files.
  for f in $lda_alidir/tree $lda_alidir/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done

  steps/nnet2/get_lda.sh --feat-type raw $lda_opts "${extra_opts[@]}" $data $lang $lda_alidir $dir || exit 1;
fi

if [ $stage -le -3 ];then
  echo "$0: initializing neural net";
  $cmd $dir/log/nnet_init.log \
    nnet-init --use-gpu=yes $dir/configs/init.config - \| \
    nnet2-ctc-init-model $alidir/tree $lang/topo - - \| \
    nnet-am-copy --learning-rate=$initial_learning_rate - $dir/0.mdl || exit 1;
fi

if [ $stage -le -2 ] && [ -z "$egs_dir" ]; then
  echo "$0: calling get_egs2.sh"            
  steps/ctc/get_egs2.sh $egs_opts "${extra_opts[@]}"  --io-opts "$io_opts" \
    --utts-per-eg $utts_per_eg --stage $get_egs_stage \
    --cmd "$cmd" $egs_opts $data $alidir $dir/egs || exit 1;
fi

if [ -z $egs_dir ]; then
  if [ -d $dir/egs ];then
    egs_dir=$dir/egs
  else
    echo "$0: no egs dir detected." && exit 1;
  fi
fi

feat_dim=$(cat $egs_dir/info/feat_dim) || exit 1;
ivector_dim=$(cat $egs_dir/info/ivector_dim) || exit 1;

num_archives=$(cat $egs_dir/info/num_archives) || { echo "error: no such file $egs_dir/info/num_archives"; exit 1; }

echo $frame_subsampling_factor >$dir/frame_subsampling_factor || exit 1;
# num_archives_expanded considers each separate label-position from
num_archives_expanded=$[$num_archives*$frame_subsampling_factor]

if [ $num_jobs_nnet -gt $num_archives_expanded ]; then
  echo "$0: --num-jobs-nnet cannot exceed num-archives*frames-per-eg which is $num_archives_expanded"
  echo "$0: setting --num-jobs-nnet to $num_archives_expanded"
  num_jobs_nnet=$num_archives_expanded
fi

# set num_iters so that as close as possible, we process the data $num_epochs
# times, i.e. $num_iters*$num_jobs_nnet == $num_epochs*$num_archives_expanded
num_iters=$[($num_epochs*$num_archives_expanded)/$num_jobs_nnet]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

finish_add_layers_iter=$[$num_hidden_layers * $add_layers_period]
# This is when we decide to mix up from: halfway between when we've finished
# adding the hidden layers and the end of training.
mix_up_iter=$[($num_iters + $finish_add_layers_iter)/2]

if [ $num_threads -eq 1 ]; then
  parallel_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
  fi
else
  echo "$0: CTC training only support single-threaded now." && exit 1;
  parallel_suffix="-parallel"
  parallel_train_opts="--num-threads=$num_threads"
fi

if [ $minibatch_size -le 1 ];then
  echo "$0: minibatch_size should bigger than 1." && exit 1
fi

approx_iters_per_epoch=$[$num_iters/$num_epochs]

cur_egs_dir=$egs_dir
rnnlayer="CuDNNRecurrentComponent"

x=0
num_archives_processed=0
frame_subsampling_opts="--frame-subsampling-factor=$frame_subsampling_factor"

while [ $x -lt $num_iters ]; do
  local_num_jobs_nnet=$num_jobs_nnet
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    if [ $x -gt 0 ] && [ $[$x%$cv_period] -eq 0 ]; then
      # Set off jobs doing some diagnostics, in the background.
      # Use the egs dir from the previous iteration for the diagnostics
      frame_shift=$[($x/$cv_period)%$frame_subsampling_factor];
      if [ $num_jobs_nnet -lt 2 ];then
        $cmd $dir/log/compute_prob_valid.$x.log \
          nnet2-ctc-compute-prob "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |" \
          "ark:nnet-ctc-shuffle-egs $frame_subsampling_opts --frame-shift=$frame_shift ark:$cur_egs_dir/valid_diagnostic.egs ark:- |"
        $cmd $dir/log/compute_prob_train.$x.log \
          nnet2-ctc-compute-prob "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |" \
          "ark:nnet-ctc-shuffle-egs $frame_subsampling_opts --frame-shift=$frame_shift ark:$cur_egs_dir/train_diagnostic.egs ark:- |"
      else
        $cmd $dir/log/compute_prob_valid.$x.log \
          nnet2-ctc-compute-prob "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |" \
          "ark:nnet-ctc-shuffle-egs $frame_subsampling_opts --frame-shift=$frame_shift ark:$cur_egs_dir/valid_diagnostic.egs ark:- |" &
        $cmd $dir/log/compute_prob_train.$x.log \
          nnet2-ctc-compute-prob "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |" \
          "ark:nnet-ctc-shuffle-egs $frame_subsampling_opts --frame-shift=$frame_shift ark:$cur_egs_dir/train_diagnostic.egs ark:- |" &
        wait
      fi
    fi

    learning_rate=$(perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters $initial_learning_rate $final_learning_rate)
    echo "Training neural net (pass $x / $num_iters) learning-rate $learning_rate at $(date)"

    mdl=$dir/$x.mdl

    # discriminative per layer training
    if [ $x -gt 1 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[$x % $add_layers_period] -eq 0 ]; then

      if [ $model_type == 'DS2' ];then
        echo "Not support DS2 now!";
        exit 1;
      fi

      layeridx=$[$x / $add_layers_period]
      inp=-1

      is_rnn=$(grep -c $rnnlayer $dir/configs/layer$layeridx.config)
      dropout=$(grep -c "Dropout" $dir/configs/layer$layeridx.config)
      if [ $is_rnn -eq 0 ];then
        inp=$(nnet-am-info $dir/$x.mdl | grep $rnnlayer | head -n 1 | awk '{print $2}')
        echo "insert before first RNN layer(insert-at $inp)"
        inp=$[$inp - 1]
      else
        inp=$(nnet-am-info $dir/$x.mdl | grep $rnnlayer | tail -n 1 | awk '{print $2}')
        inp=$((inp+2+dropout))
        echo "insert after lastest RNN layer(insert-at $inp)"
      fi
      $cmd $dir/log/insert.$x.log \
        nnet-insert --insert-at=$inp --randomize-next-component=false $dir/$x.mdl "nnet-init --use-gpu=yes --srand=$x $dir/configs/layer$layeridx.config - |" - \| \
        nnet-am-copy --learning-rate=$learning_rate - $mdl || exit 1;
    fi

    if [ $x -eq 0 ]; then
      # on iteration zero or when we just added a layer, use a smaller minibatch
      # size and just one job: the model-averaging doesn't seem to be helpful
      # when the model is changing too fast (i.e. it worsens the objective
      # function), and the smaller minibatch size will help to keep
      # the update stable.
      this_minibatch_size=$[$minibatch_size/2];
      do_average=false
    else
      this_minibatch_size=$minibatch_size
      do_average=true
    fi

    rm $dir/.error 2>/dev/null

    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.
      
      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      for n in $(seq $local_num_jobs_nnet); do
        k=$[$num_archives_processed + $n - 1]; # k is a zero-based index that we'll derive
                                               # the other indexes from.
        archive=$[($k%$num_archives)+1]; # work out the 1-based archive index.
        frame_shift=$[($k/$num_archives)%$frame_subsampling_factor];

        $cmd $parallel_opts $dir/log/train.$x.$n.log \
          nnet2-ctc-train$parallel_suffix --max-allow-frames=$max_allow_frames --momentum=$momentum --max-param-change=$max_param_change $parallel_train_opts --verbose=$verbose \
            --minibatch-size=$this_minibatch_size "$mdl" \
            "ark,bg:nnet-ctc-shuffle-egs $frame_subsampling_opts --frame-shift=$frame_shift --buffer-size=$shuffle_buffer_size --srand=$x ark:$cur_egs_dir/egs.$archive.ark ark:- |" \
            $dir/$[$x+1].$n.mdl || touch $dir/.error &
      done
      wait
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error on iteration $x of training" && exit 1;

    nnets_list=
    for n in $(seq 1 $local_num_jobs_nnet); do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    if $do_average; then
      # average the output of the different jobs.
      $cmd $dir/log/average.$x.log \
        nnet-am-average $nnets_list - \| \
        nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].mdl || exit 1;
    else
      # choose the best from the different jobs.
      n=$(perl -e '($nj,$pat)=@ARGV; $best_n=1; $best_logprob=-1.0e+10; for ($n=1;$n<=$nj;$n++) {
          $fn = sprintf($pat,$n); open(F, "<$fn") || die "Error opening log file $fn";
          undef $logprob; while (<F>) { if (m/log-prob-per-frame=(\S+)/) { $logprob=$1; } }
          close(F); if (defined $logprob && $logprob > $best_logprob) { $best_logprob=$logprob; 
          $best_n=$n; } } print "$best_n\n"; ' $local_num_jobs_nnet $dir/log/train.$x.%d.log) || exit 1;
      [ -z "$n" ] && echo "Error getting best model" && exit 1;
      $cmd $dir/log/select.$x.log \
        nnet-am-copy --learning-rate=$learning_rate $dir/$[$x+1].$n.mdl $dir/$[$x+1].mdl || exit 1;
    fi

    rm $nnets_list
    [ ! -f $dir/$[$x+1].mdl ] && exit 1;
    if [ -f $dir/$[$x-8].mdl ] && $cleanup && [ $[($x-8)%100] -ne 0  ]; then
      rm $dir/$[$x-8].mdl
    fi
  fi
  num_archives_processed=$[$num_archives_processed + $local_num_jobs_nnet]
  x=$[$x+1]
done

if [ $stage -le $num_iters ];then
  echo "Doing final diagnostic, copy $dir/$num_iters.mdl -> $dir/final.mdl"
  rm -f $dir/final.mdl || exit 1;
  $cmd $dir/log/remove_dropout.final.log \
    nnet-am-copy --remove-dropout=true $dir/$num_iters.mdl $dir/final.mdl || exit 1;
  $cmd $dir/log/compute_prob_valid.final.log \
    nnet2-ctc-compute-prob $dir/final.mdl ark:$cur_egs_dir/valid_diagnostic.egs
  $cmd $dir/log/compute_prob_train.final.log \
    nnet2-ctc-compute-prob $dir/final.mdl ark:$cur_egs_dir/train_diagnostic.egs
fi

if [[ $adjust_priors && $stage -le $((num_iters+1)) ]]; then
  echo "Getting average posterior for purposes of adjusting the priors."
  # append Softmax/LogSoftmax
  softmax=$(nnet-am-info $dir/final.mdl | grep -c 'Softmax')
  if [ $softmax -eq 0 ];then
    inp=$(nnet-am-info $dir/final.mdl | grep 'num-components' | awk '{print $2}')
    $cmd $dir/log/append_softmax.final.log \
      nnet-init $dir/configs/softmax.config - \| \
      nnet-insert --insert-at=$inp --randomize-next-component=false $dir/final.mdl - $dir/final.mdl || exit 1;
  fi

  if [ $google_prior_const -eq 0 ];then
    if $posterior_priors;then
      echo "Re-adjusting priors based on computed posteriors"
      # Note: this just uses CPUs, using a smallish subset of data.
      x=final
      rm $dir/post.$x.*.vec 2>/dev/null
      if [ $num_jobs_compute_prior -gt $num_archives ]; then num_jobs_compute_prior=num_archives; fi
      egs_part=JOB
      $cmd --max-jobs-run $num_jobs_nnet JOB=1:$num_jobs_compute_prior $dir/log/get_post.$x.JOB.log \
        nnet-ctc-shuffle-egs $frame_subsampling_opts ark:$cur_egs_dir/egs.$egs_part.ark ark:- \| \
        nnet-ctc-compute-from-egs $dir/final.mdl ark:- ark:- \| \
        matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.$x.JOB.vec || exit 1;
      $cmd $dir/log/vector_sum.$x.log \
       vector-sum $dir/post.$x.*.vec $dir/post.$x.vec || exit 1;
      rm $dir/post.$x.*.vec;
    else
      echo "Re-adjusting priors based on label_counts"
      # TODO sort and uniq
      gunzip -c $alidir/ali.*.gz | copy-int-vector ark:- ark,t:- | awk '{line=$0; gsub(" "," 0 ",line); print line;}' | \
        analyze-counts --verbose=1 --binary=false ark:- $dir/post.final.vec >& $dir/log/compute_label_counts.log || exit 1
    fi
    sleep 3;  # make sure there is time for $dir/post.$x.*.vec to appear.

    $cmd $dir/log/adjust_priors.final.log \
      nnet-adjust-priors --binary=false $dir/final.mdl $dir/post.final.vec $dir/final.mdl || exit 1;
  else
    echo "Re-adjusting priors google_prior_const is: $google_prior_const"
    $cmd $dir/log/adjust_priors.final.log \
      nnet-adjust-priors --binary=false --google-prior-const=$google_prior_const $dir/final.mdl $dir/final.mdl || exit 1;
  fi
fi

if [ ! -f $dir/final.mdl ]; then
  echo "$0: $dir/final.mdl does not exist."
  # we don't want to clean up if the training didn't succeed.
  exit 1;
fi

sleep 2

if $cleanup_egs; then
  echo Cleaning up data
  if [[ $cur_egs_dir =~ $dir/egs* ]]; then
    steps/nnet2/remove_egs.sh "$cur_egs_dir"
  fi
fi

if $cleanup;then
  echo Removing most of the models
  for x in $(seq 0 $num_iters); do
    if [ $((x%100)) -ne 0 ] && [ $x -ne $num_iters ] && [ -f $dir/$x.mdl ]; then
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm "$dir/$x.mdl"
    fi
  done
fi

echo "$0: Training Done."
exit 0;
