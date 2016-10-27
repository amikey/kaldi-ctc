# kaldi-ctc


Connectionist Temporal Classification (CTC) Automatic Speech Recognition. `Training` and `Decoding` are extremely fast.

# Intoduction
kaldi-ctc is based on [kaldi](https://github.com/kaldi-asr/kaldi), [warp-ctc](https://github.com/baidu-research/warp-ctc) and [cudnn](https://developer.nvidia.com/cudnn).

| Components |    Role    |
| -----------|:-------------:|
| kaldi      | Parent body, data prepare / build decoding WFST  |
| warp-ctc   | Fast parallel implementation of CTC  |
| cudnn(>=5.0)|Fast recurrent neural networks(LSTM,GRU,ReLU,Tanh)  |

# Compilation

```
# install dependents
cd tools
make -j
make openblas
# Install cudnn, reference script `extras/install_cudnn.sh`
bash extras/install_cudnn.sh

cd ../src
./configure --cudnn-root=CUDNN_ROOT --openblas-root=../tools/OpenBLAS/install
make depend -j
make -j
```

# Example scripts
Make sure the GPU's memory is enough, default setting can run on GTX TITAN X/1080( >= 8G).  
Using smaller `minibatch_size(default 16)` / `max_allow_frames(default 2000)` or bigger `frame_subsampling_factor(default 1)` if your GPUs are older.

## librispeech

### CTC-monophone
```
cd egs/librispeech/ctc
bash run.sh --stage -2 --num-gpus 4(change to your GPU devices amount)
```

### Edit Distance Accuracy
```
steps/ctc/report/generate_plots.py exp/ctc/cudnn_google_fs3 reports/ctc-google
```
<img src="./egs/librispeech/ctc/reports/ctc-google/accuracy.jpg" width="500">

### WER RESULITS (LM tgsmall)
| Models | Real Time Factor(RTF) | test_clean | dev_clean | test_other | dev_other |
| -------|:----:|:------:| :-------|:----------:|:----------:|
|chain   |         |  6.20  | 5.83| 14.73 |14.56|
| CTC-monophone    | (0.05 ~ 0.06) / `frame_subsampling_factor` |  8.63 | 9.02 | 20.75 |  22.16 |
| CTC-character    |  |

* There are many Out Of Vocabularies(OOVs) in training transcriptions now

```
awk 'FNR==NR{T[$1]=1;} FNR<NR{for(i=2;i<=NF;i++) {if (!($i in T)) print $i;}}' data/lang_nosp/words.txt   data/train_960/text | sort -u | wc -l
14291
```

* CTC system gets better results than chain system on a larger corpu.


# TODO
### Cleanup librispeech corpus(Fix OOVs), Fine tune parameters
### CTC-character example script
### FLAT START TRAINING CTC-RNN ACOUSTIC MODELS, CTC-triphone
* [google - FLAT START TRAINING OF CD-CTC-SMBR LSTM RNN ACOUSTIC MODELS](http://ieeexplore.ieee.org/document/7472710/)


