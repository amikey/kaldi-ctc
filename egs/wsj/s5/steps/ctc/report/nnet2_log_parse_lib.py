# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.

from __future__ import division
import sys, glob, re, math, datetime, argparse
import imp

ntl = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

def ParseDifferenceString(string):
    dict = {}
    for parts in string.split():
        sub_parts = parts.split(":")
        dict[sub_parts[0]] = float(sub_parts[1])
    return dict

def ParseTrainLogs(exp_dir):
  train_log_files = "%s/log/train.*.log" % (exp_dir)
  train_log_lines = ntl.RunKaldiCommand('grep -e Accounting {0}'.format(train_log_files))[0]
  parse_regex = re.compile(".*train\.([0-9]+).*\.log:# Accounting: time=([0-9]+) thread.*")

  train_times = {}
  for line in train_log_lines.split('\n'):
    mat_obj = parse_regex.search(line)
    if mat_obj is not None:
        groups = mat_obj.groups()
        try:
            train_times[int(groups[0])][int(groups[1])] = float(groups[2])
        except IndexError, KeyError:
            train_times[int(groups[0])] = {}
            train_times[int(groups[0])][0] = float(groups[1])
  iters = train_times.keys()
  for iter in iters:
      values = train_times[iter].values()
      train_times[iter] = max(values)
  return train_times

def ParseProbLogs(exp_dir, key = 'accuracy'):
    train_prob_files = "%s/log/compute_prob_train.*.log" % (exp_dir)
    valid_prob_files = "%s/log/compute_prob_valid.*.log" % (exp_dir)
    train_prob_strings = ntl.RunKaldiCommand('grep -e {0} {1}'.format(key, train_prob_files), wait = True)[0]
    valid_prob_strings = ntl.RunKaldiCommand('grep -e {0} {1}'.format(key, valid_prob_files))[0]
    parse_regex = re.compile(".*compute_prob_.*\.([0-9]+).log:LOG .nnet.*compute-prob:main.*:nnet.*compute-prob.cc:[0-9]+. Saw [0-9]+ examples, average ([a-zA-Z\-]+) is ([0-9.\-e]+) and ([a-zA-Z\-]+) is ([0-9.\-e]+) with total .*")
    train_loss={}
    valid_loss={}

    for line in train_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                train_loss[int(groups[0])] = groups[2]
            elif groups[3] == key:
                train_loss[int(groups[0])] = groups[4]
    for line in valid_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                valid_loss[int(groups[0])] = groups[2]
            elif groups[3] == key:
                valid_loss[int(groups[0])] = groups[4]
    iters = list(set(valid_loss.keys()).intersection(train_loss.keys()))
    iters.sort()
    return map(lambda x: (int(x), float(train_loss[x]), float(valid_loss[x])), iters)

def GenerateAccuracyReport(exp_dir, key = "accuracy"):
    times = ParseTrainLogs(exp_dir)
    data = ParseProbLogs(exp_dir, key)
    report = []
    report.append("%Iter\tduration\ttrain_loss\tvalid_loss\tdifference")
    for x in data:
        try:
            report.append("%d\t%s\t%g\t%g\t%g" % (x[0], str(times[x[0]]), x[1], x[2], x[2]-x[1]))
        except KeyError:
            continue

    total_time = 0
    for iter in times.keys():
        total_time += times[iter]
    report.append("Total training time is {0}\n".format(str(datetime.timedelta(seconds = total_time))))
    return ["\n".join(report), times, data]
