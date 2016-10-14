#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.

import warnings
import imp
import argparse
import os
import errno
import logging
import re
import subprocess

train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    # import seaborn as sns
    # sns.set(color_codes=True)
    plot = True
except ImportError:
    warnings.warn("""
This script requires matplotlib and numpy. Please install them to generate plots. Proceeding with generation of tables.
If you are on a cluster where you do not have admin rights you could try using virtualenv.""")

nlp = imp.load_source('nlp', 'steps/ctc/report/nnet2_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Generating plots')


def GetArgs():
    parser = argparse.ArgumentParser(description="""
Parses the training logs and generates a variety of plots.
example : steps/nnet3/report/generate_plots.py --comparison-dir exp/nnet3/tdnn1 --comparison-dir exp/nnet3/tdnn2 exp/nnet3/tdnn exp/nnet3/tdnn/report
""")
    parser.add_argument("--comparison-dir", type=str, action='append', help="other experiment directories for comparison. These will only be used for plots, not tables")
    parser.add_argument("--start-iter", type=int, help="Iteration from which plotting will start", default = 1)
    parser.add_argument("--is-chain", type=str, default = False, action = train_lib.StrToBoolAction, help="Iteration from which plotting will start")
    parser.add_argument("exp_dir", help="experiment directory, e.g. exp/nnet3/tdnn")
    parser.add_argument("output_dir", help="experiment directory, e.g. exp/nnet3/tdnn/report")

    args = parser.parse_args()
    if args.comparison_dir is not None and len(args.comparison_dir) > 6:
        raise Exception("max 6 --comparison-dir options can be specified. If you want to compare with more comparison_dir, you would have to carefully tune the plot_colors variable which specified colors used for plotting.")
    assert(args.start_iter >= 1)
    return args

plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow', 'cyan' ]
# plot_colors = sns.xkcd_palette(['green', 'blue', 'red', 'orange', 'light purple', 'blue green', 'poo', 'muted blue', 'sickly yellow', 'carmine', 'dirty green'])

class LatexReport:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.document=[]
        self.document.append("""
\documentclass[prl,10pt,twocolumn]{revtex4}
\usepackage{graphicx}    % Used to import the graphics
\\begin{document}
""")

    def AddFigure(self, figure_pdf, title):
        # we will have keep extending this replacement list based on errors during compilation
        # escaping underscores in the title
        title = "\\texttt{"+re.sub("_","\_", title)+"}"
        fig_latex = """
%...
\\newpage
\\begin{figure}[h]
  \\begin{center}
    \caption{""" + title + """}
    \includegraphics[width=\\textwidth]{""" + figure_pdf + """}
  \end{center}
\end{figure}
\clearpage
%...
"""
        self.document.append(fig_latex)

    def Close(self):
        self.document.append("\end{document}")
        return self.Compile()

    def Compile(self):
        root, ext = os.path.splitext(self.pdf_file)
        dir_name = os.path.dirname(self.pdf_file)
        latex_file = root + ".tex"
        lat_file = open(latex_file, "w")
        lat_file.write("\n".join(self.document))
        lat_file.close()
        logger.info("Compiling the latex report.")
        try:
            proc = subprocess.Popen(['pdflatex', '-output-directory='+str(dir_name), latex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.communicate()
        except Exception as e:
            logger.warning("There was an error compiling the latex file {0}, please do it manually.".format(latex_file))
            return False
        return True

def GenerateAccuracyPlots(exp_dir, output_dir, plot, key = 'accuracy', file_basename = 'accuracy', comparison_dir = None, start_iter = 1, latex_report = None):
    assert(start_iter >= 1)

    if plot:
        fig = plt.figure()
        plots = []

    comparison_dir = [] if comparison_dir is None else comparison_dir
    dirs = [exp_dir] + comparison_dir
    index = 0
    for dir in dirs:
        [accuracy_report, accuracy_times, accuracy_data] = nlp.GenerateAccuracyReport(dir, key)
        if index == 0:
            # this is the main experiment directory
            acc_file = open("{0}/{1}.log".format(output_dir, file_basename), "w")
            acc_file.write(accuracy_report)
            acc_file.close()

        if plot:
            color_val = plot_colors[index]
            data = np.array(accuracy_data)
            if data.shape[0] == 0:
                raise Exception("Couldn't find any rows for the accuracy plot")
            data = data[data[:,0]>=start_iter, :]
            plot_handle, = plt.plot(data[:, 0], data[:, 1], color = color_val, linestyle = "--", label = "train {0}".format(dir))
            plots.append(plot_handle)
            plot_handle, = plt.plot(data[:, 0], data[:, 2], color = color_val, label = "valid {0}".format(dir))
            plots.append(plot_handle)

        index += 1
    if plot:
        plt.xlabel('Iteration')
        plt.ylabel(key)
        lgd = plt.legend(plots, loc='lower center', bbox_to_anchor=(0.5, -0.2 + len(dirs) * -0.1 ), ncol=1)
        plt.grid(True)
        fig.suptitle("Unique Phone Sequence(CTC-RNN Predicted) {0}".format(key))
        figfile_name = '{0}/{1}.png'.format(output_dir, file_basename)
        plt.savefig(figfile_name, bbox_inches='tight', dpi=160)
        if latex_report is not None:
            latex_report.AddFigure(figfile_name, "Plot of {0} vs iterations".format(key))

def GeneratePlots(exp_dir, output_dir, comparison_dir = None, start_iter = 1, is_chain = False):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise e
    if plot:
        latex_report = None
        # latex_report = LatexReport("{0}/report.pdf".format(output_dir))
    else:
        latex_report = None

    if is_chain:
        logger.info("Generating log-probability plots")
        GenerateAccuracyPlots(exp_dir, output_dir, plot, key = 'log-probability', file_basename = 'log_probability', comparison_dir = comparison_dir, start_iter = start_iter, latex_report = latex_report)
    else:
        logger.info("Generating accuracy plots")
        GenerateAccuracyPlots(exp_dir, output_dir, plot, key = 'accuracy', file_basename = 'accuracy', comparison_dir = comparison_dir, start_iter = start_iter, latex_report = latex_report)

        logger.info("Generating probability plots")
        GenerateAccuracyPlots(exp_dir, output_dir, plot, key = 'probability', file_basename = 'probability', comparison_dir = comparison_dir, start_iter = start_iter, latex_report = latex_report)

def Main():
    args = GetArgs()
    GeneratePlots(args.exp_dir, args.output_dir,
                  comparison_dir = args.comparison_dir,
                  start_iter = args.start_iter,
                  is_chain = args.is_chain)

if __name__ == "__main__":
    Main()
