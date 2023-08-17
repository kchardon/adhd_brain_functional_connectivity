# ADHD Brain Functional Connectivity with Omega

## Introduction

During my internship, I worked on brain age prediction \[[1](#brain-age1), [2](#brain-age2)\] with the Omega dataset \[[3](#omega-paper), [4](#omega-data)\]. Unfortunately, we didn't obtain good scores for this task with Omega (see my internship report for details [here]()). We wanted to understand why age prediction didn't work: did we apply wrong preprocessing or was the dataset just not adapted for this task?

We thus decided to reproduce some results from other papers using Omega with our preprocessing. To do this, we chose \[[5](#adhd-connectivity)\] in which the authors use MEG resting-state recordings from Omega to detect ADHD based on Brain Functional Connectivity.

## Method

In \[[5](#adhd-connectivity)\], the authors present a classification between Control subjects and subjects with ADHD. However, in the version of Omega that I used, I didn't have any information on ADHD and the different classes were Control, Parkinson and Chronic Pain. I thus decided to make a classification between Control and Parkinson's using Omega.

To do that, I first wrote a function (in *get_data_omega.py*) to collect Omega data, compute the coherence (connectivity) and get the class for each subject. At first, I used the function *spectral_connectivity_epochs* from MNE \[[8](#mne)\] and MNE-Connectivity \[[9](#mne-connectivity)\] but the resulting coherence matrices were not as expressive as the in the paper. I thus wrote a function in *compute_coh.py* to compute the coherence with the same formula as in \[[5](#adhd-connectivity)\] :

For each band $\in$ { $\delta, \theta, \alpha, \beta, \gamma$ }, for each pair of sensors $xy$, the coherence is computed as

$$COH^{band}\_{xy} = \sum\_{f \in band}{\frac{|S_{xy}(f)|^2}{S_{xx}(f)S_{yy}(f)}}$$

with $S_{xy}(f)$ the cross-power spectral density, and $S_{xx}(f)$ and $S_{yy}(f)$ the respective auto-power spectral densities.

With the coherence computed like this, the matrices were still pretty bad. To see if it was a dataset problem, I wanted to test other ones. Some other studies have been made on the same task with MEG datasets \[[11](#conn), [12](#conn2)\] but the datasets were not available. We thus decided to test the model on an EEG dataset that we had \[[10](#ds)\]. The code to get the data is available in *get_data_ds.py*. Furthermore, other studies using EEG datasets for classification between Control subjects and subjects with Parkinson's are available \[[13](#eeg), [14](#eeg2)\], which makes it possible to compare the results found in this second task.

In *select_features.py*, I use Neighborhood Component Analysis (NCA) for feature selection. I use the package *ncafs* from \[[6](#ncafs)\] that implements NCA feature selection presented in \[[7](#nca)\] and which is used in \[[5](#adhd-connectivity)\]. To select the features, I run a leave-one-out and fit the NCA on the training set. I store the 5 selected features at each fold and, at the end, select the features that appear in at least 50% of the folds. If fewer than 5 features are selected at the end, I take the most selected features to have 5 features to use.

I made a function to get the name of the selected features in *get_features.py* to compare the features with the paper.

Finally, in *model_classif.py*, I wrote the function to test the three different models : SVM with RBF, KNN with k = 3 and Decision trees with a leave-one-out cross-validation.

## Results



## References

<div id="brain-age1">[1] David Sabbagh et al. <em>Predictive regression modeling with MEG/EEG : from source power to signals and cognitive states</em>. In : NeuroImage 222 (2020), p. 116893. https://doi.org/10.1016/j.neuroimage.2020.116893</div>

<div id="brain-age2">[2] Denis A. Engemann et al. <em>A reusable benchmark of brain-age prediction from M/EEG resting-state signals</em>. In : NeuroImage 262 (2022), p. 119521. https://doi.org/10.1016/j.neuroimage.2022.119521.</div>

<div id="omega-paper">[3] Guiomar Niso et al. <em>OMEGA : The Open MEG Archive</em>. In : NeuroImage 124 (2016), p. 1182-1187. https://doi.org/10.1016/j.neuroimage.2015.04.028</div> 

<div id="omega-data">[4] <em>Open MEG Archive Repository</em>. https://doi.org/10.23686/0015896</div>

<div id="adhd-connectivity">[5] Nastaran Hamedi et al. <em>Detecting ADHD Based on Brain Functional Connectivity Using Resting-State MEG Signals</em>. In : Frontiers in Biomedical Technologies 9 (2022), p. 110-118. https://doi.org/10.18502/fbt.v9i2.8850</div>

<div id="ncafs">[6] Pedro Paiva. <em>Neighborhood Component Analysis Feature Selection library</em>. https://pypi.org/project/ncafs/</div>

<div id="nca">[7] Wei Yang et al. <em>Neighborhood component feature selection for high-dimensional data</em>. In : JCP 7 (2012), p. 161-168. https://doi.org/10.4304/jcp.7.1.161-168</div>

<div id="mne">[8] Alexandre Gramfor et al. <em>MEG and EEG data analysis with MNE-Python</em>. In : Frontiers in Neuroscience 7 (2013), p. 1–13. https://doi.org/doi:10.3389/fnins.2013.00267</div>

<div id="mne-connectivity">[9] <em>MNE-Connectivity</em>. https://mne.tools/mne-connectivity/stable/index.html</div>

<div id="ds">[10] Arun Singhet al. <em>Rest eyes open</em>. In : OpenNeuro (2023). https://doi.org/doi:10.18112/openneuro.ds004584.v1.0.0</div>

<div id="conn">[11] Amine Khadmaoui et al. <em>MEG Analysis of Neural Interactions in Attention-Deficit/Hyperactivity Disorder</em>. In : Computational Intelligence and Neuroscience (2016), p. 1-10. https://doi.org/10.1155/2016/8450241</div>

<div id="conn2">[12] Muthuraman Muthuraman et al. <em>Multimodal alterations of directed connectivity profiles in patients with attention-deficit/hyperactivity disorders</em>. In : Scientific Reports 9 (2019), p. 20028. https://doi.org/10.1038/s41598-019-56398-8</div>

<div id="eeg">[13] Rajamanickam Yuvaraj et al. <em>Inter-hemispheric EEG coherence analysis in Parkinson’s disease: Assessing brain activity during emotion processing</em>. In : J Neural Transm 122 (2015), p. 237–252. https://doi.org/10.1007/s00702-014-1249-4</div>

<div id="eeg2">[14] Mariana Gongora et al. <em>EEG Coherence as a diagnostic tool to measure the initial stages of Parkinson Disease</em>. In : Medical Hypotheses 123 (2018), p. 74-78. https://doi.org/10.1016/j.mehy.2018.12.014</div>
