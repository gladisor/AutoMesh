# AutoMesh

## Setup:

To setup the conda environment
```
conda env create -n automesh python=3.8
conda activate automesh
conda install pytorch torchvision torchaudio -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

pip install class_resolver
pip install pytorch
pip install torchmetrics
pip install open3d
pip install optuna
```


## Paper:
# Landmark Detection on Heart Meshes

## Report for the G-RIPS Program 2022

```
Handed in from:
Tristan Shah and Paul Zimmer
on 12.08.
```
```
Supervisors:
Stefan Zachow (ZIB, 1000Shapes), Abdullah Barhoum (1000shapes)
```
```
Code: https://github.com/gladisor/AutoMesh
```
## 1 Introduction

For the GRIPS 2022 internship, our team was assigned the task of identifying key landmark points on the
surface of a 3d mesh. Landmark Detection on 3d meshes has many uses and is an active topic of research
in the field of deep learning. One such use of landmark detection is to find corresponding points on two
separate shapes which give information on how to orient those shapes. Landmark Detection on faces is
another popular domain, however, for this project we used a small dataset of heart meshes. Our industry
sponsor (1000shapes) is interested in identifying 8 key points which will allow them to split the heart into
consistent patches. 1000shapes will feed these patches into their robust Statistical Shape Models.

One desired outcome of this project was to utilize Automatic Machine Learning (AutoML) in order to
intelligently tune the hyperparameters of our solution. This contrasts with human intuition-guided
hyperparameter optimization which is commonly used to obtain models which are deployed in production.
In this project, we construct a hierarchical hyperparameter space that encapsulates a set of solutions to the
problem of identifying the landmarks. A Tree-structured Parzen Estimator algorithm is used to sample this
space and converge on an optimal solution. Solutions are evaluated based on a metric that computes the
average distance between each predicted point and its corresponding target branching point.


## 2 Motivation

Statistical Shape Models are robust geometric models that describe a collection of semantically similar
objects in a very compact way. At 1000shapes they are used to reconstruct approximations of healthy bone
structures given some bone affected by a pathology. These models are also capable of simulating what a
3d body part would look like in a person of different ages.

For these powerful models to be applied to 3d shapes the shape meshes must adhere to a particular
condition. Each point on each mesh must have a corresponding point on all other meshes. Therefore, there
should be an equal number of points on every mesh. This condition is not easy to enforce on irregular-
shaped meshes with areas of high curvature.

One solution proposed by 1000shapes is to divide a given shape up into several contiguous
regions known as patches. These regions have correspondence between all shapes within a
class. For example, the forehead region of a skull corresponds to a forehead on any other
skull. This property is useful because a point-to-point correspondence can be more readily
made when comparing distinct regions on the mesh.

The first step in developing point-to-point correspondence is to encircle the semantically
distinct regions using line segments. Junction points that connect multiple line segments are
known as branching points. Second, one mesh is selected as a reference mesh and all of the
branching points of a patch are mapped on the border of a unit circle. The edges of the patch
are constrained to the edge of the circle and the distance between these edge points is preserved.
A corresponding patch on another mesh is selected and its points are also projected in the same
way. Now there are two invertible transformations that map points from a mesh onto a circle.
Finally, the projected points from the reference mesh are transformed with the inverse of the
non-reference mesh projection matrix. Like this, the reference mesh is projected on the new
geometry and a direct point-to-point correspondence is given.

Figure 1 : Workflow for remeshing new geometries for achieving point to point correspondence with the reference mesh.
(modified from Lamecker, Lange, and Seebaß (2004))


## 3 Related Work

### 3.1 Landmark Detection

Most landmark detection research happens in face scan analysis. However, the algorithms and ideas

behind it match mostly to the problem of landmark detection on meshes of the left atrium. Hence the field

of facial landmark detection served as the major orientation point for our project. Khabarlak and

Koriashkina (2022) listed different methods in their survey and divided them into two major groups, direct

regression methods where the exact coordinates are predicted directly for each landmark, and indirect or

heatmap-based methods where the prediction is a probability map (Khabarlak & Koriashkina, 2022).
The heatmap regression approach was especially in our interest and Wang, Bo, et al. (2019) served as an
inspiration. According to Wang, Bo, et al. (2019), a heatmap-based target is smoothing the gradients of
the loss function and it provides more information than a boolean classification would do. Furthermore,
Wang et al. (2019) propose to train on a heatmap which is representing the probability of a vertex being
the landmark or not. The heatmap is a gaussian distribution which is taking the distance of surrounding
points to the landmark into account. The parameter sigma is then scaling the size of the heatmap. Wang et
al. (2019) put the focus moreover on a specific loss function, which can be used for this kind of problem,
Adaptive Wing loss. In order to be able to perceive the small amount of true positive predictions in a
comparably higher number of true negative predictions, loss functions should be focusing on the true
positive and false negative predictions. Adaptive wing loss (Wang et al., 2019), Focal Loss (Lin, Goyal, et
al. 2017), Tversky Loss (Salehi, Erdogmus, & Gholipour, 2017) and Focal Tversky Loss (Abraham &
Khan, 2018) are all aiming to do so in different ways. As dice loss is one of the most popular loss functions
for segmentation problems, which can also suffer from the high imbalance of true positives versus true
negatives, (Sudre, Li, et al. 2017) was also evaluated. All respective loss functions can be seen in Appendix
I Table 7.

### 3.2 Auto Machine Learning

The search for suitable machine learning models and their optimal hyperparameters usually takes a high
level of expertise and a lot of time. AutoML is therefore a method that can automatically optimize data
preparation, feature engineering, hyperparameter optimization, and neural architecture search (He, Zhao,
& Chu, 2021).

## 4 Material and Methods

As it is described in Chapter 1 the goal of this project is to develop an AutoML framework, which can find
and optimize a suitable ML model with respect to the performance in automatic landmark detection on 3d
mesh data. The following chapter is describing the material and the methods which were used to get the
best results.

### 4.1 Data:

For training of the ML models, 1000shapes provided a Dataset of 69 data points. Each data point consists
of a 3d triangular mesh representation of the left atrium of a human heart and its specific eight branching
points which are stored in a separate file.


### 4.2 Software Tools

For the implementation of the model, we used several libraries. The most influential ones with a short
description are listed in the following:

PyTorch is one of the most popular deep learning frameworks used in research which has a rich ecosystem
of libraries that are built upon it.
PyTorch Lightning is used to easily adapt and upscale training to the available computational power. It
simplifies the process of distributing the computation across multiple devices in addition to providing a
clean interface to wrap models in.
PyTorch Geometric is another library that is built on PyTorch. It has most of the transformations, models,
model layers, and more already implemented. It was used to quickly explore as many options in the field
of geometric deep learning as possible.
Open3d is a library for 3d data processing which was used for loading the mesh data and preparing it for
the GCN.
Optuna is a framework that was used to automate hyperparameter optimization. It needs several specified
parameters to optimize and a metric, a single objective value, which is to be minimized. A sampler, in our
case the Optuna TPEsampler, is then used to test different hyperparameter combinations and so to explore
the solution space of a sweep. The used parameters of a single trial and the results of training and validation
can be logged to an SQLite database. Optuna - dashboard can then be used for visualizing the results in
the database. This makes it easy to track the different trials and compare hyperparameter choices.

### 4.3 Parallelism in Training

Due to the small dataset size, training times are relatively short with some variance from model size.
However, running a large number of sequential hyperparameter trials can make the problem difficult to
solve in a reasonable amount of time. One feature of the Optuna hyperparameter sweeping algorithm is
the ability to launch multiple hyperparameter configurations each in its own process. The processes
synchronize with a shared database and report their results when trials complete. New hyperparameters
are given to the processes based on all of their results. Unfortunately, due to the limitations of our cluster,
we were unable to utilize a fully functional database. An SQLite database was used instead. SQLite
databases are not optimized for synchronization between multiple processes and therefore model training
becomes very slow at scale. Instead of model parallelism across multiple processes, we utilized Distributed
Data Parallel training to maximize the speed of a single model training session. We observed a large speed
improvement when scaling from 1 to 4 GPUs.

### 4.4 Model

There are several valid approaches to Landmark Detection on meshes. These approaches include direct
regression on the landmark points, classification of points as any of the landmarks (or none), and heatmap
regression. There are undoubtedly other methods of solving this problem, however, these were the most
prominent in the literature.

As it is written in Cao, Zheng, et al. (2022) the most prominent architectures for training on 3d mesh data
are GCNs. And even if there are other options to deal with the data, we only used GCNs in the scope of
this project.

We found in preliminary experiments that the basic regression approach is successful if no random rotation
augmentation is applied to the data. However, once rotation is applied the points which are predicted by
the model always cluster at origin. Both the classification and heatmap regression approaches force the
model to select a point on the surface of the mesh. The classification approach suffers from the flaw of a


highly imbalanced dataset. Due to the vastly different number of branching points versus all vertices the
model is highly biased against predicting branching points.
The heatmap regression model predicts a scalar value for each vertex in the mesh which describes its
probability of being a specific landmark. For multiple landmarks, the output of the model will be a matrix,
where each column is the heatmap for a single landmark. The model’s loss function aims to minimize the
difference between all the predicted values for each landmark and the target heatmap. The “hot” regions
of the heatmap are very small compared to the total surface area of the mesh. Therefore, using a simple
regression loss function such as mean squared error will focus just as much on predicting true negatives as
true positives. A loss function that focuses on true positives more than any other criteria must be used.
These classes of loss functions have been shown to outperform mean squared error on heatmap regression
tasks (Wang et al., 2019).

### 4.5 The AutoMesh framework

AutoMesh is the name of the framework which was implemented in the scope of the project. We chose to
use a heatmap regression approach with GCNs because it is currently one of the most promising state of
the art techniques for automatic landmark detection on 3d mesh data (Cao et al., 2022; Khabarlak
& Koriashkina, 2022; Wang et al., 2019). AutoMesh is an AutoML framework that optimizes the
hyperparameters of chosen models automatically. The following section describes its most influential
functionalities.

4.5.1 Data Pipeline
The mesh is not inherently stored as a graph therefore several transformations must be applied before a
Graph Neural Network can be used. Firstly, edges are extracted from the triangles which form the faces
of the mesh. These edges are crucial in defining the topology of the Atrium rather than simply interpreting
the mesh as a point cloud and losing surface information. Next, the mesh is translated so that the center of
mass is located at origin. The coordinates are also scaled to a range of [-1, 1].
The data we are given does not have a standard rotation present across all hearts. Empirically we observe
that the Aorta is located in the same general region in each data point. However, there is still a great deal
of variation in its position. Therefore we attempted to make our models as robust as possible by randomly
rotating the vertex coordinates by +-50 degrees on each axis. Furthermore, we used different
transformation operations to derive different edge features.
Finally, to prepare our data for the task of heatmap regression we generated a normally distributed heatmap
for each landmark. Each vertex in the graph is assigned a value for each landmark in the interval [0, 1].
This value is computed based on its relative euclidean distance from the landmark.

4.5.2 Sweep Configuration Pipeline
Because the AutoMesh framework should find the optimal model with the optimal hyperparameters for
our task, Optuna is used as described in the section above. It optimizes the hyperparameters which are
specified within a given range with respect to minimizing a single objective value. This objective value
was chosen to be the normalized mean error between the predicted and the ground truth branching points.
With the NME as the main metric, also different loss functions can be compared as it is invariant to the
loss values itself. The default Optuna sampler TPESamlper is used and with an SQLite database, the results
of different trials can be visualized in the Optuna-Dashboard.
To provide a flexible solution space of hyperparameters to Optuna, the hyperparameter selection in
AutoMesh with is done with .yml configuration files. The .yml files contain sets of choseable categorical
options. These options are hierarchically choseable since the hyperparameter selection itself is also
partially hierarchical. On the highest level of the hyperparameter selection are the different models, the
loss functions, and the optimization algorithms which can all be chosen independently of each other. From
these choices onwards a decision tree is opened and in every trial, and different parameter choices can be
made by the Optuna sampler which are dependent on previous choices. In Figure 2 a comprehensive, short


extract of the whole decision tree can be seen. In AutoMesh the suggestions for a parameter value by
Optuna are either coming from a categorical list (e.g. Loss function), a range of float numbers, or a range
of int numbers. To keep the overview of the different levels of hierarchy and to handle the cases, where
multiple categorical options require the same hyperparameter (e.g. Adam and every other optimizer need
a learning rate (see Figure 2 )), every .yml file is restricted to two categorical options in a row. If branches
of the decision tree exceed these two levels the respective categorical parameter options need to be
specified in a new .yml file. This gets particularly important when starting from the top-level parameter
choice “model”.

Figure 2 : Comprehensive visualization of the nested solution space of the AutoMesh framework.

With this configuration pipeline up to 52 different hyperparameters were optimized using a single Optuna
sweep. The complete list of categorical parameters which could have been chosen with the hierarchical
hyperparameter suggestion algorithm with Optuna can be found in .Appendix I. The parameters are there
organized in the same way as in the original .yml files. Additionally, some of the Parameters are explained
in the following:
Model Architectures (see Appendix I, Table 3 ): The gray models in Table 3 are built-in models from the
pytorch-geometric library and ParamGCN is a customized parameterized graph convolutional neural
network, which is able to use different convolutional layers. The convolutional layers of Table 4 in
Appendix I have been tested. All layers are built-in layers of the PyTorch-Geometric library.
As described in Chapter 3.1 testing different loss functions can be of advantage. Especially loss functions
which do not reward true negatives were tested. The list of all loss functions which were tested can be seen
in Table 7 in Appendix I.
Norms, Optimizers, have also been evaluated. In both categories, already built-in PyTorch-Geometric/
PyTorch solutions have been evaluated.
The numerical ranges and the categorical options to choose from can be seen in the source code of the
AutoMesh implementation.

### 4.6 Experiment setup

In the first phase, several preliminary Optuna sweeps have been carried out. After empirically testing
different parameters, some performed significantly worse with respect to minimizing the objective value
NME than others at all times. These options have been directly excluded from further testing. In this phase
of experimenting, also different transformations were tested, too. These transformations affect the


representation of the node features and also add specific edge features. Without these transformations, no
edge features are added. For further testing, just the most promising transformations were applied.
In the second phase of experiments, one sweep was made with the leftover parameters. 500 trials and 100
epochs served to explore the solution space in a wide range efficiently. Optuna was used as it is described
in chapter 4.2.
In the third phase, the best hyperparameter combination was evaluated in a single trial running for 200
Epochs in order to fully train the model. The trained model was cross-validated.

## 5 Results

Categorical hyperparameters which significantly performed worse in terms of minimizing the NME with
Optuna were excluded after several sweeps in phase one of the experiments. The remaining parameters
out of the lists of Appendix I which were included in the extensive sweep of phase two can be seen in
Table 1

Table 1 : Empirically selected categorical parameters for further testing.

```
Category Used categorical hyperparameters for future optimization sweeps
```
```
Model ParamGCN
```
```
Convolutional
Layers
```
```
GATConv, GATv2Conv, TransformerConv (others excluded because
no edge features can be used)
```
```
Normalization GraphNorm
```
```
Activation
functions
```
```
GELU, ReLU, LeakyReLU
```
```
Loss Functions Jaccard Loss, Tversky Loss, Focal Tversky Loss
```
```
Optimizers Adam
```
Moreover, for the next phase of experiments all listed transformations of

## APPENDIX II were excluded except for PointPairFeatures which derive rotationally invariant edge

features.

After finishing a hyperparameter sweep the best configuration of parameters was selected. The
configuration that reached the lowest NME while training all models for 100 epochs is considered to be
the best available option. The hyperparameters which showed the lowest NME in all evaluated trials are
listed in Table 2.


Table 2 : Best hyperparameter values found with Optuna parameter optimization sweep.

```
Best Hyperparameter Values found with Optuna Parameter optimization Sweep
```
```
Activation Function GELU
```
```
Add Self Loops False
```
```
Graph Neural Network Layer GATv2Conv
```
```
Loss Function TverskyLoss
```
```
alpha 0.
```
```
Learning Rate 0.
```
```
Weight Decay 0.
```
```
Number of Layers 6
```
```
Number of Hidden Channels 256
```
```
Number of Attention Heads 4
```
```
Dropout 0.
```
```
Batch Size 1
```
```
Sigma (heatmap width) 0.
```
```
Fixed Hyperparameters Chosen Empirically:
```
```
Model ParamGCN
```
```
Optimizer Adam
```
```
Normalization GraphNorm
```

Figure 3 displays the history of the Optuna optimization process over the span of 400 trials. The parameter
selection is conducted with a built-in algorithm called Tree Structured Parzen Estimation. The optimization
process converges on a set of hyperparameters which yield an NME of approximately 3.5 millimeters.

Figure 3 : Plot containing the minimal objective values of each trial from the extensive Optuna parameter optimization sweep.

The Optuna dashboard gives, besides the final best hyperparameter selection also insights into the
optimization process. Figure 4 shows the influence of each hyperparameter on the objective value. In some
cases, this importance score may not necessarily indicate an important hyperparameter but rather one or
more option yields an extremely poor score.

Figure 4 : Hyperparameter Importance from extensive Optuna sweep


According to the sweep results. The number of layers in the model is by far the most influential
hyperparameter; accounting for 68% of the variation in objective value. Optuna allows the inspection of
objective value with respect to the chosen hyperparameter as it can be seen in Figure 5 :

Figure 6 : Lowest objective values (NME) for each hyperparameter co

Interestingly, in Figure 6 the optimization does not simply choose the maximum number of allowed layers
(8). Instead, it shows that objective value improves as layers increase up to 6. Higher values beyond 6
result in diminishing performance.

In the third phase of experiments, the trained with the parameters from Table 2 and afterward cross-
validated 5 times With the parameters from Table 2 on different random splits of 64 training points and 4
validation points. The training and validation loss curves as well as the normalized mean error curves were
averaged over the 5 runs and plotted. The best average NME appears to be between 6 and 8 millimeters.
However, on certain validation splits this error was as low as 3.5 millimeters.


Figure 7 : Training results after cross-validation with the best-found hyperparameter set from Table 2.

## 6 Discussion

Optuna – dashboard gives insights as they can be seen in Figure 6 to every evaluated hyperparameter. Like
this, categorical some categorical hyperparameters were tested in preliminary tests but directly excluded
as they significantly performed worse compared to the other options.
The AutoML approach was shown to optimize all relevant hyperparameters successfully to a local
minimum. More hyperparameters could have been added to the search but due to the limited time of the
project, the ones, which, from a subjective point of view, seemed to be relevant could be explored to
sufficient optimality. It can be seen in Figure 3 , that the minimal NME was significantly decreased over
the different trials of the sweep.
Figure 7 shows further that the chosen GCN base model within the heatmap regression approach
could be trained to convergence. The training loss and validation loss is decreasing over the
number of training epochs. However, after 75 epochs the training and validation loss begin to
diverge. The final cross-validated NME can be seen in Figure 7 which is between 6 and 8
millimeters. Interestingly this distance is almost double what was achieved in the
hyperparameter sweep conducted in Figure 4. It appears that the train / val split which is used
is highly influential on the final performance of the model. Adding more validation data could
potentially resolve this issue, however, it would require removing training examples. As this
approach is already training with a small dataset, it is unwise to reduce it further.


## 7 Conclusion and Outlook

In this project, we developed a solution to the pacification problem given by 1000shapes. We were able to
demonstrate that a GNN-based, heatmap regression approach is capable of learning to select branching
points on a mesh. While the error of its predictions is quite high once cross-validation, we believe that this
is not a reflection of the approach to the problem but a lack of data that is usable by the model. We can
increase the number of datapoints through additional augmentation processes. So far only randomized
rotations were used for augmentation. However, we can create new geometries by randomly scaling the
points along an axis through stretching and shrinking. Alternatively, more data can be collected and passed
to the model. Increasing the number of data points from 68 to several hundred could yield huge
performance improvements

## References

Abraham, N., & Khan, N. M. (2018, October 18). A Novel Focal Tversky loss function with improved
Attention U-Net for lesion segmentation. Retrieved from https://arxiv.org/pdf/1810.

Cao, W., Zheng, C., Yan, Z., & Xie, W. (2022). Geometric deep learning: Progress, applications and
challenges. Science China Information Sciences, 65 (2). https://doi.org/10.1007/s11432- 020 - 3210 - 2

He, X., Zhao, K., & Chu, X. (2021). AutoML: A survey of the state-of-the-art. Knowledge-Based
Systems, 212 (3), 106622. https://doi.org/10.1016/j.knosys.2020.1 06622

Khabarlak, K., & Koriashkina, L. (2022). Fast Facial Landmark Detection and Applications: A Survey.
Journal of Computer Science and Technology, 22 (1), e02. https://doi.org/10.24215/16666038.22.e

Lamecker, H., Lange, T., & Seebaß, M. (2004). Segmentation of the Liver using a 3D Statistical Shape
Model. ZIB-Report, 04 - 09.

Lin, T.‑Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017, August 7). Focal Loss for Dense Object
Detection. Retrieved from https://arxiv.org/pdf/1708.

Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017, June 18). Tversky loss function for image
segmentation using 3D fully convolutional deep networks. Retrieved from
https://arxiv.org/pdf/1706.

Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Jorge Cardoso, M. (2017). Generalised Dice
Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. Retrieved from
https://arxiv.org/pdf/1707.03237 https://doi.org/10.1007/978- 3 - 319 - 67558 - 9_

Wang, X., Bo, L., & Fuxin, L. (2019, April 16). Adaptive Wing Loss for Robust Face Alignment via
Heatmap Regression. Retrieved from https://arxiv.org/pdf/1904.


## Appendix I: Categorical Parameters of the Solution Space

Table 3 : List of different used base models and their modified hyperparameters. Grey marked models are built in models
from pytorch-geometric

```
Model: Mofied Parameters:
```
ParamGCN (^) act, conv_layer, hidden_channels, in_channels, out_channels, norm, num_layers,
GraphUnet depth, hidden_channels, in_channels, out_channels, pool_rations
Table 4 : List of different Convolutional Layers and their modified hyperparameters that were used in combination with
ParamGCN. Layers with the parameter ‘edge_dim’ can process edge features. All Convolutional Layers are built in layers
from pytorch-geometric.
Convolutional
Layer
Modified Parameters
GCNConv In_channels, out_channels, add_self_loops
SAGEConv In_channels, out_channels, aggr, root_weigt, project
GraphConv In_channels, out_channels, aggr
GATConv In_channels, out_channels, heads, concat, dropout, heads, edge_dim,
add_self_loops
GATv2Conv In_channels, out_channels, heads, concat, dropout, heads, edge_dim,
add_self_loops
TransformerConv In_channels, out_channels, heads, concat, heads, edge_dim, beta
Table 5 : list of different normalizations that were used. All norms are built in modules from pytorch-geometric.
Normalizations
GraphNorm
LayerNorm


Table 6 : List of different activation functions that were used and their modified hyperparameters. All activation functions are
built in modules from pytorch.

```
Activation Functions Modified Parameters
```
```
ELU -
```
```
GELU -
```
```
LeakyReLU negative_slope
```
```
ReLU -
```
```
Sigmoid -
```
```
Tanh -
```
Table 7 : List of different loss functions, that were used and their modified hyperparameters and forumla.

```
Loss Function Modified
Parameters
```
```
Formula
```
```
Binary Cross Entropy
loss?
```
```
Ln=−ω[yn∗logxn+( 1 −yn)∗log( 1 −xn)]
```
```
Adaptive Wing Loss Omega, Theta
AEing(y,ŷ)={ωln(^1 +|
```
```
y−ŷ
ε |
```
```
α−y
)if |y−ŷ|<θ
A |y−ŷ|−C otherwise
```
Dice Loss - (^) Ldice=^2 ∗∑ptrue∗ppred
∑ptrue^2 +∑ppred^2 +ε^
BCE Dice Loss -
J(w)=^1 N∑H(pn,
N
n= 1
qn)= −^1 N∑[ynlogŷn+( 1 −yn)log ( 1 −ŷn)]
N
n= 1
Jaccard Loss - (^) J(A,B)=|A∩B|
|A∪B|^
Focal Loss Alpha, Gamma FL(pt)=−(^1 −pt)γlog^ (pt)^
Tversky Loss Alpha (^) TL= TP
TP+aFN+βFP^
Focal Tversky Loss Gamma (^) FTL= ∑(
c
1 −TLc)
(^1) γ


Table 8 : List of different optimizers that were used and their modified hyperparameters. All optimizers are built in modules
from pytorch.

```
Optimizer Modified Parameters
```
```
Adam learning_rate, weight_decay
```
```
Adagrad Learning_rate, learning_rate_decay, weigh_decay
```
```
SGD Learning_rate, momentum, weight_decay
```
## APPENDIX II: Empirically tested transformations and edge features:

Table 9 : Empirically tested transformations and edge features. All Transformations of the data are built in pytorch-geometric
transforms modules

```
Transformation Description
```
```
Cartesian Saves the relative Cartesian coordinates of linked nodes in its edge
attributes
```
```
Spherical Saves the spherical coordinates of linked nodes in its edge attributes
```
```
PointPairFeatures Computes rotation-invariant Point Pair Features and sets them as edge
features
```
```
VirtualNode Appends a virtual node to the given homogeneous graph that is connected
to all other nodes
```
```
SVDFeatureReduction Dimensionality reduction of node features via Singular Value
Decomposition
```
