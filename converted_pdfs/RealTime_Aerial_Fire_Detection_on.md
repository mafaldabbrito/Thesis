# Real-Time Aerial Fire Detection on

Real-Time Aerial Fire Detection on
Resource-Constrained Devices Using Knowledge
Distillation

Sabina Jangirova, Branislava Jankovic, Waseem Ullah, Latif U. Khan, Mohsen Guizani

1

5
2
0
2

b
e
F
8
2

]

V
C
.
s
c
[

1
v
9
7
9
0
2
.
2
0
5
2
:
v
i
X
r
a

Abstract—Wildfire catastrophes cause significant environmen-
tal degradation, human losses, and financial damage. To mitigate
these severe impacts, early fire detection and warning systems
are crucial. Current systems rely primarily on fixed CCTV
cameras with a limited field of view, restricting their effectiveness
in large outdoor environments. The fusion of intelligent fire
detection with remote sensing improves coverage and mobility,
enabling monitoring in remote and challenging areas. Existing
approaches predominantly utilize convolutional neural networks
and vision transformer models. While these architectures provide
high accuracy in fire detection, their computational complexity
limits real-time performance on edge devices such as UAVs. In
our work, we present a lightweight fire detection model based on
MobileViT-S, compressed through the distillation of knowledge
from a stronger teacher model. The ablation study highlights the
impact of a teacher model and the chosen distillation technique
on the model’s performance improvement. We generate activation
map visualizations using Grad-CAM to confirm the model’s
ability to focus on relevant fire regions. The high accuracy
and efficiency of the proposed model make it well-suited for
deployment on satellites, UAVs, and IoT devices for effective fire
detection. Experiments on common fire benchmarks demonstrate
that our model suppresses the state-of-the-art model by 0.44%,
2.00% while maintaining a compact model size. Our model
delivers the highest processing speed among existing works,
achieving real-time performance on resource-constrained devices.

Index Terms—Aerial images; Knowledge distillation; Vision
Transformer; Convolution Neural Network; Fire detection; Wild-
fires.

I. INTRODUCTION
Over many years, fire remains one of the most significant
natural disasters, dangerous in its destructive character and
speed of spreading. In 2023, the northern parts of Kazakhstan
experienced massive wildfires, which burned more than 60,000
hectares of forest and killed 15 people [1]. Together with the
loss of lives and environmental damage, fires lead to financial
harm. These consequences can be reduced by early detection
and correct classification of the ignited fire, ensuring a reactive
intelligent fire detection systems are
response. Nowadays,
mainly deployed on CCTV cameras, which have a fixed line of
vision and position. Therefore, there is a high risk of missing
the start of the fire if it is located in the ”blind” spot not
covered by the cameras. Thus, there is still a need for more
reliable solutions capable of classifying different types of fire
under varied conditions and complex environments. By using
unmanned aerial vehicles, we can monitor much larger, distant
territories and also come closer to suspicious objects if it is
hard to classify them as fire or non-fire. The constraint of such

devices is that they have limited storage and computational
capabilities. Therefore, they won’t be able to utilize computer
vision models with large, complex architectures. Motivated
by the described challenges, our research takes advantage of
state-of-the-art (SOTA) vision transformers and knowledge
distillation (KD) techniques to enhance the accuracy and
efficiency of fire detection systems on remote sensing devices,
contributing to more effective fire prevention and management.
In the early stages, fire detection was performed using
scalar sensor-based methods, such as the installation of smoke,
particle, temperature, and flame detection sensors [2]. These
methods are cheap and easy to install, but scalar sensors can
monitor only indoor environments and thus have limited usage
scenarios. Vision sensor-based methods work with video and
image data, presenting a broad region coverage, reduced hu-
man intervention requirements, rapid response times, environ-
mental resilience, and additional information on fire character-
istics (e.g. the size of the affected area). Mostly, conventional
machine learning (CML) and deep learning (DL) models are
used for these methods [3]. CML methods commonly employ
features such as motion, color, shape, and texture [4], [5], and
the performance of such models is correlated with the quality
of the features. Moreover, they failed to generalize to cases
with poor weather conditions and complex, unseen scenarios.
Alternatively, DL methods proved to be effective in extract-
ing characteristics, especially for the fire detection task [6]–
[11]. A more complex architecture of such models allows for
capturing intricate patterns and dependencies from the images.
DL methods have demonstrated a prominent ability to enhance
classification performance in adverse weather conditions and
complex scenes, further validating their use despite the in-
creased computational demands.

Although DL models reduce false alarm rates compared
to CML models, they require heavy computations and have
limited capabilities to distinguish between fire and fire-like ob-
jects [12]. Many researchers developed solutions to overcome
these limitations ( [3], [13]–[21]), yet there’s usually a trade-
off between the model’s processing speed and the accuracy of
predictions. In practical scenarios, quick detection and correct
response to the forming fire is necessary to prevent significant
losses. This limitation drives the demand for solutions that
balance accuracy with efficiency, enabling their deployment
in real-time applications.

To address these challenges, we propose a novel approach
utilizing MobileViT-S [22] as the best backbone model op-
timized with KD techniques. Our approach combines the

precision of large-scale teacher models with the compactness
and efficiency of student models, allowing for robust and
scalable fire detection systems. The main contributions of this
work include:

• We develop a model for fire detection employing the
best backbone and implementing a KD approach, trans-
ferring crucial insights from a larger teacher model to a
lightweight student model.

• We conduct an extensive ablation study to evaluate the
effectiveness of the proposed teacher model, student
model, and KD technique for performance.

• We evaluated the performance of our model on three fire
classification benchmarks. Our model achieves the same
results and even exceeds the accuracy of the existing
methods while being significantly more compact.

• We demonstrate our model’s ability to focus on relevant
areas within the images by using the Grad-CAM tool.
Our proposed method provides meaningful information
on the decision-making process of our proposed model,
increasing its explainability.

Section II describes the related work in the domain. Section
III contains the framework proposed in this project, while Sec-
tion IV discusses the experiments and their results and presents
the ablation study on the effect of different teacher models
and KD techniques. Finally, Section V presents the conclusion
of the work done in this research, possible implications, and
future directions.

II. RELATED WORK
Early CML methods for fire detection relied on color analy-
sis and image processing techniques to extract fire and smoke
features [23]–[27], while later approaches integrated motion
features, such as optical flow and spatiotemporal analysis
[28]–[30]. Chen et al. [23] proposed a method that used
color segmentation in the RGB color space to isolate fire-
like regions in images, while Marbach et al. [24] explored
dynamic color modeling to adapt to varying fire hues. Celik
and Demirel [26] introduced a statistical model for fire pixel
detection based on brightness and color properties. Borges and
Izquierdo [27] took a probabilistic approach, combining color
and temporal information for fire classification. However, these
methods suffered from high false alarm rates due to the diverse
characteristics of fire. To improve robustness, later methods
incorporated motion features. Foggia et al. [28] employed
optical flow to capture the dynamic nature of fire, while Chen
et al. [29] used spatiotemporal analysis to distinguish between
fire and non-fire motion patterns. Ha et al. [30] combined
motion and texture features to enhance detection reliability.
Recently, Xu et al. combined a Modified Pixel Swapping
Algorithm with mixed-pixel unmixing and threshold-weighted
fusion to detect forest fires, which improved accuracy and
reduced false alarms [5]. Although these methods reduced
false alarms to some extent, they struggled in scenarios with
camera movements or other moving objects that could mimic
fire behavior.

Deep learning (DL) methods, particularly convolutional
neural network (CNN) models, have shown improved perfor-
mance in fire detection [16], [31], [32]. Lightweight CNNs

2

like those proposed by Muhammad et al. [6] and Daoud et
al. [19] addressed computational constraints, but challenges
in detecting small, distant fires in adverse conditions remain.
Some researchers employed CNN-based models with attention
mechanisms. Li et al. proposed a fire detection approach with
multiscale feature extraction, deep supervision, and channel
attention mechanism [17]. Wang et al. proposed a Dynamic
Equilibrium Network to detect fire based on the data from
different types of sensors [9]. In [13], the authors integrated the
spatial attention (SA) and channel attention (CA) modules into
the Inceptionv3 architecture and improved the performance
of the backbone model. Similarly, [15] introduced the SA
and CA modules to the ConvNeXtTiny architecture. Yar et
al. [3] proposed a modified MobileNetV3 architecture with
an added Modified Soft Attention Mechanism (MSAM) and
3D convolutional operations. Dilshad et al. [20] developed an
optimized fire attention network (OFAN) that consisted of a
MobileNetV3Small as a backbone model, CA and SA mech-
anisms to capture global dependencies. Rui et al. developed
a multi-modal RGB-T wildfire segmentation framework that
learns both modality-specific and shared features via parallel
encoders and a shared decoder [11]. Alternatively, Yar et
al. [21] proposed a ViT-inspired model with a shifted patch
tokenization (SPT) module for spatial details, a locality self-
attention (LSA) module to optimize the softmax temperature,
and dense layers instead of the multi-head to reduce the
complexity of the model. However, these methods still need
more robustness and capability to capture small fire regions in
complex scenes, like fog or hazy weather.

The challenge of detecting small, distant fire sources in
poor weather persists in the current research. While CNNs
with attention modules enhance feature representation through
channel-wise attention, they primarily focus on local spatial
information and channel dependencies. To address this limi-
tation, we utilize the architecture that combines the efficiency
of CNNs with the global modeling capabilities of Vision
Transformers [22].

III. PROPOSED FRAMEWORK

The challenges mentioned in the previous section must be
addressed with more sophisticated approaches. We propose a
framework for developing an effective, compact, and robust
model for fire detection. This section describes the model
architecture and the training process with KD. The overall
process is depicted in Fig. 1 and Algorithm 1. The training
phase begins with the pretraining of the teacher model, ViT-
Base Patch32 (ViT/32), on the selected fire dataset. The
teacher model learns to extract complex features through its
transformer-based architecture. Then, the student model, based
on MobileViT [22], is trained using a KD framework, where
the teacher model guides the student by transferring its learned
knowledge. This process ensures the student model inherits
the teacher’s capability to recognize fire-related patterns while
maintaining a more compact and lightweight architecture.
Once training is complete, the trained student model is de-
ployed on monitoring devices, such as drones. These drones
patrol assigned areas, periodically capturing aerial images of

3

Fig. 1: Proposed framework for fire detection using KD. The training phase involves distilling knowledge from a transformer-
based teacher model (ViT/32) to the student model (MobileViT-S). The teacher model processes image patches with linear
projections, positional embeddings, and a transformer encoder to produce logits, which guide the student model’s learning
through the distillation loss (LKD). The student model combines convolutional layers and MobileViT modules to efficiently
learn both local and global features. The trained student model is deployed on resource-constrained devices, such as drones,
for real-time fire detection. The framework enables effective identification of fire regions, as illustrated by attention heatmaps
generated during inference.

the environment. The deployed model processes these images
in real-time, accurately detecting fire instances. This approach
enables an efficient response to potential fire hazards, which
can be detected even by resource-constrained gadgets.

1) Feature extraction and Model Architecture: The pro-
posed model’s architecture uses transformers as convolutions
[22]; in other words, by using a stack of transformers, the
MobileViT module can capture global representations while
also keeping the spatial order of pixels. The architecture
layer, followed by Mo-
begins with a 3×3 convolutional
bileNetV2 blocks, to extract local spatial features, capturing
fine-grained details essential for object recognition. To model
long-range dependencies and global context, the architecture
utilizes MobileViT blocks. In these blocks, feature maps are
unfolded into non-overlapping patches and processed using
transformer layers without losing the spatial order of pixels
within each patch. This approach maintains spatial inductive
bias, preserving critical spatial relationships. The patches
are folded back to reconstruct the feature map with local
and global representations. This reconstructed feature map
is projected back to a lower-dimensional space using point-
wise convolutions and combined with the original features via
concatenation. A final convolutional layer is then used to fuse
these combined features. In fire detection, MobileViT’s ability

to model fine-grained details and global context enhances its
capability to detect small or distant fires under challenging
conditions.

2) Teacher Model Architecture: The teacher model, ViT/32,
processes an input image by dividing it into non-overlapping
patches, each of size 32x32 pixels, which are then linearly
projected into a fixed-dimensional embedding space using a
fully connected layer. To preserve the spatial order of the
patches, positional encodings are added to these embeddings.
The embedded patch tokens are then fed into a transformer
encoder, which comprises multiple layers of multi-head self-
attention and feed-forward networks. The self-attention mech-
anism allows the model to capture global dependencies across
the entire image, enabling it to understand both local and
contextual information critical for tasks like fire detection. The
final output from the encoder is passed through a multilayer
perceptron (MLP) head to generate logits representing the
model’s predictions [33].

3) Knowledge Distillation: KD is a technique that allows
the transfer of knowledge from a complex model or an
ensemble of models, known as a ”teacher” model, to a simpler,
smaller ”student” model [34]. We employ KD because it is
critical that the compact and fast-inference model deployed in
UAVs and surveillance systems also has high performance.

Algorithm 1 Teacher Model Training and KD Framework
Require: Dataset D = {(xi, yi)}N

i=1, Teacher model Mt,
Student model Ms, Temperature T , Weighting factor α,
Learning rates ηt, ηs, Number of epochs Et, Es.

1: Teacher Model Training:
2: Initialize Mt.
3: for epoch e = 1 to Et do
Shuffle dataset D.
4:
for each mini-batch B = {(xb, yb)} in D do
5:
Compute teacher predictions st = Mt(xb).
6:
LCE
Calculate
7:
(cid:80)|B|
1
|B|
Update Mt parameters using optimizer: θt ← θt −
ηt∇θtLCE.

cross-entropy
b=1 CrossEntropy(st, yb).

loss:

=

8:

end for

9:
10: end for
11: KD to Student Model:
12: Initialize Ms.
13: for epoch e = 1 to Es do
Shuffle dataset D.
14:
for each mini-batch B = {(xb, yb)} in D do
15:
st
16:

predictions

Compute
teacher
Softmax(Mt(xb)/t).
Compute
student
Softmax(Ms(xb)/t).
Calculate distillation loss:

predictions

ss

=

=

4

based on extensive experiments. Their results can be found in
subsection IV-D.

IV. EXPERIMENTAL RESULTS

This section describes the experimental setup, datasets used
for evaluation, the performance and visual evaluation results,
the complexity of our proposed model, and the ablation study.

A. Model Implementation Details and Evaluation Metrics

The proposed fire detection model was implemented using
the PyTorch deep learning framework. We conducted the
experiments on one NVIDIA A100 GPU and AMD EPYC
7402 CPU with a 2.80 GHz processor. The model was trained
for 300 epochs with early stopping after 10 epochs, using a
batch size of 32, and images had a resolution of 224x224.
The training was done using a learning rate of 1e-4, AdamW
optimizer with a weight decay of 1e-3 to prevent overfitting.
We divided all datasets into train, validation, and test splits
with 70%, 20%, and 10% of images, respectively, applying
the approach from previous research for fair comparison.

The performance evaluation metrics include precision (P),
recall (R), F1-score (F1), and accuracy (Acc). These metrics
provide a fundamental assessment of the model’s effectiveness
in making accurate predictions across the entire dataset [7],
[13], [35].

LKD = T 2 · LKLD(st, ss).

B. Datasets

loss:

cross-entropy
b=1 CrossEntropy(ss, yb).

Compute
(cid:80)|B|
1
|B|
Combine losses: L = (1 − α)LCE + αLKD.
Update Ms parameters using optimizer: θs ← θs −
ηs∇θsL.

LCE

=

end for

22:
23: end for
24: return Ms.

In our proposed framework, we implement soft target KD
as described in [34]. The soft target KD involves training
the student model using the teacher model’s softened output
probabilities (soft targets). Specifically, the total loss L can be
expressed as:

L = (1 − α)LCE(y, ys) + αT 2LKLD(st, ss),

(1)

where LCE(y, ys) is the cross-entropy loss between the true
labels y and the predicted probabilities of the student model
ys. LKLD(st, ss) is the Kullback-Leibler divergence (KLD)
between the teacher’s soft targets teacher st and the student’s
output ss, that are computed with a temperature-scaled soft-
max function. T is the temperature parameter, and α is a
weighting factor.

We use ViT/32 [33] as the teacher model to implement
the KD techniques. This combination of the teacher model
architecture and the KD technique proved the most effective

In this section, we present the datasets used to evaluate the
performance of our model. Some sample images from each
dataset are depicted in Fig. 2.

1) BowFire: The BoWFire dataset [12] is a small-scale fire
detection dataset consisting of 119 fire images and 107 non-
fire images of different resolutions. The fire images present
various emergency scenarios, while non-fire images contain
images without fire and images with fire-like objects (sunsets,
red and yellow objects).

2) ADSF: The ASDF dataset was introduced in [35], con-
taining images from drones and satellites. This dataset consists
of 3000 fire images and 3000 normal images shot outdoors.
The ADSF dataset provides a range of images in different
conditions, such as time of the day, landscape, and altitude.

3) DFAN: The DFAN dataset is a medium-scale dataset that
consists of 3,804 images of different fire scenarios, split into
12 imbalanced classes. Proposed by Yar et al. [13], this dataset
challenges models with the diversity of classes. Training a
model on this dataset allows us to identify the characteristics
of the fire and respond to it according to the level of the crisis.

C. Visual Results

The visual results showcased in Fig. 3 demonstrate the
effectiveness of the proposed model in localizing fire regions
across diverse environments. The left column displays the in-
put images, while the right column presents the corresponding
attention heatmaps visualized with Grad-CAM. Input images
and attention heatmaps demonstrate the model’s ability to

17:

18:

19:

20:
21:

5

BoWFire: Fire

ADSF: Fire

DFAN: Forest Fire

Fig. 2: Sample images from the fire benchmarks showcasing the diverse nature of fire detection scenarios. Each image is
labeled with its respective class for training and evaluation purposes.

detect flames in drone and satellite imagery, even in chal-
lenging scenarios. For instance, in the first row, the model
correctly highlights the area of active flames in a satellite
image of a building fire. Similarly, in the second row, the
model successfully identifies fire spread over vegetation in a
drone-captured image. However, limitations are observed, such
as misinterpreting clouds as fire smoke due to their visual
similarity. This signifies the need for further improvement in
distinguishing fire-related features from non-fire elements in
complex scenes.

Moreover, Fig. 4 displays sample images from the DFAN
dataset, the predictions made by our proposed model, and
the ground truth labels for each image. Our model clearly
differentiates between visually unalike classes but can make
mistakes in related classes. For example, ”Car Fire”, ”SUV
Fire”, and ”Van Fire” classes are often confused with each
other. These examples highlight the model’s challenges in
handling visually related categories, particularly in scenarios
where subtle differences in object shape or fire intensity can
mislead predictions.

The visual evaluation provides valuable insights into the

possible improvement directions of the model.

D. Ablation study

We conducted numerous experiments to distill knowledge
from stronger models to improve our proposed model’s per-
formance on ADSF and DFAN datasets. The knowledge
techniques used in the experiments include soft target KD [34],
Distillation from A Stronger Teacher (DIST) [36], and One-
for-All KD (OFA-KD) [37].

Moreover, we employed ViT/32 [33] from the transformers
family and ConvNeXt-Base (ConvNeXt) [38] from the CNN
family as the teacher models. These architectures were selected
for their ability to provide different knowledge to our proposed
model. The student models tested included MobileViT-S and
MobileViT-XS to examine the effects of model size on perfor-
mance. The baseline performance of MobileViT-S on the test
splits of the DFAN and ADSF datasets without KD is 90.29%

Fig. 3: Visualization of the model attention on drone and
satellite images. The left column displays the input images,
while the right column presents the corresponding Grad-CAM-
based attention visualizations. The top row shows a fire in an
urban environment captured by a drone, with the attention map
clearly highlighting the fire region amidst surrounding objects.
The model is able to effectively focus on fire regions across
diverse environmental conditions and input modalities.

and 95.50%, respectively. For MobileViT-XS, the performance
is 87.93% and 94.00%.

Given resource constraints, we avoided an exhaustive grid
search for hyperparameter optimization. Instead, we incremen-
tally optimized individual parameters based on their observed
impact on performance. For soft target KD, we found that
T = 2 and α = 0.1 provided the best balance between the

TABLE I: Performance comparison of our model using different KD techniques on the fire benchmarks. The table highlights
the accuracies achieved by the MobileViT-S and MobileViT-XS student models under three distillation techniques: Soft Target
KD, DIST, and OFA. The results in bold signify the best accuracies for each model architecture.

6

Dataset

Teacher

DFAN

ADSF

ViT/32
ConvNeXt
ViT/32
ConvNeXt

Soft Target KD

DIST
MobileViT-S MobileViT-XS MobileViT-S MobileViT-XS MobileViT-S MobileViT-XS

OFA

91.08
88.98
91.33
92.00

90.55
88.71
94.83
92.33

89.50
90.55
94.17
95.00

89.24
89.76
93.00
95.00

89.76
88.71
95.33
95.17

86.09
88.45
95.00
95.50

for MobileViT-S, indicating that the global feature extraction
of ViT/32 is better suited for challenging datasets like DFAN.
On the ADSF dataset, MobileViT-XS achieves the best
accuracy of 95.50% using OFA-KD with ConvNeXt as the
teacher. This result slightly surpasses MobileViT-S, which
achieves a maximum accuracy of 95.33% under the same
configuration. The ADSF dataset, with only two classes,
is relatively simpler than DFAN, making it less reliant on
the global contextual features provided by ViT/32. Conse-
quently, the lightweight MobileViT-XS model performs com-
petitively on this dataset. Interestingly, the baseline accuracy
for MobileViT-S on ADSF is already 95.50%, indicating that
the dataset’s simplicity limits the impact of KD.

While MobileViT-XS achieves competitive performance on
the ADSF dataset, MobileViT-S outperforms it on the more
challenging DFAN dataset, with an accuracy of 91.08% com-
pared to 90.55%. This suggests that MobileViT-S is better
suited for complex scenarios requiring robust feature extrac-
tion and generalization.

E. Performance Evaluation

In this section, we compare the performance and the com-
plexity of our proposed model with the existing solutions. The
methods are compared on the datasets described above.

1) Performance on the Evaluation Metrics: Table II com-
pares the performance of our proposed model
to existing
methods on the three fire datasets. The evaluation highlights
the effectiveness of our approach across multiple scenarios,
showcasing both strengths and areas for improvement. On the
BoWFire dataset, our model achieves perfect scores across
all metrics even without implementing KD, with 100% Acc,
F1, Rec, and Pre, demonstrating its exceptional capability
to generalize on this dataset. In comparison, previous SOTA
methods, such as MAFire-Net [15], achieved strong results
with 97.82% Acc and an F1 of 97.77%, but our model still
outperforms them. However, it is important to note that the
small size of the BoWFire dataset limits its representativeness
and may lead to inflated performance metrics. Due to this, we
further evaluated our model on other datasets.

Fig. 5b shows the confusion matrix of our proposed model
on the test split of the ADSF dataset. The results demonstrate
that our model outperforms all previous works across all
metrics, achieving an Acc, F1, Rec, and Pre of 95.50%.
Among the existing methods, MobileNetV3 + MSAM [3]
shows strong performance, with an accuracy and F1-score
of 93.50% and 93.51%, respectively. However, our model

Fig. 4: Demonstration of correctly and incorrectly labeled
DFAN images. The top row displays correctly classified ex-
amples, including a ”Building Fire,” ”Forest Fire,” and ”Non-
Fire” scene. The bottom row presents misclassified examples,
where a ”Cargo Fire” was predicted as ”Car Fire,” an ”SUV
Fire” was correctly labeled as ”Car Fire,” and a ”Van Fire”
was predicted as ”Car Fire.”

distillation loss and the standard cross-entropy loss. For DIST,
we used α = 0.1, β = 2, γ = 2, and τ = 1, while for OFA-
KD, the optimal parameters were ϵ = 1.2 and T = 3, as
described in their corresponding papers [36], [37].

Table I highlights that KD significantly enhances the per-
formance of MobileViT-S on the DFAN dataset. The best
result, an accuracy of 91.08%, was achieved using soft target
KD with ViT/32 as the teacher model. This improvement
underscores the importance of global contextual knowledge
provided by the transformer-based teacher. In comparison,
MobileViT-XS achieves slightly lower performance, with a
maximum accuracy of 90.55% under the same configuration.
This demonstrates that while MobileViT-XS is lightweight,
it is less effective in handling the complex scenarios present
in the DFAN dataset. OFA-KD and DIST also improved
performance compared to the baseline but showed slightly
lower results than soft target KD, likely due to the specific
properties of the DFAN dataset, which benefits from the
global context provided by ViT/32. The results also reveal that
ConvNeXt, a CNN-based teacher, does not provide as much
performance improvement as ViT/32. For instance, the best
accuracy achieved with ConvNeXt as the teacher was 88.98%

7

(a) BoWFire dataset.

(b) ADSF dataset.

(c) DFAN dataset.

Fig. 5: Confusion matrices for the fire benchmarks. For the BoWFire dataset, the model achieves perfect classification with
no misclassifications. On the ADSF dataset, the confusion matrix demonstrates high accuracy, with minor misclassifications
between fire and non-fire classes. The DFAN dataset’s confusion matrix captures the complexity of multiclass fire detection,
with most classes achieving high classification accuracy, but for some classes, the accuracy falls behind, such as ”Car Fire,”
”SUV Fire,” and ”Van Fire”.

surpasses this by a margin of 2.0%, reflecting its stronger capa-
bility in fire detection tasks. Overall, the consistent superiority
of our model across all metrics demonstrates its robustness
and effectiveness in accurately identifying fire regions under
diverse scenarios of the ADSF dataset.

As seen from Table II, our model achieves an accuracy of
91.08%, an F1-score of 90.75%, a recall of 90.27%, and a
precision of 91.43%. While these results position our model
competitively among existing works, it falls slightly short in
some metrics. Specifically, our model achieves the second-best
result in accuracy, with a 0.12% gap between MobileNetV3
+ MSAM [3]. Our model shows the best F1 and precision
scores, but ADFireNet [35] and MobileNetV3 + MSAM have
higher recall of 90.49% and 91.17%, respectively. Notably, our
model outperforms earlier approaches, indicating improved
generalization compared to earlier architectures. Insights from
the confusion matrix in Fig. 5c further validate the robustness
of our model. The matrix highlights its ability to accurately

classify critical fire categories like ”Forest Fire” and ”Car
Fire,” achieving high classification counts in these categories.
However,
limitations are observed in classes like ”Cargo
Fire” and ”Pickup Fire,” where some misclassifications occur,
potentially due to visual similarities with other categories. This
underscores an area for improvement in further refining the
model’s attention mechanism to reduce misclassification of
visually similar categories.

2) Complexity Analysis: Due to the resource constraints of
surveillance systems, UAVs, and IoT devices, the fire detection
model should be able to quickly predict the class of an image
on any system. Table III presents the system specifications,
model size, and frames-per-second (FPS) of our proposed
model and the existing work.

The proposed model demonstrates significant improvements
in performance compared to existing methods, as shown in
Table III. It achieved an impressive 431.07 FPS on an Nvidia
A100, 28.04 FPS on an AMD EPYC 7402 (2.80 GHz), and

8

TABLE II: Comparison of the performance of our proposed model to the existing work across the fire benchmarks. The metrics
in bold represent the best performance, while the underscored metrics are the second-best performance.

Methods

EFDNet [17]
ANetFire [39]
Xception [8]
EMNFire [6]
DFAN (comp.) [13]
DFAN [13]
OFAN [20]
MAFire-Net [15]
FireClassNet [19]
ResNet50 + FAN [13]
NASNetM + FAN [13]
MobileNet + FAN [13]
ADFireNet [35]
MobileNetV3 + MSAM [3]
Our Model

BoWFire Dataset

F1
81.85
88.00
-
92.00
93.10
95.00
96.00
97.77
99.58
-
-
-
-
-
100

Rec
83.00
98.00
-
93.00
92.00
94.00
95.00
98.15
99.44
-
-
-
-
-
100

Pre
81.81
80.00
-
90.00
94.30
95.00
96.00
97.05
99.72
-
-
-
-
-
100

Acc
83.33
88.05
91.41
92.04
93.00
95.00
96.23
97.82
99.56
-
-
-
-
-
100

Acc
88.00
-
-
-
-
89.36
-
-
-
-
-
-
90.86
93.50
95.50

ADSF Dataset
Rec
88.00
-
-
-
-
94.00
-
-
-
-
-
-
90.86
93.51
95.50

F1
87.75
-
-
-
-
89.84
-
-
-
-
-
-
89.84
93.51
95.50

Pre
87.50
-
-
-
-
86.01
-
-
-
-
-
-
90.90
93.57
95.50

Acc
77.50
-
-
-
86.50
88.00
-
88.83
-
86.12
82.56
85.30
90.00
91.20
91.08

DFAN Dataset
Rec
77.00
-
-
-
87.00
88.00
-
86.44
-
86.00
82.00
85.00
90.49
91.17
90.27

F1
77.49
-
-
-
86.00
87.00
-
87.53
-
85.00
81.00
85.00
89.99
90.63
90.75

Pre
78.00
-
-
-
86.00
88.00
-
89.35
-
88.00
82.00
85.00
90.43
90.36
91.43

9.36 on Raspberry Pi 4 with a compact model size of 19.73
MB. In comparison, EMNFire [6] has the smallest model size
of 13.0 MB, but its FPS values were lower, achieving 34.0
on a TITAN X (12GB) and 5.0 on a Raspberry Pi. While
MobileNetV3 + MSAM [3] provides competitive FPS on high-
performance systems (75.15 FPS on a GeForce RTX-3090), it
falls behind on Raspberry Pi with 8.0 FPS and has a larger
model size of 25.20 MB. Similarly, DFAN (compressed) [13]
achieves a high 125.33 FPS on an RTX 2070, but its larger
model size of 41.09 MB makes it less suitable for resource-
constrained devices, where our model delivers a better balance
of compactness and speed. Despite the strong performance of
these models, the proposed model outperformed all others in
terms of FPS and model size. It achieved the highest FPS
values on all system specifications while maintaining a smaller
model size, demonstrating its efficiency and scalability for
deployment on various devices, including resource-constrained
environments.

TABLE III: Complexity analysis of the proposed model com-
pared with existing research on different devices.

Model

EMNFire [6]

GNetFire [18]

SE-EFFNet [40]

DFAN (comp.) [13]

OFAN [20]

MAFire-Net [15]

MobileNetV3
+ MSAM [3]

Our Model

System
TITAN X (12GB)
Raspberry Pi
TITAN X (12GB)
Raspberry Pi
RTX 2070 (12GB)
Raspberry Pi
RTX 2070 (12GB)
Intel i9 (3.60GHz)
Raspberry Pi
Intel i9 (5.00GHz)
Raspberry Pi
GeForce RTX-3090
Intel i10 (5.3GHz)
Raspberry Pi
GeForce RTX-3090
Intel i9 (3.60GHz)
Raspberry Pi
A100
EPYC 7402 (2.80 GHz)
Raspberry Pi

Size (MB)

13.0

43.3

47.75

41.09

12.20

74.43

25.20

19.73

FPS
34.0
5.0
20.0
4.0
45.0
6.0
125.33
22.73
3.21
25.50
8.37
78.31
14.32
0.92
75.15
24.0
8.0
431.07
28.04
9.36

V. CONCLUSION
In this work, we proposed a lightweight and efficient
fire detection model based on the MobileViT-S architecture,
optimized through KD techniques to achieve high accu-
racy and real-time inference on resource-constrained devices.
By leveraging the inherent hybrid structure of MobileViT-
S, which combines the local feature extraction capabilities
of CNNs with the global context modeling of transformers,
our model demonstrates exceptional performance in detecting
fire and wildfire regions under diverse surveillance condi-
tions. Through rigorous experiments on benchmark datasets
such as BoWFire, ADSF, and DFAN, the proposed model
achieved 100%, 95.50%, and 91.08% accuracies and lowered
false positive rate. Notably, the model not only surpassed
or matched SOTA results but also achieved the highest FPS
across all tested devices, demonstrating its suitability for real-
time applications.

Nevertheless, our approach has some limitations. First,
the model’s ability to differentiate between visually similar
elements, such as smoke and clouds, needs improvement to
minimize false positives. Second, exploring advanced data
augmentation techniques or incorporating temporal informa-
tion from video sequences could enhance the model’s gen-
eralization capability in dynamic environments. Lastly, future
work could investigate more sophisticated KD strategies to
better utilize diverse teacher models and further improve the
student’s performance. By addressing the identified limitations
and exploring the proposed directions,
the robustness and
feasibility of fire monitoring systems can be further enhanced.

REFERENCES

[1] [Online].

Available:

https://eurasianet.org/

kazakhstan-mass-wildfire-deaths-provoke-anger-at-corruption

[2] H. Yar, A. S. Imran, Z. A. Khan, M. Sajjad, and Z. Kastrati, “Towards
smart home automation using iot-enabled edge-computing paradigm,”
Sensors, vol. 21, no. 14, p. 4932, 2021.

[3] H. Yar, Z. A. Khan, I. Rida, W. Ullah, M. J. Kim, and S. W. Baik, “An
efficient deep learning architecture for effective fire detection in smart
surveillance,” Image and Vision Computing, vol. 145, p. 104989, 2024.
[4] H. Harkat, J. M. Nascimento, A. Bernardino, and H. F. T. Ahmed, “Fire
images classification based on a handcraft approach,” Expert Systems
with Applications, vol. 212, p. 118594, 2023.

9

[26] T. Celik and H. Demirel, “Fire detection in video sequences using a
generic color model,” Fire safety journal, vol. 44, no. 2, pp. 147–158,
2009.

[27] P. V. K. Borges and E. Izquierdo, “A probabilistic approach for vision-
based fire detection in videos,” IEEE transactions on circuits and systems
for video technology, vol. 20, no. 5, pp. 721–731, 2010.

[28] P. Foggia, A. Saggese, and M. Vento, “Real-time fire detection for video-
surveillance applications using a combination of experts based on color,
shape, and motion,” IEEE TRANSACTIONS on circuits and systems for
video technology, vol. 25, no. 9, pp. 1545–1556, 2015.

[29] J. Chen, Y. He, and J. Wang, “Multi-feature fusion based fast video
flame detection,” Building and Environment, vol. 45, no. 5, pp. 1113–
1122, 2010.

[30] C. Ha, U. Hwang, G. Jeon, J. Cho, and J. Jeong, “Vision-based fire
detection algorithm using optical flow,” in 2012 Sixth international
conference on complex, Intelligent, and Software Intensive Systems.
IEEE, 2012, pp. 526–530.

[31] S. Frizzi, R. Kaabi, M. Bouchouicha, J.-M. Ginoux, E. Moreau, and
F. Fnaiech, “Convolutional neural network for video fire and smoke de-
tection,” in IECON 2016-42nd Annual Conference of the IEEE Industrial
Electronics Society.
IEEE, 2016, pp. 877–882.

[32] W. Lee, S. Kim, Y.-T. Lee, H.-W. Lee, and M. Choi, “Deep neural
networks for wild fire detection with unmanned aerial vehicle,” in 2017
IEEE international conference on consumer electronics (ICCE).
IEEE,
2017, pp. 252–253.

[33] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for

image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[34] G. Hinton, “Distilling the knowledge in a neural network,” arXiv preprint

arXiv:1503.02531, 2015.

[35] H. Yar, W. Ullah, Z. A. Khan, and S. W. Baik, “An effective attention-
based cnn model for fire detection in adverse weather conditions,” ISPRS
Journal of Photogrammetry and Remote Sensing, vol. 206, pp. 335–346,
2023.

[36] T. Huang, S. You, F. Wang, C. Qian, and C. Xu, “Knowledge distillation
from a stronger teacher,” Advances in Neural Information Processing
Systems, vol. 35, pp. 33 716–33 727, 2022.

[37] Z. Hao, J. Guo, K. Han, Y. Tang, H. Hu, Y. Wang, and C. Xu,
“One-for-all: Bridge the gap between heterogeneous architectures in
knowledge distillation,” Advances in Neural Information Processing
Systems, vol. 36, 2024.

[38] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A
convnet for the 2020s,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2022, pp. 11 976–11 986.
[39] K. Muhammad, J. Ahmad, and S. W. Baik, “Early fire detection using
convolutional neural networks during surveillance for effective disaster
management,” Neurocomputing, vol. 288, pp. 30–42, 2018.

[40] Z. A. Khan, T. Hussain, F. U. M. Ullah, S. K. Gupta, M. Y. Lee, and
S. W. Baik, “Randomly initialized cnn with densely connected stacked
autoencoder for efficient fire detection,” Engineering Applications of
Artificial Intelligence, vol. 116, p. 105403, 2022.

[5] H. Xu, G. Zhang, R. Chu, J. Zhang, Z. Yang, X. Wu, and H. Xiao,
“Detecting forest fire omission error based on data fusion at subpixel
scale,” International Journal of Applied Earth Observation and Geoin-
formation, vol. 128, p. 103737, 2024.

[6] K. Muhammad, S. Khan, M. Elhoseny, S. H. Ahmed, and S. W. Baik,
“Efficient fire detection for uncertain surveillance environment,” IEEE
Transactions on Industrial Informatics, vol. 15, no. 5, pp. 3113–3122,
2019.

[7] T. Khan, H. ˙I. Aslan et al., “Performance evaluation of enhanced
convnexttiny-based fire detection system in real-world scenarios,” 2023.
[8] V. E. Sathishkumar, J. Cho, M. Subramanian, and O. S. Naren, “Forest
fire and smoke detection using deep learning-based learning without
forgetting,” Fire ecology, vol. 19, no. 1, p. 9, 2023.

[9] M. Wang, D. Yu, W. He, P. Yue, and Z. Liang, “Domain-incremental
learning for fire detection in space-air-ground integrated observation
network,” International Journal of Applied Earth Observation and
Geoinformation, vol. 118, p. 103279, 2023.

[10] S. Jin, T. Wang, H. Huang, X. Zheng, T. Li, and Z. Guo, “A self-
adaptive wildfire detection algorithm by fusing physical and deep
learning schemes,” International Journal of Applied Earth Observation
and Geoinformation, vol. 127, p. 103671, 2024.

[11] X. Rui, Z. Li, X. Zhang, Z. Li, and W. Song, “A rgb-thermal based adap-
tive modality learning network for day–night wildfire identification,” In-
ternational Journal of Applied Earth Observation and Geoinformation,
vol. 125, p. 103554, 2023.

[12] D. Y. Chino, L. P. Avalhais, J. F. Rodrigues, and A. J. Traina, “Bowfire:
detection of fire in still images by integrating pixel color and texture
analysis,” in 2015 28th SIBGRAPI conference on graphics, patterns and
images.

IEEE, 2015, pp. 95–102.

[13] H. Yar, T. Hussain, M. Agarwal, Z. A. Khan, S. K. Gupta, and S. W.
Baik, “Optimized dual fire attention network and medium-scale fire
classification benchmark,” IEEE Transactions on Image Processing,
vol. 31, pp. 6331–6343, 2022.

[14] M. Park, J. Bak, S. Park et al., “Advanced wildfire detection using
generative adversarial network-based augmented datasets and weakly
supervised object localization,” International Journal of Applied Earth
Observation and Geoinformation, vol. 114, p. 103052, 2022.

[15] T. Khan, Z. A. Khan, and C. Choi, “Enhancing real-time fire detection:
an effective multi-attention network and a fire benchmark,” Neural
Computing and Applications, pp. 1–15, 2023.

[16] J. Sharma, O.-C. Granmo, M. Goodwin, and J. T. Fidje, “Deep convo-
lutional neural networks for fire detection in images,” in Engineering
Applications of Neural Networks: 18th International Conference, EANN
2017, Athens, Greece, August 25–27, 2017, Proceedings.
Springer,
2017, pp. 183–193.

[17] S. Li, Q. Yan, and P. Liu, “An efficient fire detection method based
on multiscale feature extraction, implicit deep supervision and channel
attention mechanism,” IEEE Transactions on Image Processing, vol. 29,
pp. 8467–8475, 2020.

[18] K. Muhammad, J. Ahmad, I. Mehmood, S. Rho, and S. W. Baik,
“Convolutional neural networks based fire detection in surveillance
videos,” Ieee Access, vol. 6, pp. 18 174–18 183, 2018.

[19] Z. Daoud, A. Ben Hamida, and C. Ben Amar, “Fireclassnet: a deep
convolutional neural network approach for pjf fire images classification,”
Neural Computing and Applications, vol. 35, no. 26, pp. 19 069–19 085,
2023.

[20] N. Dilshad, S. U. Khan, N. S. Alghamdi, T. Taleb, and J. Song, “Toward
efficient fire detection in iot environment: A modified attention network
and large-scale data set,” IEEE Internet of Things Journal, vol. 11, no. 8,
pp. 13 467–13 481, 2024.

[21] H. Yar, Z. A. Khan, T. Hussain, and S. W. Baik, “A modified vision
transformer architecture with scratch learning capabilities for effective
fire detection,” Expert Systems with Applications, vol. 252, p. 123935,
2024.

[22] S. Mehta and M. Rastegari, “Mobilevit:

light-weight, general-
purpose, and mobile-friendly vision transformer,” arXiv preprint
arXiv:2110.02178, 2021.

[23] T.-H. Chen, P.-H. Wu, and Y.-C. Chiou, “An early fire-detection method
based on image processing,” in 2004 International Conference on Image
Processing, 2004. ICIP’04., vol. 3.

IEEE, 2004, pp. 1707–1710.

[24] G. Marbach, M. Loepfe, and T. Brupbacher, “An image processing
technique for fire detection in video images,” Fire safety journal, vol. 41,
no. 4, pp. 285–289, 2006.

[25] A. Rafiee, R. Dianat, M. Jamshidi, R. Tavakoli, and S. Abbaspour, “Fire
and smoke detection using wavelet analysis and disorder characteristics,”
in 2011 3rd International conference on computer research and devel-
opment, vol. 3.

IEEE, 2011, pp. 262–265.

