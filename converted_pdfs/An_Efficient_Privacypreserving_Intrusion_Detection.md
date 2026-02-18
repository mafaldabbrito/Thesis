# An Efficient Privacy-preserving Intrusion Detection

An Efficient Privacy-preserving Intrusion Detection
Scheme for UAV Swarm Networks

Kanchon Gharami
Dept. of Electrical Engineering and Computer Science
Embry-Riddle Aeronautical University, Florida, USA
gharamik@my.erau.edu, kanchon2199@gmail.com

Shafika Showkat Moni
Dept. of Electrical Engineering and Computer Science
Embry-Riddle Aeronautical University, Florida, USA
monis@erau.edu, shafika1403@gmail.com

5
2
0
2

v
o
N
7
2

]

R
C
.
s
c
[

1
v
1
9
7
2
2
.
1
1
5
2
:
v
i
X
r
a

Abstract—The rapid proliferation of unmanned aerial vehicles
(UAVs) and their applications in diverse domains, such as
surveillance, disaster management, agriculture, and defense, have
revolutionized modern technology. While the potential benefits of
swarm-based UAV networks are growing significantly, they are
vulnerable to various security attacks that can jeopardize the
overall mission success by degrading their performance, disrupt-
ing decision-making, and compromising the trajectory planning
process. The Intrusion Detection System (IDS) plays a vital role
in identifying potential security attacks to ensure the secure
operation of UAV swarm networks. However, conventional IDS
primarily focuses on binary classification with resource-intensive
neural networks and faces challenges, including latency, privacy
breaches, increased performance overhead, and model drift. This
research aims to address these challenges by developing a novel
lightweight and federated continuous learning-based IDS scheme.
Our proposed model facilitates decentralized training across di-
verse UAV swarms to ensure data heterogeneity and privacy. The
performance evaluation of our model demonstrates significant
improvements, with classification accuracies of 99.45% on UKM-
IDS, 99.99% on UAV-IDS, 96.85% on TLM-UAV dataset, and
98.05% on Cyber-Physical datasets.

Index Terms—UAV, Intrusion detection, UAV swarm, Feder-
ated learning, Cybersecurity, Anomaly detection, Heterogeneous
learning, Privacy-preserving

I. INTRODUCTION

U NMANNED Aerial Vehicles (UAVs) have emerged as

transformative tools across a wide range of fields,
including disaster response,
logistics, precision agriculture,
environmental monitoring, and military operations [1], [2].
Their ability to perform tasks autonomously in inaccessible
or hazardous areas, combined with real-time adaptability and
scalability, makes them indispensable for critical applications.
However, the growing reliance on UAVs introduces significant
security challenges, especially in swarm networks where mul-
tiple UAVs collaborate to achieve shared objectives. In swarm-
based systems, where decisions and actions are interdependent,
a single compromised UAV can cascade and compromise the
entire network. Intrusion Detection Systems (IDSs) play a vital
role in safeguarding UAV swarm networks by identifying and
mitigating threats in real-time. While traditional IDSs have

This paper has been accepted for publication in the Proceedings of the
44th AIAA/IEEE Digital Avionics Systems Conference (DASC) 2025, where
it received the Best Paper of Session Award.
The
for
code
SPIRE-Lab-2025/UAV-IDS-FL

at: https://github.com/

this work is

available

source

demonstrated effectiveness in static or terrestrial networks,
UAV swarm environments introduce unique challenges, such
as limited computational resources, high mobility, and the
need for decentralized decision-making. To address these
challenges, researchers are focusing on developing advanced
IDSs tailored to the dynamic and resource-constrained nature
of UAV swarm networks. Intrusion detection systems (IDSs)
can mitigate the potential risk of these security vulnerabilities.
Despite notable progress in swarm-based IDS for UAVs,
unexplored avenues remain. Firstly,
they lack support for
handling heterogeneous data and require manual tuning to
adapt to new datasets, at least to match the dataset dimension.
Developing a unified framework that can seamlessly adapt
to diverse datasets without changes to model architecture or
hyperparameters while maintaining high accuracy is crucial for
UAV swarm networks. Secondly, studies utilizing resource-
intensive deep learning models make it difficult for deploy-
ment in resource-constraint UAV swarm networks. Thirdly,
conventional IDS encounters difficulties in enabling the model
to retain knowledge from earlier datasets while seamlessly
learning from new data or swarm groups. Thus, this results in a
decline in the accuracy of identifying intrusion and a potential
increase in false positives or false negatives. Finally,
the
existing model designed under controlled laboratory conditions
trained on any particular dataset does not reflect all the vast
complexities of real-world scenarios.

We aim to overcome the above challenges that conventional
intrusion detection models face including latency, privacy
breaches, increased performance overhead, and model drift.
We propose a lightweight and efficient multiclass intrusion
detection framework for UAV swarm networks, leveraging
federated continuous learning [3], [4]. Unlike existing research
that focuses on one specific dataset, we introduce a unified
three-component architecture to support data heterogeneity
in our IDS model. Firstly, the swarm-specific input layer is
designed to accommodate the heterogeneity of UAV swarm
datasets, where the number of input features may differ across
swarms due to varying sensor configurations or attack types.
This layer transforms datasets with varying feature dimensions
into a fixed-dimensional representation, ensuring compatibility
with subsequent components and federated learning (FL) [5].
It consists of a dense layer followed by a custom activation
function, ReTeLU, which combines the noise-filtering proper-

ties of ReLU with the selective thresholding of TeLU. This
hybrid activation effectively eliminates irrelevant noise while
retaining significant features, reducing the impact of minor
sensor fluctuations common in UAV data. As a result, the
model can focus on meaningful and stable patterns, enhancing
its robustness in dynamic environments.

Secondly, the shared CNN-LSTM encoder serves as the
backbone of the architecture, featuring a parallel setup where
the same input data is processed simultaneously by Convo-
lutional Neural Network (CNN) [6] and Long Short-Term
Memory (LSTM) [7]. This design allows the model to capture
spatial and temporal patterns concurrently, making it highly
efficient for intrusion detection in resource-constrained UAV
environments. Additionally, this encoder also participates in
federated learning, enabling collaborative knowledge sharing
across heterogeneous swarms. Then, by integrating Elastic
Weight Consolidation (EWC) into the FL setup, we transform
it into a Federated Continuous Learning system to enable
the model to retain knowledge from earlier datasets while
seamlessly learning from new data or swarm groups. More-
over, adapting FL, where data remain securely stored on local
devices and only aggregated model updates are communicated,
facilitates the development of a privacy-preserving IDS model.
Finally, the swarm-specific classifier layer acts as a multi-
head classifier, tailored to the unique classification demands of
each swarm group. This layer consists of a lightweight two-
layer neural network and employs our custom ReTeLU activa-
tion function, ensuring efficient and swarm-specific anomaly
detection.

A. Related Works

In recent years, a wide range of methods have been
developed to tackle the challenge of intrusion detection in
UAV networks. This section provides an overview of recent
approaches focusing on intrusion detection, analyzing their
strengths, and identifying their limitations, which serve as a
foundation for developing our proposed framework.

In 2019, Bithas et al. [8] conducted an in-depth assessment
of UAV anomaly detection using deep learning techniques.
They evaluated four deep learning architectures: autoencoders,
Deep Belief Networks (DBN), and Long Short-Term Memory
(LSTM) networks. Their study used several datasets, including
KDD Cup 1999 [9], NSL-KDD [10], CICIDS2017 [11], and
CICIDS2018 [12]. A notable strength of this research is its
approach to addressing challenges such as channel modeling
and resource management. However, a significant limitation
is the reliance on outdated datasets like KDD Cup 1999 and
NSL-KDD, which may not reflect the complexities of modern
UAV systems.

Raju et al. [13],

in 2023, conducted a UAV anomaly
detection study using a Variational Autoencoder (VAE) on
the ALFA dataset [14]. Their work employs unsupervised
learning, which enhances the system’s ability to detect unseen
attacks within a lightweight model and to cluster various
attack types into groups. However, this study has limitations,
including an inability to explicitly identify the underlying

causes of detected anomalies and a potential data bias due
to over-reliance on a single dataset.

In 2024, Hadi et al. [15] introduced a collaborative in-
trusion detection framework using a feedforward network.
Their approach emphasizes collaborative detection through
a trust score mechanism rather than relying solely on deep
learning models, offering a new perspective on UAV intrusion
detection. Additionally, they combine ReLU and TeLU acti-
vation functions to handle noisy sensor data effectively. The
framework achieves high detection accuracy on the UAV IDS
[16] dataset, even in real-world deployments. However, their
work lacks support for handling heterogeneous data and swarm
networks, privacy preservation, and multiclass classification.
Furthermore, while their system has been deployed in real-
world UAV test runs, the experiments were still conducted
within a limited testbed, which may not fully capture the
complexity of real-life attack patterns.

Zhao et al. [17] propose a security situation assessment
approach for UAV swarm networks using a Transformer-
ResNeXt-SE-based model, one of the most advanced neural
network architectures among all the anomaly detection litera-
ture. Their focus is on providing proactive network protection
rather than detecting anomalies post-attack. The model
is
exceptionally well-designed, and the collaborative detection
strategy is logically sound, yielding strong evaluation results
on four test datasets. However, the main limitation of this
system is its heavy computational requirements, making it
unsuitable for resource-constrained UAVs. In addition to that,
the lack of support for handling heterogeneous data could
introduce system performance biases.

Most recently, Lu et al. [18] proposed a swarm anomaly
detection model for IoT UAVs using a multimodal denoising
autoencoder. They successfully implemented their model in a
simulated federated learning setup, ensuring privacy preserva-
tion. The denoising autoencoder effectively detects multiclass
attacks, even in the presence of noisy sensor data. Their
experiments demonstrate outstanding performance across four
advanced intrusion detection datasets. However, the model
has some limitations: it is designed for homogeneous swarm
groups (with homogeneous datasets), and by adding Gaussian
there is a noticeable drop
noise to the sensor data [19],
in detection accuracy on the Cyber Physical dataset, which
contains both sequential cyber data and non-linear sensor data.
All of the above research has significantly contributed to the
UAV intrusion detection field. Through a review and analysis
of the literature, it is clear that while traditional UAV intrusion
detection models have unique strengths, none serve as a unified
IDS framework. To the best of our knowledge, none of the
existing methodologies addresses data heterogeneity, which is
crucial for developing generalized decentralized ad hoc UAV
networks.

B. Contributions

This paper aim to address the above research gaps in
existing IDS for UAV swam networks. We propose a unified
multiclass intrusion detection framework leveraging federated

continuous learning. Our system’s core is an encoder-classifier
network built on a CNN-LSTM multimodal architecture [20].
By combining the ability of the CNN to handle non-linear data
with the strength of LSTM to process time series information,
our model efficiently handles intrusion detection data sets
that
include both cyber attributes (linear data types) and
physical attributes (non-linear data types). Our framework
adopts a federated continuous learning approach [3], [4],
enabling the system to train simultaneously on heterogeneous
datasets while preserving data privacy [21]. We test and
train the model across the four distinct swarms for our four
heterogeneous datasets: UAV-IDS, UKM-IDS, TLM-IDS, and
Cyber-Physical. We summarize the key contributions of this
paper as follows:

• The proposed framework integrates federated learning
to collaboratively train on four heterogeneous datasets,
effectively reducing bias toward any specific dataset or
environment, and improving generalization. The modular
and scalable design supports the seamless addition of new
swarm groups with new datasets and attributes, making
the learning process more generalized and adaptive.
strengthens

the
model’s ability to recognize anomaly patterns, which we
leverage to detect attacks early and effectively through
binary classification in our supervised model.

• Training on heterogeneous datasets

• To address the challenge of limited perspective, we
integrate the Elastic Weight Consolidation (EWC) [22]
regularization technique into our FL setup, transforming
it into a Federated Continuous Learning system. This
integration prevents catastrophic forgetting [23], enabling
the model to retain knowledge from earlier datasets while
seamlessly learning from new data or swarm groups.
• Our lightweight Encoder-Classifier network incorporates
a shared CNN-LSTM encoder
to efficiently process
heterogeneous datasets, while the multilayer perceptron
(MLP) classifier [24] dimensions vary across swarms to
enable multiclass classification. This architecture mini-
mizes hardware requirements while supporting diverse
swarm groups with different class label sets.

• FL’s decentralized approach ensures that each swarm
group retains its data locally and trains through its edge
server, eliminating the need to transmit data to the Global
Control Server (GCS). This approach guarantees robust
data privacy for every swarm group.

The rest of the paper is organized as follows: Section II
outlines the proposed Encoder-Classifier model. We discuss
the implementation details, including datasets, experimental
setups, and federated learning integration in Section III. Sec-
tion IV presents the results and analysis, including perfor-
mance metrics and comparative benchmarks. Finally, Section
V concludes with a summary of findings.

II. PROPOSED ENCODER-CLASSIFIER MODEL

The proposed Encoder-Classifier model combines CNN for
spatial feature extraction with LSTM networks to capture tem-
poral dependencies, enabling efficient multiclass classification.

It is designed to handle the heterogeneity of UAV swarm
datasets, which can vary in both input features and attack class
labels. To address this, the model includes a swarm-specific
input layer that adapts to the varying input dimensions of each
dataset across different swarm, along with a local classifier
that adjusts its output based on the number of attack classes
for each swarm. As shown in Fig. 1, the CNN-LSTM encoder
extracts a unified latent representation from the input, while the
MLP classifier tailors the final classification output to match
the specific needs of each swarm. This structure ensures that
the model can perform multiclass intrusion detection across
different UAV networks, maintaining flexibility and accuracy
in diverse environments.

A. Swarm-Specific Input Layer

The swarm-specific input layer is designed to accommodate
the heterogeneity of UAV swarm datasets, where the number
of input features may differ across swarms due to varying
sensor configurations or attack types. This layer adapts each
swarm’s dataset to a fixed-dimensional representation, ensur-
ing consistency and compatibility for federated learning with
the rest of the model, regardless of differences in input size.
Given an input dataset Xi ∈ RB×Fi, where B is the batch
size and Fi is the number of features for the i-th swarm, we
project the data to a fixed hidden dimension hi ∈ RB×D using
a fully connected layer:

hi = ReTeLU





Fi(cid:88)

WijXij + bi





(1)

j=1

whereWij and bi are the weight matrix and bias, respec-
tively, and ReTeLU is a customized activation function. The
ReTeLU function combines ReLU and TeLU (Thresholded
ReLU) to enhance non-linearity and improve the model’s
robustness to noisy data. Specifically, ReLU sets negative
values to zero, eliminating irrelevant noise [15], while TeLU
introduces a threshold θ, ensuring only the most significant
positive features are retained, like following:

ReTeLU(x) =






0,
θ,
x,

if x ≤ 0
if 0 < x ≤ θ
if x > θ

(2)

This hybrid activation reduces the impact of minor sensor
fluctuations, which are common in UAV data, allowing the
model to focus on more meaningful and stable patterns.

The Client-Specific Input Layer operates independently
within each swarm, meaning its weights are updated locally
and do not participate in federated learning aggregation. This
localized adaptation ensures that each swarm can process its
unique dataset effectively without requiring synchronization
with other swarms. The fixed output dimension of this layer,
regardless of the input variability, maintains consistency across
swarms, ensuring that the model’s latent space is uniform.
This enables the model
to integrate data from heteroge-
neous swarms and perform multiclass classification effectively,

Fig. 1: The model diagram of the proposed Encoder-Classifier framework.
.

leveraging federated learning while preserving each swarm’s
specific characteristics.

passed through a fully connected layer to project the combined
features into the latent space:

B. Shared CNN-LSTM Encoder

The shared CNN-LSTM encoder combines CNN for spatial
feature extraction and LSTM networks for capturing temporal
dependencies [25], making it ideal for intrusion detection with
resource-limited UAV hardware. Given the input h ∈ RB×D,
where B is the batch size and D is the hidden dimension, the
encoder first processes the data through two parallel layers:
CNN and LSTM.

The CNN path extracts spatial features from the input data
by applying a convolution operation, where Wcnn ∈ RD×Ccnn
is the weight matrix and bcnn ∈ RCcnn is the bias term. The
output of the convolution is then passed through the ReTeLU
activation function (for each filter k) and followed by a max-
pooling layer to reduce the sequence length and retain the most
important features. This process is represented by:

xcnn,k = ReTeLU

(cid:33)

Wcnn,k,ihi + bcnn,k

(cid:32) D
(cid:88)

i=1

x′

cnn = MaxPool1D (xcnn)

(3)

(4)

Meanwhile, the LSTM path processes the data sequentially
to capture temporal dependencies. The LSTM operations are
given by:

˜ft = σ (Wf · zt + bf )
˜it = σ (Wi · zt + bi)
˜ot = σ (Wo · zt + bo)
˜ct = ˜ft ◦ ˜ct−1 + ˜it ◦ tanh (Wc · zt + bc)
˜ht = ˜ot ◦ tanh(˜ct)

(5)

(6)

(7)

(8)

(9)

Where, zt = [˜ht−1; xt]. The final output from the LSTM
is the last hidden state, hlstm = ht. After both the CNN and
LSTM paths process the data, their outputs are concatenated
into a single feature vector, z ∈ RB×(Ccnn+Llstm), where Ccnn is
the number of channels after the CNN layer and Llstm is the
number of LSTM features. This concatenated vector is then

z = concat (x′

cnn, hlstm)

(10)

zencoder = ReTeLU (Wfcz + bfc)
where Wfc ∈ R(Ccnn+Llstm)×D and bfc ∈ RD are the weight
matrix and bias for the fully connected layer, respectively.

(11)

C. Swarm-Specific Classifier

The swarm-specific classifier supports the model

to the
varying number of attack classes for heterogeneous UAV
groups. After the shared CNN-LSTM encoder processes the
input data, the resulting latent vector zencoder ∈ RB×Lencoder is
passed through a multi-layer perceptron (MLP) layer to project
the features into a smaller latent space and produce the final
class probabilities. The MLP operations are given by:

yfc1 = Wfc1zencoder + bfc1

yclassifier = ReTeLU(Woutyfc1 + bout)

(12)

(13)

where Wfc1, bfc1, Wout, and bout are the weight matrices
and biases for the fully connected layers. The softmax func-
tion [26] ensures that the output is a probability distribution
over the attack classes.

III. IMPLEMENTATION

In this section, we discuss the four heterogeneous datasets:
UAV IDS, UKM IDS, TLM IDS, and the Cyber-Physical
dataset, and our unified system model to effectively detect
intrusion in UAV swarm networks. To address the challenges
of catastrophic forgetting and improve learning efficiency, we
incorporate the Elastic Weight Consolidation (EWC) strategy
into our federated learning system, transforming it into a fed-
erated continuous learning model. This integration allows the
system to retain previously learned knowledge while efficiently
acquiring new information.

The shared encoder of our proposed encoder-classifier par-
ticipates directly in federated aggregation, enabling the unified
model to detect intrusions across all datasets. Meanwhile, the
swarm-specific input layer and local classifiers handle the

varying dimensions of each dataset, preserving local knowl-
edge specific to each swarm. This framework ensures fairness
without any bias towards specific datasets. It also enables the
model to detect intrusions in any new dataset or environment
while maintaining the privacy of each swarm’s data. This
section provides a detailed description of the datasets, system
architecture, federated learning implementation, experimental
parameters, and setup.

A. Dataset Descriptions

Our proposed system model integrates a federated learning
framework to combine four heterogeneous datasets: UAV IDS,
UKM IDS, TLM IDS, and the Cyber-Physical dataset. The
details of these datasets are described in this section and
summarized in Table I.

TABLE I: Dataset Summary

Dataset
UAV IDS
UKM IDS
TLM IDS
Cyber-Physical

Samples
98,736
12,887
12,254
33,102

Attributes
54
46
18
36

Classes
2
9
5
3

Source
UAV Traffic
Network Logs
UAV Simulation
Tello Drone

The UAV IDS [16] dataset consists of real-world network
traffic data collected from three popular UAV model: DJI
Spark, DBPower UDI, and Parrot Bebop. It includes 98,736
unique and up-to-date Wi-Fi traffic log samples, categorized
into two classes: Benign and Anomaly. The anomaly class en-
compasses a broad range of attack vectors. For our experiment,
we use the bi-directional flow version of this dataset, which
features 54 attributes (+1 label).

The UKM-IDS [27] dataset collected from real-world net-
work traffic consists of 12,887 samples with 46 features (+2
label), covering nine types of attacks: Normal, TCP flood,
Port scanning, ARP poisoning, UDP data flood, Mass HTTP
requests, Metasploit exploits, and BeEF HTTP exploits. The
dataset’s complexity is shown by analyzing its features and
classes with rough-set theory and testing it with a dynamic
neural network, which demonstrates its higher complexity and
relevance compared to other intrusion detection datasets.

The TLM dataset [28] was created using a software-in-the-
loop simulation setup, where typical UAV failure scenarios are
simulated by varying internal physical parameters. It consists
of 12,254 samples with 18 attributes (+1 label) and five class
labels: Benign, RC failure, GPS failure, ACC failure, and
Engine failure. A quadrotor UAV is used to simulate com-
mon UAV anomalies, such as engine failures, accelerometer
malfunction, and remote control issues for data collection.

The Cyber-Physical Dataset [29] integrates both cyber and
physical attributes to create an intrusion detection dataset,
featuring five different attack classes. The data was collected
using the DJI Tello EDU drone. Due to differing data dimen-
sions across the attack classes, we use a minimized version of
the dataset, which contains 33,102 samples, 36 features, and
one label with three classes: Benign, Replay, and DoS.

B. System Architecture

Our proposed system architecture integrates four separate
UAV swarm networks, each corresponding to one of our
heterogeneous datasets,
to perform decentralized intrusion
detection using Federated Learning. Each swarm consists of
several leaf UAV nodes and an edge server, with the edge
server acting as a local master. The edge server maintains a
copy of the Encoder-Classifier model, aggregates data from the
UAV nodes, and performs local anomaly detection. A central
cloud server coordinates the aggregation of model updates
from all edge servers to maintain a global model. While
all servers share the same Encoder-Classifier framework, the
shared encoder weights are synchronized across the system,
with separate input
layers and classifiers specific to each
swarm. The proposed system architecture is depicted in Fig. 2.

Fig. 2: Federated learning based system architecture
.

To train the global model without sharing raw data, we
adopt the Federated Averaging (FedAvg) algorithm [30]. In
each round, a subset of clients (UAV swarms) trains their local
models, and the local model updates are averaged to update
the global model. The client-specific update w(r)
is computed
as:

i

i = w(r−1)
w(r)

i

− η∇L(w(r−1)

i

, Di)

(14)

where L(wi, Di) is the local loss and η is the learning rate.
The global model is updated by averaging the local updates:
1
|Sr|

w(r) =

w(r)
i

(15)

(cid:88)

i∈Sr

where Sr is the set of selected clients in round r.

Elastic Weight Consolidation (EWC) [31] is used to mit-
igate catastrophic forgetting when training across multiple
heterogeneous tasks, as well as obtain the continuous learning
capability [32]. The EWC penalty term is added to the local
loss:

L(w) = Llocal(w, Di) + LEWC(w)

(16)

where LEWC(w) = (cid:80)
i λi(wi − ˆwi)2 and ˆwi represents the
optimal parameter from the previous task. The step-by-step
process of the FedAvg loop with EWC integration is given in
Algorithm 1.

Algorithm 1 Federated Learning for Intrusion Detection

TABLE IV: FL Client Configuration Parameters

Parameter
Num UAV Client
Min fit UAV clients
Min evaluate UAV clients
Min available UAV clients

Value
4
4
4
4

1: Input: θenc, θclf, Di, λi
2: Output: Global model parameters θglobal
3: Initialize global parameters θ(0)
global
4: for each round r = 1, . . . , 50 do
5:
6:
7:
8:

Select a subset Sr ⊂ K of clients
Send θ(r−1)
global
for each client k ∈ Sr do

to clients in Sr

Perform local training and compute update w(r)

using:

k

w(r)

k = w(r−1)

k

− η∇L(w(r−1)

k

, Dk) + λLEWC(w(r−1)

k

, θk)

9:
10:

end for
Aggregate updates:

θ(r)
global =

1
|Sr|

(cid:88)

k∈Sr

w(r)
k

11: end for

The cloud server aggregates model updates from edge
servers after each round. The global model is then distributed
to the edge servers for real-time anomaly detection, ensur-
ing privacy-preserving training and continuous improvement
through federated updates and EWC regularization.

C. Experimental Parameters

In this study, we use the Flower framework1 for federated
learning simulations and PyTorch2 for deep learning model
development. The experiments were conducted on a Windows
11 PC with an Intel Core i7-7500U 2.7GHz Processor, an
NVIDIA GeForce 940MX 4GB DDR3 GPU, and 8GB DDR4
RAM. The dataset was split into a training set (80%) and a
test set (20%). The configurations and characteristics of the
test environment are summarized in Tables II, III, and IV.

TABLE II: Encoder-Classifier Model Summary

Layer Name
Input Layer
CNN Layer
BatchNorm1d
MaxPool1d
LSTM Layer
Dense
Classifier

Input Dimension Output Dimension
Dynamic
128 × 1
16 × 1
16 × 1
16 × 1
144
64

128
16 × 1
16 × 1
16 × 1
128 × 1
64
Dynamic

TABLE III: Model Training Parameters

Parameter
Batch Size
Epochs
Rounds
Learning Rate
Lambda EWC (Regularization)

Value
32
1
50
0.001
0.4

1https://flower.ai/
2https://pytorch.org/

We employ the Cross-Entropy Loss function [33] to op-
timize the model, as it effectively measures how far the
predicted probability distribution p deviates from the actual
label distribution y. The loss is computed as:

LCE = −

C
(cid:88)

c=1

yc log(pc)

(17)

where C represents the number of classes, yc is the true
label, and pc is the predicted probability for class c. This loss
function is effective in classification tasks, driving the model
towards accurate class predictions.

Given the federated learning setup, which involves multi-
party collaboration, distributed computation, and data privacy,
the model can efficiently train on heterogeneous UAV swarm
data while safeguarding privacy, even with limited computa-
tional resources.

IV. RESULTS AND ANALYSIS

In this section, we conduct a comprehensive evaluation
of the proposed system in terms of classification accuracy,
computational efficiency, and communication overhead.

A. Classification Performance

To evaluate the effectiveness of our proposed system, we
utilize a multiclass confusion matrix analysis on the testing
datasets. The confusion matrix reveals the distribution and
recognition patterns relative to the true labels, providing in-
sights into the model’s performance. It is important to note
that this evaluation is conducted after post-federated learning
training, during which identical models (with shared weights)
are tested on four distinct swarms corresponding to our four
heterogeneous datasets. The evaluation results, derived from
the confusion matrices, are shown in Fig. 3.

Fig. 3 shows the multiclass confusion matrices for the
four swarm datasets. In all cases, most predictions lie along
the diagonal, indicating that the model accurately classifies
both benign and attack samples. The UAV IDS dataset shows
near-perfect performance with only one error. UKM IDS also
performs strongly, with very few misclassifications across mul-
tiple attack types. For TLM IDS, the model correctly identifies
most classes, though there are a few confusions between sim-
ilar anomaly types. On the Cyber Physical dataset, the model
shows high accuracy in detecting DoS and replay attacks, with
only minor errors in classifying benign samples. Overall, the
model demonstrates reliable and consistent performance across
all datasets.

Moreover, additional performance evaluation measures, in-
cluding the Detection Accuracy, Recall, F1 Score, Detection

(a) UAV IDS

(b) UKM IDS

(c) TLM IDS

(d) Cyber Physical

Fig. 3: Confusion Matrix of Four Heterogeneous Swarm

Precision, Detection Error Rate [15] are calculated using the
confusion matrix parameters, which are presented in Table V.

TABLE V: Evaluation Scores of Four Heterogeneous Swarm

Dataset
UAV IDS
UKM IDS
TLM IDS
CyberPhysical

Accuracy
99.99%
99.46%
96.85%
98.05%

Recall
99.99%
99.61%
98.64%
98.00%

F1 Score
99.99%
99.03%
94.83%
98.08%

Precision
99.99%
99.62%
92.13%
98.16%

Error
0.01%
0.54%
3.43%
1.95%

To demonstrate the improvements, we conducted exper-
iments using several existing benchmarks,
including the
Multi-modal Denoising Auto-encoder (L-MADE) [18], Feed
Forward CNN (FFCNN) [15], MLP Auto-encoder (MLP-
AE)
[34], and our CNN-LSTM-based Encoder-Classifier
model. The results presented in Table VI and Fig. 4 highlight
the performance differences across these models.

Table VI shows that our approach consistently performs
at or near the best across all datasets. It clearly outperforms
the baselines on UAV-IDS and CyberPhysical, and performs
competitively on UKM-IDS and TLM-IDS. Most notably, our

TABLE VI: Performance Comparison Across Various Models

Datasets

UAV IDS

UKM IDS

TLM IDS

CyberPhysical

Model
L-MADE
FFCNN
MLP-AE
Ours
L-MADE
FFCNN
MLP-AE
Ours
L-MADE
FFCNN
MLP-AE
Ours
L-MADE
FFCNN
MLP-AE
Ours

99.23%
99.00%
100%

98.67%
98.88%
100%

F1
98.10%
99.00%
99.99%

Recall
97.84%
99.87%
99.97%

Precision
Accuracy
97.84%
98.15%
99.64%
98.35%
99.97%
99.97%
99.99% 99.99% 99.99% 99.99%
98.76%
98.81%
99.35%
96.35%
100%
100%
99.46% 99.61% 99.03% 98.62%
98.52%
97.86%
91.40%
94.36%
96.57%
90.94%
96.85% 98.64% 94.83% 92.13%
83.06%
82.72%
83.80%
94.94%
91.83%
88.56%
96.80%
96.36%
96.36%
98.16%
98.05% 98.00%

82.72%
92.67%
96.49%
98.08

97.60%
79.70%
96.93%

98.86%
84.00%
95.12%

model stands out for its stable and strong performance across
all datasets, highlighting its robustness and efficiency.

Fig. 5a, we observe that CPU usage gradually rises during
training and stabilizes around 40–50% for most clients. This
confirms that the model trains well even on basic UAV-grade
processors. Fig. 5b confirms that GPU utilization is almost
zero, which means our system does not rely on GPUs and can
easily run on edge devices without accelerators. In Fig. 5c, we
see that both upload and download sizes remain fixed at around
3.45 MB, showing low communication cost and consistency
in each round.

To understand how fast each client can process data during
training, we measure the number of samples trained per second
in every round. As shown in Fig. 6, clients start with high
throughput (1200–1400 samples/s) and later stabilize around
100–150 samples/s. This steady pattern confirms that training
remains smooth and manageable on all clients, even when
hardware capabilities differ.

Fig. 6: Client-side throughput across federated training rounds.

It is crucial to detect intrusion in real-time or near-real-time
to ensure UAV safety in a dynamic environment. Fig. 7 shows
the average inference latency per sample for each client. The
latency stays well below 15 milliseconds per sample, with
most clients operating between 8–12 ms. This observation
states that our model responds quickly and is suitable for real-
world deployment.

Fig. 4: Accuracy Comparison across different dataset

B. Computational and Communication Performance

In addition to achieving high detection accuracy, our pro-
posed system is optimized to be lightweight and suitable for
deployment in real UAV swarms. This section evaluates the
system’s training cost, inference delay, and communication
overhead throughout the federated learning process.

(a) CPU utilization

(b) GPU utilization

(c) Bandwidth per round

Fig. 5: System resource utilization during 50 federated rounds.

Fig. 5 shows three key system-level metrics: CPU usage,
GPU usage, and communication bandwidth per round. From

Fig. 7: Per-sample inference latency (lower is better).

Fig. 8 tracks the total time each client takes to complete
training in each round. The median fit-time remains under 150
seconds throughout the 50 rounds. This trend is consistent and
predictable, which helps when coordinating training schedules
in real UAV fleets.

To give a more detailed picture, we also collect static system
profiling information during a test inference run. Table VII
summarizes key statistics such as model size, Floating-point

privacy preservation, noise reduction, and false alarm mini-
mization, they lack a comprehensive and unified framework
to tackle the existing research gaps. In contrast, we introduce
a unified, decentralized, and lightweight system framework
to handle heterogeneous data across UAV swarms effectively.
The proposed system model ensures unbiased detection, pre-
serves privacy, reduces false alarms due to noise, supports
continuous learning without forgetting previous knowledge,
and enhances the detection accuracy of new attack patterns.
The performance evaluation of our proposed system model
demonstrates robust and competitive performance compared
to leading benchmarks. By integrating federated learning with
lightweight design, our system not only addresses current
challenges but also paves the way for Advanced General
Intelligence (AGI) in intrusion detection.

REFERENCES

[1] A. B. Mohammed, L. C. Fourati, and A. M. Fakhrudeen, “Comprehen-
sive systematic review of intelligent approaches in uav-based intrusion
detection, blockchain, and network security,” Computer Networks, p.
110140, 2023.

[2] R. A. AL-Syouf, R. M. Bani-Hani, and O. Y. AL-Jarrah, “Machine
learning approaches to intrusion detection in unmanned aerial vehicles
(uavs),” Neural Computing and Applications, vol. 36, no. 29, pp. 18 009–
18 041, 2024.

[3] L. Li, Y. Fan, M. Tse, and K.-Y. Lin, “A review of applications in
federated learning,” Computers & Industrial Engineering, vol. 149, p.
106854, 2020.

[4] Z. Yang, “Federated class continual learning in medical imaging: a
study of fedewc and fedlwf strategies against catastrophic forgetting,”
in Fifth International Conference on Computer Vision and Data Mining
(ICCVDM 2024), vol. 13272. SPIE, 2024, pp. 231–240.

[5] J. Wen, Z. Zhang, Y. Lan, Z. Cui, J. Cai, and W. Zhang, “A survey on
federated learning: challenges and applications,” International Journal
of Machine Learning and Cybernetics, vol. 14, no. 2, pp. 513–535, 2023.
[6] M. Priyadarsini and N. Sonekar, “A cnn-based approach for anomaly
detection in smart grid systems,” Electric Power Systems Research, vol.
238, p. 111077, 2025.

[7] Y. Yu, X. Si, C. Hu, and J. Zhang, “A review of recurrent neural
networks: Lstm cells and network architectures,” Neural computation,
vol. 31, no. 7, pp. 1235–1270, 2019.

[8] P. S. Bithas, E. T. Michailidis, N. Nomikos, D. Vouyioukas, and
A. G. Kanatas, “A survey on machine-learning techniques for uav-based
communications,” Sensors, vol. 19, no. 23, p. 5170, 2019.

[9] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed
analysis of the kdd cup 99 data set,” in 2009 IEEE symposium on
computational intelligence for security and defense applications.
Ieee,
2009, pp. 1–6.

[10] R. Bala and R. Nagpal, “A review on kdd cup99 and nsl nsl-kdd
dataset.” International Journal of Advanced Research in Computer
Science, vol. 10, no. 2, 2019.

[11] R. Panigrahi and S. Borah, “A detailed analysis of cicids2017 dataset
for designing intrusion detection systems,” International Journal of
Engineering & Technology, vol. 7, no. 3.24, pp. 479–482, 2018.
[12] M. Cantone, C. Marocco, and A. Bria, “Generalization challenges
in network intrusion detection: A study on cic-ids2017 and cse-cic-
ids2018 datasets,” in 1st INTERNATIONAL PhD SYMPOSIUM ON
ENGINEERING AND SPORT SCIENCE, p. 185.

[13] R. Dhakal, C. Bosma, P. Chaudhary, and L. N. Kandel, “Uav fault
and anomaly detection using autoencoders,” in 2023 IEEE/AIAA 42nd
Digital Avionics Systems Conference (DASC).

IEEE, 2023, pp. 1–8.

[14] A. Keipour, M. Mousaei, and S. Scherer, “Alfa: A dataset for uav fault
and anomaly detection,” The International Journal of Robotics Research,
vol. 40, no. 2-3, pp. 515–520, 2021.

[15] H. J. Hadi, Y. Cao, S. Li, Y. Hu, J. Wang, and S. Wang, “Real-time
collaborative intrusion detection system in uav networks using deep
learning,” IEEE Internet of Things Journal, 2024.

[16] “Unmanned Aerial Vehicle (UAV) Intrusion Detection,” UCI Machine
Learning Repository, 2020, DOI: https://doi.org/10.24432/C56P6X.

Fig. 8: Client training time per round (median and IQR).

TABLE VII: System Profiling Summary

Metric
Latency (ms/sample)
CPU Memory (MB)
GPU Memory (MB)
Power Usage (Watt)

CPU
2.12
831.04
8.52
–

Quant. CPU
2.77
833.71
8.52
–

Model FLOPs
Trainable Params
Model Size (bytes)

GPU
3.49
789.60
9.47
9.78

236,736
231,541
955,959

operations per second (FLOPs), and resource consumption on
CPU and GPU.

Finally, Fig. 9 compares our model’s weight and parameter
footprint with existing works. The radar plots confirm that our
method achieves excellent accuracy while staying compact in
size, making it ideal for memory-constrained UAV boards.

(a) Trainable parameter count

(b) Model size in MB

Fig. 9: Model weight comparison across different architec-
tures.

Our proposed system model requires low CPU and GPU
power, uses minimal bandwidth, and finishes training quickly.
These evaluation metrics demonstrate the effectiveness and
robustness of the proposed IDS system for real-world deploy-
ment.

V. CONCLUSION

In the emerging field of intrusion detection within UAV
swarm networks, many existing solutions excel in controlled
lab environments but struggle to perform effectively in real-
world deployments. These challenges often arise from relying
on biases in specific datasets or configurations with limited in-
trusion patterns, centralized structure, and high computational
costs. While some solutions address specific issues such as

[17] D. Zhao, P. Shen, X. Han, and S. Zeng, “Security situation assessment
in uav swarm networks using transrese: A transformer-resnext-se based
approach,” Vehicular Communications, vol. 50, p. 100842, 2024.
[18] Y. Lu, T. Yang, C. Zhao, W. Chen, and R. Zeng, “A swarm anomaly
detection model for iot uavs based on a multi-modal denoising autoen-
coder and federated learning,” Computers & Industrial Engineering, vol.
196, p. 110454, 2024.

[19] D. Guo, Y. Wu, S. S. Shitz, and S. Verd´u, “Estimation in gaussian noise:
Properties of the minimum mean-square error,” IEEE Transactions on
Information Theory, vol. 57, no. 4, pp. 2371–2385, 2011.

[20] L. Praharaj, M. Gupta, and D. Gupta, “Hierarchical federated transfer
learning and digital twin enhanced secure cooperative smart farming,”
in 2023 IEEE International Conference on Big Data (BigData).
IEEE,
2023, pp. 3304–3313.

[21] F. Ilhan, G. Su, and L. Liu, “Scalefl: Resource-adaptive federated
learning with heterogeneous clients,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2023, pp.
24 532–24 541.

[22] J. Thorne and A. Vlachos, “Elastic weight consolidation for better bias

inoculation,” arXiv preprint arXiv:2004.14366, 2020.

[23] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins,
A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska
et al., “Overcoming catastrophic forgetting in neural networks,” Pro-
ceedings of the national academy of sciences, vol. 114, no. 13, pp.
3521–3526, 2017.

[24] M. W. Gardner and S. Dorling, “Artificial neural networks (the multi-
layer perceptron)—a review of applications in the atmospheric sciences,”
Atmospheric environment, vol. 32, no. 14-15, pp. 2627–2636, 1998.
[25] D. Kuettel, M. D. Breitenstein, L. Van Gool, and V. Ferrari, “What’s
going on? discovering spatio-temporal dependencies in dynamic scenes,”
in 2010 IEEE computer society conference on computer vision and
pattern recognition.
IEEE, 2010, pp. 1951–1958.

[26] M. Wang, S. Lu, D. Zhu, J. Lin, and Z. Wang, “A high-speed and low-
complexity architecture for softmax function in deep learning,” in 2018
IEEE asia pacific conference on circuits and systems (APCCAS).
IEEE,
2018, pp. 223–226.

[27] M. S. Al-Daweri, S. Abdullah, and K. A. Z. Ariffin, “An adaptive method
and a new dataset, ukm-ids20, for the network intrusion detection
system,” Computer Communications, vol. 180, pp. 57–76, 2021.
[28] T. Yang, Y. Lu, H. Deng, J. Chen, and X. Tang, “Acquisition and
processing of uav fault data based on time line modeling method,”
Applied Sciences, vol. 13, no. 7, p. 4301, 2023.

[29] T. Puccetti, S. Nardi, C. Cinquilli, T. Zoppi, and A. Ceccarelli, “Rospace:
Intrusion detection dataset for a ros2-based cyber-physical system and
iot networks,” Scientific Data, vol. 11, no. 1, p. 481, 2024.

[30] M. M. Rahman and S. Purushotham, “Fair federated survival analysis,”
in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
no. 19, 2025, pp. 20 104–20 112.

[31] S. Aslam, A. Rasool, X. Li, and H. Wu, “Cel: A continual learning model
for disease outbreak prediction by leveraging domain adaptation via
elastic weight consolidation,” Interdisciplinary Sciences: Computational
Life Sciences, pp. 1–19, 2025.

[32] B. Maschler, H. Vietz, N. Jazdi, and M. Weyrich, “Continual learning
of fault prediction for turbofan engines using deep learning with elastic
weight consolidation,” in 2020 25th IEEE international conference on
emerging technologies and factory automation (ETFA), vol. 1.
IEEE,
2020, pp. 959–966.

[33] A. Mao, M. Mohri, and Y. Zhong, “Cross-entropy loss functions:
Theoretical analysis and applications,” in International conference on
Machine learning. PMLR, 2023, pp. 23 803–23 828.

[34] T. Yashwanth, K. Ashwini, G. S. Chaithanya, and A. Tabassum, “Net-
work intrusion detection using auto-encoder neural networks and mlp,”
in 2024 Third International Conference on Distributed Computing and
Electrical Circuits and Electronics (ICDCECE).
IEEE, 2024, pp. 1–6.

