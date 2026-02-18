# LiPar: A Lightweight Parallel Learning Model for

LiPar: A Lightweight Parallel Learning Model for
Practical In-Vehicle Network Intrusion Detection
Aiheng Zhang, Qiguang Jiang, Kai Wang∗, Ming Li

1

4
2
0
2

c
e
D
1
2

]

R
C
.
s
c
[

2
v
0
0
0
8
0
.
1
1
3
2
:
v
i
X
r
a

Abstract—With the development of intelligent transportation
systems, vehicles are exposed to a complex network environment.
As the main network of in-vehicle networks, the controller area
network (CAN) has many potential security hazards, resulting in
higher generalization capability and lighter security requirements
for intrusion detection systems to ensure safety. Among intrusion
detection technologies, methods based on deep learning work best
without prior expert knowledge. However, they all have a large
model size and usually rely on large computing power such as
cloud computing, and are therefore not suitable to be installed
on the in-vehicle network. Therefore, we explore computational
resource allocation schemes in in-vehicle network and propose
a lightweight parallel neural network structure, LiPar, which
achieve enhanced generalization capability for identifying normal
and abnormal patterns of in-vehicle communication flows to
achieve effective intrusion detection while improving the utiliza-
tion of limited computing resources. In particular, LiPar adap-
tationally allocates task loads to in-vehicle computing devices,
such as multiple electronic control units, domain controllers,
computing gateways through evaluates whether a computing
device is suitable to undertake the branch computing tasks
according to its real-time resource occupancy. Through exper-
iments, we prove that LiPar has great detection performance,
running efficiency, and lightweight model size, which can be well
adapted to the in-vehicle environment practically and protect the
in-vehicle CAN bus security. Furthermore, with only the common
multi-dimensional branch convolution networks for detection,
LiPar can have a high potential for generalization in spatial
and temporal feature fusion learning.

Index Terms—Lightweight neural network, parallel structure,
intrusion detection, spatial and temporal feature fusion, in-vehicle
network.

I. INTRODUCTION

W ITH the development of mobile communication tech-

nology and Internet of Things (IoT) technology, the
Internet of vehicles and in-vehicle networks (IVNs) have been
widely applied in key functions of intelligent transportation
systems. The Internet of vehicles can help vehicles acquire
environmental information promptly and make vehicles more
intelligent by sharing information with neighboring vehicles
and infrastructure such as roadside surveillance [1]. However,
increased vehicular connectivity with external terminals pro-

This work was supported in part by National Natural Science Foundation of
China (NSFC) (grant number 62272129) and Taishan Scholar Foundation of
Shandong Province (grant number tsqn202408112). (Corresponding author:
Kai Wang.)

Aiheng Zhang, Qiguang Jiang, and Kai Wang are with the School
of Computer Science and Technology, Harbin Institute of Technology,
Weihai, China. (e-mail: zahboyos@163.com;
jiangqiguang 971@163.com;
dr.wangkai@hit.edu.cn)

Ming Li is with Shandong Inspur Database Technology Co., Ltd, China.

(email: liming2017@inspur.com)

vides more opportunity for external attackers and results in
more risk of being attacked [2], [3].

In-vehicle communication among Electronic Control Units
(ECUs) is mainly transmitted via the Controller Area Network
(CAN), which is a standard backbone network of in-vehicle
communication. CAN is a message-broadcast-based protocol
without node information for the sender and receiver. It lacks
authentication information and message encryption mecha-
nisms, making it easy for attackers to intrude and corrupt CAN
communication [4], [5]. Therefore,
improving the security
of the in-vehicle CAN bus is an important aspect to ensure
vehicle security and passenger safety.

The intrusion detection system (IDS) has always been a
focus of research in the field of vehicle security. IDS can detect
attacks timely and accurately, so as to help vehicle-mounted
systems make appropriate responses in time. As a result, intru-
sion detection is of paramount importance for in-vehicle CAN
bus [6]. Current intrusion detection techniques can mainly fall
into four categories [7]: (1) the specification-based approaches,
(2) the fingerprint-based approaches, (3) the statistics-based
approaches, and (4) the learning-based approaches. Among
them, the dominant techniques are learning-based approaches.
Deep learning-based approaches, in particular, have better real-
time detection performance in the face of continuously chang-
ing attacks through multiple training. The reasons are that
deep learning-based approaches can provide more intelligent
IDS to detect attacks without prior expert knowledge [8],
[9]. Compared with the other approaches, deep learning-based
methods have the capability to handle huge amounts of data,
learn attack characteristics, predict the occurrence of attacks,
and achieve more accurate detection results [10].

In recent years, most of the in-vehicle CAN bus intru-
sion detection technologies are developed based on Convolu-
tional Neural Networks (CNN) and Long Short-Term Memory
(LSTM) networks [11], [12], due to the multidimensional time
series CAN data. Although these methods have extremely high
detection accuracy, they all use linear stacking of neural net-
work layers to build models, which are usually characterized
by the large model size. They consume considerable comput-
ing resources and may lead to a large delay in detection.

Although deep learning models generally rely on the Cloud
Computing (CC) server or Mobile Edge Computing (MEC)
server which has sufficient resources, they are not as reliable
and timely as local computing [13]. In fact, vehicles have
extremely high requirements for the reliability and timeliness
of intrusion detection technology, and the computing resources
in the in-vehicle environment are very limited [14]. A linear
stacked deep learning model might only run on one ECU

device or one in-vehicle computing component, which may
cause overload and long delay, and affect the original functions
of the vehicle. In our previous study [15], we selected 10
representative intrusion detection models and evaluated their
adaptability for the in-vehicle environment, and proposed
corresponding baseline selection schemes, which provided a
basis for this study.

To solve the above problems, we propose a novel
lightweight method for IVN intrusion detection. The main
contributions of our work can be summarized as follows:

• We design a lightweight parallel network model (Li-
Par), which is the first parallel-based intrusion detection
method for IVN. The model allocates the task load
to multiple parallel branches and multiple computing
devices according to a resource adaptation algorithm, and
adopts lightweight structures to compose each branch.
• LiPar is able to identify normal and abnormal patterns
embedded in time series data of in-vehicle communi-
cation flows accurately. Specifically, a multidimensional
feature extraction approach based on the combination
of CNN and LSTM structures is designed to make
information fusion from spatial and temporal views to
improve the generalization capability.

• A resource adaptation algorithm is designed to make full
use of multi computing resources, which can evaluate
whether a computing device is suitable to undertake
the branch computing tasks according to its real-time
resource occupancy.

• Extensive experimental evaluation and comparative anal-
ysis are conducted to demonstrate that
the proposed
LiPar has outstanding detection performance, running
efficiency, and a lightweight degree, which is significant
to be applied in IVN.

The paper is organized as follows. Section II introduces the
important background knowledge related to our research. Sec-
tion III explains the proposed framework and design details.
Section IV describes the experimental settings, results, and
evaluation. Section V summarizes the related research work
in this field, and Section VI concludes this paper.

II. BACKGROUND

A. In-vehicle controller area network

The in-vehicle network is important

to realize complex
functions, high performance, and good comfort of vehicles.
The structure of the in-vehicle network is related to the
manufacturer and the type of vehicle and has no unified
standard [16], most of which is mainly implemented by CAN,
LIN, and Ethernet. According to the development trend of in-
vehicle networks, CAN is in the dominant position.

CAN is developed by Robert Bosch in 1985 to reduce
communication costs and complexity. Due to its high effi-
ciency and stability, CAN has been the most representative
in-vehicle network technology and is widely used in the On-
Board Diagnostics II (OBD-II) standard as a major protocol
[17]. As shown in Fig. 1, the topology of in-vehicle CAN
can be divided into different control units according to their
functional domains, such as the infotainment units, comfort

2

TABLE I
SAMPLE OF CAN PACKETS: DIFFERENT ID, DLC, AND PAYLOAD FIELDS

Timestamp

ID

DLC

Payload

1479121434.854108
1479121434.854290
1479121434.854947
1479121434.869396
1479121434.870212

0545
02b0
043f
05f0
0350

8
5
8
2
8

d8 00 00 8a 00 00 00 00
8d ff 00 07 02
00 40 60 ff 5a 6c 08 00
f4 00
05 28 a4 66 6d 00 00 82

units, chassis units, and powertrain units. Benefiting from the
message-broadcast-based characteristic of the in-vehicle CAN
bus, the ECUs can be interconnected with each other and make
there is only one gateway
efficient cooperation. However,
between the external interfaces, diagnostic, and the CAN bus
which has low security without permission authentication and
information encryption. Fig. 1 also illustrates that the OBD-
II port and the external interfaces, such as Bluetooth, WiFi,
and telematic services, have provided a wide attack surface
to the in-vehicle CAN bus. Once the attacker breaks through
the gateway, the entire in-vehicle network will be threatened.
Therefore, the CAN bus is very fragile and is surrounded by
various threats, and the gateway is the most fatal place to
install the IDS.

Generally, the in-vehicle CAN protocol has two formats:
the standard format and the extended format. The standard
format was standardized by the International Organization for
Standardization (ISO) 11898 [18], as shown in Fig. 2, which
has an 11-bit identifier (ID), while the extended format has
a 29-bit ID frame. The standard data frame is composed of
a 1-bit Start of Frame (SOF), a 12-bit arbitration field which
consists of an 11-bit ID and 1-bit RTR, a 6-bit control field
mainly including a 4-bit DLC, a data field (called payload) in
the range of 0 to 8 bytes (the length is related to the value of
DLC), a CRC field, an ACK field, and a 7-bit End of Frame.
In the field of intrusion detection, researchers usually pay
more attention to the CAN ID, DLC, and payload. They are
the three most important data fields for vehicle control and
sensor feedback and will contain more attack features if an
attack occurs. The CAN ID determines the priority of CAN
messages, the DLC determines the length of the data field
and the payload contains most of the control information and
status information of ECUs. A representative and widely used
dataset for in-vehicle CAN intrusion detection, called the Car-
Hacking dataset, is collected by Seo et al [19]. Some samples
in the Car-Hacking dataset with different values of ID, DLC,
and data payload are shown in Table I.

In addition to the normal in-vehicle CAN communication
message data, there are four types of network-based attacks
in this dataset, DoS attack, fuzzy attack, and impersonation
attack, including spoofing the drive gear and spoofing the RPM
gauze. They can all attack vehicles via network connections
and have different attacking behaviors. The details of these
attacks are given as follows.

1) DoS attack: An attack that blocks other communication
information by occupying the communication channel.

3

Fig. 1. The topology of in-vehicle CAN bus and the possible attack routes: (a) attacking through the OBD-II port; (b) attacking from the external interfaces;
(c) attacking by infected ECU to occupy the CAN bus.

Fig. 2. The standard data frame of CAN bus.

Since the CAN bus structure is based on message broad-
casting, the order of message sending is determined by
the priority of messages. The lower the value of the
CAN ID, the higher the priority of the CAN message.
Therefore, a DoS attacker always forges numerous CAN
messages with the lowest ID values of 0x0000 to occupy
the CAN bus resources and cause system paralysis.
2) Fuzzing attack: An attack that uses malicious ECU to
send fake messages to the in-vehicle CAN bus. Fuzzy
attacks use random ID values to reduce the recognition
of attack features and avoid being detected. Unlike DoS,
fuzzy attacks are much slower to send fake messages.
If the sending speed is the same as the normal CAN
messages,
to be
discovered.

this type of attack will be difficult

3) Impersonation attack: An attack that can implement
unauthorized access by spoofing legitimate authentica-
tion. The attacker will simulate a legitimate ECU node
and send effective simulated messages to the in-vehicle
CAN bus, but cause the component behavior to be
abnormal. Spoofing the drive gear and spoofing the RPM
gauze are two attacks to simulate the drive gear and RPM
behavior, respectively.

In order to improve the security of in-vehicle networks, this
paper proposed a novel intrusion detection technology that is
lightweight enough to be installed on the gateway. And, its
detection effect is evaluated on the four types of attack data
in the Car-Hacking dataset mentioned above. The details of

experimental settings and results are described in Section IV.

B. Electronic control units

The in-vehicle CAN is actually designed to provide more
efficient and faster communication between ECUs. Actually,
ECU is indispensable to the realization of various complex
functions in modern intelligent vehicles.

There are many sensors and actuators in the vehicle, which
are respectively used to obtain the status data of vehicle
components and make the components perform the specified
actions according to the commands. The ECU is responsible
for receiving the electrical signals from the sensor, calculating
and evaluating these electrical signals according to the control
program set in advance. In addition, if the ECU receives a
control command from the driver, it will also calculate the
triggering signal for the actuator according to the program.
As shown in Fig. 3, the control program is stored in a special
memory, and the calculation is done by a microprocessor in the
ECU. Besides, it mainly contains some data memory, IO ports,
and a clock generator. On the whole, the internal structure
of ECU is very simple, but this also leads to very limited
memory and computing resources [20], especially for running
and storing a deep learning model which has a large size and
computation.

In the automotive industry, companies and manufacturers
are equipping their vehicles with a large number of ECUs
to control various vehicle components, such as ventilation
systems, window control systems, and engine control systems,

4

Fig. 4. The schematic diagram of convolution calculation. The convolution
kernel is used as the weight to multiply and add the corresponding input data
pixel points to obtain a neuron of the feature map. Then, the convolution
kernel is sliding according to the step size (1 in this figure) to calculate other
neurons, forming a complete feature map.

can reduce the parameters to be calculated, and different
feature maps (also called channels) can be obtained by
convoluting the image with multiple convolution kernels.
Its main ability is to detect the same type of features
at different positions,
it has good translation
invariance.

that

is,

• Pooling: It mainly includes two operations, average pool-
ing and maximum pooling, which take the average or
maximum value of all pixels in the receptive field as a
neuron of the feature map. Its main function is to select
features and reduce the number of features, thus reducing
the number of parameters.

Furthermore, we can get the formula for calculating the size

of the receptive field from the convolution operations:

F (i) = (F (i + 1) − 1) × Stride + Ksize,

(1)

where F (i) is the side length of the receptive field on the ith
level, Stride is the step size of sliding on the ith level and
Ksize is the side length of the convolution or pooling kernel.
According to the Equation (1), some research has proved that
the 5 ∗ 5 convolution kernel can be replaced by stacking two
3 ∗ 3 convolution kernels, and the 7 ∗ 7 convolution kernel can
be replaced by stacking three 3 ∗ 3 convolution kernels with
the same size of the receptive field [21]. Obviously, the 3 ∗ 3
kernel has much fewer parameters than a larger-scale kernel.
Therefore, in our model design, we stack multiple layers of
3 ∗ 3 convolution kernels instead of a large-scale convolution
kernel.

D. Recurrent neural network

Fig. 3. The schematic diagram of ECU internal structure.

in order to support advanced system functions [6]. Generally,
the number of ECUs is about dozens in the vehicle and can
be as high as hundreds in a high-end automobile. Also, these
ECUs are serial on their own in-vehicle buses as illustrated
in Fig. 1. Therefore, in order to solve the resource limitation
problem of a single ECU, we designed a parallel lightweight
neural network based on the above characteristics to make full
use of multiple ECU resources.

In fact, different ECU resources in the car have different
utilization and importance. For example, as shown in Fig. 1,
the ECUs can be divided into different domains according to
the functional modules they are responsible for. When con-
sidering vehicle security, the ECUs in the powertrain domain
must be more important than the ECUs in the infotainment
domain. Considering the frequency of utilization, the ECUs in
the chassis domain, like the brake control unit, may be busier
than ECUs in the comfort domain, like the window control
unit. Because different ECUs have different risk indexes
and resource occupancy, we also designed an algorithm to
dynamically allocate the running load of the model according
to the ECU status.

C. Convolutional neural network

The deep neural network uses high-level features to repre-
sent the abstract semantics of data by building a multi-layer
network, so it has excellent feature learning ability, but also
has a large memory consumption and computing resource
consumption. CNN is a kind of deep neural network with
convolution structures, but convolution structures can reduce
the amount of memory occupied by a neural network. There
are three key operations: receptive field localization, weight
sharing, and pooling, which effectively reduce the number of
network parameters and alleviate the problem of overfitting
the model. As illustrated in Fig. 4:

• Receptive field localization: each neuron does not need to
receive the whole image, but only needs to feel the local
features. Then, the global information can be obtained
by synthesizing these different local neurons at a higher
level, which can reduce the number of connections.
• Weight sharing: A convolution operation of an image
uses the same convolution kernel for each neuron. That

The convolutional neural network can effectively learn the
spatial features of each input, but it can not learn the temporal
features, which is the sequence relationship between inputs.
The Recurrent Neural Network (RNN) is a kind of neural
network with memory ability [22]. In the RNN, neurons can
not only learn their own input features but also receive infor-
mation from other neurons, forming a long-term “memory”.

5

values of the tth in-vehicle CAN message with N data fields.
The training data consists of normal data and four types of
attack data. Our purpose is to detect attacks in testing data
(cid:104)
test, . . . , x(T ′)
x(1)
which is denoted as xtest =
. The output of
our model is the corresponding set of multiple classification
result labels.

test

(cid:105)

Fig. 5. The schematic diagram of recurrent neural networks.

As shown in Fig. 5, xt is a vector, which represents the value
of the input layer at time t. ht is a vector representing the value
of the hidden layer at time t. ot is a vector representing the
value of the output layer at time t. U and V are the weight
matrices respectively for the computation between the input
layer and the hidden layer, and between the hidden layer and
the output layer. The weight matrix W is for the transfer of
information contained in the hidden layer between different
times. As a result, the value h of the hidden layer depends
not only on the current input x but also on the value h at
the previous time, which contains all the information of h at
previous times. In the same hidden layer, the weight matrices
W , V , and U at different times are equal, which forms the
parameter sharing in the RNN.

III. THE PROPOSED FRAMEWORK

A. The problem statement

Deep learning algorithms are widely used in many fields, but
most of them are only available in cloud computing because
of their large calculation and ample cloud resources. However,
the computing resources on devices, especially ECU devices,
are too limited to execute deep learning algorithms with
large model sizes. The existing in-vehicle CAN bus intrusion
detection technologies have the problems of massive model
parameters, large size, high detection delay, and high resource
consumption for one computing device.

Actually, in the field of deep learning, there is always a
contradiction between the lightweight degree and detection
effect of a learning model, which is often impossible to
have both. There is also a certain contradiction between the
timeliness and complexity of the model and the detection
effect of the model in the field of deep learning, as well as
the field of intrusion detection. Also, high model complexity
is always required to achieve a high detection accuracy but
can lead to low timeliness. However, in-vehicle IDS has a
high requirement for timeliness because even a very short
breakdown can be fatal to a speeding car. Therefore, our
goal is to design a learning model that can detect intrusion
attacks with high accuracy, low time delay, and low resource
consumption for each computing device.

In our study,

in-vehicle CAN bus data is obtained in
chronological order: the training data is denoted as xtrain =
(cid:104)
train, . . . , x(T )
x(1)
, which is used to train our models. The
value xt
train is an N dimensional vector representing the

train

(cid:105)

B. The framework overview

According to the characteristics of the in-vehicle CAN
bus, which is based on message broadcast and equipped
with hundreds of ECUs and other computing devices. We
make full use of its natural advantages of parallel resources
and propose an intrusion detection method for the in-vehicle
CAN bus based on parallel lightweight multi-feature learning
architecture, which can dynamically adapt the running load
to resources of multiple computing devices. Specifically, we
present ECU devices as the example of the use for LiPar to
allocate computational resources. In fact, LiPar is applicable
to all scenarios where computing resources are allocated to
multiple devices.

lightweight

Fig. 6 shows the proposed parallel

learning
architecture, named LiPar. The steps of data reading, data
pre-processing, and final feature fusion and classification of
in-vehicle CAN messages are all completed in the central
electronic module (CEM), which can be the OBD-II or the
gateway of the in-vehicle CAN bus. Each branch network is
the main network structure for feature extraction and feature
learning, which contains a large amount of computation.
Therefore, the computation tasks of each branch will be as-
signed to different ECUs according to the resource adaptation
algorithm proposed in this study. Also, the network design
of each branch is shallow (no more than 10 layers), so as
to ensure that each branch network is lightweight enough
and will not generate excessive load on ECU. In addition,
the parallel structure of DWParNet can enable each branch
to conduct down-sampling and spatial feature extraction from
different dimensions. STParNet can use one more branch for
temporal feature extraction to learn more comprehensive and
full features and improve the detection performance of the
model after feature fusion.

C. DWParNet: the parallel learning network for spatial fea-
ture extraction

The DWParNet we proposed plays an important role in the
whole model, mainly reflected in two aspects: 1) It undertakes
almost all the tasks of extracting and learning spatial feature
information, and 2) the main parts of the parallel structure are
embodied in the DWParNet.

We adopt CNN structure in DWParNet for spatial fea-
ture extraction and limit the size of convolution kernels to
3 ∗ 3 instead of larger-scale kernels to reduce the number
of training parameters. However, traditional convolution still
has some redundant computations that can be pruned. For
further lightweight design in DWParNet, we adopt a lot of
Depthwise (DW) Convolutions, which is a special form of
group convolution, instead of traditional convolution to reduce
the model parameters and computation.

6

Fig. 6. The architecture of our parallel lightweight learning method based on
resource adaptation.

In general, the convolution on one layer will use multiple
convolution kernels to obtain more features, the number of
which is consistent with the number of convolution kernels.
As Fig. 7(a) illustrated, each channel of the input image is
fully connected with each convolution kernel. Furthermore,
group convolution is proposed to reduce computation and
parameters [23], which divides the channels of the input data
and convolution kernels into different groups for calculation,
as shown in Fig. 7(b). For example, we assume that the input
data shape is Hi∗Wi∗Ci; the output data shape is Ho∗Wo∗Co;
and the number of groups as g. Since the number of channels
of convolution kernels is equal to the input channels and the
number of convolution kernels is equal to the output features,
the number of standard convolution parameters is

N = Ksize ∗ Ksize ∗ Ci ∗ Co,

(2)

and the number of group convolution parameters is

Ksize ∗ Ksize ∗

Ci
g

∗

Co
g

∗ g = N/g.

(3)

Therefore, group convolution can greatly reduce the parameter
amount according to the number of groups. Especially, when
Co = Ci, we can set g = Ci and it becomes DW convolution
as shown in Fig. 7(c). The disadvantage of DW convolution
is that it can only get the output features with the same shape
as the input data.

To solve the drawbacks of DW convolution, we only use
the traditional convolution operation with a convolution core
size of 1 ∗ 1 in DWParNet when the number of channels needs
to be adjusted by down-sampling. According to experimental
experience, the number of network branches is set to 3. The
details of DWParNet are shown in Fig. 8. We use batch
normalization (BN) to accelerate network training by reducing
internal covariate and the ReLU activation function to prevent
gradient vanishing during model training. Each branch adopts
different granularity feature extraction from multi-dimension,
but they all obtain 2 ∗ 2 multi-channel feature matrices so that
they can be merged in the direction of the channel and learn
more comprehensive features.

(a)

(b)

(c)

Fig. 7. The schematic diagram of conventional convolution, grouping convo-
lution, and DW convolution. (a) Each input channel is fully connected with
Co kernels to get Co output channels. (b) Each color represents each group,
in which Ci/g input channels are fully connected with Co/g kernels to get
Co/g output channels. (c) Each input channel is connected with one kernel
to get one output channel.

D. STParNet: integrating temporal feature extraction

In order to improve the detection effect and generalization
ability of the model, we integrate the temporal feature learning
structure on the basis of DWParNet.

Although RNN is a classical and efficient temporal feature
extraction network, during the training of the RNN, with
the extension of training time and the increased number of
network layers, the problem of gradient explosion or gradient
disappearance occurs easily. To solve this problem, we adopt
the LSTM network which is an improved scheme of RNN [24].
The LSTM network has improved the calculation between the
input layer and the hidden layer by adding a memory cell Ct
to selectively memorize information. In addition, the LSTM
network also sets a forgetting gate ft, update gate it, and
output gate ot, to discard useless memory, learn the essence
from the new input, memorize it, and send it to the next
moment state ht. The specific calculation process is as follows:

ft = σ (Wf · [ht−1, xt] + bf ) ,

it = σ (Wi · [ht−1, xt] + bi) ,

ot = σ (Wo · [ht−1, xt] + bo) ,
˜Ct = tanh (WC · [ht−1, xt] + bC) ,
Ct = ft ∗ Ct−1 + it ∗ ˜Ct,

ht = ot ∗ tanh (Ct) ,

(4)

(5)

(6)

(7)

(8)

(9)

where W is the weight matrix of each gate, b is the bias of
each gate, and ˜Ct is the cellular state of the current memory
cell Ct.

To explore the temporal relationship between in-vehicle
CAN messages, we add the LSTM network structure to the
model, analyze it in combination with the spatial features
extracted by DWParNet, and obtain the final model, STParNet,
as shown in Fig. 9. The LSTM layer is mainly responsible for
learning the temporal features of in-vehicle CAN messages.
The full connection layer is to decode the hidden state of
the last time step and get a logit, which is the output of
the temporal feature learning branch. Then, we calculate the

7

Fig. 8. The network structure of DWParNet.

Fig. 9. The schematic diagram of STParNet structure.

TABLE II
THE HYPERPARAMETERS OF THE PROPOSED MODEL

Hyperparameter

Values

Branch1-Conv1
Branch2-Conv1
Branch2-Conv2
Branch3-Conv1
Branch3-Conv2
Branch3-Conv3
Fusion-Conv
LSTM-Layer
LSTM-Hidden Size
Fusion-Pooling
Fusion-Dropout

64
128
256
32
96
192
192
2
32
Adaptive avg
0.5

average of the logit obtained by DWParNet and the logit of
temporal features and take it as the final classification basis.
After parameter adjustments and model optimization, we
determine the hyperparameters that make the model perfor-
mance optimal, as shown in Table II. Because the number
of convolution cores in DW convolution operation is deter-
mined by the number of channels of input data, no human
intervention is required, so it is not listed in Table II. Except
for the convolution layer of the fusion part, other convolution
operations are all identical to the DWParNet shown in Fig. 8.

E. Resource Adaptation Based on Vehicular Environment

Through different down-sampling processing of sub-neural
networks, the number of parameters to be processed and the

memory usage are different for each sub-neural network. The
less computation of the sub-network, the fewer ECU resources
and less running time it occupies. Similarly, different ECU
devices in the original vehicle network will produce different
resource occupancy according to the functional modules in
charge.

To make full use of numerous ECU resources in the vehicle,
we also propose a resource adaptation algorithm based on
the in-vehicle CAN bus. The algorithm mainly quantifies the
importance and resources of each ECU module and collects
indicators from three aspects: processor idle rate, memory idle
rate, and importance level of functional modules in ECUs.
Then, we analyze the memory occupancy and calculation rate
of each branch network on the corresponding ECU, and define
the resource occupancy index of the branch network. We
compare the branch network occupancy index with the ECU
availability index to determine whether to allocate the branch
network to the corresponding ECU. The mathematical model
we constructed is as follows.

In this method, n shallow sub-neural network structures are
allocated to n different ECUs to run. Let the total number of
ECUs in the vehicle be N , and the processor idle rate and
the memory idle rate of the i th ECU be Pi and Mi when
it is running normally. Then, the total resource idle rate Si
of the ECU can be comprehensively evaluated and defined as
follows:

Si =

2Pi · Mi
Pi + Mi

.

(10)

Automobile manufacturers always have a risk rating for
the car’s module system, indicating the risk level the module
malfunction could lead to. We use these risk indexes to
describe the importance index of the module, set as a positive
integer Ri (usually no more than 10). The higher the index
shows the higher the importance index is, and the more
inappropriate it is to be assigned the task of a branch network.
Therefore, the overall availability index Ui of the i th ECU is
defined as follows:

Ui = (cid:0)1 + α2(cid:1)

Si · 1
Ri
+ Si

α2 · 1
Ri

,

(11)

where α is a positive integer coefficient, used to adjust the
balance between the importance and the resource idle rate.
Then, by substituting Equation (10) into Equation (11) and
simplify, we have:

Ui =

2 (cid:0)1 + α2(cid:1) · Pi · Mi
α2 · (Pi + Mi) + 2Pi · Mi · Ri

.

(12)

In addition, we define the memory occupation ratio of the
j th branch network to be allocated on the i th ECU as
mij (mij < 1), that is, the ratio of the memory occupied by the
j th branch network to the total memory of the i th ECU. Since
the main calculation amount of the learning network model
is forward/backward propagation, the calculation amount of
the forward/backward propagation process model can reflect
the situation of CPU resource occupation. We define the
calculation rate of the j th branch network structure as:

cj =

F orward/backward pass size
T otal model size
Based on a comprehensive evaluation, the resource occu-
pation index of the j th branch network on the i th ECU is
defined as:

(13)

.

Oij =

(cid:0)1 + β2(cid:1) · cj · mij
β2 · mij + cj

,

(14)

where β is a positive integer coefficient, used to adjust
the balance between the memory occupation ratio and the
calculation rate.

The algorithm compares Ui with Oij to determine whether
the task of the j th branch network is suitable to be assigned
to the i th ECU. If Ui ≥ Oij, the i th ECU can be selected to
install the j th branch shallow neural network, otherwise the
branch module cannot be loaded onto the ECU.

IV. EXPERIMENTAL RESULTS AND PERFORMANCE
EVALUATION

A. Experimental setup

This experiment was carried out on a MacBook Pro note-
book. The hardware conditions include a 2.2GHz quad-core
Inter Core i7 processor, 16GB 1600MHz DDR3 memory, and
an Inter Iris Pro 1536MB graphics card. The experiment is
based on the operating system of MacOS Monterey version
12.6 and PyCharm 2022.2.2 Community version, in which
Python 3.9.12 and Pytorch 1.12.1 are installed.

8

TABLE III
THE DATASET PARTITION FOR TRAINING, VALIDATION, AND TESTING

Type

Training set

Validation Set

Testing set

25,637
Normal
28,145
DoS
33,439
Fuzzy
Spoofing Gear
49,063
Spoofing RPM 53,646

7,325
8,042
9,554
14,018
15,328

3,662
4,020
4,776
7,009
7,663

spatial feature extraction and temporal feature extraction, there
are two forms of data for branch network input.

First, the in-vehicle CAN message data is processed into
image form. We extract CAN ID and 8-byte payload from
CAN message data to form a feature vector of a CAN message
with 9 feature values. Then, every 9 feature vectors are taken
to form a 9 * 9 two-dimensional feature matrix which is
taken as a channel of an image, and every 3 channels form
an image which is stored in a three-dimensional tensor. That
is, the shape of each input data for the convolution branch is
3 * 9 * 9. The image data distribution after pre-processing is
shown in Table III. If the image is entirely composed of normal
messages, it will be marked as normal data, otherwise, it will
be marked as the corresponding type of attack data.

For

the temporal

feature extraction branch,

the three-
tensor will be expanded and reshaped into a
dimensional
two-dimensional tensor with a shape of 27 * 9. Every 27
vectors are input into the LSTM network as a sequence with
a temporal relationship. It should be noted that the LSTM
network structure needs to learn the temporal characteristics
between CAN messages, so the sequence of 27 data in each
group that form an image cannot be disturbed during data
processing.

C. Model training

We have adopted a validation dataset to determine the best
hyperparameters of the model and model training. The trend
of loss value and accurate value during cross-validation is
shown in Fig. 10. According to the changing trend of loss
and accuracy, we decide to take 14 as the best training epoch
because there is not too much performance improvement of
the model after epoch 14 and further training will lead to
over-fitting. We train the proposed STParNet on the Adam
optimizer with a 0.0001 initial learning rate and use a sparse
categorical cross-entropy to calculate the loss value. All the
best-performing hyperparameters for model training are sum-
marized in Table IV.

B. Data pre-processing

D. Evaluation metrics

The dataset we used is the Car-Hacking dataset [19], which
is obtained by recording CAN traffic through OBD-II port in a
real vehicle and includes one normal dataset and four different
attack datasets: DoS, Fuzzy, spoofing GEAR and spoofing
RPM. Since the model proposed in this study combines
branches of two types of network structure, respectively for

In this study, we not only evaluate the detection ability
and generalization ability of the model but also evaluated the
lightweight degree and running efficiency of the model. We
used the confusion matrices, top-1 accuracy, and AUC values
to evaluate the detection effect of models. The AUC is selected
because it can reflect the processing ability of the model for

9

(a) The training loss and validation loss

(a) The confusion matrix of DWParNet

(b) The training accuracy and validation accuracy

Fig. 10. The curves of loss and accuracy during model training.

TABLE IV
THE HYPERPARAMETERS FOR TRAINING THE MODEL

Hyperparameter for training

Values

Learning rate
Optimizer
Loss function
Batch size
Epoch

0.0001
Adam
sparse categorical cross-entropy
32
14

the unbalanced dataset, that is, the generalization ability of the
model. The calculation method of top-1 accuracy is

Accuracy =

1
N

N
(cid:88)

i

1 (yi = ˆyi) .

(15)

To evaluate the lightweight degree of the model, we used
the total parameter quantity and total memory consumption
to describe the model size. Among them, the total memory
consumption is about the sum of forward/backward pass usage
and parameter usage. In addition, training speed and inference
speed are used to evaluate the running efficiency of the model.

E. Results and analysis

One of the purposes of our research is to verify whether the
integration of temporal and spatial feature learning is better
than the single spatial feature learning structure. Therefore,
we compare and analyze the intrusion detection results of
DWParNet and STParNet. The results are shown in Fig. 11.
Both DWParNet and STParNet have great detection accuracy

(b) The confusion matrix of STParNet

Fig. 11. The detection effect of DWParNet and STParNet.

of each kind of attack data and normal data, all higher than
0.9998. Compared with the single spatial feature learning
network DWParNet,
the STParNet proposed in this paper
has improved the detection and classification performance of
attacks. The classification accuracy of the four attacks has
been improved by 0.00050, 0.00084, 0.00043 and 0.00013
respectively, which illustrates that adding LSTM structure to
extract temporal features and analyzing with spatial features
can effectively improved the detection ability of the model
compared with only learning data spatial features.

Moreover, we select three baseline methods for comparative
experiments: MobileNetV3 [25], EfficientNet [26], and CANet
[27]. Although MobileNetV3 and EfficientNet are not models
developed for vehicular environments, nor are they models for
intrusion detection, they are the most classic models of deep
learning algorithms developed from cloud to end, and from
deep learning to lightweight learning. They all have advanced
performance and small model sizes and are the best among the

10

TABLE V
THE RESULTS OF DETECTION PERFORMANCE

Models

Accuracy

AUC

DWParNet (Ours)
STParNet (Ours)
MobileNetV3
EfficientNet
CANet

0.9994
0.9998
0.9990
0.9993
0.9942

0.99981
1.00000
0.99990
0.99996
0.92146

Fig. 13. The training speed and inference speed of the models. “item” refer
to the data that the model can process at one time and “items/s” refers to the
batches of data that the model can process per second.

eter quantity and calculation quantity of each branch network
are collected through experiments, the results are shown in
Table VI. The four branch structures will be allocated to four
different ECUs according to the algorithm. The fusion part is
undertaken by the central electronic module. That is because
the task load of fusion is relatively large and the calculation
is concentrated, and the central electronic module has more
resources than ECU and can fully undertake the feature fusion
part. The results prove that the design of parallel architecture
can greatly reduce the model size and resource consumption
for each ECU, even less than 0.5MB, while ensuring detection
capability. Among them, the size and task load of branch 4 are
0.06MB which is similar with that of the CANet model, about
0.03MB, but from the perspective of the intrusion detection
effect, STParNet has more advantages and higher security as
mentioned before.

According to the mathematical model proposed in Sec-
tion III-E, the calculation rates of the four branch networks are
0.8889, 0.6216, 0.5833, and 0.1667, calculated by Equation
(13). We assumed that the total size of each ECU memory
is 1MB. Then, calculated by Equation (14), their memory
occupancy rates are 0.09, 0.37, 0.24, and 0.06. Based on
experience, the memory occupation ratio has a greater impact
on the resource occupation index, so we set β = 2. After cal-
culation, the resource occupation indexes of the four branches
are 0.3203, 0.5472, 0.4535, and 0.1230. The result illustrates
that the four branches all have a very low demand for available
resources of ECU. As long as Ui ≥ Oij is satisfied, these four
branches can be allocated to the corresponding ECU.

V. RELATED WORKS
The IDSs for the in-vehicle CAN bus can be divided into
four types: (1) the specification-based approaches which detect
abnormal behavior that does not match system specifications,
such as protocols and frame formats [28]; (2) the fingerprint-
based approaches which use contours defined based on ECU
features rather than human-defined specifications, such as the
clock skew of ECUs [29] and the voltage fingerprint [30],
[31]; (3) the statistics-based approaches, such as entropy-
based [32] and frequency-based [33]; (4) the learning-based

Fig. 12. The parameter quantity and memory consumption of the models.

lightweight models of deep learning networks so far. CANet is
the most lightweight model with better performance in the field
of in-vehicle CAN bus intrusion detection, which is selected
by a large number of experiments in the research [15]. The
comparative experimental results of detection performance,
lightweight degree, and running efficiency are shown in Ta-
ble V, Fig. 12, and Fig. 13, respectively.

These experimental results show that the STParNet pro-
posed in our study has the highest detection accuracy and
AUC value, meaning that it has the greatest intrusion de-
tection performance among these models. Compared with
MobileNetV3 and EfficientNet, it can be seen that STParNet
can greatly reduce the model parameter quantity and memory
usage to half and a quarter of the two models’ original sizes
respectively, which has been able to meet the constraints of
in-vehicle resources. Also, STParNet has obvious advantages
in both training speed and inference speed, about two times
of MobileNetV3’s speed and ten times of EfficientNet’s speed.
For DWParNet, the training speed of STParNet is also slightly
improved, from 10.30 to 12.79 items/s. Compared with the
CANet model, STParNet still has a gap in the size and running
speed of the model, but the detection effect is much better
than CANet, improving 0.0056 in accuracy, and especially, its
generalization ability described by AUC value is improved
from about 0.92146 to 1.00000, which means that
it can
provide better security for the in-vehicle network.

Furthermore, we calculate and analyze the resource con-
sumption of each branch network of the STParNet. The param-

11

TABLE VI
THE STATISTICAL RESULTS OF EACH BRANCH NETWORK OF STPARNET AND BASELINE CANET

Branch/model

Forward/backward pass size (MB)

Parameter size (MB)

Total size (MB)

Calculation rate cj

Occupation index Oij

Branch1-Conv
Branch2-Conv
Branch3-Conv
Branch4-LSTM
Fusion
STParNet (Ours)
CANet
MobileNetV3
EfficientNet

0.08
0.23
0.14
0.01
0.01
0.47
0.01
4.20
8.52

0.00
0.15
0.10
0.05
3.38
3.68
0.02
5.81
15.31

0.09
0.37
0.24
0.06
3.40
4.15
0.03
10.70
24.51

0.8889
0.6216
0.5833
0.1667
/
/
0.3333
/
/

0.3203
0.5472
0.4535
0.1230
/
/
0.1103
/
/

approaches, such as [34], [11], and [12], which perform better
than other types without prior expert knowledge, especially
deep learning-based methods, but rely on sufficient computing
resources.

In order to enable the IDSs to be installed on the terminal so
that it can achieve more efficient and stable detection capabil-
ity, the existing research direction of learning-based technol-
ogy has been developing towards lightweight models. There
are two kinds of lightweight directions we have investigated in
the field of IDS: (1) using a simple and shallow CNN structure
combined with other technologies to enhance the detection
performance, such as combining an LSTM structure [35] and
using recursive graphs to process the data [36], and (2) using
autoencoder structure with lightweight neural units, such as
[27], [37], and [38]. In the research [15], these models have
been analyzed in the same experimental environment, and it
is found that CANet has the best comprehensive performance.
Therefore, we chose the CANet as one of the baseline methods
in our experiment.

In fact, the development direction of deep learning model
lightweight is not only in the field of IDS but also in the field
of deep learning image processing which began the research
on lightweight structure earlier. Two of the lightweight tech-
nologies are to use the DW convolution instead of traditional
convolution and to adopt Squeeze-and-Excitation (SE) mod-
ular structure. They are both mentioned in [25] and [26]. In
addition, channel shuffle and channel split can also be used to
improve the running efficiency of deep learning models, such
as [39].

For parallel network structure, although there is no corre-
sponding application in the field of IDS, there are parallel
structures and models of neural network learning in the field
of image processing. For example,
the Inception structure
proposed in research [40] is a parallel structure, although the
whole network is still composed of these Inception structures
that are deeply stacked. Or, the algorithm parallelizes the
learning process of a deep network, such as [41]. The one
using parallel branches as network structure we found is the
ParNet proposed by Ankit et al. in [42].

VI. CONCLUSION

We have proposed a lightweight intrusion detection model,
LiPar, for the in-vehicle CAN bus, including the STParNet

and the resource adaptation algorithm. Through comparative
experiments on STParNet and DWParNet, we have proved that
the fusion of temporal and spatial feature extraction structures
can effectively improve the intrusion detection accuracy and
the generalization ability of the model. Moreover, we found
that the design of parallel structures with multi-dimension
feature extraction can help to enhance a lot of the detection
performance as well as reduce the model size and task load for
each ECU. We have also proposed a resource adaptation algo-
rithm to allocate the branch task to the appropriate ECU. After
calculating,
the resource occupancy indexes also illustrate
that the branched STParNet are lightweight enough to easily
find ECU suitable for allocation. In conclusion, the LiPar
model we proposed has great advantages in terms of detection
performance, generalization ability, running efficiency, and
lightweight degree. It can be loaded on the vehicle practically
and be a better choice for the in-vehicle CAN bus intrusion
detection system.

ACKNOWLEDGMENTS

This work is supported by National Natural Science Foun-
dation of China (NSFC) (grant number 62272129) and Taishan
Scholar Foundation of Shandong Province (grant number
tsqn202408112).

REFERENCES

[1] H. H.

Jeong, Y. C. Shen,

Jeong, and T. T. Oh, “A
J. P.
comprehensive survey on vehicular networking for safe and efficient
driving in smart transportation: A focus on systems, protocols, and
applications,” Vehicular Communications, vol. 31, p. 100349, 2021.
[Online]. Available: https://www.sciencedirect.com/science/article/pii/
S2214209621000188

[2] J. Petit and S. E. Shladover, “Potential cyberattacks on automated
vehicles,” IEEE Transactions on Intelligent Transportation Systems,
vol. 16, no. 2, pp. 546–556, 2015.

[3] S. Checkoway, D. McCoy, B. Kantor, D. Anderson, H. Shacham, S. Sav-
age, K. Koscher, A. Czeskis, F. Roesner, and T. Kohno, “Comprehensive
experimental analyses of automotive attack surfaces,” in Proceedings of
the 20th USENIX Conference on Security, ser. SEC’11. USA: USENIX
Association, 2011, p. 6.

[4] C. Young, J. Zambreno, H. Olufowobi, and G. Bloom, “Survey of
automotive controller area network intrusion detection systems,” IEEE
Design Test, vol. 36, no. 6, pp. 48–55, 2019.

[5] E. Aliwa, O. Rana, C. Perera, and P. Burnap, “Cyberattacks and
countermeasures for in-vehicle networks,” ACM Comput. Surv., vol. 54,
no. 1, mar 2021. [Online]. Available: https://doi.org/10.1145/3431233

[6] H. J. Jo and W. Choi, “A survey of attacks on controller area networks
and corresponding countermeasures,” IEEE Transactions on Intelligent
Transportation Systems, pp. 1–19, 2021.

[7] H. M. Song and H. K. Kim, “Self-supervised anomaly detection for in-
vehicle network using noised pseudo normal data,” IEEE Transactions
on Vehicular Technology, vol. 70, no. 2, pp. 1098–1108, Feb. 2021.
[8] A. Jolfaei, N. Kumar, M. Chen, and K. Kant, “Guest editorial introduc-
tion to the special issue on deep learning models for safe and secure
transportation systems,” IEEE Transactions on Intelligent
intelligent
Transportation Systems, vol. 22, no. 7, pp. 4224–4229, Jul. 2021.
[9] A. Mchergui, T. Moulahi, and S. Zeadally, “Survey on artificial in-
telligence (AI) techniques for vehicular ad-hoc networks (VANETs),”
Vehicular Communications, vol. In Press, pp. 1–16, Aug. 2021.
[10] W. Wu, R. Li, G. Xie, J. An, Y. Bai, J. Zhou, and K. Li, “A survey
of intrusion detection for in-vehicle networks,” IEEE Transactions on
Intelligent Transportation Systems, vol. 21, no. 3, pp. 919–933, 2020.

[11] H. M. Song, J. Woo, and H. K. Kim, “In-vehicle network intrusion
detection using deep convolutional neural network,” Vehicular Commu-
nications, vol. 21, no. 100198, pp. 1–13, Jan. 2020.

[12] S. Tariq, S. Lee, and S. S. Woo, “CANTransfer: transfer learning based
intrusion detection on a controller area network using convolutional
LSTM network,” in Proceedings of the 35th Annual ACM Symposium
on Applied Computing (SAC 20). Brno, Czech Republic: ACM, Mar.
2020, pp. 1048–1055.

[13] S. Yi, C. Li, and Q. Li, “A survey of fog computing: Concepts,
the 2015 Workshop
applications and issues,” in Proceedings of
on Mobile Big Data, ser. Mobidata ’15. New York, NY, USA:
Association for Computing Machinery, 2015, p. 37–42.
[Online].
Available: https://doi.org/10.1145/2757384.2757397

[14] A. H. Saleh and A. Anpalagan, “AI empowered computing resource
allocation in vehicular ad-hoc NETworks,” in 2022 7th International
Conference on Business and Industrial Research (ICBIR), 2022, pp.
221–226.

[15] K. Wang, A. Zhang, H. Sun, and B. Wang, “Analysis of recent deep-
learning-based intrusion detection methods for in-vehicle network,”
IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 2,
pp. 1843–1854, 2023.

[16] X. Xiaojuan and L. Zhiyuan, “Simulation technique of in-vehicle can
network based on matlab,” in 2007 Chinese Control Conference, 2007,
pp. 661–665.

[17] S. Park and J.-Y. Choi, “Hierarchical anomaly detection model for
in-vehicle networks using machine learning algorithms,” Sensors,
vol. 20, no. 14, 2020. [Online]. Available: https://www.mdpi.com/
1424-8220/20/14/3934

[18] International Organization for Standardization (ISO), “ISO 11898-
1:2015 Road vehicles — Controller Area Network (CAN),” 2015,
available online: https://www.iso.org/standard/63648.html (accessed on
20 Dec 2022).

[19] E. Seo, H. M. Song, and H. K. Kim, “GIDS: GAN based intrusion de-
tection system for in-vehicle network,” in 2018 16th Annual Conference
on Privacy, Security and Trust (PST), 2018, pp. 1–6.

[20] M. Kaiser, U. Schaefer, and G. Haaf, Electronic control unit.
Wiesbaden: Springer Fachmedien Wiesbaden, 2015, pp. 18–43.
[Online]. Available: https://doi.org/10.1007/978-3-658-03975-2 3
[21] S. Karen and Z. Andrew, “Very deep convolutional networks for
large-scale image recognition,” arXiv, no. 1409.1556, Apr 2015.
[Online]. Available: https://doi.org/10.48550/arXiv.1409.1556

[22] C. Kyunghyun, M. Bart, van, G. Caglar, B. Dzmitry, B. Fethi, S. Holger,
and B. Yoshua, “Learning phrase representations using rnn encoder-
decoder for statistical machine translation,” arXiv, no. 1406.1078, Sep
2014. [Online]. Available: https://doi.org/10.48550/arXiv.1406.1078
[23] Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi, “Deep roots:
Improving cnn efficiency with hierarchical filter groups,” in 2017 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2017,
pp. 5977–5986.

[24] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural

Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[25] H. Andrew, S. Mark, C. Grace, C. Liang-Chieh, C. Bo, T. Mingxing,
W. Weijun, Z. Yukun, P. Ruoming, V. Vijay, V. L. Quoc, and
A. Hartwig, “Searching for MobileNetV3,” arXiv, no. 1905.02244, Nov
2019. [Online]. Available: https://doi.org/10.48550/arXiv.1905.02244

[26] T. Mingxing and V. L. Quoc, “Efficientnet: Rethinking model scaling
for convolutional neural networks,” arXiv, no. 1905.11946, Sep 2020.
[Online]. Available: https://doi.org/10.48550/arXiv.1905.11946

[27] M. Hanselmann, T. Strauss, K. Dormann, and H. Ulmer, “CANet: An
unsupervised intrusion detection system for high dimensional CAN bus
data,” IEEE Access, vol. 8, pp. 58 194–58 205, 2020.

12

[28] U. Larson, D. Oka, and E. Jonsson, “An approach to specification-based
attack detection for in-vehicle networks,” 07 2008, pp. 220 – 225.
[29] S. U. Sagong, X. Ying, A. Clark, L. Bushnell, and R. Poovendran,
“Cloaking the clock: Emulating clock skew in controller area networks,”
in 2018 ACM/IEEE 9th International Conference on Cyber-Physical
Systems (ICCPS), 2018, pp. 32–42.

[30] W. Choi, K. Joo, H. J. Jo, M. C. Park, and D. H. Lee, “Voltageids: Low-
level communication characteristics for automotive intrusion detection
system,” IEEE Transactions on Information Forensics and Security,
vol. 13, no. 8, pp. 2114–2129, 2018.

[31] O. Schell and M. Kneib, “Valid: Voltage-based lightweight intrusion
detection for the controller area network,” in 2020 IEEE 19th Inter-
national Conference on Trust, Security and Privacy in Computing and
Communications (TrustCom), 2020, pp. 225–232.

[32] Z. Yu, Y. Liu, G. Xie, R. Li, S. Liu, and L. T. Yang, “Tce-ids:
Time interval conditional entropy- based intrusion detection system for
automotive controller area networks,” IEEE Transactions on Industrial
Informatics, vol. 19, no. 2, pp. 1185–1195, 2023.

[33] A. Taylor, N. Japkowicz, and S. Leblanc, “Frequency-based anomaly
detection for the automotive can bus,” in 2015 World Congress on
Industrial Control Systems Security (WCICSS), 2015, pp. 45–49.
[34] A. R. Javed, S. u. Rehman, M. U. Khan, M. Alazab, and T. R. G,
“Canintelliids: Detecting in-vehicle intrusion attacks on a controller
area network using cnn and attention-based gru,” IEEE Transactions on
Network Science and Engineering, vol. 8, no. 2, pp. 1456–1466, 2021.
[35] W. Lo, H. Alqahtani, K. Thakur, A. Almadhor, S. Chander, and
G. Kumar, “A hybrid deep learning based intrusion detection
system using spatial-temporal
in-vehicle network
traffic,” Vehicular Communications, vol. 35, p. 100471, 2022.
[Online]. Available: https://www.sciencedirect.com/science/article/pii/
S2214209622000183

representation of

[36] A. K. Desta, S. Ohira, I. Arai, and K. Fujikawa, “Rec-CNN: In-vehicle
networks intrusion detection using convolutional neural networks
trained on recurrence plots,” Vehicular Communications, vol. 35,
p. 100470, 2022. [Online]. Available: https://www.sciencedirect.com/
science/article/pii/S2214209622000171

[37] R. Zhao, J. Yin, Z. Xue, G. Gui, B. Adebisi, T. Ohtsuki, H. Gacanin,
and H. Sari, “An efficient intrusion detection method based on dynamic
autoencoder,” IEEE Wireless Communications Letters, vol. 10, no. 8, pp.
1707–1711, 2021.

[38] Y. Lin, C. Chen, F. Xiao, O. Avatefipour, K. Alsubhi, and A. Yunianta,
“An evolutionary deep learning anomaly detection framework for in-
vehicle networks - CAN bus,” IEEE Transactions on Industry Applica-
tions, vol. Early Access, pp. 1–9, Jul. 2020.

[39] M. Ningning, Z. Xiangyu, Z. Hai-Tao, and S. Jian, “ShuffleNet V2:
Practical guidelines for efficient cnn architecture design,” arXiv, no.
1807.11164, Jul 2018. [Online]. Available: https://doi.org/10.48550/
arXiv.1807.11164

[40] S. Christian, L. Wei, J. Yangqing, S. Pierre, R. Scott, A. Dragomir,
E. Dumitru, V. Vincent, and R. Andrew, “Going deeper with
convolutions,” arXiv, no. 1409.4842, Sep 2014. [Online]. Available:
https://doi.org/10.48550/arXiv.1409.4842

[41] H. Lee, C.-J. Hsieh, and J.-S. Lee, “Local critic training for model-
parallel learning of deep neural networks,” IEEE Transactions on Neural
Networks and Learning Systems, vol. 33, no. 9, pp. 4424–4436, 2022.
[42] G. Ankit, B. Alexey, D. Jia, and K. Vladlen, “Non-deep network,”
arXiv, no. 2110.07641, Oct 2021. [Online]. Available: https://doi.org/
10.48550/arXiv.2110.07641

Aiheng Zhang received the B.S. degree in computer
science and technology from the Beijing University
of Technology, Beijing, China. She is currently pur-
suing a master’s degree in computer technology with
the Harbin Institute of Technology (HIT), China. Her
research interests include intelligent and lightweight
in-vehicle intrusion detection models.

Qiguang Jiang received the B.S. degree in Infor-
mation and Computing Science from the Harbin
Institute of Technology (HIT), Weihai, China. She is
currently pursuing the master’s degree in computer
science and technology with the Harbin Institute
of Technology (HIT), China. Her research interests
include intelligent and efficient in-vehicle intrusion
detection models.

13

Kai Wang received the B.S. and Ph.D. degrees
from Beijing Jiaotong University. He is currently a
Professor with the School of Computer Science and
Technology, Harbin Institute of Technology (HIT),
Weihai. Before joined HIT, he was a postdoc re-
searcher in computer science and technology with
Tsinghua University. He has published more than 40
papers in prestigious international journals, includ-
ing IEEE TITS, IEEE TCE, ACM TOIT, ACM TIST,
etc. His current research interest is on security for
cyber-physical systems and emerging networks (e.g.,
autonomous vehicles, industrial control systems, IoT), and applied machine
learning for network attacks detection and information forensics. He is a
Member of the IEEE and ACM, and a Senior Member of the China Computer
Federation (CCF).

Ming Li obtained his B.S. degree from Shandong
University, M.Sc. degree from Ulm University, and
Ph.D. degree from Hamburg University of Technol-
ogy. He is currently the Vice General Manager and
CTO at Shandong Inspur Database Technology Co.,
Ltd. Before joining the Inspur Group, he worked
as a senior engineer at Intel Corporation. He has
published over 10 papers and books, including IEEE
GLOBOCOM, ICC, ITC, WCNC, VTC, etc., and
has also participated in several national standards.
His current research interests include HTAP cloud-
native high-performance databases, big data analytics systems, and edge
computing. He serves as an Executive Director of the China Industry-
University-Research Institute Collaboration Association (CIUR) and is the
Director of the Jinan Key Laboratory of Distributed Databases.

