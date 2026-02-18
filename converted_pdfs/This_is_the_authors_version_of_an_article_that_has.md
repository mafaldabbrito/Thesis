# This is the author’s version of an article that has been published in IEEE Transactions on Network and Service Management. Changes were made to

This is the author’s version of an article that has been published in IEEE Transactions on Network and Service Management. Changes were made to
this version by the publisher prior to publication. The ﬁnal version of record is available at https://doi.org/10.1109/TNSM.2020.2971776.
The source code associated with this project is available at https://github.com/doriguzzi/lucid-ddos.

LUCID: A Practical, Lightweight Deep Learning
Solution for DDoS Attack Detection
R. Doriguzzi-Corinα, S. Millarβ, S. Scott-Haywardβ, J. Mart´ınez-del-Rinc´onβ, D. Siracusaα
αICT, Fondazione Bruno Kessler - Italy
βCSIT, Queen’s University Belfast - Northern Ireland

0
2
0
2

g
u
A
8
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
2
0
9
4
0
.
2
0
0
2
:
v
i
X
r
a

Abstract—Distributed Denial of Service (DDoS) attacks are one
of the most harmful threats in today’s Internet, disrupting the
availability of essential services. The challenge of DDoS detection
is the combination of attack approaches coupled with the volume
of live trafﬁc to be analysed. In this paper, we present a practical,
lightweight deep learning DDoS detection system called LUCID,
which exploits the properties of Convolutional Neural Networks
(CNNs) to classify trafﬁc ﬂows as either malicious or benign.
We make four main contributions; (1) an innovative application
of a CNN to detect DDoS trafﬁc with low processing overhead,
(2) a dataset-agnostic preprocessing mechanism to produce trafﬁc
observations for online attack detection, (3) an activation analysis
to explain LUCID’s DDoS classiﬁcation, and (4) an empirical
validation of the solution on a resource-constrained hardware
platform. Using the latest datasets, LUCID matches existing state-
of-the-art detection accuracy whilst presenting a 40x reduction
in processing time, as compared to the state-of-the-art. With
our evaluation results, we prove that the proposed approach
is suitable for effective DDoS detection in resource-constrained
operational environments.

Index Terms—Distributed Denial of Service, Deep Learning,

Convolutional Neural Networks, Edge Computing

I. INTRODUCTION

DDoS attacks are one of the most harmful threats in today’s
Internet, disrupting the availability of essential services in
production systems and everyday life. Although DDoS attacks
have been known to the network research community since the
early 1980s, our network defences against these attacks still
prove inadequate.

In late 2016, the attack on the Domain Name Server (DNS)
provider, Dyn, provided a worrying demonstration of the
potential disruption from targeted DDoS attacks [1]. This
particular attack leveraged a botnet (Mirai) of unsecured IoT
(Internet of Things) devices affecting more than 60 services.
At the time, this was the largest DDoS attack recorded, at
600 Gbps. This was exceeded in February 2018 with a major
DDoS attack towards Github [2]. At its peak, the victim saw
incoming trafﬁc at a rate of 1.3 Tbps. The attackers leveraged
in memcached, a popular database
a vulnerability present
caching tool. In this case, an ampliﬁcation attack was executed
using a spoofed source IP address (the victim IP address).
If globally implemented, BCP38 “Network Ingress Filtering”
[3] could mitigate such an attack by blocking packets with
spoofed IP addresses from progressing through the network.
However, these two examples illustrate that scale rather than
sophistication enables the DDoS to succeed.

In recent years, DDoS attacks have become more difﬁcult
to detect due to the many combinations of attack approaches.

For example, multi-vector attacks where an attacker uses a
combination of multiple protocols for the DDoS are common.
In order to combat the diversity of attack techniques, more
nuanced and more robust defence techniques are required.
Traditional signature-based intrusion detection systems cannot
react to new attacks. Existing statistical anomaly-based de-
tection systems are constrained by the requirement to deﬁne
thresholds for detection. Network Intrusion Detection Sys-
tems (NIDSs) using machine learning techniques are being
explored to address the limitations of existing solutions. In
this category, deep learning (DL) systems have been shown to
be very effective in discriminating DDoS trafﬁc from benign
trafﬁc by deriving high-level feature representations of the
trafﬁc from low-level, granular features of packets [4], [5].
However, many existing DL-based approaches described in
the scientiﬁc literature are too resource-intensive from the
training perspective, and lack the pragmatism for real-world
deployment. Speciﬁcally, current solutions are not designed for
online attack detection within the constraints of a live network
where detection algorithms must process trafﬁc ﬂows that can
be split across multiple capture time windows.

Convolutional Neural Networks (CNNs), a speciﬁc DL
technique, have grown in popularity in recent times leading
to major innovations in computer vision [6]–[8] and Natural
Language Processing [9], as well as various niche areas such
as protein binding prediction [10], [11], machine vibration
analysis [12] and medical signal processing [13]. Whilst their
use is still under-researched in cybersecurity generally, the
in
application of CNNs has advanced the state-of-the-art
certain speciﬁc scenarios such as malware detection [14]–[17],
code analysis [18], network trafﬁc analysis [4], [19]–[21] and
intrusion detection in industrial control systems [22]. These
successes, combined with the beneﬁts of CNN with respect
to reduced feature engineering and high detection accuracy,
motivate us to employ CNNs in our work.

While large CNN architectures have been proven to provide
state-of-the-art detection rates, less attention has been given to
minimise their size while maintaining competent performance
in limited resource environments. As observed with the Dyn
attack and the Mirai botnet, the opportunity for launching
DDoS attacks from unsecured IoT devices is increasing as
we deploy more IoT devices on our networks. This leads to
consideration of the placement of the defence mechanism.
Mitigation of attacks such as the Mirai and Memcached
examples include the use of high-powered appliances with the
capacity to absorb volumetric DDoS attacks. These appliances
are located locally at the enterprise or in the Cloud. With the

Copyright (c) 2020 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

drive towards edge computing to improve service provision, it
becomes relevant to consider the ability to both protect against
attacks closer to the edge and on resource-constrained devices.
Indeed, even without resource restrictions, it is valuable to
minimize resource usage for maximum system output.

Combining the requirements for advanced DDoS detection
with the capability of deployment on resource-constrained
devices, this paper makes the following contributions:

• A DL-based DDoS detection architecture suitable for
online resource-constrained environments, which lever-
ages CNNs to learn the behaviour of DDoS and benign
trafﬁc ﬂows with both low processing overhead and attack
detection time. We call our model LUCID (Lightweight,
Usable CNN in DDoS Detection).

• A dataset-agnostic preprocessing mechanism that pro-
duces trafﬁc observations consistent with those collected
in existing online systems, where the detection algorithms
must cope with segments of trafﬁc ﬂows collected over
pre-deﬁned time windows.

• A kernel activation analysis to interpret and explain to
which features LUCID attaches importance when making
a DDoS classiﬁcation.

• An empirical validation of LUCID on a resource-
constrained hardware platform to demonstrate the appli-
cability of the approach in edge computing scenarios,
where devices possess limited computing capabilities.
The remainder of this paper is structured as follows: Sec.
II reviews and discusses the related work. Sec. III details the
methodology with respect to the network trafﬁc processing
and the LUCID CNN model architecture. Sec. IV describes the
experimental setup detailing the datasets and the development
of LUCID with the hyper-parameter tuning process. In Sec.
V, LUCID is evaluated and compared with the state-of-the-art
approaches. Sec. VI introduces our kernel activation analysis
for explainability of LUCID’s classiﬁcation process. Sec. VII
presents the experiment and results for the DDoS detection at
the edge. Finally, the conclusions are provided in Sec. VIII.

II. RELATED WORK

DDoS detection and mitigation techniques have been ex-
plored by the network research community since the ﬁrst
reported DDoS attack incident in 1999 [23]. In this section, we
review and discuss anomaly-based DDoS detection techniques
categorised by statistical approaches and machine learning
approaches, with a speciﬁc focus on deep learning techniques.

A. Statistical approaches to DDoS detection

Measuring statistical properties of network trafﬁc attributes
is a common approach to DDoS detection, and generally
involves monitoring the entropy variations of speciﬁc packet
header ﬁelds. By deﬁnition, the entropy is a measure of the
diversity or the randomness in a data set. Entropy-based DDoS
detection approaches have been proposed in the scientiﬁc
literature since the early 2000s, based on the assumption that
during a volumetric DDoS attack, the randomness of trafﬁc
features is subject to sudden variations. The rationale is that
volumetric DDoS attacks are typically characterised by a huge

2

number of attackers (in the order of hundreds of thousands
[24]), often utilising compromised devices that send a high
volume of trafﬁc to one or more end hosts (the victims). As
a result, these attacks usually cause a drop in the distribution
of some of the trafﬁc attributes, such as the destination IP
address, or an increase in the distribution of other attributes,
such as the source IP address. The identiﬁcation of a DDoS
attack is usually determined by means of thresholds on these
distribution indicators.

In one of the ﬁrst published works using this approach,
Feinstein et al. [25] proposed a DDoS detection technique
based on the computation of source IP address entropy and
Chi-square distribution. The authors observed that the variation
in source IP address entropy and chi-square statistics due to
ﬂuctuations in legitimate trafﬁc was small, compared to the
deviations caused by DDoS attacks. Similarly, [26] combined
entropy and volume trafﬁc characteristics to detect volumetric
DDoS attacks, while the authors of [27] proposed an entropy-
based scoring system based on the destination IP address
entropy and dynamic combinations of IP and TCP layer
attributes to detect and mitigate DDoS attacks.

A common drawback to these entropy-based techniques is
the requirement to select an appropriate detection threshold.
Given the variation in trafﬁc type and volume across different
networks, it is a challenge to identify the appropriate detection
threshold that minimizes false positive and false negative rates
in different attack scenarios. One solution is to dynamically
adjust the thresholds to auto-adapt to the normal ﬂuctuations
of the network trafﬁc, as proposed in [28], [29].

Importantly, monitoring the distribution of trafﬁc attributes
does not provide sufﬁcient information to distinguish between
benign and malicious trafﬁc. To address this, some approaches
apply a rudimentary threshold on the packet rate [30] or
traceback techniques [31], [32].

An alternative statistical approach is adopted in [33], where
Ahmed et al. use packet attributes and trafﬁc ﬂow-level statis-
tics to distinguish between benign and DDoS trafﬁc. However,
this solution may not be suitable for online systems, since
some of the ﬂow-level statistics used for the detection e.g. total
bytes, number of packets from source to destination and from
destination to source, and ﬂow duration, cannot be computed
when the trafﬁc features are collected within observation time
windows. Approaches based on ﬂow-level statistics have also
been proposed in [34]–[39], among many others. In particular,
[36]–[39] use ﬂow-level statistics to feed CNNs and other DL
models, as discussed in Sec. II-C. To overcome the limitations
of statistical approaches to DDoS detection, machine learning
techniques have been explored.

B. Machine Learning for DDoS detection

As identiﬁed by Sommer and Paxson in [40], there has
been extensive research on the application of machine learning
to network anomaly detection. The 2016 Buczak and Guven
survey [41] cites the use of Support Vector Machine (SVM),
k-Nearest Neighbour (k-NN), Random Forest, Na¨ıve Bayes
etc. achieving success for cyber security intrusion detection.
However, due to the challenges particular to network intrusion

detection, such as high cost of errors, variability in trafﬁc
etc., adoption of these solutions in the “real-world” has been
limited. Over recent years, there has been a gradual increase
in availability of realistic network trafﬁc data sets and an
increased engagement between data scientists and network
researchers to improve model explainability such that more
practical Machine Learning (ML) solutions for network attack
detection can be developed. Some of the ﬁrst application
of machine learning techniques speciﬁc to DDoS detection
has been for trafﬁc classiﬁcation. Speciﬁcally, to distinguish
between benign and malicious trafﬁc, techniques such as extra-
trees and multi-layer perceptrons have been applied [42], [43].
In consideration of the realistic operation of DDoS attacks
from virtual machines, He et al. [44] evaluate nine ML
algorithms to identify their capability to detect the DDoS from
the source side in the cloud. The results are promising with
high accuracy (99.7%) and low false positives (< 0.07%) for
the best performing algorithm; SVM linear kernel. Although
there is no information provided regarding the detection time
or the datasets used for the evaluation, the results illustrate
the variability in accuracy and performance across the range
of ML models. This is reﬂected across the literature e.g. [45],
[46] with the algorithm performance highly dependent on the
selected features (and datasets) evaluated. This has motivated
the consideration of deep learning for DDoS detection, which
reduces the emphasis on feature engineering.

C. Deep Learning for DDoS detection

There is a small body of work investigating the application
of DL to DDoS detection. For example, in [47], the authors
address the problem of threshold setting in entropy-based
techniques by combining entropy features with DL-based
classiﬁers. The evaluation demonstrates improved performance
over the threshold-based approach with higher precision and
recall. In [48], a Recurrent Neural Network (RNN)-Intrusion
Detection System (IDS) is compared with a series of pre-
viously presented ML techniques (e.g. J48, Artiﬁcial Neural
Network (ANN), Random Forest, and SVM) applied to the
NSL-KDD [49] dataset. The RNN technique demonstrates a
higher accuracy and detection rate.

Some CNN-based works [36]–[39], as identiﬁed in Sec.
II-A, use ﬂow-level statistics (total bytes, ﬂow duration, total
number of ﬂags, etc.) as input to the proposed DL-based
architectures. In addition, [36] and [37] combine the statistical
features with packet payloads to train the proposed IDSs.

In [19], Kehe Wu et al. present an IDS based on CNN for
multi-class trafﬁc classiﬁcation. The proposed neural network
model has been validated with ﬂow-level features from the
NSL-KDD dataset encoded into 11x11 arrays. Evaluation
results show that the proposed model performs well compared
to complex models with 20 times more trainable parameters.
A similar approach is taken by the authors of [20], where
the CNN-based IDS is validated over datasets NSL-KDD and
UNSW-NB-15 [50]. In [51], the authors study the application
of CNNs to IDS by comparing a series of architectures (shal-
low, moderate, and deep, to reﬂect the number of convolution
and pooling layers) across 3 trafﬁc datasets; NSL-KDD, Kyoto

3

Honeypot [52], and MAWILab [53]. In the results presented,
the shallow CNN model with a single convolution layer and
single max. pooling layer performed best. However, there is
signiﬁcant variance in the detection accuracy results across the
datasets, which indicates instability in the model.

More speciﬁc to our DDoS problem, Ghanbari et al. propose
a feature extraction algorithm based on the discrete wavelet
transform and on the variance fractal dimension trajectory
to maximize the sensitivity of the CNN in detecting DDoS
attacks [5]. The evaluation results show that the proposed ap-
proach recognises DDoS attacks with 87.35% accuracy on the
CAIDA DDoS attack dataset [54]. Although the authors state
that their method allows real-time detection of DDoS attacks
in a range of environments, no performance measurements are
reported to support this claim.

DeepDefense [4] combines CNNs and RNNs to translate
original trafﬁc traces into arrays that contain packet features
collected within sliding time windows. The results presented
demonstrate high accuracy in DDoS attack detection within
the selected ISCX2012 dataset [55]. However, it is not clear if
these results were obtained on unseen test data, or are results
from the training phase. Furthermore, the number of trainable
parameters in the model is extremely large indicating a long
and resource-intensive training phase. This would signiﬁcantly
challenge implementation in an online system with constrained
resources, as will be discussed in Sec. V and VII.

Although deep learning offers the potential for an effective
DDoS detection method, as described, existing approaches
are limited by their suitability for online implementation
in resource-constrained environments. In Sec. V, we com-
pare our proposed solution, LUCID, with the state-of-the-art,
speciﬁcally [4], [35], [36], [38], [47] and demonstrate the
contributions of LUCID.

III. METHODOLOGY

In this paper we present LUCID, a CNN-based solution
for DDoS detection that can be deployed in online resource-
constrained environments. Our CNN encapsulates the learning
of malicious activity from trafﬁc to enable the identiﬁcation
of DDoS patterns regardless of their temporal positioning.
This is a fundamental beneﬁt of CNNs; to produce the same
output regardless of where a pattern appears in the input.
This encapsulation and learning of features whilst training
the model removes the need for excessive feature engineering,
ranking and selection. To support an online attack detection
system, we use a novel preprocessing method for the network
trafﬁc that generates a spatial data representation used as input
to the CNN. In this section, we introduce the network trafﬁc
preprocessing method, the CNN model architecture, and the
learning procedure.

A. Network Trafﬁc preprocessing

Network trafﬁc is comprised of data ﬂows between end-
points. Due to the shared nature of the communication link,
packets from different data ﬂows are multiplexed resulting in
packets from the same ﬂow being separated for transmission.
This means that the processing for live presentation of trafﬁc

TABLE I
GLOSSARY OF SYMBOLS.

α
f
h
id
k
m

Learning rate
Number of features per packet
Height of convolutional ﬁlters
5-tuple ﬂow identiﬁer
Number of convolutional ﬁlters
Max pooling size

n
s
t
τ
E
L

Number of packets per sample
Batch size
Time window duration
Time window start time
Array of labelled samples
Set of labels

to a NIDS is quite different to the processing of a static dataset
comprising complete ﬂows. For the same reason, the ability
to generate ﬂow-level statistics, as relied upon by many of the
existing works described in Sec. II, is not feasible in an online
system.

In order to develop our online NIDS, we created a tool
that converts the trafﬁc ﬂows extracted from network trafﬁc
traces of a dataset into array-like data structures and splits
them into sub-ﬂows based on time windows. Shaping the input
as packet ﬂows in this manner creates a spatial data repre-
sentation, which allows the CNN to learn the characteristics
of DDoS attacks and benign trafﬁc through the convolutional
ﬁlters sliding over such input to identify salient patterns. This
form of input is compatible with trafﬁc captured in online
deployments. The process is illustrated in Algorithm 1 and
described next. The symbols are deﬁned in Table I.

Algorithm 1 Network trafﬁc preprocessing algorithm
Input: Network trafﬁc trace (N T T ), ﬂow-level labels (L),

time window (t), max packets/sample (n)

Output: List of labelled samples (E)

1: procedure PREPROCESSING(N T T , L, t, n)
2:
3:
4:
5:

E ← ∅
τ ← −1
for all pkt ∈ N T T do
id ← pkt.tuple
if τ == −1 or pkt.time > τ + t then

(cid:46) Initialise the set of samples
(cid:46) Initialise the time window start-time
(cid:46) Loop over the packets
(cid:46) 5-tuple ﬂow identiﬁer

τ ← pkt.time

(cid:46) Time window start time

end if
if (cid:12)

(cid:12)E[τ, id](cid:12)
E[τ, id].pkts.append(pkt.f eatures)

(cid:12) < n then

(cid:46) Max n pkts/sample

6:
7:
8:
9:
10:

end for
E ← normalization padding(E)
for all e ∈ E do

(cid:46) Labelling
e.label ← L[e.id] (cid:46) Apply the label to the sample

end if

11:
12:
13:
14:
15:
16:
17:
18: end procedure

end for
return E

Feature extraction. Given a trafﬁc trace ﬁle from the
dataset and a pre-deﬁned time window of length t seconds,
the algorithm collects all the packets from the ﬁle with capture
time between t0, the capture time of the ﬁrst packet, and time
t0 + t. From each packet, the algorithm extracts 11 attributes
(see Table II). We intuitively exclude those attributes that
would be detrimental to the generalization of the model, such
as IP addresses and TCP/UDP ports (speciﬁc to the end-hosts
and user applications), link layer encapsulation type (linked to

4

Fig. 1. Graphical representation of E.

the network interfaces) and application-layer attributes (e.g.,
IRC or HTTP protocol attributes).

Data processing algorithm. This procedure, described in
Algorithm 1 at
lines 4-12, simulates the trafﬁc capturing
process of online IDSs, where the trafﬁc is collected for a
certain amount of time t before being sent to the anomaly
detection algorithms. Hence, such algorithms must base their
decisions on portions of trafﬁc ﬂows, without the knowledge
of their whole life. To simulate this process, the attributes of
the packets belonging to the same bi-directional trafﬁc ﬂow
are grouped in chronological order to form an example of
shape [n, f ] (as shown in Table II), where f is the number
of features (11) and n is the maximum number of packets the
parsing process collects for each ﬂow within the time window.
t and n are hyper-parameters for our CNN. Flows longer than
n are truncated, while shorter ﬂows are zero-padded at the end
during the next stage after normalization. The same operations
are repeated for the packets within time window [t0 +t, t0 +2t]
and so on, until the end of the ﬁle.

Logically, we hypothesize that short time windows enable
the online systems to detect DDoS attacks within a very short
time frame. Conversely, higher values of t and n offer more
information on ﬂows to the detection algorithms, which we
expect to result in higher detection accuracy. The sensitivity
of our CNN to the values of t and n is evaluated in Sec. IV.
The output of this process can be seen as a bi-dimensional
array of samples (E[τ, id] in Algorithm 1). A row of the
array represents the samples whose packets have been captured
in the same time window, whilst a column represents the
samples whose packets belong to the same bi-directional ﬂow.
A graphical representation of array E is provided in Fig. 1.

Normalization and padding. Each attribute value is nor-
malized to a [0, 1] scale and the samples are zero-padded so
that each sample is of ﬁxed length n, since having samples of
ﬁxed length is a requirement for a CNN to be able to learn
over a full sample set. In Fig. 1, each non-empty element of
the array E is a compact graphical representation of a sample.

5

TABLE II
A TCP FLOW SAMPLE BEFORE NORMALIZATION.

Pkt #

0
1
...
j
j + 1
...
n

Time
(sec)1
0
0.092
...
0.513
0
...
0

Packet
Len
151
135
...
66
0
...
0

Highest
Layer2
99602525
99602525
...
78354535
0
...
0

IP
Flags
0x4000
0x4000
...
0x4000
0
...
0

Protocols3

0011010001000b
0011010001000b
...
0010010001000b
0000000000000b
...
0000000000000b

TCP
Len
85
69
...
0
0
...
0

TCP
Ack
336
453
...
405
0
...
0

TCP
Flags
0x018
0x018
...
0x010
0
...
0

TCP
Window Size
1444
510
...
1444
0
...
0

UDP
Len
0
0
...
0
0
...
0

ICMP
Type
0
0
...
0
0
...
0









s
t
e
k
c
a
P

g
n
i
d
d
a
P

1 Relative time from the ﬁrst packet of the ﬂow.
2 Numerical representation of the highest layer recognised in the packet.
3 Binary representation of the list of protocols recognised in the packet using the well-known Bag-of-Words (BoW) model. It includes protocols from Layer 2 (arp)

to common clear text application layer protocols such as http, telnet, ftp and dns.

In each E element, coloured rows are the packets in the form
of 11 normalized attributes (i.e., the upper part of Table II),
while the white rows represent the zero-padding (i.e., the lower
part of Table II). Please note that, empty elements in Fig. 1
are for visualization only and are not included in the dataset.
An empty E[τ, id] means that no packets of ﬂow id have been
captured in time window [τ, τ + t] (e.g. E[t0, F 4]).

Labelling. Each example E[τ, id] is labelled by matching
its ﬂow identiﬁer id with the labels provided with the original
dataset (lines 14-16 in Algorithm 1). This also means that the
value of the label is constant along each column of array E,
as represented in Fig. 1.

B. LUCID Model Architecture

We take the output from Algorithm 1 as input to our CNN
model for the purposes of online attack detection. LUCID
classiﬁes trafﬁc ﬂows into one of two classes, either malicious
(DDoS) or benign. Our objective is to minimise the com-
plexity and performance time of this CNN model for feasible
deployment on resource-constrained devices. To achieve this,
the proposed approach is a lightweight, supervised detection
system that incorporates a CNN, similar to that of [9] from the
ﬁeld of Natural Language Processing. CNNs have shared and
reused parameters with regard to the weights of the kernels,
whereas in a traditional neural network every weight is used
only once. This reduces the storage and memory requirements
of our model. The complete architecture is depicted in Fig. 2
and described in the next sections, with the hyper-parameter
tuning and ablation studies being discussed in Sec. IV.

Input layer. Recall that each trafﬁc ﬂow has been reshaped
into a 2-D matrix of packet features as per Sec. III-A, creating
a novel spatial representation that enables the CNN to learn
the correlation between packets of the same ﬂow. Thus, this
ﬁrst layer takes as input a trafﬁc ﬂow represented by a matrix
F of size n × f . F contains n individual packet vectors, such
that F = {pkt1, ... , pktn} where pktn is the nth packet in a
ﬂow, and each packet vector has length f = 11 features.

CNN layer. As per Fig. 2, each input matrix F is operated
on by a single convolutional layer with k ﬁlters of size h × f ,
with h being the length of each ﬁlter, and again f = 11. Each
ﬁlter, also known as a kernel or sliding window, convolves

over F with a step of 1 to extract and learn local features that
contain useful information for detection of DDoS and benign
ﬂows. Each of the k ﬁlters generates an activation map a of
size (n − h + 1), such that ak = ReLU (Conv(F )Wk, bk),
where Wk and bk are the weight and bias parameters of
the kth ﬁlter that are learned during the training stage. To
introduce non-linearity among the learned ﬁlters, we use the
rectiﬁed linear activation function ReLU (x) = max{0, x},
as per convention for CNNs. All activation maps are stacked,
creating an activation matrix A of size (n − h + 1) × k, such
that A = [a1|...|ak].

There are two main beneﬁts of including a CNN in our
architecture. Firstly,
to beneﬁt from
it allows the model
efﬁciency gains compared to standard neural networks, since
the weights in each ﬁlter are reused across the whole input.
Sharing weights, instead of the full end-to-end connectivity
with a standard neural net, makes the model more lightweight
and reduces its memory footprint as the number of learnable
parameters is greatly reduced. Secondly, during the training
phase, the CNN automatically learns the weights and biases
of each ﬁlter such that the learning of salient characteristics
and features is encapsulated inside the resulting model during
training. This reduces the time-consuming feature engineering
and ranking involved in statistical and traditional machine
learning methods, which relies on expert human knowledge.
As a result, this model is more adaptable to new subtleties of
DDoS attack, since the training stage can be simply repeated
anytime with fresh training data without having to craft and
rank new features.

Max pooling layer. For max pooling, we down-sample
along the ﬁrst dimension of A, which represents the temporal
nature of the input. A pool size of m produces an output
matrix mo of size ((n − h + 1)/m) × k, which contains the
largest m activations of each learned ﬁlter, such that mo =
[max(a1)|...|max(ak)]. In this way,
the model disregards
the less useful information that produced smaller activations,
instead paying attention to the larger activations. This also
means that we dispose of the positional information of the
activation, i.e. where it occurred in the original ﬂow, giving a
more compressed feature encoding, and, in turn, reducing the
complexity of the network. mo is then ﬂattened to produce

6

not require signiﬁcant expert input to craft bespoke features
and statistically assess their importance during preprocessing,
unlike many existing methods, as outlined in Sec. II.

IV. EXPERIMENTAL SETUP

A. Datasets

Our CNN model is validated with recent datasets ISCX2012
[55], CIC2017 [56] and CSECIC2018 [57] provided by the
Canadian Institute for Cybersecurity of the University of
New Brunswick (UNB), Canada. They consist of several
days of network activity, normal and malicious,
including
DDoS attacks. The three datasets are publicly available in
the form of trafﬁc traces in pcap format including full packet
payloads, plus supplementary text ﬁles containing the labels
and statistical details for each trafﬁc ﬂow.

The UNB researchers have generated these datasets by
using proﬁles to accurately represent the abstract properties
of human and attack behaviours. One proﬁle characterises the
normal network activities and provides distribution models for
applications and protocols (HTTP, SMTP, SSH, IMAP, POP3,
and FTP) produced with the analysis of real trafﬁc traces.
Other proﬁles describe a variety of attack scenarios based on
recent security reports. They are used to mimic the behaviour
of the malicious attackers by means of custom botnets and
well-known DDoS attacking tools such as High Orbit Ion
Cannon (HOIC) [58] and its predecessor, the Low Orbit Ion
Cannon (LOIC) [59]. HOIC and LOIC have been widely
used by Anonymous and other hacker groups in some highly-
publicized attacks against PayPal, Mastercard, Visa, Amazon,
Megaupload, among others [60].

Table III shows the parts of the three datasets used in this
work. In the table, the column Trafﬁc trace speciﬁes the name
of the trace, according to [55], [56] and [57]. Speciﬁcally,
the ISCX2012-Tue15 trace contains a DDoS attack based
on an IRC botnet. The CIC2017-Fri7PM trace contains a
HTTP DDoS generated with LOIC, while the CSECIC2018-
Wed21 trace contains a HTTP DDoS generated with HOIC.
With respect to the original ﬁle, the trace CIC2017-Fri7PM
is reduced to timeslot 3.30PM-5.00PM to exclude malicious
packets related to other cyber attacks (port scans and back-
doors).

TABLE III
THE DATASETS FROM UNB [61].

Dataset

Trafﬁc trace

#Flows

#Benign

#DDoS

ISCX2012

Tue15

571698

534320

37378

CIC2017

Fri7PM

225745

97718

128027

CSECIC2018 Wed21

1048575

360832

687743

In an initial design, the model was trained and validated
on the ISCX2012 dataset producing high accuracy results.
However, testing the model on the CIC2017 dataset conﬁrmed
the generally held observation that a model trained on one
dataset will not necessarily perform well on a completely
new dataset. In particular, we obtained a false negative rate
of about 17%. This can be attributed to the different attacks

Fig. 2. LUCID architecture.

the ﬁnal one-dimensional feature vector v to be input to the
classiﬁcation layer.

Classiﬁcation layer. v is input to a fully-connected layer
of the same size, and the output layer has a sole node. This
output x is passed to the sigmoid activation function such that
σ(x) = 1/(1 + e−x). This constrains the activation to a value
of between 0 and 1, hence returning the probability p ∈ [0, 1]
of a given ﬂow being a malicious DDoS attack. The ﬂow is
classiﬁed as DDoS when p > 0.5, and benign otherwise.

C. The Learning Procedure

When training LUCID, the objective is to minimise its cost
function through iteratively updating all the weights and biases
contained within the model. These weights and biases are also
known as trainable, or learnable, parameters. The cost function
calculates the cost, also called the error or the loss, between the
model’s prediction, and the ground truth of the input. Hence by
minimising this cost function, we reduce the prediction error.
At each iteration in training, the input data is fed forward
through the network, the error calculated, and then this error
is back-propagated through the network. This continues until
convergence is reached, when further updates don’t reduce
the error any further, or the training process reaches the set
maximum number of epochs. With two classes in our problem
the binary cross-entropy cost function is used. Formally this
cost function c that calculates the error over a batch of s
samples can be written as:

c = −

1
s

s
(cid:88)

j=1

(yj log pj + (1 − yj) log(1 − pj))

(1)

where yj is the ground truth target label for each ﬂow j in
the batch of s samples, and pj is the predicted probability ﬂow
j is malicious DDoS. This is supervised learning because each
ﬂow in our datasets is labelled with the ground truth, either
DDoS or benign. To reduce bias in our learning procedure, we
ensure that these datasets are balanced with equal numbers of
malicious and benign ﬂows, which gives a greater degree of
conﬁdence that the model is learning the correct feature repre-
sentations from the patterns in the trafﬁc ﬂows. As previously
highlighted, the learning is encapsulated inside the model by
all the weights and biases, meaning that our approach does

represented in the two datasets, as previously described. What
we attempt in this work is to develop a model that when trained
and validated across a mixed dataset can reproduce the high
performance results on completely unseen test data. To achieve
this, a combined training dataset is generated as described in
Sec. IV-B.

B. Data preparation

We extract

randomly select 37378 benign ﬂows

the 37378 DDoS ﬂows from ISCX2012,
from the
plus
same year
this process with
97718/97718 benign/DDoS ﬂows for CIC2017 and again with
360832/360832 benign/DDoS ﬂows for CSECIC2018.

to balance. We

repeat

After the pre-preprocessing stage, where ﬂows are translated
into array-like data structures (Sec. III-A), each of the three
datasets is split into training (90%) and test (10%) sets, with
10% of the training set used for validation. Please note that, the
split operation is performed on a per-ﬂow basis to ensure that
samples obtained from the same trafﬁc ﬂow end up in the same
split, hence avoiding the “contamination” of the validation and
test splits with data used for the training. We ﬁnally combine
the training splits from each year by balancing them with
equal proportions from each year to produce a single training
set. We do the same with the validation and test splits, to
obtain a ﬁnal dataset referred to as UNB201X in the rest of
the paper. UNB201X training and validation sets are only used
for training the model and tuning the hyper-parameters (Sec.
IV-D), while the test set is used for the evaluation presented
in Sec. V and VII, either as a whole combined test set, or as
individual per-year test sets for state-of-the-art comparison.

A summary of the ﬁnal UNB201X splits is presented in
Table IV, which reports the number of samples as a function of
time window duration t. As illustrated in Table IV, low values
of this hyper-parameter yield larger numbers of samples.
Intuitively, using short time windows leads to splitting trafﬁc
ﬂows into many small fragments (ultimately converted into
samples), while long time windows produce the opposite
result. In contrast, the value of n has a negligible impact on
the ﬁnal number of samples in the dataset.

TABLE IV
UNB201X DATASET SPLITS.

Time
Window

Total
Samples

Training Validation

Test

t=1s
t=2s
t=3s
t=4s
t=5s
t=10s
t=20s
t=50s
t=100s

480519
353058
310590
289437
276024
265902
235593
227214
224154

389190
285963
251574
234438
223569
215379
190827
184041
181551

43272
31782
27957
26055
24852
23931
21204
20451
20187

48057
35313
31059
28944
27603
26592
23562
22722
22416

C. Evaluation methodology

As per convention in the literature, we report the metrics
Accuracy (ACC), False Positive Rate (FPR), Precision (or

7

Positive Predictive Value (PPV)), Recall (or True Positive Rate
(TPR)) and F1 Score (F1), with a focus on the latter. Accuracy
is the percentage of correctly classiﬁed samples (both benign
and DDoS). FPR represents the percentage of samples that
are falsely classiﬁed as DDoS. PPV is the ratio between the
correctly detected DDoS samples and all the detected DDoS
samples (true and false). TPR represents the percentage of
DDoS samples that are correctly classiﬁed as such. The F1
Score is an overall measure of a model’s performance; that is
the harmonic mean of the PPV and TPR. These metrics are
formally deﬁned as follows:

ACC =

T P +T N

T P +T N +F P +F N F P R = F P

F P +T N

P P V = T P

T P +F P T P R = T P

T P +F N F 1 = 2 · P P V ·T P R

P P V +T P R

where TP=True Positives, TN=True Negatives, FP=False Pos-
itives, FN=False Negatives.

The output of the training process is a combination of
trainable and hyper parameters that maximizes the F1 Score
on the validation set or, in other words, that minimizes the
total number of False Positives and False Negatives.

Model training and validation have been performed on a
server-class computer equipped with two 16-core Intel Xeon
Silver 4110 @2.1 GHz CPUs and 64 GB of RAM. The models
have been implemented in Python v3.6 using the Keras API
v2.2.4 [62] on top of Tensorﬂow 1.13.1 [63].

D. Hyper-parameter tuning

Tuning the hyper-parameters is an important step to opti-
mise the model’s accuracy, as their values inﬂuence the model
complexity and the learning process. Prior to our experiments,
we empirically chose the hyper-parameter values based on the
results of preliminary tuning and on the motivations described
per parameter. We then adopted a grid search strategy to
explore the set of hyper-parameters using F1 score as the
performance metric. At each point in the grid, the training
continues indeﬁnitely and stops when the loss does not de-
crease for a consecutive 25 times. Then, the search process
saves the F1 score and moves to the next point.

As per Sec. IV-B, UNB201X is split into training, validation
and testing sets. For hyper-parameter tuning, we use only the
validation set. It is important to highlight that we do not tune to
the test set, as that may artiﬁcially improve performance. The
test set is kept completely unseen, solely for use in generating
our experimental results, which are reported in Sec. V.

Maximum number of packets/sample. n is important
for the characterization of the trafﬁc and for capturing the
temporal patterns of trafﬁc ﬂows. The value of n indicates
the maximum number of packets of a ﬂow recorded in
chronological order in a sample.

The resulting set of packets describes a portion of the life
of the ﬂow in a given time window, including the (relative)
time information of packets. Repetition-based DDoS attacks
use a small set of messages at approximately constant rates,
therefore a small value of n is sufﬁcient to spot the temporal
patterns among the packet features, hence requiring a limited
number of trainable parameters. On the other hand, more

e
r
o
c
S
1
F

1
0.99
0.98

0.95

0.90

0.88

α = 0.01, s = 2048, k = 64, h = 3, m = n − h + 1

t=1

t=10

t=100

1

2

3

4 5

10

20

50

100

Value of hyper-parameter n (packets/example) in logarithmic scale

Fig. 3. Sensitivity of our model to hyper-parameter n.

complex attacks, such as the ones performed with the HOIC
tool, which uses multiple HTTP headers to make the requests
appear legitimate, might require a larger number of packets to
achieve the desired degree of accuracy. Given the variety of
DDoS tools used to simulate the attack trafﬁc in the dataset
(IRC-based bot, LOIC and HOIC), we experimented with n
ranging between 1 and 100, and we compared the performance
in terms of F1 score. The results are provided in Fig. 3 for
different durations of time window t, but at ﬁxed values of
the other hyper-parameters for the sake of visualisation.

The F1 score steadily increases with the value of n when
n < 5, and then stabilises when n ≥ 5. However, an increase
in F1 score is still observed up to n = 100. Although, a
low value of n can be used to speed up the detection time
(less convolutions) and to reduce the requirements in terms of
storage and RAM (smaller sample size), which links to our
objective of a lightweight implementation, we wish to balance
high accuracy with low resource consumption. This will be
demonstrated in Sec. VII.

Time Window. The time window t is used to simulate
the capturing process of online systems (see Sec. III-A). We
evaluated the F1 score for time windows ranging between 1
and 100 seconds (as in the related work e.g. [4]) at different
values of n. The results are shown in Fig. 4.

e
r
o
c
S
1
F

1
0.99
0.98

0.95

0.90

0.88

α = 0.01, s = 2048, k = 64, h = 3, m = n − h + 1

n=1

n=10

n=2

n=100

1

2

3

4 5

10

20

50

100

Value of hyper-parameter t (seconds) in logarithmic scale

8

correlate the attributes of different packets within the same
sample, the F1 score is more inﬂuenced by the number of
samples in the training set (the more samples, the better).

Height of convolutional ﬁlters. h determines the height of
the ﬁlters (the width is ﬁxed to 11, the number of features),
i.e. the number of packets to involve in each matrix operation.
Testing with h = 1, 2, 3, 4, 5, we observed a small, but
noticeable, difference in the F1 score between h = 1 (0.9934)
and h = 3 (0.9950), with no major improvement beyond
h = 3.

Number of convolutional ﬁlters. As per common practice,
we experimented by increasing the number of convolutional
ﬁlters k by powers of 2, from k = 1 to k = 64. We observed
a steady increase in the F1 score with the value of k, which
is a direct consequence of the increasing number of trainable
parameters in the model.

Resulting hyper-parameter set. After conducting a com-
prehensive grid search on 2835 combinations of hyper-
parameters, we have selected the CNN model conﬁguration
that maximises the F1 score on the UNB201X validation set
(Table V). That is:

n = 100, t = 100, k = 64, h = 3, m = 98

The resulting model, trained with batch size s = 2048 and
using the Adam optimizer [64] with learning rate α = 0.01,
consists of 2241 trainable parameters, 2176 for the convolu-
tional layer (h · f units for each ﬁlter plus bias, multiplied by
the number of ﬁlters K) and 65 for the fully connected layer
(64 units plus bias).

As previously noted, other conﬁgurations may present lower
resource requirements at the cost of a minimal decrease in F1
score. For example, using k = 32 would reduce the number of
convolutions by half, while n = 10, 20, 50 would also require
fewer convolutions and a smaller memory footprint. However,
setting n = 100 not only maximises the F1 score, but also
enables a fair comparison with state-of-the-art approaches such
as DeepDefense [4] (Sec. V), where the authors trained their
neural networks using n = 100 (in [4], the hyper-parameter is
denoted as T ). Furthermore, the chosen conﬁguration enables
a worst-case analysis for resource-constrained scenarios such
as that presented in Sec. VII.

These hyper-parameters are kept constant throughout our

experiments presented in Sec. V and VII.

TABLE V
SCORES OBTAINED ON THE UNB201X VALIDATION SET.

Validation set

ACC

FPR

PPV

TPR

F1

UNB201X

0.9950

0.0083

0.9917

0.9983

0.9950

Fig. 4. Sensitivity of our model to hyper-parameter t.

V. RESULTS

Although the number of samples in the training set de-
creases when t increases (see Table IV), the CNN is relatively
insensitive to this hyper-parameter for n > 1. With n = 1,
the trafﬁc ﬂows are represented by samples of shape [1, f ],
i.e. only one packet/sample, irrespective of the duration of the
time window. In such a corner case, since the CNN cannot

In this section, we present a detailed evaluation of the
proposed approach with the datasets presented in Sec. IV-A.
Evaluation metrics of Accuracy (ACC), False Positive Rate
(FPR), Precision (PPV), Recall (TPR) and F1 Score (F1) have
been used for performance measurement and for comparison
with state-of-the-art models.

9

A. Detection accuracy

In order to validate our approach and the results obtained on
the validation dataset, we measure the performance of LUCID
in classifying unseen trafﬁc ﬂows as benign or malicious
(DDoS). Table VI summarizes the results obtained on the
various test sets produced through the procedure described
in Sec. IV-B. As illustrated, the very high performance is
maintained across the range of test datasets indicating the
robustness of the LUCID design. These results are further
discussed in Sec. V-B, where we compare our solution with
state-of-the-art works reported in the scientiﬁc literature.

the classiﬁcation of DDoS trafﬁc. Therefore, we have imple-
mented 3LSTM for comparison purposes. The architecture
of this model includes 6 LSTM layers of 64 neurons each,
2 fully connected layers of 128 neurons each, and 4 batch
normalization layers. To directly compare the DL models,
we have trained 3LSTM on the UNB201X training set with
n = 100 and t = 100 as done with LUCID. We have compared
our implementation of 3LSTM with LUCID on each of the four
test sets, and present the F1 score results in Table VII.

TABLE VII
LUCID-DEEPDEFENSE COMPARISON (F1 SCORE).

TABLE VI
LUCID DETECTION PERFORMANCE ON THE TEST SETS.

Model

Trainable
Parameters

ISCX
2012

CIC
2017

CSECIC
2018

UNB
201X

Test set

ACC

FPR

PPV

TPR

F1

LUCID

2241

0.9889

0.9966

0.9987

0.9946

ISCX2012

0.9888

0.0179

0.9827

0.9952

0.9889

3LSTM

1004889

0.9880

0.9968

0.9987

0.9943

CIC2017

0.9967

0.0059

0.9939

0.9994

0.9966

CSECIC2018

0.9987

0.0016

0.9984

0.9989

0.9987

UNB201X

0.9946

0.0087

0.9914

0.9979

0.9946

The results show that thanks to the properties of its CNN,
LUCID learns to distinguish between patterns of malicious
DDoS behaviour and benign ﬂows. Given the properties of
convolutional methods, these patterns are recognised regard-
less of the position they occupy in a ﬂow, demonstrating that
our spatial representation of a ﬂow is robust. Irrespective of
whether the DDoS event appears at the start or the end of
the input, LUCID will produce the same representation in
its output. Although the temporal dynamics in DDoS attacks
might suggest that alternative DL architectures may seem more
suitable (e.g. Long Short-Term Memory (LSTM)), our novel
preprocessing method combined with the CNN removes the
requirement for the model to maintain temporal context of
each whole ﬂow as the data is pushed through the network.
In comparison, LSTMs are known to be very difﬁcult to train,
and their performance is inherently slower for long sequences
compared to CNNs.

B. State-Of-The-Art Comparison

For a fair comparison between LUCID and the state-of-the-
art, we focus our analysis on solutions that have validated the
UNB datasets for DDoS attack detection.

We have paid particular attention to DeepDefense [4] as,
similar to our approach, the model is trained with packet
attributes rather than ﬂow-level statistics used in other works.
DeepDefense translates the pcap ﬁles of ISCX2012 into arrays
that contain packet attributes collected within sliding time
windows. The label assigned to a sample is the label of the last
packet in the time window, according to the labels provided
with the original dataset. The proposed data preprocessing
technique is similar to LUCID’s. However, in LUCID, a sample
corresponds to a single trafﬁc ﬂow, whereas in DeepDefense
a sample represents the trafﬁc collected in a time window.

The results presented in Table VII show that LUCID and
3LSTM are comparable in terms of F1 score across the
range of test datasets. However, in terms of computation time,
LUCID outperforms 3LSTM in detection time. Speciﬁcally,
as measured on the Intel Xeon server in these experiments,
LUCID can classify more than 55000 samples/sec on average,
while 3LSTM barely reaches 1300 samples/sec on average
(i.e., more than 40 times slower). Indeed, LUCID’s limited
number of hidden units and trainable parameters contribute to
a much lower computational complexity compared to 3LSTM.
As previously noted, there are a number of solutions in the
literature that present performance results for the ISCX2012
and CIC2017 datasets. Notably, these works do not all specify
whether the results presented are based on a validation dataset
or a test dataset. For LUCID, we reiterate that the results
presented in this section are based on a test set of completely
unseen data.

TABLE VIII
PERFORMANCE COMPARISON WITH STATE-OF-THE-ART APPROACHES
USING THE ISCX2012 DATASET FOR DDOS DETECTION.

Model

LUCID

DeepDefense
3LSTM [4]

ACC

FPR

PPV

TPR

F1

0.9888

0.0179

0.9827

0.9952

0.9889

0.9841

N/A

0.9834

0.9847

0.9840

TR-IDS [36]

0.9809

0.0040

E3ML [47]

N/A

N/A

N/A

N/A

0.9593

0.9474

N/A

N/A

In Table VIII, we compare the performance of LUCID
against state-of-the-art works validated on ISCX2012. Table
VIII also includes the performance of 3LSTM as reported in
the DeepDefense paper [4]. With respect to our version of
3LSTM, the scores are slightly lower, which we propose is
due to the different pcap preprocessing mechanisms used in
the two implementations. This indicates a performance beneﬁt
when using the LUCID preprocessing mechanism.

Of the four DL models presented in the DeepDefense
paper, the one called 3LSTM produces the highest scores in

TR-IDS [36] is an IDS which adopts a text-CNN [9] to
extract features from the payload of the network trafﬁc. These

features, along with a combination of 25 packet and ﬂow-
level attributes, are used for trafﬁc classiﬁcation by means of
a Random Forest algorithm. Accuracy and TPR of TR-IDS are
above 0.99 for all the attack proﬁles available in ISCX2012
except the DDoS attack, for which the performance results are
noticeably lower than LUCID.

E3ML [47] uses 20 entropy-based trafﬁc features and three
ML classiﬁers (a RNN, a Multilayer Perceptron and an Al-
ternating Decision Tree) to classify the trafﬁc as normal or
DDoS. Despite the complex architecture, the TPR measured
on ISCX2012 shows that E3ML is inclined to false negatives.
For the CIC2017 dataset, we present the performance com-

parison with state-of-the-art solutions in Table IX.

TABLE IX
PERFORMANCE COMPARISON WITH STATE-OF-THE-ART APPROACHES
USING THE CIC2017 DATASET FOR DDOS DETECTION.

Model

LUCID

ACC

FPR

PPV

TPR

F1

0.9967

0.0059

0.9939

0.9994

0.9966

DeepGFL [35]

N/A

MLP [38]

0.8634

1D-CNN [38]

0.9514

LSTM [38]

0.9624

N/A

N/A

N/A

N/A

0.7567

0.3024

0.4321

0.8847

0.8625

0.8735

0.9814

0.9017

0.9399

0.9844

0.8989

0.8959

1D-CNN +
LSTM [38]

0.9716

N/A

0.9741

0.9910

0.9825

DeepGFL [35] is a framework designed to extract high-
order trafﬁc features from low-order features forming a hier-
archical graph representation. To validate the proposed frame-
work, the authors used the graph representation of the features
to train two trafﬁc classiﬁers, namely Decision Tree and
Random Forest, and tested them on CIC2017. Although the
precision scores on the several attack types are reasonably
good (between 0.88 and 1 on any type of trafﬁc proﬁle except
DDoS),
the
the results presented in the paper reveal
proposed approach is prone to false negatives, leading to very
low F1 scores.

that

The authors of [38] propose four different DL models for
DDoS attack detection in Internet of Things (IoT) networks.
The models are built with combinations of LSTM, CNN and
fully connected layers. The input
the models
consists of 82 units, one for each ﬂow-level feature available
in CIC2017, while the output layer returns the probability of
a given ﬂow being part of a DDoS attack. The model 1D-
CNN+LSTM produces good classiﬁcation scores, while the
others seem to suffer from high false negatives rates.

layer of all

To the best of our knowledge, no DDoS attack detection
solutions validated on the CSECIC2018 dataset are available
yet in the scientiﬁc literature.

C. Discussion

From the results presented and analysed in the previous
sections, we can conclude that using packet-level attributes
of network trafﬁc is more effective, and results in higher clas-
siﬁcation accuracy, than using ﬂow-level features or statistic

10

information such as the entropy measure. This is not only
proved by the evaluation results obtained with LUCID and
our implementation of DeepDefense (both based on packet-
level attributes), but also by the high classiﬁcation accuracy
of TR-IDS, which combines ﬂow-level features with packet
attributes, including part of the payload.

In contrast, E3ML, DeepGFL and most of the solutions
proposed in [38], which all rely on ﬂow-level features, seem
to be more prone to false negatives, and hence to classify
DDoS attacks as normal activity. The only exception is the
model 1D-CNN+LSTM of [38], which produces a high TPR
by combining CNN and RNN layers.

Furthermore, we highlight that LUCID has not been tuned
to the individual datasets but rather to the validation portion of
a combined dataset, and still outperforms the state-of-the-art
on totally unseen test data.

VI. ANALYSIS

We now present interpretation and explanation of the inter-
nal operations of LUCID by way of proving that the model
is learning the correct domain information. We do this by
analysing the features used in the dataset and their activations
in the model. To the best of our knowledge, this is the ﬁrst
application of a speciﬁc activation analysis to a CNN-based
DDoS detection method.

A. Kernel activations

This approach is inspired by a similar study [65] to interpret
CNNs in the rather different domain of natural
language
processing. However, the kernel activation analysis technique
is transferable to our work. As each kernel has the same width
as the input matrix, it is possible to remove the classiﬁer,
push the DDoS ﬂows through the convolutional layer and
capture the resulting activations per kernel. For each ﬂow,
we calculate the total activations per feature, which in the
spatial input representation means per column, resulting in
11 values that map to the 11 features. This is then repeated
for all kernels, across all DDoS ﬂows, with the ﬁnal output
being the total column-wise activation of each feature. The
intuition is that
the higher a feature’s activation when a
positive sample i.e. a DDoS ﬂow is seen, the more importance
the CNN attaches to that particular feature. Conversely, the
lower the activation, the lower the importance of the feature,
and since our model uses the conventional rectiﬁed linear
activation function, ReLU (x) = max{0, x}, this means that
any negative activations become zero and hence have no
impact on the Sigmoid classiﬁer for detecting a DDoS attack.
Summing these activations over all kernels is possible
since they are of the same size and operate over the same
spatial representations. We analyse DDoS ﬂows from the same
UNB201X test set used in Sec. V-A.

Table X presents the ranking of the 11 features based on
the post-ReLU average column-wise feature activation sums,
and highlights two features that activate our CNN the most,
across all of its kernels.

Highest Layer. We assert that the CNN may be learning
from the highest layer at which each DDoS ﬂow operates.

TABLE X
RANKING OF THE TOTAL COLUMN-WISE FEATURE KERNEL ACTIVATIONS
FOR THE UNB201X DATASET

Feature

Highest Layer
IP Flags
TCP Flags
TCP Len
Protocols
Pkt Len

Total Kernel
Activation

Feature

Total Kernel
Activation

0.69540
0.30337
0.19693
0.16874
0.14897
0.14392

Time
TCP Win Size
TCP Ack
UDP Len
ICMP Type

0.11108
0.09596
0.00061
0.00000
0.00000

Recall that highest layer links to the type of DDoS attack e.g.
network, transport, or application layer attack. We propose that
this information could be used to extend LUCID to predict
the speciﬁc type of DDoS attack taking place, and there-
fore, to contribute to selection of the appropriate protection
mechanism. We would achieve the prediction by extending
the dataset labeling, which we consider for future work.

IP Flags. In our design, this attribute is a 16-bit integer
value which includes three bits representing the ﬂags Reserved
Bit, Don’t Fragment and More Fragments, plus 13 bits for the
Fragment offset value, which is non-zero only if bit “Don’t
Fragment” is unset. Unlike the IP fragmented ﬂood DDoS
in which the IP ﬂags are manipulated to exploit
attacks,
the datagram fragmentation mechanisms, 99.99% of DDoS
packets in the UNB datasets present an IP ﬂags value of
0x4000, with only the “Don’t Fragment” bit set to 1. A
different distribution of IP ﬂags is observed in the UNB
benign trafﬁc, with the “Don’t Fragment” bit set to 1 in
about 92% of the packets. Thus, the pattern of IP ﬂags is
slightly different between attack and benign trafﬁc, and we
are conﬁdent that LUCID is indeed learning their signiﬁcance
in DDoS classiﬁcation, as evidenced by its 2nd place in our
ranking.

B. Future Directions

However, even given this activation analysis, there is no
deﬁnitive list of features that exist for detecting DDoS attacks
with which we can directly compare our results. Analysing
the related work, we identify a wide range of both stateless
and stateful features highlighted for their inﬂuence in a given
detection model, which is not unexpected as the features of
use vary depending on the attack trafﬁc. This is highlighted by
the 2014 study [66], which concludes that different classes of
attack have different properties, leading to the wide variance
in features identiﬁed as salient for the attack detection. The
authors also observe that the learning of patterns speciﬁc to
the attack scenario would be more valuable than an effort to
produce an attack-agnostic ﬁnite list of features. We, therefore,
conclude from our analysis that LUCID appears to be learning
the importance of relevant features for DDoS detection, which
gives us conﬁdence in the prediction performance.

Linked to this activation analysis, we highlight adversarial
robustness as a key consideration for the deployment of ML-
based IDSs. As detailed in [67], the two main attacks on IDSs
are during training via a poisoning attack (i.e. corruption of the

11

training data), or in testing, when an evasion attack attempts
to cause incorrect classiﬁcation by making small perturbations
to observed features. Our activation analysis is a ﬁrst step in
the investigation of the model behaviour in adversarial cases
with the feature ranking in Table X highlighting the features
for perturbation for evasion attacks. Of course, the adversary
model (goal, knowledge, and capability) dictates the potential
for a successful attack. For example, the attacker would require
full knowledge of the CNN and kernel activations, and have
the ability to forge trafﬁc within the network. The construction
of defences robust to adversarial attacks is an open problem
[68] and an aspect which we will further explore for LUCID.

VII. USE-CASE: DDOS DETECTION AT THE EDGE

Edge computing is an emerging paradigm adopted in a
variety of contexts (e.g. fog computing [69], edge clouds
[70]), with the aim of improving the performance of applica-
tions with low-latency and high-bandwidth requirements. Edge
computing complements centralised data centres with a large
number of distributed nodes that provide computation services
close to the sources of the data.

The proliferation of attacks leveraging unsecured IoT de-
vices (e.g., the Mirai botnet [71] and its variants) demonstrate
the potential value in edge-based DDoS attack detection.
Indeed, with edge nodes close to the IoT infrastructure, they
can detect and block the DDoS trafﬁc as soon as it leaves
the compromised devices. However, in contrast to cloud high-
performance servers, edge nodes cannot exploit sophisticated
solutions against DDoS attacks, due to their limited computing
and memory resources. Although recent research efforts have
demonstrated that the mitigation of DDoS attacks is feasible
even by means of commodity computers [72], [73], edge
computing-based DDoS detection is still at an early stage.

In this section, we demonstrate that our DDoS detection
solution can be deployed and effectively executed on resource-
constrained devices, such as edge nodes or IoT gateways, by
running LUCID on an NVIDIA Jetson TX2 development board
[74], equipped with a quad-core ARM Cortex-A57@2 GHz
CPU, 8 GB of RAM and a 256-core Pascal@1300 MHz
Graphics Processing Unit (GPU). For the experiments, we
used Tensorﬂow 1.9.0 with GPU support enabled by cuDNN,
a GPU-accelerated library for deep neural networks [75].

A. Detection

In the ﬁrst experiment, we analyse the applicability of
our approach to online edge computing environments by
estimating the prediction performance in terms of samples
processed per second. As we are aware that edge nodes do not
necessarily mount a GPU device, we conduct the experiments
with and without the GPU support on the UNB201X test set
and discuss the results.

We note that in an online system, our preprocessing tool pre-
sented in Section III-A can be integrated into the server/edge
device. The tool would process the live trafﬁc collected from
the NICs of the server/edge device, collecting the packet
attributes, organising them into ﬂows and, after a predeﬁned
time interval, T , pass the data structure to the CNN for

inference. We acknowledge that the speed of this process will
inﬂuence the overall system performance. However, as we
have not focused on optimising our preprocessing tool, rather
on optimising detection, its evaluation is left as future work.
Instead, in these experiments, we load the UNB datasets from
the hard disk rather than processing live trafﬁc.

With respect to this, one relevant parameter is the batch
size, which conﬁgures how many samples are processed by the
CNN in parallel at each iteration. Such a parameter inﬂuences
the speed of the detection, as it determines the number of
iterations and, as a consequence, the number of memory reads
required by the CNN to process all the samples in the test set
(or the samples collected in a time window, in the case of
online detection).

d
n
o
c
e
s
/
s
e
p
m
a
S

l

23K
22K
21K
20K
19K
18K
17K
16K
15K
14K
13K
12K

64

128

256

512

1024

2048

4096

8192

Batch size (logarithimc scale)

GPU

CPU

Fig. 5.

Inference performance on the NVIDIA Jetson TX2 board.

Fig. 5 shows the performance of LUCID on the development
board in terms of processed samples/second. As the shape
of each sample is [n, f ] = [100, 11], i.e. each sample can
contain the features of up to 100 packets, we can estimate
that the maximum number of packets per second (pps) that the
device can process without the GPU and using a batch size
of 1024 samples is approximately 1.9 Mpps. As an example,
the content of the UNB201X test set
is 602,547 packets
distributed over 22,416 samples, which represents a processing
requirement of 500 Kpps without the GPU, and 600 Kpps when
the GPU is enabled. This illustrates the ability to deploy
LUCID on a resource-constrained platform.

The second measurement regarding resource-constrained
systems is the memory requirement to store all the samples
collected over a time window. The memory occupancy per
sample is 8,800 bytes, i.e. 100·11 = 1100 ﬂoating point values
of 8 bytes each. As per Fig. 5, the CNN can process around
23K samples/second with the help of the GPU and using a
batch size of 1024. To cope with such a processing speed,
the device would require approximately 20 GB RAM for a
t = 100 time window. However, this value greatly exceeds the
typical amount of memory available on edge nodes, in general
(e.g., 1 GB on Raspberry Pi 3 [76], 2 GB on the ODROID-
XU board [77]), and on our device, in particular. Indeed, the
memory resources of nodes can represent the real bottleneck
in an edge computing scenario.

Therefore, assuming that our edge node is equipped with
1 GB RAM, the maximum number of samples that can be
stored in RAM is approximately 100K (without taking into
account RAM used by the operating system and applications).

12

We have calculated that this memory size would be sufﬁcient
for an attack such as the HTTP-based DDoS attack in the
CSECIC2018 dataset, for which we measured approximately
30K samples on average over a 100 s time window. For
more aggressive attacks, however, a strategy to overcome the
memory limitation would be to conﬁgure the CNN model with
lower values of t and n. For instance, setting the value of both
parameters to 10 can reduce the memory requirement by a
factor of 100, with a low cost in detection accuracy (F1 score
0.9928 on the UNB201X test set, compared to the highest
score obtained with t = n = 100, i.e. 0.9946). The dynamic
conﬁguration of the model itself is out of scope of this work.
The measurements based on our test datasets demonstrate
the LUCID CNN is usable on a resource-constrained
that
platform both with respect to processing and memory require-
ments. These results are promising for effective deployment
of LUCID in a variety of edge computing scenarios, including
those where the nodes execute latency-sensitive services. A
major challenge in this regard is balancing between resource
usage of LUCID (including trafﬁc collection and preprocess-
ing) and detection accuracy, i.e. ensuring the required level of
protection against DDoS attacks without causing delays to the
services. A deep study of this trade-off is out of scope of this
paper and is reserved for future work.

B. Training time

In a real-world scenario, the CNN model will require re-
training with new samples of benign and malicious trafﬁc
to update all
the weights and biases. In edge computing
environments, the traditional approach is to send large amounts
of data from edge nodes to remote facilities such as private
or commercial datacentres. However, this can result in high
end-to-end latency and bandwidth usage. In addition, it may
raise security concerns, as it requires trust in a third-party
entity (in the case of commercial cloud services) regarding
the preservation of data conﬁdentiality and integrity.

A solution to this issue is to execute the re-training task
locally on the edge nodes. In this case, the main challenge is
to control the total training time, as this time determines how
long the node remains exposed to new DDoS attacks before
the detection model can leverage the updated parameters.

To demonstrate the suitability of our model for this situation,
we have measured the convergence training time of LUCID
on the development board using the UNB201X training and
validation sets with and without the GPU support. We have
experimented by following the learning procedure described
in Sec. III-C, thus with a training termination criterion based
on the loss value measured on the validation set. The results
are presented in Table XI along with the performance obtained
on the server used for the study in Sec. IV-D.

As shown in Table XI,

the CNN training time on the
development board without using the GPU is around 2 hours
(184 epochs). This is approximately 4 times slower than
training on the server, but clearly outperforms the training time
of our implementation of DeepDefense 3LSTM, which we
measured at more than 1000 sec/epoch with the GPU (i.e., 40
times slower than LUCID under the same testing conditions).

TABLE XI
TRAINING CONVERGENCE TIME.

Setup

LUCID Server

LUCID Dev. board (GPU)

LUCID Dev. board (CPU)

3LSTM Dev. board (GPU)

Time/epoch
(sec)

Convergence
time (sec)

10.2

25.8

40.5

1070

1880

4500

7450

>90000

In application scenarios where a faster convergence is
required, the time can be further reduced by either terminating
the training process early after a pre-deﬁned number of epochs,
or limiting the size of the training/validation sets. As adopting
one or both of such strategies can result in a lower detection
accuracy, the challenge in such scenarios is ﬁnding the trade-
off between convergence time and detection accuracy that
meets the application requirements.

VIII. CONCLUSIONS

The challenge of DDoS attacks continues to undermine the
availability of networks globally. In this work, we have pre-
sented a CNN-based DDoS detection architecture. Our design
has targeted a practical, lightweight implementation with low
processing overhead and attack detection time. The beneﬁt
of the CNN model is to remove threshold conﬁguration as
required by statistical detection approaches, and reduce feature
engineering and the reliance on human experts required by
alternative ML techniques. This enables practical deployment.
In contrast to existing solutions, our unique trafﬁc pre-
processing mechanism acknowledges how trafﬁc ﬂows across
network devices and is designed to present network trafﬁc
to the CNN model for online DDoS attack detection. Our
evaluation results demonstrate that LUCID matches the existing
state-of-the-art performance. However, distinct from existing
work, we demonstrate consistent detection results across a
range of datasets, demonstrating the stability of our solution.
Furthermore, our evaluation on a resource-constrained device
demonstrates the suitability of our model for deployment in
resource-constrained environments. Speciﬁcally, we demon-
strate a 40x improvement in processing time over similar
state-of-the-art solutions. Finally, we have also presented an
activation analysis to explain how LUCID learns to detect
DDoS trafﬁc, which is lacking in existing works.

ACKNOWLEDGMENT

This work has received funding from the European Union’s
Horizon 2020 Research and Innovation Programme under
grant agreement no. 815141 (DECENTER project).

REFERENCES

[1] Krebs

on

Security,

“DDoS

Reddit,”
Spotify,
ddos-on-dyn-impacts-twitter-spotify-reddit,
Oct-2019].

on

Dyn

Twitter,
https://krebsonsecurity.com/2016/10/
31-

[Accessed:

Impacts

2016,

13

[2] Radware,

“Memcached

DDoS

Attacks,”

https://security.

radware.com/ddos-threats-attacks/threat-advisories-attack-reports/
memcached-under-attack/, 2018, [Accessed: 31-Oct-2019].

[3] IETF Network Working Group, “Network Ingress Filtering: Defeating
Denial of Service Attacks which employ IP Source Address Spooﬁng,”
https://tools.ietf.org/html/bcp38, 2000, [Accessed: 31-Oct-2019].

[4] X. Yuan, C. Li, and X. Li, “DeepDefense: Identifying DDoS Attack via

Deep Learning,” in Proc. of SMARTCOMP, 2017.

[5] M. Ghanbari and W. Kinsner, “Extracting Features from Both the Input
and the Output of a Convolutional Neural Network to Detect Distributed
Denial of Service Attacks,” in Proc. of ICCI*CC, 2018.

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image

recognition,” in Proc. CVPR, 2016, pp. 770–778.

[7] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classiﬁcation
with deep convolutional neural networks,” in Advances in Neural Infor-
mation Processing Systems 25, 2012, pp. 1097–1105.

[8] M. Sabokrou, M. Fayyaz, M. Fathy, Z. Moayed, and R. Klette, “Deep-
anomaly: Fully convolutional neural network for fast anomaly detection
in crowded scenes,” Computer Vision and Image Understanding, vol.
172, pp. 88 – 97, 2018.

[9] Y. Kim, “Convolutional neural networks for sentence classiﬁcation,” in

Proc. of EMNLP, 2014.

[10] B. Alipanahi, A. Delong, M. Weirauch, and B. J Frey, “Predicting the
sequence speciﬁcities of dna- and rna-binding proteins by deep learning,”
Nature biotechnology, vol. 33, 07 2015.

[11] D. Quang and X. Xie, “DanQ: a hybrid convolutional and recurrent deep
neural network for quantifying the function of DNA sequences,” Nucleic
Acids Research, vol. 44, no. 11, pp. e107–e107, 2016.

[12] O. Janssens, V. Slavkovikj, B. Vervisch, K. Stockman, M. Loccuﬁer,
S. Verstockt, R. V. de Walle, and S. V. Hoecke, “Convolutional Neural
Network Based Fault Detection for Rotating Machinery,” Journal of
Sound and Vibration, vol. 377, pp. 331 – 345, 2016.

[13] A. Vilamala, K. H. Madsen, and L. K. Hansen, “Deep Convolutional
Neural Networks for Interpretable Analysis of EEG Sleep Stage Scor-
ing,” Proc. of MLSP, 2017.

[14] N. McLaughlin, J. Martinez del Rincon, B. Kang, S. Yerima, P. Miller,
S. Sezer, Y. Safaei, E. Trickel, Z. Zhao, A. Doup´e, and G. Joon Ahn,
“Deep android malware detection,” in Proc. of CODASPY, 2017.
[15] T. Kim, B. Kang, M. Rho, S. Sezer, and E. G. Im, “A multimodal deep
learning method for android malware detection using various features,”
IEEE Transactions on Information Forensics and Security, vol. 14, no. 3,
pp. 773–788, March 2019.

[16] Wei Wang, Ming Zhu, Xuewen Zeng, Xiaozhou Ye, and Yiqiang Sheng,
“Malware trafﬁc classiﬁcation using convolutional neural network for
representation learning,” in Proc. of ICOIN, 2017.

[17] M. Yeo, Y. Koo, Y. Yoon, T. Hwang, J. Ryu, J. Song, and C. Park,
“Flow-based malware detection using convolutional neural network,” in
Proc. of International Conference on Information Networking, 2018.

[18] R. Russell, L. Kim, L. Hamilton, T. Lazovich, J. Harer, O. Ozdemir,
P. Ellingwood, and M. McConley, “Automated Vulnerability Detection in
Source Code Using Deep Representation Learning,” in Proc. of ICMLA,
2018.

[19] K. Wu, Z. Chen, and W. Li, “A Novel Intrusion Detection Model for a
Massive Network Using Convolutional Neural Networks,” IEEE Access,
vol. 6, pp. 50 850–50 859, 2018.

[20] S. Potluri, S. Ahmed, and C. Diedrich, “Convolutional Neural Networks

for Multi-class Intrusion Detection System,” in Proc. of MIKE, 2018.

[21] R. Vinayakumar, K. P. Soman, and P. Poornachandran, “Applying
convolutional neural network for network intrusion detection,” in Proc.
of ICACCI, 2017.

[22] M. Abdelaty, R. Doriguzzi-Corin, and D. Siracusa, “AADS: A Noise-
Robust Anomaly Detection Framework for Industrial Control Systems,”
in Proc. of ICICS, 2019.

[23] P. Criscuolo, “Distributed denial of service, tribe ﬂood network 2000,
and stacheldraht CIAC-2319, Department of Energy Computer Incident
Advisory Capability (CIAC),” UCRLID-136939, Rev, vol. 1, 2000.
[24] H. A. Herrera, W. R. Rivas, and S. Kumar, “Evaluation of Internet
Connectivity Under Distributed Denial of Service Attacks from Botnets
of Varying Magnitudes,” in Proc. of ICDIS, 2018.

[25] L. Feinstein, D. Schnackenberg, R. Balupari, and D. Kindred, “Statistical
Approaches to DDoS Attack Detection and Response,” in Proceedings
DARPA Information Survivability Conference and Exposition, 2003.
[26] P. Bojovi´c, I. Baˇsiˇcevi´c, S. Ocovaj, and M. Popovi´c, “A practical
approach to detection of distributed denial-of-service attacks using a
hybrid detection method,” Computers & Electrical Engineering, vol. 73,
pp. 84–96, 2019.

[27] K. Kalkan, L. Altay, G. Gr, and F. Alagz, “JESS: Joint Entropy-Based
DDoS Defense Scheme in SDN,” IEEE Journal on Selected Areas in
Communications, vol. 36, no. 10, pp. 2358–2372, Oct 2018.

[28] S. B. I. Shah, M. Anbar, A. Al-Ani, and A. K. Al-Ani, “Hybridizing
entropy based mechanism with adaptive threshold algorithm to detect
ra ﬂooding attack in ipv6 networks,” in Computational Science and
Technology. Singapore: Springer Singapore, 2019, pp. 315–323.
[29] P. Kumar, M. Tripathi, A. Nehra, M. Conti, and C. Lal, “Safety: Early
detection and mitigation of tcp syn ﬂood utilizing entropy in sdn,” IEEE
Transactions on Network and Service Management, vol. 15, no. 4, pp.
1545–1559, 2018.

[30] J.-H. Jun, C.-W. Ahn, and S.-H. Kim, “Ddos attack detection by using
packet sampling and ﬂow features,” in Proc. of the 29th Annual ACM
Symposium on Applied Computing, 2014.

[31] S. Yu, W. Zhou, R. Doss, and W. Jia, “Traceback of DDoS Attacks Using
Entropy Variations,” IEEE Transactions on Parallel and Distributed
Systems, 2011.

[32] R. Wang, Z. Jia, and L. Ju, “An entropy-based distributed ddos de-
tection mechanism in software-deﬁned networking,” in 2015 IEEE
Trustcom/BigDataSE/ISPA, 2015.

[33] M. E. Ahmed, S. Ullah, and H. Kim, “Statistical application ﬁnger-
printing for ddos attack mitigation,” IEEE Transactions on Information
Forensics and Security, vol. 14, no. 6, pp. 1471–1484, 2019.

[34] J. Wang, L. Yang, J. Wu, and J. H. Abawajy, “Clustering Analysis for

Malicious Network Trafﬁc,” in Proc. of IEEE ICC, 2017.

[35] Y. Yao, L. Su, and Z. Lu, “DeepGFL: Deep Feature Learning via Graph
for Attack Detection on Flow-Based Network Trafﬁc,” in Proc. of IEEE
Military Communications Conference (MILCOM), 2018.

[36] E. Min, J. Long, Q. Liu, J. Cui, , and W. Chen, “TR-IDS: Anomaly-
Based Intrusion Detection through Text-Convolutional Neural Network
and Random Forest,” Security and Communication Networks, 2018.
[37] J. Cui, J. Long, E. Min, Q. Liu, and Q. Li, “Comparative Study of
CNN and RNN for Deep Learning Based Intrusion Detection System,”
in Cloud Computing and Security, 2018, pp. 159–170.

[38] M. Roopak, G. Yun Tian, and J. Chambers, “Deep Learning Models for
Cyber Security in IoT Networks,” in Proc. of IEEE CCWC, 2019.
[39] S. Homayoun, M. Ahmadzadeh, S. Hashemi, A. Dehghantanha, and
R. Khayami, “BoTShark: A Deep Learning Approach for Botnet Trafﬁc
Detection,” in Cyber Threat Intelligence. Springer, 2018, pp. 137–153.
[40] R. Sommer and V. Paxson, “Outside the closed world: On using machine
learning for network intrusion detection,” in Proc. of IEEE symposium
on security and privacy, 2010.

[41] A. L. Buczak and E. Guven, “A survey of data mining and machine
learning methods for cyber security intrusion detection,” IEEE Commu-
nications Surveys & Tutorials, vol. 18, no. 2, pp. 1153–1176, 2016.
[42] M. Idhammad, K. Afdel, and M. Belouch, “Semi-supervised machine
learning approach for ddos detection,” Applied Intelligence, vol. 48,
no. 10, pp. 3193–3208, 2018.

[43] K. J. Singh, T. Khelchandra, and T. De, “Entropy-Based Application
Layer DDoS Attack Detection Using Artiﬁcial Neural Networks,” En-
tropy, vol. 18, p. 350, 2016.

[44] Z. He, T. Zhang, and R. B. Lee, “Machine learning based DDoS attack
detection from source side in cloud,” in Proc. of CSCloud, 2017.
[45] K. S. Hoon, K. C. Yeo, S. Azam, B. Shunmugam, and F. De Boer,
“Critical review of machine learning approaches to apply big data
analytics in DDoS forensics,” in Proc of ICCCI, 2018.

[46] R. Primartha and B. A. Tama, “Anomaly detection using random forest:

A performance revisited,” in Proc. of ICoDSE, 2017.

[47] A. Koay, A. Chen, I. Welch, and W. K. G. Seah, “A new multi classiﬁer
system using entropy-based features in ddos attack detection,” in Proc.
of ICOIN, 2018.

[48] C. Yin, Y. Zhu, J. Fei, and X. He, “A deep learning approach for intrusion
detection using recurrent neural networks,” IEEE Access, vol. 5, 2017.
[49] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed

analysis of the kdd cup 99 data set,” in Proc. of IEEE CISDA, 2009.

[50] N. Moustafa and J. Slay, “Unsw-nb15: a comprehensive data set for
network intrusion detection systems (unsw-nb15 network data set),” in
Proc. of MilCIS, 2015.

[51] D. Kwon, K. Natarajan, S. C. Suh, H. Kim, and J. Kim, “An empir-
ical study on network anomaly detection using convolutional neural
networks,” in Proc. of IEEE ICDCS, 2018.

[52] J. Song, H. Takakura, and Y. Okabe, “Description of Kyoto
University Benchmark Data,” http://www.takakura.com/Kyoto data/
BenchmarkData-Description-v5.pdf, [Accessed: 31-Oct-2019].

[53] C. Callegari, S. Giordano, and M. Pagano, “Statistical network anomaly

detection: An experimental study,” in Proc. of FNSS, 2016.

14

[54] CAIDA, “DDoS Attack 2007 Dataset,” https://www.caida.org/data/

passive/ddos-20070804 dataset.xml, 2019, [Accessed: 31-Oct-2019].

[55] A. Shiravi, H. Shiravi, M. Tavallaee, and A. A. Ghorbani, “Toward
developing a systematic approach to generate benchmark datasets for
intrusion detection,” Computers & Security, vol. 31, 2012.

[56] I. Sharafaldin, A. Habibi Lashkari, and A. A. Ghorbani, “Toward
Generating a New Intrusion Detection Dataset and Intrusion Trafﬁc
Characterization,” in Proc. of ICISSP, 2018.

[57] The Canadian Institute for Cybersecurity, “CSE-CIC-IDS2018 dataset,”
[Accessed: 31-

https://www.unb.ca/cic/datasets/ids-2018.html, 2018,
Oct-2019].

[58] Imperva, “HOIC,” https://www.imperva.com/learn/application-security/

high-orbit-ion-cannon, 2019, [Accessed: 31-Oct-2019].

[59] Imperva, “LOIC,” https://www.imperva.com/learn/application-security/

low-orbit-ion-cannon, 2019, [Accessed: 31-Oct-2019].

[60] The Guardian, “Thousands download LOIC software for Anonymous at-
tacks - but are they making a difference?” https://www.theguardian.com/
technology/blog/2010/dec/10/hackers-loic-anonymous-wikileaks, 2010,
[Accessed: 31-Oct-2019].

[61] The Canadian Institute for Cybersecurity, “Datasets,” https://www.unb.

ca/cic/datasets/index.html, 2019, [Accessed: 31-Oct-2019].

[62] Keras-team, “Keras: Deep Learning for humans,” https://github.com/

keras-team/keras, 2019, [Accessed: 31-Oct-2019].

[63] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin,
S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga,
S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden,
M. Wicke, Y. Yu, and X. Zheng, “Tensorﬂow: A system for large-
scale machine learning,” in Proc. of the 12th USENIX Conference on
Operating Systems Design and Implementation, 2016.

[64] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,”

in Proc. of ICLR, 2014.

[65] A.

and Y. Goldberg,

Jacovi, O. Sar Shalom,

“Understanding
convolutional neural networks for text classiﬁcation,” in Proceedings of
the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting
Neural Networks
Brussels, Belgium: Association for
Computational Linguistics, Nov. 2018, pp. 56–65. [Online]. Available:
https://www.aclweb.org/anthology/W18-5408

for NLP.

[66] V. Bukac, “Trafﬁc characteristics of common dos tools,” Masaryk
University, Technical report FIMU-RS-2014-02, pp. 74–78, 2014.
[67] I. Corona, G. Giacinto, and F. Roli, “Adversarial Attacks Against
Intrusion Detection Systems: Taxonomy, Solutions and Open Issues,”
Inf. Sci., vol. 239, pp. 201–225, 2013.

[68] N. Carlini, A. Athalye, N. Papernot, W. Brendel, J. Rauber, D. Tsipras,
I. Goodfellow, A. Madry, and A. Kurakin, “On Evaluating Adversarial
Robustness,” CoRR, vol. abs/1902.06705, 2019.

[69] F. Bonomi, R. Milito, P. Natarajan, and J. Zhu, “Fog computing: A
platform for internet of things and analytics,” in Big Data and Internet
of Things: A Roadmap for Smart Environments. Springer, 2014.
[70] H. Chang and A. Hari and S. Mukherjee and T. V. Lakshman, “Bringing
the cloud to the edge,” in 2014 IEEE Conference on Computer Com-
munications Workshops (INFOCOM WKSHPS), 2014.

[71] M. Antonakakis, T. April, M. Bailey, M. Bernhard, E. Bursztein,
J. Cochran, Z. Durumeric, J. A. Halderman, L. Invernizzi, M. Kallitsis,
D. Kumar, C. Lever, Z. Ma, J. Mason, D. Menscher, C. Seaman,
N. Sullivan, K. Thomas, and Y. Zhou, “Understanding the Mirai Botnet,”
in USENIX Security Symposium, 2017.

[72] S. Miano, R. Doriguzzi-Corin, F. Risso, D. Siracusa, and R. Sommese,
“Introducing SmartNICs in Server-Based Data Plane Processing: The
DDoS Mitigation Use Case,” IEEE Access, vol. 7, 2019.

[73] T. Høiland-Jørgensen, J. D. Brouer, D. Borkmann, J. Fastabend, T. Her-
bert, D. Ahern, and D. Miller, “The eXpress Data Path: Fast Pro-
grammable Packet Processing in the Operating System Kernel,” in Proc.
of ACM CoNEXT, 2018.

[74] NVIDIA

Corporation,

datasheet,”
jetson-tx2-series-modules-data-sheet, 2018, [Accessed: 31-Oct-2019].

“NVIDIA
Series
http://developer.nvidia.com/embedded/dlc/

Jetson

TX2

[75] NVIDIA

Corporation,

“cuDNN

Developer

Guide,”

https:

//docs.nvidia.com/deeplearning/sdk/pdf/cuDNN-Developer-Guide.pdf,
2019, [Accessed: 31-Oct-2019].

[76] Raspberry Pi Foundation, “Raspberry Pi 3 Model B,” https://www.
raspberrypi.org/products/raspberry-pi-3-model-b/, 2019, [Accessed: 31-
Oct-2019].

[77] N. Wang, B. Varghese, M. Matthaiou, and D. S. Nikolopoulos,
“ENORM: A Framework For Edge NOde Resource Management,” IEEE
Transactions on Services Computing, 2018.

