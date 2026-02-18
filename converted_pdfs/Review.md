# Review

Review
A Review of Machine Learning and Transfer Learning Strategies
for Intrusion Detection Systems in 5G and Beyond

Kinzah Noor 1, Agbotiname Lucky Imoize 2,*

, Chun-Ta Li 3,*

and Chi-Yao Weng 4,*

1 Office of Research Innovation and Commercialization (ORIC), University of Management and

Technology (UMT), Lahore 54770, Pakistan; kinzah.noor@umt.edu.pk

2 Department of Electrical and Electronics Engineering, Faculty of Engineering, University of Lagos, Akoka,

3

Lagos 100213, Nigeria
Bachelor’s Program of Artificial Intelligence and Information Security, Fu Jen Catholic University,
New Taipei City 24206, Taiwan

4 Department of Computer Science and Information Engineering, National Chiayi University,

Chiayi City 600355, Taiwan

* Correspondence: aimoize@unilag.edu.ng (A.L.I.); 157278@mail.fju.edu.tw (C.-T.L.);

cyweng@mail.ncyu.edu.tw (C.-Y.W.)

Abstract: This review systematically explores the application of machine learning (ML)
models in the context of Intrusion Detection Systems (IDSs) for modern network security,
particularly within 5G environments. The evaluation is based on the 5G-NIDD dataset, a
richly labeled resource encompassing a broad range of network behaviors, from benign user
traffic to various attack scenarios. This review examines multiple machine learning (ML)
models, assessing their performance across critical metrics, including accuracy, precision,
recall, F1-score, Receiver Operating Characteristic (ROC), Area Under the Curve (AUC), and
execution time. Key findings indicate that the K-Nearest Neighbors (KNN) model excels in
accuracy and ROC AUC, while the Voting Classifier achieves superior precision and F1-
score. Other models, including decision tree (DT), Bagging, and Extra Trees, demonstrate
strong recall, while AdaBoost shows underperformance across all metrics. Naive Bayes (NB)
stands out for its computational efficiency despite moderate performance in other areas. As
5G technologies evolve, introducing more complex architectures, such as network slicing,
increases the vulnerability to cyber threats, particularly Distributed Denial-of-Service
(DDoS) attacks. This review also investigates the potential of deep learning (DL) and Deep
Transfer Learning (DTL) models in enhancing the detection of such attacks. Advanced DL
architectures, such as Bidirectional Long Short-Term Memory (BiLSTM), Convolutional
Neural Networks (CNNs), Residual Networks (ResNet), and Inception, are evaluated,
with a focus on the ability of DTL to leverage knowledge transfer from source datasets
to improve detection accuracy on sparse 5G-NIDD data. The findings underscore the
importance of large-scale labeled datasets and adaptive security mechanisms in addressing
evolving threats. This review concludes by highlighting the significant role of ML and
DTL approaches in strengthening network defense and fostering proactive, robust security
solutions for future networks.

Keywords: machine learning (ML); deep learning (DL); 5G communication; network
intrusion detection systems (NIDSs); artificial intelligence (AI); Internet of Things (IoT);
NIDS datasets; security threats

MSC: 68R07; 68M25; 94A13

Academic Editor: Antanas Cenys

Received: 19 February 2025

Revised: 19 March 2025

Accepted: 24 March 2025

Published: 26 March 2025

Citation: Noor, K.; Imoize, A.L.; Li,

C.-T.; Weng, C.-Y. A Review of

Machine Learning and Transfer

Learning Strategies for Intrusion

Detection Systems in 5G and Beyond.

Mathematics 2025, 13, 1088. https://

doi.org/10.3390/math13071088

Copyright: © 2025 by the authors.

Licensee MDPI, Basel, Switzerland.

This article is an open access article

distributed under the terms and

conditions of the Creative Commons

Attribution (CC BY) license

(https://creativecommons.org/

licenses/by/4.0/).

Mathematics 2025, 13, 1088

https://doi.org/10.3390/math13071088

Mathematics 2025, 13, 1088

2 of 63

1. Introduction

The rapid proliferation of Internet of Things (IoT) devices and the expansion of broad-
band connectivity have fueled the evolution of 5G networks, enabling transformative
applications such as augmented/virtual reality, smart factories, autonomous vehicles, and
intelligent urban infrastructures [1]. These networks provide significant advancements in
capacity, throughput, and latency that drive innovation across diverse industries, including
healthcare and precision agriculture [2]. At the same time, emerging 6G networks are
envisioned to support even more sophisticated, data-intensive applications by leverag-
ing artificial intelligence (AI) and machine learning (ML) to transition from mere device
connectivity to comprehensive, connected intelligence [3].

The increased complexity and dynamism of modern 5G networks, however, have
introduced significant challenges in network security. Traditional Intrusion Detection Sys-
tems (IDSs) struggle to cope with the evolving threat landscape, largely due to the rapid
growth in connected devices, exponential data traffic, and dynamic network architectures.
These challenges are further compounded by the scarcity of labeled data tailored to 5G
environments, making it difficult to develop highly accurate detection models. Conse-
quently, the expanded threat surface necessitates the development of robust, proactive, and
self-adaptive Network Intrusion Detection Systems (NIDSs) capable of identifying and
classifying anomalous behaviors in real time [4].

Recent advances in deep learning (DL) and ML have shown promise in enhancing
the efficiency and accuracy of IDS. While traditional ML methods—such as K-Nearest
Neighbors (KNN), Naive Bayes (NB), and support vector machines (SVMs)—often depend
heavily on manual feature engineering, time-series models and AI-driven approaches
offer improved capabilities for real-time threat detection and mitigation [5]. The evolution
toward 6G networks, with their extreme performance requirements and deeper integration
of AI, further emphasizes the need for adaptive security mechanisms that can effectively
utilize the vast data streams generated by modern networks to detect anomalous behaviors
and vulnerabilities [6]. A critical challenge in the realm of network security is the detection
of Distributed Denial-of-Service (DDoS) attacks. These attacks are particularly difficult to
detect in 5G and Beyond-5G (B5G) networks due to dynamic network topologies and high
traffic volumes. Although DL models have shown potential in recognizing complex attack
patterns, their performance is often constrained by variations in data distribution across
different networks. To address these challenges, Transfer Learning (TL) and, in particular,
Deep Transfer Learning (DTL) have been employed to repurpose models trained in one
domain for application in another, thereby enhancing performance while mitigating issues
related to data scarcity and overfitting [7].

Another critical aspect of developing a reliable IDS is preventing data leakage dur-
ing preprocessing—a phenomenon where external information inadvertently influences
the training process, leading to misleading high-performance metrics [8]. This issue is
exacerbated by the scarcity of up-to-date, comprehensive datasets that accurately reflect
the complexities of real-world 5G network traffic. Publicly available datasets are often
outdated or generated under controlled conditions, which limits their applicability for
contemporary security research. Furthermore, privacy concerns and the risk of exposing
vulnerabilities have led mobile network operators to withhold live network data, creating
a significant barrier to AI-/ML-based security research.

The accuracy and reliability of ML-based IDSs are heavily influenced by the quality and
diversity of the datasets employed for both training and performance assessment. An ideal
dataset should capture a wide range of attack vectors, include diverse protocols, provide
comprehensive traffic records, and offer meticulous labeling and metadata. Although
datasets such as CICIDS2017 have advanced the field, a significant gap remains between

Mathematics 2025, 13, 1088

3 of 63

the simulated environments used to generate these datasets and the dynamic conditions of
operational 5G networks. To address this gap, the present review focuses on the 5G-NIDD
dataset—an authentic dataset derived from a real-world 5G network implementation. This
dataset, which comprises both benign and malicious traffic, has been used to evaluate the
performance of 13 distinct machine learning (ML) algorithms, ranging from simple linear
models, such as Logistic Regression (LR), to complex ensemble methods like random forest
(RF), Gradient Boosting (GB), and Voting Classifiers [9].

This research addresses key questions regarding 5G network security and the evolving
role of IDS. This study examines the major security hurdles faced in 5G systems and
explains the capabilities of IDSs to address these threats. The evaluation reveals that
existing IDS frameworks require improvements because their datasets do not perform
reliably, necessitating more effective methods to enhance overall effectiveness. The research
focuses on evaluating a combinatorial algorithm that operates in sequence to detect and
analyze keyloggers, making this the primary area of investigation. This review finds
that integrating a solid testbed architecture into a 5G Test Network (5GTN) enhances the
precision of security evaluation more effectively than traditional testbeds or simulation
testing methods. This review evaluates various machine learning (ML) models for intrusion
detection, analyzing the 5G-NIDD dataset to assess their suitability in actual 5G system
deployments. The research presents substantial information about three main obstacles:
data leakage protection, the limited availability of network datasets, and the limitations of
TL, which guide the future development of network security, AI, and ML techniques.

The contributions of this study include the following:

■

■

■

■

This paper presents an in-depth examination of 5G network security challenges and
the evolving role of IDSs in mitigating these threats. Existing IDS frameworks and
datasets are critically evaluated to underscore the need for more reliable and repre-
sentative data in developing effective NIDSs. The use of a back-to-back combinatorial
algorithm for keylogger detection and analysis is also proposed.
This paper introduces a robust testbed architecture integrated with the 5GTN platform.
This architecture facilitates the generation of realistic datasets, enabling a more accu-
rate assessment of security solutions compared to traditional testbed or simulation
environments.
It reviews the application of diverse ML models for intrusion detection using the 5G-
NIDD dataset. The analysis highlights the strengths and limitations of each algorithm
under the unique conditions of 5G networks, providing insights into their practical
applicability for real-world intrusion detection.
This study outlines critical challenges, including data leakage prevention, the scarcity
of live network datasets, and the effective use of time-series analysis (TL). These
insights offer valuable guidance for future AI-/ML-based network security research.

The remainder of this paper is structured as follows: Section 2 provides a compre-
hensive review of the background and related work, focusing on the evolution of network
security challenges and the limitations of existing datasets and IDS frameworks in 5G
environments. Section 3 explores the fundamental concepts of modern wireless networks
and examines the available datasets for 5G network security. Section 4 presents an in-depth
discussion of IDS, including IDS classification, NIDSs for IoT, a comparison of open-source
NIDS, and the role of datasets in NIDS development, along with a comparative analysis
of existing datasets. Section 5 describes the development of 5G NIDS datasets, covering
aspects such as simulated attack types, system architecture, testbed design, and the method-
ology for dataset generation. Additionally, this section provides a comparative evaluation
of different architectures and frameworks, along with an analysis of cyberattack types and
relevant performance metrics. Section 6 explores Knowledge Discovery in Databases (KDD)

Mathematics 2025, 13, 1088

4 of 63

for Network Intrusion Detection System (NIDS) preprocessing, covering data cleaning,
feature transformation, data acquisition, and processing. Furthermore, it examines the data
mining stage, performance assessment, evaluation techniques, challenges in incremental
evaluation, attack impact analysis, and the application of ML and DL models for perfor-
mance evaluation, including a comparative study of various models. Section 7 highlights
the key challenges and limitations associated with 5G NIDS research. Section 8 discusses
recent trends, future research directions, and lessons learned. Finally, Section 9 concludes
the paper by summarizing the main findings and their broader implications for designing
and implementing secure 5G and 6G networks.

2. Background and Related Work

Recent advancements in AI have significantly enhanced machine learning (ML) ap-
proaches for threat detection. For example, one study [10] comprehensively reviewed
various data mining techniques used in NIDS. It proposed an ensemble-based feature
selection and anomaly detection framework, emphasizing the necessity for real-time so-
lutions. In a similar effort, another investigation [11] developed an optimized ML-based
NIDS that balances computational complexity and detection performance. Self-adaptive
ML models in 5G environments further leverage AI to enable real-time anomaly detection,
dynamically adjusting to traffic patterns and optimizing resource allocation. These AI-/ML-
based approaches are credited with the capacity to identify and neutralize sophisticated
threats in 5G and next-generation networks, thanks to the substantial computational power
available [12]. Moreover, such systems continually enhance their detection capabilities over
time by employing various learning paradigms, including Supervised Learning (SL), Unsu-
pervised Learning (USL), Semi-Supervised Learning (SSL), and Reinforcement Learning
(RL) [13]. In particular, SL methods, which rely on labeled datasets, utilize models such as
decision trees (DTs), random forests (RFs), Naive Bayes (NB), and Deep Neural Networks
(DNNs), all of which require sufficient data for practical training.

A robust dataset comprising six million traffic flows, including both benign and
attack scenarios, has been collected from a dedicated 5G laboratory testbed [14]. This
dataset serves as the source domain for developing DTL models. Within this testbed,
the network is segmented into two distinct slices, each providing valuable traffic data.
In contrast, the 5G-NIDD dataset [15] represents the target domain and contains limited
annotated traffic data related to various DDoS attacks observed in a real-world 5G network
environment. Researchers have explored several advanced DL architectures as potential
classifiers for detecting DDoS attacks, including Bidirectional Long Short-Term Memory
(BiLSTM) networks [16], Convolutional Neural Networks (CNNs) [17], Residual Networks
(ResNet), and Inception models. The curated dataset is used to train robust pre-trained
models, which, in turn, facilitate the application of DTL methods to various algorithms on
the target dataset.

An overview of prior studies on DTL highlights its potential in DDoS detection for
5G networks [18]. Although TL has been extensively applied in fields such as computer
vision and natural language processing, its use in cybersecurity—particularly for detecting
DDoS attacks in 5G systems using DTL—remains relatively underexplored. To bridge this
gap, the review [19] investigates the application of DTL in IDSs for 5G networks, aiming to
enhance DDoS attack detection. Additional research [20] demonstrates that incorporating
TL with the AdaBoost algorithm can boost classification accuracy and overall network
performance. Another study [21] addressed network traffic classification in 5G IoT systems
by employing DTL with pre-trained models like EfficientNet and Big Transfer, achieving
near-perfect accuracy with only 10% of the data labeled, as validated on the USTC-TFC2016

Mathematics 2025, 13, 1088

5 of 63

dataset. These findings underscore the potential of TL to improve IoT data classification
and optimize resource utilization in data-constrained environments.

Beyond these applications, further research has investigated the integration of IoT and
5G in the context of the Industrial Internet of Things (IIoT). One study [22] applied TL to
reduce data and resource requirements for model training, enhancing predictive mainte-
nance and fault detection by leveraging pre-trained models on similar equipment while
optimizing the 5G infrastructure. In another approach, a deep deterministic policy gradient
algorithm, enhanced with TL, was utilized for network slicing in IIoT scenarios [23]. This
method optimized bandwidth allocation and transmission power, resulting in improved
Quality of Service (QoS), energy efficiency, and reliability, with TL expediting the training
process across multiple gateways. Moreover, the Fortis-Economical Denial of Sustainability
(EDoS) framework [24] was introduced to mitigate EDoS attacks in 5G network slicing. By
combining Control Gated–Gated Recurrent Unit (CG-GRU)-based anomaly detection with
DTL, FortisEDoS can detect malicious behavior, prevent unwarranted resource scaling, and
employ a comprehensive surveillance system—comprising auto-scaling modules and EDoS
mitigation mechanisms—to monitor and counteract attacks in real time, thereby reducing
the financial impact of such threats [25].

Other studies have further explored the potential of ML techniques in NIDS. For in-
stance, one investigation [26] utilized the UNSW-NB15 dataset to train various ML models,
employing Extreme Gradient Boost (XGB) for feature selection and finding that Artificial
Neural Networks (ANNs) provided superior performance. In a separate study [27], re-
searchers introduced a novel feature selection method based on statistical measures such as
standard deviation and the difference between mean and median values. When tested on
datasets like NSL-KDD, UNSW-NB15, and CICIDS2017, this approach outperformed estab-
lished techniques such as Recursive Feature Elimination, Chi-Square, and RF to enhance
classification accuracy.

Historically, the DARPA dataset was one of the first benchmarks for ML-based IDS,
encompassing simulated attacks such as DoS, password guessing, buffer overflow, SYN
flood, and network mapper exploits. However, due to its inherent limitations, subsequent
datasets—such as KDD Cup 99, NSL-KDD, DEFCON, CAIDA, LBNL, CTU-13, UNSW-
NB15, and Bot-IoT—have been developed to better evaluate IDS performance [28]. Despite
their contributions, many of these datasets are now considered outdated due to the rapid
evolution of network technologies and the emergence of new cyber threats. IDSs can
generally be classified into pattern-matching and AI-based anomaly detection approaches.
While pattern-matching methods tend to suffer from a high False Positive Rate (FPR),
AI-based methods are heavily dependent on the quality of feature selection. They can
face challenges such as inadequate feature representation and overfitting [29]. Although
conventional ML techniques like SVM and KNN sometimes struggle to achieve high
accuracy without an increase in False Positives (FPs), certain studies [30] have demonstrated
excellent performance on datasets like CIC-IDS-2017 and UNSW-NB15. Furthermore,
hybrid models—such as those integrating Convolutional Neural Networks (CNNs) and
Long Short-Term Memory (LSTM) in a hierarchical structure or employing feature fusion
techniques—have shown promising results, with one study reporting an accuracy of up to
84% using the UNSW-NB15 dataset [31].

The authors [32] address critical cybersecurity challenges by noting that outdated
datasets, such as KDDCup99 and NSL-KDD, fail to capture modern attack techniques,
resulting in ineffective IDS performance. To remedy this, they propose TestCloudIDS—a
new, labeled dataset reflecting current threat landscapes across diverse attack types and
cloud environments—along with SparkShield. This IDS model leverages big data analytics
and Apache Spark for faster, proactive threat detection. Similarly, the study [33] introduces

Mathematics 2025, 13, 1088

6 of 63

a dynamic Models Orchestrator that deploys adaptive ML/DL models as a service within a
Dockerized 5G environment to secure network slices and devices. By utilizing algorithms
ranging from DTs and RF to CNNs and Large Language Models and training on updated
datasets like CICDDOS2019, this orchestrator operates within the Radio Access Network
(RAN) to detect and mitigate real-time attacks such as scans and floods.

Despite advancements in DTL-based IDSs, several critical challenges persist. These
include data imbalance, where intrusion samples are scarce compared to normal data,
leading to biased models; difficulties in effective feature extraction from diverse data
sources such as network traffic and system logs; and vulnerability to adversarial attacks
that can evade detection. Overfitting remains a concern, particularly with limited training
data, which can negatively impact model generalization. Additionally, the lack of inter-
pretability in DL models makes it difficult to understand detection decisions, while the
complexity of securing IDS communication in IoT environments poses further risks [34].
This review focuses on improving model robustness against adversarial attacks, enhancing
interpretability, optimizing resource efficiency, integrating Intrusion Detection Systems
(IDSs) with other security tools such as firewalls, and leveraging blockchain technology for
secure collaborative learning. Addressing these gaps is crucial for advancing DTL-based
IDSs to ensure comprehensive, adaptive, and resilient cybersecurity solutions. RL- and
deep RL-based IDSs offer adaptive intrusion detection but face key challenges, including
limited interpretability, scalability issues, and the need for adaptability to dynamic threats.
Real-time efficiency, robustness against adversarial attacks, and security concerns in cloud
and IoT environments require further research. Energy efficiency is critical for unmanned
aerial vehicles and other resource-constrained settings. Additionally, IDSs must evolve for
edge and fog computing, handle imbalanced datasets, and reduce reliance on outdated or
simulated data to enhance resilience and effectiveness [35]. This research bridges the gap
by incorporating real-world examples, analyzing network threats, and evaluating datasets
tailored for IoT applications, enhancing the practical relevance and applicability of IDSs.
While DL excels at capturing complex temporal patterns, traditional ML techniques
can be highly effective for simpler patterns or when working with smaller datasets. Re-
gardless of the approach, robust feature engineering and domain expertise remain critical.
Ultimately, the choice between ML and DL methods should be informed by the complexity
of the data, resource constraints, and computational demands, with hybrid approaches
often providing an optimal balance by leveraging the strengths of both methodologies.
Table 1 presents a comprehensive literature review of intrusion detection approaches in
5G networks from 2020 to 2025. It summarizes methodologies, key findings, limitations,
and future scope, highlighting advancements and areas requiring further exploration in
the field.

Table 1. Related work on intrusion detection and datasets.

Reference Methodology

Findings

Limitations

Future Scope

[11]

Multi-stage ML-based NIDS;
oversampling to reduce
training size; compare feature
selection (info gain vs.
correlation); hyperparameter
tuning; evaluated on CICIDS
2017 and UNSW-NB 2015.

Training samples
were reduced by up
to 74%, features by
up to 50%, with a
detection accuracy of
over 99%, and a
1–2% improvement
over recent works.

Evaluation is limited
to specific datasets
due to potential
computational
overhead from
multi-stage
processing.

Real-time
implementation,
broader dataset testing,
integration of deep
learning, and further
optimization.

Mathematics 2025, 13, 1088

7 of 63

Table 1. Cont.

Reference Methodology

Findings

Limitations

Future Scope

[12]

[13]

[14]

[15]

[16]

[19]

[20]

[21]

Comparative study of ML
algorithms (LDA, CART, RF)
for IDSs across various
domains (fog computing, IoT,
big data, smart city, 5G) using
the KDD-CUP dataset.

Measured and
compared the
efficiency of different
ML models in
detecting intrusions.

Evaluation is limited
to the KDD-CUP
dataset and selected
ML algorithms; it
may not reflect the
latest advancements.

Review of ML algorithms (SL,
USL, SSL, RL DL) and their
applications in Industry 4.0.

ML is key for
processing diverse
digital data and
enabling smart
applications.

Broad scope; limited
empirical depth.

Extend research to
diverse, real-world
datasets; incorporate
advanced ML
techniques; explore
applicability to
emerging network
environments.

Encourage
domain-specific studies
and tackle
implementation
challenges.

Analyzed DoS/DDoS impact
on 5G slices; created a dataset
from a simulated testbed;
evaluated a bidirectional
LSTM (SliceSecure).

DoS/DDoS attacks
reduce
bandwidth/latency;
SliceSecure achieved
99.99% accuracy.

Based on simulation,
it may not reflect
real-world
conditions.

Collect real-world data;
further refine detection
models.

Proposed an ML-based IDS
integrated into the 5G core;
compared ML and DL
algorithms using the
CICIDS2017 and
CSE-CIC-IDS-2018 datasets.

Hybrid BiLSTM using wind
speed and climate indices;
three-stage feature selection
(partial
auto-/cross-correlation,
RReliefF, Boruta-RF) with
Bayesian tuning;
benchmarked vs. LSTM,
RNN, multilayer perceptron
(MLP), RF.

TL-enabled edge-CNN
framework pre-trained on an
image dataset and fine-tuned
with limited device data; joint
energy–latency optimization
via uploading decision and
bandwidth allocation.

GB achieved 99.3%
(secure) and 96.4%
(attack) accuracy.

Limited to two
benchmark datasets;
lacks real-world
implementation.

Extend to real-world 5G
environments; explore
additional datasets and
algorithms.

Achieved best
performance with
76.6–84.8% of errors
≤ 0.5 m/s; lowest
RMSE (9.6–23.8%)
and MAPE
(8.8–21.5%).

Not explicitly
discussed; may
require further
validation across
diverse conditions.

Real-time deployment;
broader testing across
various sites; scalability
and adaptability
improvements.

Achieved ~85%
prediction accuracy
with ~1% of model
parameters uploaded
(32× compression
ratio) on ImageNet.

Evaluation limited to
ImageNet; potential
challenges in
generalizing to
diverse industrial
scenarios.

Validate with real-world
industrial data; further,
optimize energy–latency
trade-offs; extend the
framework to other
applications and
datasets.

5G virtualization architecture;
TL-enhanced AdaBoost for
IoT classification;
sub-channel reuse for cellular
and D2D links.

High classification
accuracy; improved
spectrum utilization
through extensive
resource reuse.

DTL with weight transfer
and fine-tuning for network
traffic classification in 5G IoT
systems with scarce labeled
data.

With only 10%
labeled data,
accuracy nearly
matches that of
full-data training.

Focuses solely on
resource management;
limited real-world
validation.

Extend to real-world 5G
IoT scenarios; explore
further optimization
strategies.

Validation is limited
to specific datasets/
scenarios; computational
demands of
fine-tuning may be
challenging.

Extend to diverse 5G IoT
scenarios and reduce
human intervention in
model training.

Mathematics 2025, 13, 1088

8 of 63

Table 1. Cont.

Reference Methodology

Findings

Limitations

Future Scope

[23]

[24]

[25]

[27]

[30]

[31]

SDN-based network slicing
architecture for industrial IoT;
DDPG-based slice
optimization algorithm,
TL-based multiagent DDPG
for LoRa gateways.

Enhanced QoS,
energy efficiency,
and reliability;
accelerated training
process across
multiple gateways.

FortisEDoS: DTL with
CG-GRU (graph + RNN) for
EDoS detection in B5G
slicing.

Outperforms
baselines in detection
and efficiency.

Real-world
scalability and
deployment
challenges are not
fully addressed.

Validate in real
deployments; further,
optimize scalability and
integration with diverse
IoT environments.

Not explicitly
discussed: potential
real-world
deployment
challenges.

Validate in real-world
scenarios; enhance
scalability and
integration with other
security measures.

FortisEDoS: DTL with
CG-GRU (graph + RNN) for
EDoS detection in
cloud-native network slicing;
transfer learning adapts
detection across slices.

Fusion of statistical
importance (std. dev and diff.
of mean/median) for feature
selection in DNN-based IDSs;
evaluated on NSL-KDD,
UNSW_NB-15, and
CIC-IDS-2017.

Evaluated multiple ML
techniques using Weka on the
CICIDS2017 dataset;
compared full vs. reduced
attribute sets via CFS and
Zeek-based extraction.

Proposed three DL
models—early fusion, late
fusion, and late
ensemble—that use feature
fusion with fully connected
networks; evaluated on the
UNSW-NB15 and NSL-KDD
datasets.

CG-GRU achieves a
detection rate of over
92% with low
complexity; transfer
learning yields a
sensitivity of over
91% and speeds up
training by at least
61%; and it provides
explainable decisions.

Enhanced
performance
(accuracy, precision,
recall, F-score, and
FPR) and reduced
execution time;
improvements
statistically validated.

Tree-based methods
(PART, J48, RF)
achieved
near-perfect F1
scores (≈0.999 full,
0.990 with 6 CFS
attributes, 0.997 with
14 Zeek attributes)
with fast execution.

Late-fusion and
late-ensemble
models outperform
early-fusion and
state-of-the-art
methods with
improved
generalization and
robustness against
class imbalance.

False alarms may
trigger unnecessary
scaling, impacting
SLAs; they lack a
mitigation strategy.

Develop intelligent
resource provisioning,
improve VNFs/slice
selection, and integrate
FL for
privacy-preserving
mitigation.

It may be
dataset-specific;
further validation is
required across
diverse
environments.

Extend to
online/real-time IDSs
and explore integration
with other DL
architectures.

Findings are based
solely on
CICIDS2017,
potential dataset
dependency.

Analyze other Zeek logs
for additional features
and validate models on
different IDS datasets.

Limited exploration
of long-term
dependencies and
explainability.

Explore fusion with
recurrent units for long
dependencies and
integrate post hoc
Explainable AI (XAI) to
clarify attribute
contributions.

Mathematics 2025, 13, 1088

9 of 63

Table 1. Cont.

Reference Methodology

Findings

Limitations

Future Scope

[32]

[33]

[34]

[35]

Proposed the TestCloudIDS
dataset (15 DDoS variants, 76
features) and SparkShield—a
Spark-based IDS—evaluated
on UNSW-NB15, NSL-KDD,
CICIDS2017, and
TestCloudIDS.

Orchestrator in RAN via
Docker that dynamically
selects adaptive ML/DL
models (DT, RF, MLP, LLM)
trained on CICDDOS2019 for
real-time security.

Reviewed post-2015 studies
on DTL-based IDSs,
analyzing datasets,
techniques, and evaluation
metrics.

Achieves improved
threat classification
with recent attack
patterns; highlights
the inadequacy of
older datasets for
zero-day attacks.

Enhanced dynamic
security with
effective real-time
attack detection/
mitigation and
improved accuracy
via UE feedback.

DTL improves IDSs
by transferring
knowledge,
addressing data
scarcity, and
enhancing
performance.

Based on simulated
cloud data,
real-world validation
is needed.

Expand dataset, test on
real networks, and
update with evolving
attack strategies.

Evaluated in a
simulated
environment; lacks
real-world testbed
validation.

Integrate into a real 5G
testbed; refine model
selection and mitigation
strategies.

Requires labeled
data, faces
overfitting,
adversarial attacks,
and high complexity.

Improve robustness,
efficiency, and security
with
adversarial-resistant
models and blockchain
integration.

A comprehensive review of
deep RL-based IDSs,
categorizing studies,
analyzing datasets,
techniques, and evaluation
metrics.

Deep RL improves
IDS accuracy,
adaptability, and
decision-making
across IoT, ICSs, and
smart grids.

Scalability issues,
real-time constraints,
adversarial
vulnerabilities, and
dataset limitations.

Enhance interpretability,
robustness, energy
efficiency, and
integration with
edge/fog computing.

3. Fifth-Generation Core Network and Security Challenges

Fifth-generation technology constitutes a revolutionary advancement in telecommuni-
cations, delivering unparalleled improvements in data throughput, latency reduction, and
device connectivity [36]. Enhanced Mobile Broadband (eMBB) provides ultra-high data
rates, enabling bandwidth-intensive applications such as high-definition video streaming
and augmented and virtual reality. Ultra-reliable low-latency communication (uRLLC)
offers minimal delays and exceptional reliability, making it indispensable for mission-
critical applications, including autonomous vehicles, industrial automation, and remote
surgeries. Additionally, Massive Machine-Type Communications (mMTCs) support the
seamless connectivity of many devices, paving the way for smart cities and expansive IoT
deployments. These three capabilities—often referred to as the “5G Triangle”—form the
foundational pillars of 5G networks, as illustrated in Figure 1.

To realize these advanced capabilities, 5G leverages two primary architectural frame-
works. The Non-Standalone (NSA) architecture augments the existing 4G LTE infrastruc-
ture with new radio technology, enabling enhanced data rates and rapid, cost-effective
deployments. However, the NSA does not support critical features such as uRLLC and
mMTC, which limits its applicability in scenarios requiring ultra-low latency and massive
connectivity. In contrast, the Standalone (SA) architecture establishes an independent 5G
CN based on a Service-Based Architecture (SBA), modularizing network functions and
facilitating communication via the HTTP/2 protocol. This design enhances scalability and
flexibility, supporting advanced functionalities such as network slicing, which unlocks the
full potential of 5G for next-generation applications, including edge computing, IIoT, and

Mathematics 2025, 13, 1088

10 of 63

autonomous transportation [37]. Figure 2 illustrates the SA architecture, which operates
independently of the 4G network and is built on an SBA. It modularized network functions,
enabling efficient communication via the HTTP/2 protocol while supporting advanced
capabilities such as network slicing.

Figure 1. Fifth-generation services: powering the next generation of connectivity.

Figure 2. Service-Based Architecture (SBA) of the 5G-SA-CN.

While 5G offers transformative benefits, it also introduces significant security chal-
lenges. The extensive number of connected devices, including IoT and industrial systems,
dramatically expands the potential attack surface. Many of these devices lack robust secu-
rity measures, making them vulnerable to cyberattacks, including unauthorized access and

Mathematics 2025, 13, x FOR PEER REVIEW 9 of 60   techniques, and evaluation metrics. addressing data scarcity, and enhancing perfor-mance. adversarial attacks, and high complexity. models and blockchain in-tegration. [35] A comprehensive review of deep RL-based IDSs, categorizing studies, ana-lyzing datasets, tech-niques, and evaluation metrics. Deep RL improves IDS ac-curacy, adaptability, and decision-making across IoT, ICSs, and smart grids. Scalability issues, real-time constraints, adversarial vulnerabilities, and dataset limitations. Enhance interpretability, robustness, energy effi-ciency, and integration with edge/fog computing. 3. Fifth-Generation Core Network and Security Challenges Fifth-generation technology constitutes a revolutionary advancement in telecommuni-cations, delivering unparalleled improvements in data throughput, latency reduction, and device connectivity [36]. Enhanced Mobile Broadband (eMBB) provides ultra-high data rates, enabling bandwidth-intensive applications such as high-definition video streaming and augmented and virtual reality. Ultra-reliable low-latency communication (uRLLC) of-fers minimal delays and exceptional reliability, making it indispensable for mission-critical applications, including autonomous vehicles, industrial automation, and remote surgeries. Additionally, Massive Machine-Type Communications (mMTCs) support the seamless con-nectivity of many devices, paving the way for smart cities and expansive IoT deployments. These three capabilities—often referred to as the “5G Triangle”—form the foundational pil-lars of 5G networks, as illustrated in Figure 1.  Figure 1. Fifth-generation services: powering the next generation of connectivity. To realize these advanced capabilities, 5G leverages two primary architectural frame-works. The Non-Standalone (NSA) architecture augments the existing 4G LTE infrastruc-ture with new radio technology, enabling enhanced data rates and rapid, cost-effective de-ployments. However, the NSA does not support critical features such as uRLLC and mMTC, Mathematics 2025, 13, x FOR PEER REVIEW 10 of 60   which limits its applicability in scenarios requiring ultra-low latency and massive connec-tivity. In contrast, the Standalone (SA) architecture establishes an independent 5G CN based on a Service-Based Architecture (SBA), modularizing network functions and facilitating communication via the HTTP/2 protocol. This design enhances scalability and flexibility, supporting advanced functionalities such as network slicing, which unlocks the full poten-tial of 5G for next-generation applications, including edge computing, IIoT, and autono-mous transportation [37]. Figure 2 illustrates the SA architecture, which operates inde-pendently of the 4G network and is built on an SBA. It modularized network functions, enabling efficient communication via the HTTP/2 protocol while supporting advanced ca-pabilities such as network slicing.  Figure 2. Service-Based Architecture (SBA) of the 5G-SA-CN. While 5G offers transformative benefits, it also introduces significant security challenges. The extensive number of connected devices, including IoT and industrial systems, dramati-cally expands the potential attack surface. Many of these devices lack robust security measures, making them vulnerable to cyberattacks, including unauthorized access and mal-ware infections. Additionally, the network slicing feature, which enables the creation of virtual networks for specific applications (e.g., healthcare and autonomous driving), introduces fur-ther risks. A compromised slice could expose other slices or the underlying physical infra-structure to cross-slice attacks [38]. This vulnerability highlights the importance of developing advanced threat models, AI-based security systems [39], and blockchain solutions [40]. The scalability inherent in 5G also elevates the risk of large-scale DDoS attacks. With billions of connected devices, attackers can exploit botnets to overwhelm critical network functions such as the Access and Mobility Management Function (AMF) and the User Plane Function (UPF). This threat is particularly concerning for uRLLC systems used in healthcare and smart grids, where even minor disruptions can have significant consequences. To miti-gate these risks, specialized IDSs [41] and strategies to secure the control plane have been developed [42]. Moreover, virtualized network technologies, such as Software-Defined Networking (SDN) and Network Function Virtualization (NFV), introduce vulnerabilities in centralized control planes and management protocols, necessitating stronger security measures. The shift toward edge computing further complicates the security landscape by decentralizing data processing closer to end users, making edge nodes more susceptible to attacks. Often Mathematics 2025, 13, 1088

11 of 63

malware infections. Additionally, the network slicing feature, which enables the creation
of virtual networks for specific applications (e.g., healthcare and autonomous driving),
introduces further risks. A compromised slice could expose other slices or the underly-
ing physical infrastructure to cross-slice attacks [38]. This vulnerability highlights the
importance of developing advanced threat models, AI-based security systems [39], and
blockchain solutions [40].

The scalability inherent in 5G also elevates the risk of large-scale DDoS attacks. With
billions of connected devices, attackers can exploit botnets to overwhelm critical network
functions such as the Access and Mobility Management Function (AMF) and the User Plane
Function (UPF). This threat is particularly concerning for uRLLC systems used in healthcare
and smart grids, where even minor disruptions can have significant consequences. To
mitigate these risks, specialized IDSs [41] and strategies to secure the control plane have
been developed [42].

Moreover, virtualized network technologies, such as Software-Defined Networking
(SDN) and Network Function Virtualization (NFV), introduce vulnerabilities in centralized
control planes and management protocols, necessitating stronger security measures. The
shift toward edge computing further complicates the security landscape by decentralizing
data processing closer to end users, making edge nodes more susceptible to attacks. Often
lacking the robust security infrastructure of centralized cloud systems, these nodes are
more complex to monitor and protect. To counter these challenges, adopting zero-trust ar-
chitectures, end-to-end encryption, and robust authentication mechanisms is essential [43].
The research [44] examines the role of machine learning (ML) and deep learning (DL) in
enhancing 5G network security by addressing emerging vulnerabilities and threat vectors.
It analyzes key 5G components, including network slicing, mMTC, and edge computing,
alongside current security protocols and regulations. The study highlights how machine
learning (ML) and deep learning (DL) enhance anomaly detection, predictive security, and
intrusion prevention, providing insights into the future of 5G security. Table 2 outlines
suggested machine learning (ML) and trust and loyalty (TL) techniques for 5G networks,
along with dedicated attack types for each 5G domain (eMBB, mMTC, and uRLLC).

Table 2. ML and TL techniques for 5G security [44].

5G Domain

ML Techniques

TL Techniques

Dedicated Attack Types

eMBB

DL, NN, SVM, KNN

TL for anomaly detection,
TL for intrusion detection

DoS, eavesdropping, Man-in-the-Middle
(MitM), data injection

mMTC

RF, DT, RL

TL for device
authentication, TL for
network traffic analysis

Device impersonation, jamming attacks,
data injection, Sybil attacks

uRLLC

CNN, RNN, Autoencoders
(AEs)

TL for signal processing,
TL for traffic prediction

Timing attacks, spoofing, signal
interference, jamming

3.1. Intrusion Detection in 5G Networks

To enhance cybersecurity in 5G networks, the authors [45] propose integrating a ded-
icated cybersecurity module at each 5G station, which functions as an additional server
housing both a firewall and an Intrusion Detection System (IDS). The research highlights
the vulnerability of 5G cellular networks to Probe, DoS, and software-based attacks, ne-
cessitating robust intrusion detection mechanisms. The initial IDS framework utilizes
machine learning (ML) algorithms trained on multiple datasets, including the widely used
KDD99 dataset, which is crucial for academic research and prototype development. KDD99
originates from the DARPA’98 IDS evaluation program and comprises approximately
5 million labeled samples spanning four primary attack categories: DoS, Remote-to-Local

Mathematics 2025, 13, 1088

12 of 63

(R2L), User-to-Root (U2R), and Probe attacks. These categories are further divided into spe-
cific attack types, including APACHE2, NEPTUNE, BUFFER_OVERFLOW, and IPSWEEP,
among others.

To enhance detection accuracy, the IDS is further trained on two additional datasets—
DOS1 (380 MB) and DOS2 (85 MB)—containing modern DoS/DDoS attack patterns, includ-
ing LDAP, MSSQL, NetBIOS, and UDPStorm. The data partitioning strategy assigns 90%
of KDD99 for training and 10% for testing, while the DOS1 and DOS2 datasets are split
into 80% training and 20% testing. This methodology yields high accuracy rates: 96.1%
(KDD99), 99.38% (DOS1), and 99.99% (DOS2). The IDS first analyzes incoming network
traffic against KDD99 attack signatures, and then sequentially verifies it against DOS1 and
DOS2 attack patterns. If no threat is detected, the IDS classifies the traffic as benign.

The system was experimentally validated using a laboratory setup comprising
60 Raspberry Pi devices and a central server hosting the IPS module. Attack vectors were
simulated, and the IDS successfully identified key cyber threats, including DoS, R2L, U2R,
and Probe attacks, proving its effectiveness as a prototype for real-world 5G intrusion
detection. Unlike existing approaches that rely solely on KDD99, the system integrates
multiple datasets to enhance detection capabilities. Future improvements will focus on
refining real-time detection, developing complex attack vectors, and creating a proprietary
dataset for extensive testing and optimization. Ultimately, the system aims to be deployed
in real-world 5G networks, reinforcing their security posture against evolving cyber threats.
Figure 3 illustrates the security infrastructure that safeguards 5G networks against cyber-
attacks. The system features specialized 5G station-based cybersecurity components that
incorporate both a firewall and an Intrusion Detection System (IDS). Network traffic passes
through a sequential set of dataset analyses, which improves threat discovery and protects
against attacks.

Figure 3. Security framework module.

IDSs play a crucial role in securing 5G networks, particularly within SDN security
frameworks. The virtualization of key components, such as the RAN and the CN, enables

Mathematics 2025, 13, x FOR PEER REVIEW 12 of 60   reinforcing their security posture against evolving cyber threats. Figure 3 illustrates the secu-rity infrastructure that safeguards 5G networks against cyberattacks. The system features spe-cialized 5G station-based cybersecurity components that incorporate both a firewall and an Intrusion Detection System (IDS). Network traffic passes through a sequential set of dataset analyses, which improves threat discovery and protects against attacks.  Figure 3. Security framework module. IDSs play a crucial role in securing 5G networks, particularly within SDN security frameworks. The virtualization of key components, such as the RAN and the CN, enables the automation of security mechanisms, allowing copies of network traffic to be analyzed without affecting performance. However, a critical challenge in developing ML- and DL-based IDSs for SDN lies in selecting the dataset. Existing research highlights the scarcity of comprehensive datasets that accurately represent real-world cyber threats. Many available datasets, such as CICIDS2017 and CSE-CIC-IDS2018, include labeled instances of both at-tack and benign traffic; however, they suffer from limitations, including outdated attack patterns and insufficient diversity in traffic types. Several studies have compared the effectiveness of ML and DL models for IDSs in 5G networks. LR, RF, GB, AE, and DNNs were evaluated in [46] using benchmark datasets. Among these models, GB demonstrated superior performance, achieving 99.3% accuracy for secure traffic and 96.4% for attack detection. AE-based anomaly detection, on the other hand, underperformed, highlighting the limitations of USL in this domain. The study also emphasized the importance of dataset integrity, privacy concerns, and realistic attack sim-ulations to improve model robustness. As technology advances, IDS solutions must evolve to address emerging threats. Future network generations, such as 6G and beyond, are ex-pected to integrate more sophisticated AI-driven security mechanisms. An overview of se-curity vulnerabilities, along with corresponding mitigation methods and related challenges for 5G networks, is presented in Table 3. Security threats and risks within the protocol stack components are classified based on which defense mechanisms are provided. In addition to Mathematics 2025, 13, 1088

13 of 63

the automation of security mechanisms, allowing copies of network traffic to be analyzed
without affecting performance. However, a critical challenge in developing ML- and DL-
based IDSs for SDN lies in selecting the dataset. Existing research highlights the scarcity of
comprehensive datasets that accurately represent real-world cyber threats. Many available
datasets, such as CICIDS2017 and CSE-CIC-IDS2018, include labeled instances of both
attack and benign traffic; however, they suffer from limitations, including outdated attack
patterns and insufficient diversity in traffic types.

Several studies have compared the effectiveness of ML and DL models for IDSs in 5G
networks. LR, RF, GB, AE, and DNNs were evaluated in [46] using benchmark datasets.
Among these models, GB demonstrated superior performance, achieving 99.3% accuracy
for secure traffic and 96.4% for attack detection. AE-based anomaly detection, on the
other hand, underperformed, highlighting the limitations of USL in this domain. The
study also emphasized the importance of dataset integrity, privacy concerns, and realistic
attack simulations to improve model robustness. As technology advances, IDS solutions
must evolve to address emerging threats. Future network generations, such as 6G and
beyond, are expected to integrate more sophisticated AI-driven security mechanisms. An
overview of security vulnerabilities, along with corresponding mitigation methods and
related challenges for 5G networks, is presented in Table 3. Security threats and risks
within the protocol stack components are classified based on which defense mechanisms
are provided. In addition to this overview, a breakdown of essential security challenges
and upcoming considerations is provided, targeting various layers of the OSI model in
5G networks.

Table 3. Security vulnerabilities, mitigation strategies, and challenges in 5G across OSI layers [47].

Layer Name

Protocol Stack
Component

Security Threats and
Risks

Mitigation Strategies

1 Physical

Wireless signal
and hardware
layer

(i) Power leakage
leading to private data
exposure.
(ii) Eavesdropping on
wireless signals.
(iii) Injection of
fabricated data.
(iv) Malicious signal
amplification.
(v) Expansion of attack
surface due to increased
network entry points.

(i, ii) Power control,
beamforming, and clustering.
(ii, iii) Defining secure
transmission zones,
implementing device-to-device
(D2D) communication, and
leveraging Physical Layer
Security (PLS).
(iv) No established
countermeasure identified.
(v) Continuous network
monitoring and threat
detection.

2 Data link

Frame
transmission
and MAC
security

(i) Exploitation of IEEE
802.1 security gaps
using penetration tools.

(i) Firmware updates, ML-IDS,
and Received Signal Strength
(RSS)-based security
mechanisms.

Challenges and
Future
Considerations

(i–v) Prioritizing
research on the most
secure 5G physical
layer technologies
over fragmented
explorations of
multiple approaches.

(i) Firmware updates
may not always be
feasible; ML-IDS has
training inefficiencies,
and RSS-based
methods are
susceptible to noise
interference.

Mathematics 2025, 13, 1088

14 of 63

Table 3. Cont.

Layer Name

Protocol Stack
Component

Security Threats and
Risks

Mitigation Strategies

3 Network

Routing and
packet
forwarding

4 Transport

Data flow
control and
transmission
security

5 Session

Authentication
and connection
management

6 Presentation

Data formatting
and encoding

7 Application

User interface
and data
services

(i) Breaches in data
confidentiality and
integrity, along with
susceptibility to replay
attacks.

(i) DoS attacks,
unauthorized rule
modifications, and
insertion of malicious
policies in SDN
controllers.

(i) Session hijacking and
interception via
plaintext credentials.
(ii) Security flaws in
NetBIOS leading to
unauthorized resource
sharing.
(iii) Identity exposure,
DoS, and interception
vulnerabilities in
authentication protocols.

(i) Concealing malicious
payloads using
multimedia files.
(ii) Buffer overflow due
to insufficient input
validation.
(iii) Format string
vulnerabilities leading
to unauthorized code
execution.

(i) Distributed Denial of
Service (DDoS) targeting
blockchain-based
protocols.
(ii) Exploitation of
transaction malleability
in blockchain systems.
(iii) Manipulated
multimedia uploads in
connected vehicles.

Challenges and
Future
Considerations

(i) IPsec alone does
not ensure complete
end-to-end security,
requiring additional
protection across
other layers.

(i) A practical,
widely accepted
SDN security
framework remains
underdeveloped.

(i, ii) Additional
research required to
address session-layer
security risks.
(iii) Enhanced
authentication
methods improve
security but introduce
computational
overhead.

(i) Encryption via IPsec.

(i) Embedding security
measures within the SDN
architecture.

(i) No universal
countermeasure established.
(ii) Disabling null sessions and
enforcing strong admin
credentials.
(iii) Adoption of improved
authentication protocols.

(i, ii, iii) Implementing
rigorous input/output
validation at the application
and presentation layers.
(i, ii, iii) Periodic updates to
cryptographic algorithms.

(i, ii, iii) Limited
recent research on
presentation layer
security mechanisms.

(i) No widely adopted
countermeasure.
(ii) Implementation of
Segregated Witness (SegWit).
(iii) Use of public key
cryptography.

(i, ii) Blockchain
scalability challenges
necessitate stronger
cryptographic
functions.
(iii) The effectiveness
of proposed defenses
needs further
evaluation.

3.2. Existing Datasets for 5G Network Security

Recent advancements in 5G network security have yielded several innovative tools

and datasets for evaluating and enhancing the resilience of next-generation networks.

For instance, an open-source 5G network traffic fuzzer presented in [48] enables
systematic assessment of 5G components by replaying and modifying traffic patterns. This
tool dynamically generates and transmits traffic to critical network nodes—including the

Mathematics 2025, 13, 1088

15 of 63

AMF, UPF, and RAN—and supports dynamic and static packet alterations across control
and data planes. It integrates a federated learning framework to detect DDoS attacks
targeting the General Packet Radio Service Tunneling Protocol (GTP-U) within the 5G CN.
The underlying model is trained and evaluated using the UNSW-NB15 dataset within a
simulated environment, demonstrating its potential effectiveness.

In another significant contribution, ref. [49] introduces the 5GC Packet Forwarding
Control Protocol (PFCP) Intrusion Detection Dataset, tailored to identify cyberattacks on
the N4 interface’s PFCP between the Session Management Function (SMF) and the UPF.
This labeled dataset, which encompasses comprehensive network flow statistics across four
distinct PFCP attack types, is publicly available on IEEE Dataport and Zenodo.

Moreover, the study in [50] leverages the Free5GC and UERANSIM tools to generate a
dataset featuring ten attack scenarios that span reconnaissance, DoS, and network reconfig-
uration categories. The reconnaissance attacks exploit vulnerabilities in Free5GS, such as
impersonation and inadequate input validation. DoS scenarios simulate events such as net-
work repository function crashes and fabricated AMF deletions, while network reconfigu-
ration scenarios mimic fake AMF insertions and modifications. The data were standardized
to 1024 bytes, omitting layers above the application layer to ensure consistency.

A common limitation among these datasets is their reliance on trace-based data, which
requires significant processing, storage, and real-time trace collection—factors that can
delay anomaly detection. To address this challenge, a novel framework is proposed that
leverages aggregated log counters and performance metrics from CN nodes. This approach
incurs minimal overhead and enables real-time intrusion detection, making it a practical
solution for 5G networks.

Table 4 provides a comprehensive overview of the datasets employed in 5G net-
work security research, detailing their descriptions, limitations, and potential avenues
for future work. Collectively, these datasets address critical aspects of 5G security, in-
cluding intrusion detection, traffic anomaly analysis, IoT security, attack simulation, and
authentication vulnerabilities.

Table 4. Existing datasets for 5G network security.

Reference Methodology

Findings

Limitations

Future Scope

[19]

[22]

5G Threat
Intelligence
Dataset

Network traffic logs for 5G
services, including various
attack scenarios such as
DoS and DDoS attacks, are
labeled for enhanced threat
detection.

It may not represent all
attack vectors; it is limited
to specific scenarios and
conditions.

Integration of dynamic
attack patterns,
multivector attacks, and
more complex network
environments.

Green Computing
in 5G IoT

Uses improved AdaBoost
and resource reuse for data
classification and
optimization in 5G IoT.

Limited to resource
management, lacks
real-world validation, and
ignores energy trade-offs.

Focus on real-world
testing, energy-efficient
methods, and scalability
in IoT.

[48]

5greplay Tool

A tool designed to fuzz 5G
network traffic by injecting
attack vectors to assess
network vulnerabilities.

Limited to specific attack
scenarios and experimental
setups; lacks extensive
real-world evaluation.

Extend to real-world 5G
IoT scenarios; explore
further optimization
strategies.

Mathematics 2025, 13, 1088

16 of 63

Table 4. Cont.

Reference Methodology

Findings

Limitations

Future Scope

[49]

[50]

[51]

[52]

[53]

[54]

5GC PFCP
Intrusion
Detection Dataset

ML-5G attack
detection

5G Traffic
Anomaly
Detection Dataset

5G Cybersecurity
Dataset

5G IoT Security
Dataset

SPEC5G: A
Dataset for 5G
Cellular Network
Protocol Analysis

[55]

Secure SemCom
Dataset

Labeled dataset for
AI-powered intrusion
detection in 5G CNs,
focusing on PFCP-based
cyberattacks. Includes pcap
files and TCP/IP flow
statistics.

Uses ML on programmable
logic to detect 5G network
attacks in real time.

Includes traffic anomalies
in 5G networks, such as
attacks exploiting
vulnerabilities like
eavesdropping or DDoS.

The dataset contains normal
and attack traffic patterns
for 5G networks, including
DoS and DDoS attacks.

Focuses on IoT devices in
5G networks, including
malicious activities such as
botnets and unauthorized
access.

The dataset contains
3,547,586 sentences with
134M words from 13,094
cellular network
specifications and 13 online
websites to automate 5G
protocol analysis.

Explores security
challenges in SemCom and
mitigation techniques like
adversarial training,
cryptography, and
blockchain.

Limited to four PFCP
attack types, may not
cover all real-world
threats, and requires
further validation in
diverse 5G environments.

Expansion to more
attack scenarios,
real-world validation,
and integration with
advanced AI-based
security frameworks.

Limited evaluation on a
specific hardware
platform; experimental
setup only.

It may not capture all
attack vectors or reflect
the latest network
architecture changes.

Limited to a specific set of
attack types and lacks
real-world data diversity.

Primarily simulated data,
which may not fully
capture real-world IoT
device behavior.

Limited to textual data;
may not cover all aspects
of 5G network security.

High computational
complexity of encryption
and differential privacy,
evolving backdoor attacks,
challenges in smart
contract integration, and
vulnerability to semantic
adversarial attacks.

Broader real-world
testing, extended attack
scenarios, and deeper
5G integration.

Addition of new
vulnerabilities and real-
time traffic anomalies,
especially for emerging
5G applications.

Expansion to include
more diverse attack
types, real-world traffic,
and cross-domain
scenario.

Collection of real-world
IoT device data, more
diverse attack patterns,
and adaptive models for
evolving threats.

Integration with other
data types, such as
network traffic data, to
provide a more
comprehensive analysis.

Development of
dynamic data cleaning,
optimized encryption
techniques,
multi-strategy backdoor
defense, smart contract-
enabled SemCom, and
robust countermeasures
using semantic
fingerprints.

4. Intrusion Detection Systems (IDSs) and Classification

Intrusion involves unauthorized activities that threaten the confidentiality, integrity,
or availability of information systems. Any action that disrupts computer services or
poses a security risk qualifies as an intrusion. IDSs are essential tools for identifying
and responding to these threats. IDSs can be implemented as software or hardware to
detect malicious activities that traditional firewalls might overlook [51]. IDS solutions are
classified into Signature-Based Intrusion Detection Systems (SIDSs) and Anomaly-Based

Mathematics 2025, 13, 1088

17 of 63

Intrusion Detection Systems (AIDSs). SIDSs identify known attack patterns by matching
database signatures using knowledge-based or misuse detection methods [52].

Tools like Snort and NetSTAT use SIDS techniques by inspecting network packets and
comparing them to signature databases. SIDSs excel at accurately detecting previously
known intrusions but struggle with zero-day threats due to the absence of matching sig-
natures. Advanced techniques, such as state machines or formal language patterns, are
required to create effective signatures for modern malware. In contrast, AI detects devi-
ations from normal system behavior using machine learning, statistical, and knowledge-
based methods. The development process involves a training phase to learn normal traffic
patterns and a testing phase to identify anomalies in new data. AIDSs effectively de-
tect zero-day attacks and internal threats, such as unauthorized transactions. However,
they are more prone to FPs since new legitimate activities may be flagged as intrusions.
AIDSs can be divided into five subclasses based on their training methods: statistics-based,
pattern-based, rule-based, state-based, and heuristic-based systems [56].

While SIDSs can only detect known intrusions, AIDSs can identify novel threats,
including zero-day attacks. The strength of AIDSs lies in their anomaly detection capability,
though they suffer from increased FPs. IDS solutions are also categorized by deployment
type: Host-Based IDS (HIDS), Network-Based IDS (NIDS), Wireless-Based IDS (WIDS),
Network Behavior Analysis (NBA), and Mixed IDS (MIDS). HIDS monitors sensitive host
activities, NIDS analyzes specific network segments, and WIDS focuses on wireless traffic.
The NBA detects anomalies in traffic flows, while MIDS integrates multiple technologies
to enhance the accuracy of detection. Key IDS components include sensors for NIDS,
WIDS, and NBA, as well as agents for HIDS, which relay data to a management server for
processing and a database server for storage. IDSs can be deployed in managed networks,
which are isolated and secure but costly, or standard networks, which are public but can be
protected using virtual LANs.

Detection methods include anomaly-based, signature-based, and specification-based
techniques. IDSs may operate in real-time or offline, with passive (alert-only) or active
(preventive) responses. Despite their robustness, IDSs face challenges in accuracy due to
False Positives (FPs) and False Negatives (FNs), with security administrators prioritizing
the reduction in FNs to avoid undetected threats [57]. As cyber threats continue to evolve,
combining SIDS and AIS techniques may offer a comprehensive defense strategy for mod-
ern cybersecurity challenges. Table 5 compares IDS technologies—HIDS, NIDS, WIDS, and
NBA—based on components, architecture, strengths, limitations, and detection capabilities.
It highlights their unique roles in monitoring, analyzing, and securing network and host
activities against threats.

Table 5. Comparative analysis of different IDS technologies [56].

Feature

HIDS

NIDS

WIDS

NBA

Components

Agent (software,
inline); MS: 1 − n; DS:
optional

Sensor
(inline/passive); MS: 1
− n; DS: optional

Sensor (passive); MS: 1
− n; DS: optional

Detection scope

Single host

Network segment or
subnet

WLAN, wireless
clients

Sensor (mostly
passive); MS: 1 − n
(optional); DS:
optional

Network subnets
and hosts

Architecture

Managed or standard
network

Managed network

Managed or standard
network

Managed or
standard network

Mathematics 2025, 13, 1088

18 of 63

Table 5. Cont.

Feature

HIDS

NIDS

WIDS

NBA

Technology
limitations

Security
capabilities

Strengths

Effective in analyzing
encrypted end-to-end
communications

A broad analysis of AP
protocols

Accuracy challenges
due to lack of context,
host resource usage,
security conflicts

Cannot detect wireless
protocols; delays in
reporting; prone to
false positives/
negatives.

High accuracy due to
narrow focus;
uniquely monitors
wireless protocols

Excellent for
detecting
reconnaissance scans,
malware infections,
and DoS attacks

Vulnerable to physical
jamming; limited
security for wireless
protocols

Batch processing
causes delays in
detection; lacks
real-time monitoring

Monitors system calls,
file system activities,
and traffic

Monitors hosts, OS,
APs, and network
traffic

Tracks wireless
protocol activities and
devices

Inspects host services
and protocol traffic
(IP, TCP, UDP)

Detection
methodology

Combination of
signature and anomaly
detection

Primarily signature
detection, with
anomaly and
specification-based
detection

Predominantly
anomaly detection,
supplemented by
signature- and
specification-based
methods

Major reliance on
anomaly detection,
incorporating
specification-based
methods

4.1. NIDS for IoT

The rapid adoption of IoT technologies has presented significant security challenges,
driving the need for intelligent Network Intrusion Detection Systems (NIDSs). Lever-
aging ML and big data analytics has proven essential for processing and analyzing the
large volumes of unstructured data generated by web applications [58]. ML techniques,
including SL and USL, effectively detect hidden patterns and novel threats and reduce
FPs in network traffic. DL, a subset of ML utilizing ANNs, has shown immense potential
for NIDS applications by continuously learning from traffic patterns to counter zero-day
and evolving threats. It discusses the role of learning techniques in IoT-based Network
Intrusion Detection Systems (NIDSs), emphasizing how big data, machine learning (ML),
and deep learning (DL) enhance cybersecurity. Unlike traditional signature-based NIDS,
ML-based systems require fewer updates and efficiently detect zero-day attacks. This study
highlights the advantages of ML-driven NIDSs, including their classification capabilities
and potential to strengthen IoT security against emerging cyber threats. Figure 4 illustrates
the machine learning (ML) techniques and classification in Network Intrusion Detection
Systems (NIDSs) for IoT security.

Figure 4. ML techniques.

Mathematics 2025, 13, x FOR PEER REVIEW 18 of 60    Figure 4. ML techniques. Table 6. Overview of NIDS-IoT with accuracy and techniques. References  Method Description Accuracy [59] Sparse Convolutional Net-work Intrusion classification with evolutionary tech-niques using an IGA-BP autoencoder model in MATLAB 98.98% [59] Stacked Autoencoder (SAE) DL-based NIDS for IEEE 802.11 networks 98.66% [60] IP Flow-based IDS Real-time intrusion detection using flow features, outperforming Snort and Zeek Near perfect [61] Fog-layer NIDS Hybrid DNN-kNN model for low resource con-sumption 99.77% (NSL-KDD), 99.85% (CICIDS2017) [62] Efficient IDS using GWO and PSO Feature selection using Grey Wolf and Particle Swarm Optimization with RF 99.66% [63] DL-based NIDS Concatenated CNN models (VGG16, VGG19, Xception) for intrusion detection 96.23% (UNSW-NB15), 99.26% (CIC DDoS 2019) [64] Multilayer DL-NIDS Two-stage detection process achieving improved results G-mean: 78% [65] Hybrid CNN-LSTM System Device-level phishing and cloud-based botnet detection >94% [66] Neural Networks (NNs) Opti-mized by Genetic Algorithms Enhanced fog computing-based intrusion detec-tion with reduced execution time Not specified [67] ML-based Framework (SVM, GBM, RF) RF for detecting malicious traffic in the NSL-KDD dataset 85.34% [68] Two-level Anomaly Detection Abnormal traffic detection using DT and RF 99.9% 4.2. Comparison of Open-Source NIDSs Several free, open-source NIDSs are available for sniffing, analyzing, and detecting ma-licious network traffic. Among the most widely used is Snort, a lightweight, single-threaded SIDS that supports multiple operating systems. It operates in three modes: Sniffer, Packet Log-ger, and NIDS modes, enabling real-time traffic analysis and attack detection, including SQL injection and cross-site scripting attacks. Suricata is another popular tool, offering a multi-threaded architecture for high-performance intrusion detection and prevention, as well as net-work security monitoring and offline packet processing. The Open Information Security Foun-dation developed Suricata and supports various operating systems, making it a flexible and scalable solution. Bro-IDS (now known as Zeek) is a network analysis framework that com-bines SIDS and IDS. It inspects network traffic at a higher level and supports multiple appli-cation-layer protocols, including DNS, FTP, HTTP, and SMTP. Meanwhile, Kismet is a wire-less IDS that detects and sniffs Wi-Fi networks, Bluetooth devices, and sensors. It runs on Linux, BSD, Android, and Windows (with hardware restrictions) and remains undetectable [59]

[59]

[60]

[61]

[62]

[63]

[64]

[65]

[66]

[67]

[68]

Mathematics 2025, 13, 1088

19 of 63

Table 6 demonstrates various effective ML- and DL-based NIDS solutions. These ad-
vancements underscore the crucial role of ML and DL techniques in bolstering the security
of IoT networks and providing scalable and efficient solutions for evolving cyber threats.

Table 6. Overview of NIDS-IoT with accuracy and techniques.

References Method

Description

Sparse Convolutional
Network

Intrusion classification with evolutionary
techniques using an IGA-BP autoencoder
model in MATLAB

Accuracy

98.98%

Stacked Autoencoder (SAE) DL-based NIDS for IEEE 802.11 networks

98.66%

IP Flow-based IDS

Fog-layer NIDS

Real-time intrusion detection using flow
features, outperforming Snort and Zeek

Near perfect

Hybrid DNN-kNN model for low
resource consumption

99.77% (NSL-KDD), 99.85%
(CICIDS2017)

Efficient IDS using GWO
and PSO

Feature selection using Grey Wolf and
Particle Swarm Optimization with RF

99.66%

DL-based NIDS

Multilayer DL-NIDS

Concatenated CNN models (VGG16,
VGG19, Xception) for intrusion detection

96.23% (UNSW-NB15),
99.26% (CIC DDoS 2019)

Two-stage detection process achieving
improved results

G-mean: 78%

Hybrid CNN-LSTM System

Device-level phishing and cloud-based
botnet detection

>94%

Neural Networks (NNs)
Optimized by Genetic
Algorithms

Enhanced fog computing-based intrusion
detection with reduced execution time

Not specified

ML-based Framework (SVM,
GBM, RF)

RF for detecting malicious traffic in the
NSL-KDD dataset

Two-level Anomaly
Detection

Abnormal traffic detection using DT
and RF

85.34%

99.9%

4.2. Comparison of Open-Source NIDSs

Several free, open-source NIDSs are available for sniffing, analyzing, and detecting
malicious network traffic. Among the most widely used is Snort, a lightweight, single-
threaded SIDS that supports multiple operating systems. It operates in three modes: Sniffer,
Packet Logger, and NIDS modes, enabling real-time traffic analysis and attack detection,
including SQL injection and cross-site scripting attacks. Suricata is another popular tool,
offering a multi-threaded architecture for high-performance intrusion detection and pre-
vention, as well as network security monitoring and offline packet processing. The Open
Information Security Foundation developed Suricata and supports various operating sys-
tems, making it a flexible and scalable solution. Bro-IDS (now known as Zeek) is a network
analysis framework that combines SIDS and IDS. It inspects network traffic at a higher level
and supports multiple application-layer protocols, including DNS, FTP, HTTP, and SMTP.
Meanwhile, Kismet is a wireless IDS that detects and sniffs Wi-Fi networks, Bluetooth
devices, and sensors. It runs on Linux, BSD, Android, and Windows (with hardware
restrictions) and remains undetectable while monitoring network activity. OpenWIPS-ng is
another wireless IDS/IPS that monitors traffic for signature-based attacks. It consists of
sensors for data collection, a server for aggregation and intrusion detection, and a graphical
user interface (GUI) for managing and displaying threat information. However, it lacks
significant community support.

Mathematics 2025, 13, 1088

20 of 63

Security Onion is a Linux-based distribution that integrates multiple security tools,
including Snort, Suricata, Bro, Elasticsearch, and Logstash, for a more comprehensive
security solution. It enables full packet capture, network intrusion detection, and log
management for enhanced security monitoring. On the other hand, Sagan is a multi-
threaded log analysis and correlation engine initially designed as a HIDS but later extended
to function as a SIDS. It is optimized for low CPU and memory usage, making it suitable
for IoT environments.

Each of these tools has distinct advantages and limitations. While Snort is well doc-
umented and widely adopted, it may be susceptible to packet loss. Suricata offers a
multi-threaded architecture but requires more system resources. Bro-IDS offers high-level
traffic analysis but lacks a built-in graphical user interface (GUI). Kismet is useful for
wireless monitoring but is limited to wireless networks. OpenWIPS-ng is modular and
scalable but lacks strong community support. Security Onion and Sagan offer compre-
hensive security solutions, integrating multiple tools for real-time monitoring, ease of
deployment, and efficient network protection. These comparisons enable researchers and
practitioners to select the most suitable NIDS for their specific security needs, particularly
in IoT environments [53]. Table 7 presents a comparative analysis of various open-source
Network Intrusion Detection Systems (NIDSs), highlighting their key advantages and
drawbacks. The comparison aims to help select the most suitable NIDS based on specific
network security requirements.

Table 7. Comparison of open-source NIDSs [53].

NIDS

Strengths

Limitations

Snort

Suricata

Bro-IDS
(Zeek)

Kismet

Lightweight intrusion detection system with strong
industry adoption;
regular updates, extensive feature set, and multiple
administrative front-ends;
comprehensive documentation with active community
support;
proven reliability with thorough testing and a simple
deployment process.

Multi-threaded processing enables high-speed traffic
analysis, leveraging hardware acceleration for network
traffic inspection. It supports LuaJIT scripting for more
efficient threat detection and logs additional network
data, including TLS/SSL certificates, HTTP requests, and
DNS queries. Additionally, it is capable of detecting file
downloads.

Lacks an intuitive GUI; the
administrative console may be
challenging to use;
packet loss issues when handling
high-speed traffic (100–200 Mbps) before
exceeding a single CPU’s limit.

Higher memory and CPU consumption
compared to Snort.

Implements both SIDS and AIDS; uses advanced
signature detection techniques;
offers high-level network traffic analysis;
retains historical data for threat analysis and correlation,
making it suitable for high-speed networks.

Only runs on UNIX-based operating
systems, lacks a built-in GUI, primarily
relies on log files,
and requires expert-level knowledge for
setup and configuration.

Can expand capabilities to different network types via
plugins; supports channel hopping for detecting multiple
networks; remains undetectable while monitoring
wireless packets;
a well-maintained open-source tool for wireless
monitoring; enables real-time capture and live streaming
over HTTP.

Cannot directly retrieve IP addresses;
restricted to wireless network
monitoring.

Mathematics 2025, 13, 1088

21 of 63

Table 7. Cont.

NIDS

Strengths

Limitations

OpenWIPS-
ng

Modular architecture with plugin support for extended
functionality, designed for easy deployment by
non-experts, and enhanced detection capabilities due to
support for multiple sensors.

Security
Onion

Sagan

Highly flexible security monitoring solution;
integrates real-time analysis with GUI support via Sguil;
simple installation with customizable configurations;
regular updates to enhance security levels.

Optimized for real-time log analysis with a
multi-threaded architecture; supports various log formats
and normalization techniques; capable of geolocating
IP addresses;
distributes processing across multiple servers, enabling
Efficient resource utilization with lightweight CPU and
memory usage; supports active development and
provides ease of installation.

Limited to wireless networks;
lacks encrypted communication between
sensors and servers;
less popular, with minimal
documentation and community backing;
it is still underdeveloped in comparison
to other NIDS.

It inherits certain limitations from its
integrated tools. Initially, it functions as
an IDS and requires additional
configuration to work as an IPS.

Primarily focused on log analysis rather
than direct intrusion detection.

4.3. Role of Datasets in NIDS Development

NIDSs are essential for securing 5G networks by detecting and mitigating sophisticated
threats that stem from the technology’s inherent complexity and advanced capabilities.
Study [69] introduces AIS-NIDS, which employs packet-level analysis to identify threats
more precisely than traditional flow-based systems. By integrating autonomous detection
via ML with incremental learning, AIS-NIDS adapts to new attack classes, effectively learn-
ing from previously unseen threats. In a related effort, research [70] examines the challenges
in developing efficient NIDS, emphasizing the critical role of ML and the utilization of
publicly available datasets. This study highlights existing challenges and uncovers promis-
ing opportunities for future enhancements. Complementing these findings, research [71]
presents Reliable-NIDS (R-NIDS), which combines multiple datasets—including the novel
UNK22—to enhance the generalization capabilities of ML models, thereby demonstrating
significant practical benefits in deployment.

Further advancing the field, research [72] comprehensively reviews IDS types, bench-
mark datasets, and ML-/DL-based methodologies. Within this framework, a proposed
NIDS model achieved an impressive 98.11% accuracy and a 97.81% detection rate on the
UNSW-NB15 dataset. However, research [73] raises concerns about the generalizability
of ML classifiers by revealing significant statistical differences between synthetic datasets
(such as CIC_IDS, UNSW-NB15, and TON_IOT) and real-world data, thereby stressing the
need for improved training methodologies. Moreover, research [74] critically evaluates the
CICIDS2017 dataset, uncovering issues in traffic generation, flow construction, and labeling
that adversely impact dataset quality. An improved data processing methodology could
relabel over 20% of the traffic traces, enhancing model benchmarks and offering valuable
recommendations for developing more robust datasets. The experimental evaluation of
5G network security NIDS models operates using the structure illustrated in Figure 5.
The diagram illustrates the following steps for the experimental workflow, which begins
with data preprocessing, then uses (Analysis of Variance (ANOVA) for feature selection,
and continues with model training and final performance assessment. Feature selection

Mathematics 2025, 13, 1088

22 of 63

needs to be emphasized for both performance improvement and computational speed
optimization purposes.

Figure 5. Process workflow.

The feature selection process aimed to identify the most informative features in the
dataset, reducing dimensionality and improving model performance. Given its sensitivity
to linear relationships and scalability, ANOVA was used as the feature selection method,
making it suitable for the 5G-NIDD dataset. After exhaustive experimentation, the optimal
number of features selected was eight, striking a balance between model complexity and
accuracy [75].

4.4. ML Datasets

To develop a robust intrusion detection framework, the ML model was trained on the
widely recognized KDD-CUP’99 dataset, which serves as a benchmark for evaluating net-
work intrusion detection systems. This dataset comprises over 4 million training instances
and approximately 311,029 testing instances, offering a comprehensive range of network
traffic characteristics, including both fundamental connection attributes such as packet
header information and more advanced content-based features.

The study [76] ensures accurate classification by selecting five key features: protocol
type, which identifies the transport protocol (TCP, SCTP, UDP); service type, which specifies
the network service running at the destination (e.g., HTTP, FTP, SSH); connection status
flag, which denotes whether the connection is normal, rejected, or reset; source bytes,
representing the volume of data transmitted by the sender; and destination bytes, indicating
the amount of data received. These features were chosen due to their effectiveness in
distinguishing normal from malicious traffic while ensuring compatibility with real-time
packet-level analysis through Scapy. Additionally, the dataset categorizes cyber threats
into four primary attack types: Probing Attack, Remote-to-Local Attack, DoS attack, and
User-to-Root Attack.

Mathematics 2025, 13, x FOR PEER REVIEW 21 of 60   over 20% of the traffic traces, enhancing model benchmarks and offering valuable recommen-dations for developing more robust datasets. The experimental evaluation of 5G network se-curity NIDS models operates using the structure illustrated in Figure 5. The diagram illustrates the following steps for the experimental workflow, which begins with data preprocessing, then uses (Analysis of Variance (ANOVA) for feature selection, and continues with model training and final performance assessment. Feature selection needs to be emphasized for both performance improvement and computational speed optimization purposes.  Figure 5. Process workflow. The feature selection process aimed to identify the most informative features in the dataset, reducing dimensionality and improving model performance. Given its sensitivity to linear relationships and scalability, ANOVA was used as the feature selection method, making it suitable for the 5G-NIDD dataset. After exhaustive experimentation, the optimal number of features selected was eight, striking a balance between model complexity and accuracy [75]. 4.4. ML Datasets To develop a robust intrusion detection framework, the ML model was trained on the widely recognized KDD-CUP’99 dataset, which serves as a benchmark for evaluating net-work intrusion detection systems. This dataset comprises over 4 million training instances and approximately 311,029 testing instances, offering a comprehensive range of network traffic characteristics, including both fundamental connection attributes such as packet header information and more advanced content-based features. The study [76] ensures accurate classification by selecting five key features: protocol type, which identifies the transport protocol (TCP, SCTP, UDP); service type, which specifies the network service running at the destination (e.g., HTTP, FTP, SSH); connection status flag, which denotes whether the connection is normal, rejected, or reset; source bytes, representing Mathematics 2025, 13, 1088

23 of 63

To enhance data quality and model efficiency, a structured preprocessing pipeline was
implemented. This involved transforming multiclass attack labels into a binary format,
where “1” represents an attack and “0” represents normal traffic. The connection status flag
values were standardized for compatibility with Scapy, and only the most relevant features
were selected to align with real-time detection requirements. Additionally, categorical
features were encoded using OneHotEncoder, while numerical attributes were normalized
using MinMaxScaler to maintain consistency across the dataset.

Following preprocessing, multiple AI/ML models were trained and rigorously evalu-
ated using TensorFlow, including RF, One-Class SVM, Local Outlier Factor, KNN, and AEs.
Among these models, RF emerged as the most effective in terms of accuracy, training effi-
ciency, and inference speed, making it the optimal choice for real-time intrusion detection.
This ML-driven approach demonstrates the potential of artificial intelligence in enhancing
network security. It highlights the importance of real-world dataset-driven training for
detecting cyber threats in modern cyber–physical environments.

4.5. Integrating Explainable and Hybrid AI

Explainability in AI models is particularly crucial in Industry 5.0, where human
oversight and collaboration with intelligent systems play a vital role in decision-making.
To enhance interpretability and transparency, research [77] emphasized the integration of
XAI techniques in DL models, ensuring that human analysts can comprehend the decision-
making process of AI-driven systems. This transparency fosters trust in AI outcomes
and ensures compliance with security guidelines, making XAI a bridge between complex
algorithmic processes and actionable, security-driven decisions.

In parallel, the development of ensemble and hybrid AI models has significantly im-
proved cybersecurity defenses by leveraging the strengths of multiple learning paradigms.
Combining CNNs with recurrent neural networks (RNNs) or integrating RF with GB
techniques enables hybrid models to process diverse data types, such as temporal se-
quences and spatial patterns, simultaneously. These approaches offer a comprehensive
threat detection mechanism that enhances both accuracy and adaptability. For example, a
study [78] explored the use of DNNs combined with graph-based learning to analyze attack
patterns across enterprise networks, significantly improving detection accuracy against
persistent threats.

The evaluation of various AI models underscores the necessity of both explainability
and hybridization in addressing complex cybersecurity challenges. By integrating tradi-
tional and advanced AI models, a robust framework emerges that is capable of adaptive,
transparent, and highly efficient threat detection. This evolution in AI-driven security
solutions not only strengthens cybersecurity resilience but also paves the way for more
accountable and human-centric AI applications in Industry 5.0. Table 8 presents a com-
parative analysis of various AI-driven threat detection methodologies, highlighting their
objectives, limitations, and advantages.

Table 8. Comparison of AI-driven threat detection techniques [77].

Approach

Objective

Challenges

Key Strengths

DL-based AI-driven
network threat detection.

Adversarial training for
improved intrusion
detection.

Strengthening threat
identification in IoT
networks.

Enhancing detection
efficiency against
sophisticated threats.

Computational demands
are high.

Susceptible to adversarial
manipulations.

Achieves high precision in
real-time threat
identification.

Increased robustness
against evolving security
threats.

Mathematics 2025, 13, 1088

24 of 63

Table 8. Cont.

Approach

Objective

Challenges

Key Strengths

XAI for Industry 5.0
cybersecurity.

Boosting interpretability
and transparency in
AI-based security
solutions.

Difficulties in
implementing
explainability frameworks.

Provides clear AI-driven
insights for cybersecurity
decisions.

AI-driven protection
framework for cyber threat
intelligence.

Delivering comprehensive
defense mechanisms for AI
workloads.

Struggles with novel and
evolving threat patterns.

Adaptable and scalable for
diverse environments.

AI-powered automated
architecture for continuous
attack detection.

Enhancing real-time attack
detection in enterprise
systems.

Requires high-quality and
extensive data for optimal
functioning.

Reduces manual workload
while improving detection
efficiency.

Transformer-based AI for
social media threat
monitoring.

Detecting emerging
cybersecurity risks on
platforms like Twitter.

Effectiveness depends on
the volume and quality of
textual data.

Efficient at analyzing and
processing extensive
text-based datasets.

Federated learning-based
detection of adversarial
threats.

Improving adversarial
attack identification in
federated networks.

Resource-intensive nature
of federated learning.

Ensures data
confidentiality while
bolstering security in
distributed systems.

Multi-domain Trojan
identification through
domain adaptation.

Strengthening Trojan
malware detection across
diverse environments.

Complex adaptation
process when applied to
multiple domains.

High accuracy in
identifying cross-domain
Trojans.

4.6. DL Models

DL has emerged as a prominent technique in data mining, offering powerful capabil-
ities for modeling complex abstractions and relationships across multiple neural layers.
DL is extensively explored across various domains, including image recognition, speech
processing, natural language understanding, and social network analysis. Beyond these
applications, DL algorithms excel in discovering correlations across vast datasets from
diverse sources, enabling the simultaneous learning of attributes, classification, and predic-
tive analytics. The efficacy of DL in network security has led to its widespread adoption
in the development of IDSs [79]. The research contains a comparison of DL methods
implemented for IDSs in Table 9. Each approach reveals its technical design, along with its
main benefits, drawbacks, and applications in cybersecurity. The evaluation demonstrates
the effectiveness of various deep learning models in detecting and mitigating cyber threats.

Table 9. Comparison of DL models for IDSs [79].

Technique

Architecture

Advantages

Limitations

Applications

Generative
Architectures

USL, dynamically
trained on raw data

Learns without
labeled data, flexible
for different tasks

Requires large data,
complex
optimization

Data synthesis,
anomaly detection,
SSL

Autoencoder (AE)

Encoder–decoder
network

Effective for
dimensionality
reduction and
feature learning

Sensitive to noisy
inputs, requires
careful tuning

Data compression,
anomaly detection,
feature extraction

Stacked Autoencoder
(SAE)

Deep AE with
multiple hidden
layers

Captures hierarchical
feature
representations

High computational
cost, prone to
overfitting

Intrusion detection,
image recognition,
speech processing

Mathematics 2025, 13, 1088

25 of 63

Table 9. Cont.

Technique

Architecture

Advantages

Limitations

Applications

Sparse Autoencoder
(SAE)

AE with sparsity
constraints

Reduces redundant
features, improves
feature learning

Needs careful selection
of sparsity constraints

Denoising Autoencoder
(DAE)

AE trained with
corrupted inputs

Learns robust feature
representations

Requires noise level
adjustment

Restricted Boltzmann
Machine (RBM)

A probabilistic model
with two layers

Efficient feature
learning, suitable for
pre-training deep
networks

Slow convergence,
complex training
process

Intrusion detection,
compressed sensing,
representation learning

Noise reduction,
speech enhancement,
anomaly detection

Feature extraction,
recommendation
systems, anomaly
detection

Deep Belief Network
(DBN)

Stacked RBMs trained
layer-wise

Fast learning, effective
feature extraction

Computationally
intensive, needs large
datasets

Pattern recognition,
speech recognition,
cybersecurity

Recurrent Neural
Network (RNN)

Sequential network
with loops

Captures temporal
dependencies

Prone to vanishing
gradient problem

Long Short-Term
Memory (LSTM)

RNN variant with
memory cells

Handles long-term
dependencies,
mitigates vanishing
gradient

Gated Recurrent Unit
(GRU)

Simplified version of
LSTM

Faster training, fewer
parameters

High computational
cost

Less expressive than
LSTM in complex
sequences

Neural Classic
Network (NCN)

Fully connected
multilayer perceptron

Simple structure,
efficient for basic tasks

Limited in learning
complex patterns

Time-series forecasting,
speech recognition, text
processing

Speech recognition,
financial forecasting,
network intrusion
detection

Real-time speech
recognition, sequence
modeling, anomaly
detection

Image classification,
binary classification
tasks

Linear Function (LF)

Single-layer function

Nonlinear Function
(NLF)

Nonlinear activation
functions (sigmoid,
tanh, etc.)

Computationally
efficient

Limited in handling
complex problems

Linear regression,
signal processing

Can model complex
relationships

May cause vanishing
gradient issues

NNs, DL applications

Enhancing 5G Security: A Transfer Learning-Based IDS Approach

Traditional IDS solutions struggle to keep pace with sophisticated cyber threats, of-
ten hindered by the limited availability of labeled data for practical model training. To
overcome these limitations, the study [80] explores the integration of DTL to improve
IDS performance in 5G networks. By leveraging pre-trained DL models and fine-tuning
them for intrusion detection, we enable more effective anomaly detection and cyber threat
mitigation. The experimental evaluations highlight the superior performance of TL-based
models, with the Inception model achieving approximately a 10% improvement in F1-score
compared to conventional DL approaches without TL.

This study employs a two-phase transfer learning (TL) approach, utilizing a robust 5G
network dataset for pre-training (source domain) and a separate 5G-specific dataset (target
domain) for fine-tuning the model. Key deep learning (DL) architectures, including BiLSTM
and CNN, are employed to develop an intrusion detection system (IDS) framework capable
of detecting both known and emerging Distributed Denial-Of-Service (DDoS) attacks. The
proposed method involves freezing convolutional layers of the base model while adapting
higher-level layers to the target dataset, ensuring efficient knowledge transfer. Dataset
selection includes a large-scale, real-time 5G network traffic dataset as the source and
the 5G-NIDD dataset as the target, ensuring relevance to modern 5G threats. The results

Mathematics 2025, 13, 1088

26 of 63

demonstrate the effectiveness of TL in enhancing model adaptability, reducing training
time, and significantly improving intrusion detection accuracy in 5G environments. This
research contributes to the advancement of AI-driven security solutions, reinforcing the
resilience of next-generation networks against evolving cyber threats.

4.7. Quantum Machine Learning

NNs have revolutionized both industry and academia, yet their integration with quan-
tum computing remains a formidable challenge. In work [81], an innovative Quantum
Neural Network (QNN) model is tailored for quantum neural computing. The approach
leverages classically controlled single-qubit operations and measurements within real-
world quantum systems while accounting for environment-induced decoherence. This
significantly simplifies the physical implementation of QNNs, making them more practical
for near-term quantum devices. A key advantage of the model is its ability to circumvent
the exponential growth of state-space size that traditionally hinders QNNs. By doing so, it
minimizes memory overhead and enables fast optimization using classical optimization
algorithms. To validate its performance, the proposed model was benchmarked on hand-
written digit recognition and various nonlinear classification tasks. The results demonstrate
superior nonlinear classification capabilities, remarkable robustness to noise, and efficient
learning dynamics. The findings pave the way for broader applications of quantum com-
puting in neural networks and inspire the early realization of quantum neural computing
before the advent of fully developed quantum computers.

The rapid advancements in quantum technology have led to the development of
numerous communication and computational schemes that leverage quantum advantages.
These advancements indicate vast application potential, yet the experimental realization
of such schemes remains a formidable challenge due to the complexities associated with
generating high-dimensional or highly entangled quantum states. In a study [82], a quan-
tum coupon collector protocol has been proposed and analyzed that utilizes coherent states
and simple linear optical components, making it feasible for practical implementation with
existing experimental setups. The findings demonstrate that the proposed protocol signifi-
cantly reduces the number of samples required to learn a given set compared to the classical
coupon collector problem, thereby surpassing classical computational limits. Furthermore,
the concept has been extended by designing a quantum blind box game, which effectively
transmits information beyond classical constraints. These results provide strong evidence of
the superiority of quantum mechanics in ML and communication complexity, highlighting
its potential to revolutionize future computing and information-processing paradigms.

4.8. Comparison of Batch Learning Models (BLMs) and Data Streaming Models (DSMs)

DSMs exhibit superior performance in binary classification, while BLMs, particularly
J48, maintain an advantage in multiclass classification. Among DSMs, OzaBagAdwin (OBA)
stands out as the most accurate model for binary classification; however, its computational
demands necessitate optimization for large-scale deployment. These findings align with
previous research, reinforcing the effectiveness of DSMs for evolving data streams. More-
over, the results underscore the need for adaptive learning strategies to enhance real-time
Intrusion Detection Systems (IDSs), ensuring efficient and scalable threat mitigation. DSMs
significantly outperform BLMs in binary classification, highlighting their suitability for
real-time applications. In multiclass classification, J48 remains the most effective, though
DSMs offer comparable performance. Statistical analysis confirms a significant difference in
binary classification but not in multiclass scenarios. Findings [83] align with prior studies,
reaffirming that DSMs hold promise for large-scale, evolving data streams. Future research
should explore optimization strategies to enhance the efficacy of these methods in complex

Mathematics 2025, 13, 1088

27 of 63

classification tasks. Table 10 compares the BLMs and DSMs for IDSs. It describes the
implemented algorithms, along with their accuracy results for both binary and multiclass
tasks, and provides relevant insights from the evaluation phase. The comparison process
helps one understand the basic capabilities, as well as the weaknesses, of different model
types when detecting cybersecurity threats.

Table 10. Comparative analysis of BLMs and DSMs [83].

Model Type

Algorithm

Binary Accuracy (%)

Multiclass Accuracy (%) Key Insights

Batch Learning
Models (BLMs)

J48

PART

94.73

92.83

Data Streaming
Models (DSMs)

Hoeffding Tree (HT)

98.38

OBA

99.67

87.66

87.05

71.98

82.80

Best BLM for multiclass
classification; widely used in
intrusion detection

Slightly lower accuracy than
J48; rule-based classification

High binary accuracy but
weak multiclass performance

Highest binary accuracy;
computationally intensive

4.9. Dataset Comparison

Free datasets play a crucial role in the implementation and validation of NIDSs.
Among the most widely used datasets, KDDCUP99 (KDD99) is one of the earliest datasets
derived from the DARPA dataset for detecting malicious connections. It categorizes attacks
into DoS, Remote to User, User to Root, and Probing, using 41 extracted features. Despite
its popularity, KDD99 has significant drawbacks, including outdated data, imbalanced
classification, and simulation artifacts that may lead to overestimating the performance
of anomaly detection. NSL-KDD was developed as an improved version to overcome
these limitations by removing duplicate records and balancing probability distributions.
However, it still lacks modern attack scenarios.

To address the need for contemporary datasets, the Australian Centre for Cyber Secu-
rity created UNSW-NB15 in 2015. It integrates hybrid real and synthetic attack behaviors,
featuring nine attack types, making it more complex than KDD99 due to the similarities
between attack and expected behaviors. Another notable dataset is the IoT dataset, which
focuses on classifying IoT devices rather than detecting intrusions. It captures network
traffic from 28 IoT devices over a six-month period, providing valuable insights but not
explicitly targeting IDS applications.

More recent datasets, such as CICIDS and CSE-CIC-IDS2018, offer improved realism
in attack simulations. The CICIDS dataset, developed by the Canadian Institute for Cyber-
security at the University of New Brunswick, reflects real-world threats using 25 simulated
users and multiple protocols, including HTTP, HTTPS, FTP, SSH, and email. It is analyzed
using a CICFlowMeter to provide labeled flow data. Meanwhile, CSE-CIC-IDS2018 is
an AIDS dataset designed to replace traditional static datasets. It features seven attack
scenarios, including Brute Force, DDoS, Web attacks, and Local Network Infiltration, with
a large-scale attack infrastructure consisting of 50 attack nodes, 30 servers, and 420 hosts.
Eighty extracted network features provide a dynamic and realistic evaluation environment
for intrusion detection.

In comparison, despite being outdated, KDD99 remains one of the most widely
used datasets, leading to the development of NSL-KDD with improved balance and data
reliability. UNSW-NB15 presents modern attack scenarios while introducing additional
complexity. The IoT dataset is the only one specifically designed for IoT traffic analysis,
although not for IDS purposes. CICIDS and CSE-CIC-IDS2018 provide more realistic attack

Mathematics 2025, 13, 1088

28 of 63

behaviors and traffic analysis but do not directly address IoT security. While these datasets
serve as the foundation for NIDS evaluation, they still suffer from limitations such as
privacy concerns, data anonymization, and the inability to fully reflect evolving security
threats [84]. Table 11 provides a comparative analysis of various publicly available network
security datasets, highlighting their advantages and limitations.

Table 11. Comparative analysis of free datasets [84].

Datasets

Benefits

Drawbacks

Covered Attacks

Use Cases

KDD99

Widely recognized and frequently
used dataset;
contains labeled data;
includes 41 distinct features per
connection, along with class labels;
provides network traffic in PCAP
format.

Imbalanced
classification issues;
considered outdated. It
does not include IoT
and 5G-related data.

Covers various attack
types such as DoS, R2U,
User to Root U2R, and
Probing.

IDS research, classic
ML-based anomaly
detection.

NSL-KDD

Improved version of KDD99;
addresses some of KDD99’s
limitations and eliminates duplicate
records in training and testing
datasets.

Lacks scenarios for
modern, low-footprint
attacks;
does not support IoT
systems or 5G.

UNSW-
NB15

IoT
Dataset

CICIDS

Represents a blend of real and
synthetic modern network activities
and cyber threats;
offers network traffic data in PCAP
and CSV formats.

Designed for IoT network traffic
analysis,
represents real-world IoT network
environments, and provides
network traffic in PCAP and CSV
formats.

It is more complex than
KDD99 due to
similarities between
legitimate and
malicious network
behaviors.

Lacks labeled data,
no attack data are
included, aimed at IoT
device proliferation
and traffic
characterization rather
than security analysis.

It contains labeled network flow
data;
suitable for ML and DL applications;
offers network traffic in PCAP and
CSV formats.

Restricted access is not
publicly available and
does not cover
IoT-based network
security scenarios.

DoS, R2L, U2R, Probing

ML-based IDS research,
benchmark dataset for
anomaly detection.

Encompasses nine
attack categories,
including Fuzzers,
Analysis, Backdoors,
DoS, Exploits, Generic,
Reconnaissance,
Shellcode, and Worms.

Modern IDS evaluation,
ML-based attack
detection.

No attacks included,
normal traffic analysis
only.

IoT device behavior
analysis, traffic
classification.

Simulates multiple
attack types such as
Brute Force (FTP and
SSH), DoS, Heartbleed,
Web Attacks,
Infiltration, Botnet, and
DDoS.

ML-/DL-based IDS
research, DL security
models.

Network flows with labeled features;
designed for ML and DL research;
provides network traffic in PCAP,
CSV, and log formats;
dynamically generated and
adaptable dataset;
extensible, modifiable, and
reproducible.

It focuses on 5G network slicing and
NFV security; provides network
traffic at different layers; suitable for
AI-based security research.

CSE-CIC-
IDS2018

5G-NIDD
Dataset

Not publicly accessible.
It does not include
IoT-specific traffic
analysis.

Brute Force,
Heartbleed, Botnet,
DoS, DDoS, Web
Attacks, and Local
Network Infiltration.

ML-/DL-based IDS,
cyber threat detection,
forensics analysis.

Limited labeled data,
not widely available.

DDoS, network slicing
attacks, NFV exploits.

5G attack detection,
AI-based anomaly
detection.

Mathematics 2025, 13, 1088

29 of 63

Table 11. Cont.

Datasets

Benefits

Drawbacks

Covered Attacks

Use Cases

5G-
Emulation
Dataset

SDN/NFV
Security
Dataset

Simulates realistic 5G network
conditions, covering slicing and
jamming threats, and is helpful in
developing ML-/DL-based security
solutions.

Tailored for 5G SDN and NFV
research, it covers
virtualization-based threats and
helps in securing cloud-based 5G
infrastructure.

IoT/5G
Dataset
(TON_IoT)

Captures real-world IoT device
traffic in 5G environments, supports
AI/ML research, available in
multiple formats.

5G-AI
Security
Dataset

Focuses on AI-powered security,
includes adversarial attack scenarios,
and helps in developing AI-based
5G intrusion detection models.

The simulated dataset
may not fully reflect
real-world traffic.

DDoS, jamming, and
slicing-based attacks.

ML-/DL-based IDS for
5G, network slicing
security.

Does not include IoT
security and limited
real-world deployment.

Spoofing, flooding,
malware propagation,
SDN-/NFV-specific
attacks

SDN-based 5G network
security, virtualized
network security
analysis.

Limited data for
advanced 5G-specific
attacks.

Botnets, scanning,
backdoors, IoT exploits

IoT-5G security
monitoring, smart
environment attack
detection.

Not publicly available;
requires specialized AI
models.

Model evasion, AI
poisoning, adversarial
attacks in 5G.

AI-powered IDS,
adversarial attack
detection in 5G.

5. Development of the 5G NIDD Datasets

Several open-source platforms and tools are crucial in simulating, analyzing, and
securing 5G network environments. These platforms facilitate the development of IDSs by
enabling realistic network simulations, generating attack scenarios, and evaluating secu-
rity performance. Open5GS is a widely adopted open-source platform that emulates CN
functions for 4G and 5G systems. It includes essential 4G components, such as the Mobility
Management Entity, Serving Gateway, and Packet Gateway, as well as 5G components like
the AMF and SMF. Open5GS enables the selective deployment of network nodes, allowing
for the simulation of real-world network operations. Additionally, it facilitates the genera-
tion of datasets encompassing benign and malicious traffic scenarios, making it a valuable
resource for evaluating IDS performance in realistic environments [85]. UERANSIM is
another key open-source tool that simulates 5G access networks and user equipment (UE)
interactions with a 5G CN. It supports critical 5G functionalities, including authentication,
mobility management, and handovers. The flexibility of UERANSIM in handling UE
connections enables the testing of various attack scenarios, such as rogue device infiltration
and malicious traffic overload, thereby aiding in the assessment of security vulnerabilities
within 5G networks [86].

Scapy, a powerful Python library, is instrumental in crafting and manipulating custom
network packets for security testing. It supports the simulation of complex network attacks
by allowing fine-grained control over protocol parameters, including Packet Forwarding
Control Protocol (PFCP) and GTP-C (GPRS Tunneling Protocol–Control Plane). This
capability makes Scapy a valuable tool for analyzing security threats and intrusion patterns
in 5G environments [87]. Kubernetes serves as an orchestration platform for managing
It
containerized workloads, enhancing the scalability and portability of 5G testbeds.
efficiently manages components such as Open5GS and UERANSIM, enabling dynamic
scaling, automated deployment, and sophisticated attack simulations. By leveraging
Kubernetes, researchers can conduct large-scale security experiments, automate network
configurations, and replicate diverse threat landscapes within controlled environments [88].
Grafana Loki is a scalable log aggregation system designed to index log metadata
using a label-based approach. It integrates seamlessly with Grafana, providing efficient
log management and real-time data visualization. Its support for LogQL query language

Mathematics 2025, 13, 1088

30 of 63

simplifies log analysis, facilitating effective monitoring of network activity, anomaly detec-
tion, and forensic investigations in 5G security frameworks [89]. Prometheus is a widely
used monitoring and alerting tool that collects and processes time-series data, including
CPU utilization, memory consumption, and network packet throughput. When integrated
with Kubernetes, it enables real-time performance monitoring of containerized network
functions. Together with Grafana, Prometheus enhances security monitoring by offering
advanced visualizations and real-time alerts for security incidents, including DDoS attacks
and anomalous traffic patterns. These open-source platforms collectively enable robust
simulation, monitoring, and security assessment of 5G network infrastructures. Their
integration into research testbeds enhances the detection and mitigation of evolving cyber
threats, significantly advancing intrusion detection and network security strategies.

5.1. Simulated Attack Types

Denial-of-Service (DoS) attacks remain a significant threat to modern networks, aiming
to disrupt services through resource overload or deny legitimate user access [90]. These
attacks, which may involve flooding or malicious content injection, exploit vulnerabilities
across various network components—including nodes, devices, and applications. In 5G
networks, the lack of robust security measures on user devices further amplifies these
risks, underscoring the urgent need for effective intrusion detection systems. DoS and
DDoS attacks can be broadly classified into volume-based, protocol-based, and application-
layer attacks. Volume-based attacks inundate targets with overwhelming traffic, protocol-
based attacks exploit weaknesses in network protocols, and application-layer attacks
target specific services while often mimicking legitimate user behavior, which makes them
particularly challenging to detect. Standard techniques include ICMP floods, UDP floods,
SYN floods, HTTP floods, and slow-rate attacks (e.g., Slowloris, Slow POST). Port scanning
is frequently employed as a preliminary step to identify potential vulnerabilities in network
hosts, with methods such as SYN scans, TCP Connect scans, and UDP scans providing
critical reconnaissance information for subsequent attacks.

This review systematically explores various DoS/DDoS attack methodologies, cat-
egorizing them and assessing their implications for network security. It highlights the
necessity for robust detection and mitigation strategies by examining tools such as Hping3,
Goldeneye, and Torshammer. Additionally, this study evaluates the use of 5Greplay—a
tool capable of generating high-bandwidth traffic to simulate DoS/DDoS conditions—in
conducting security assessments of 5G networks. In particular, the analysis focuses on
the AMF component of the 5G-CN, which is notably susceptible to replay attacks, thereby
emphasizing the critical role of comprehensive security evaluations in next-generation
network infrastructures.

5.2. System Architecture

In [91], the proposed network architecture is designed to emulate various attack scenar-
ios and rigorously assess network behavior under adverse conditions. Central to this design
is a 5G-CN implemented via Open5GS, which supports multiple network functions. The
simulation environment incorporates legitimate and compromised user devices—modeled
using UERANSIM—to facilitate realistic emulation of attacks launched by malicious users
and compromised servers. Attack vectors include General Packet Radio Service Tunneling
Protocol (GTP-U) Denial-of-Service (DoS) attacks, attack request flooding, and PFCP-based
attacks. The architecture incorporates a sophisticated monitoring framework that utilizes
Prometheus and Loki for collecting and analyzing real-time metrics, ensuring precise detec-
tion and comprehensive analysis of these threats. This robust platform provides researchers
and security professionals with the necessary tools to evaluate the resilience of 5G networks

Mathematics 2025, 13, 1088

31 of 63

against a diverse array of attack vectors. The overall system architecture is illustrated in
Figure 6.

Figure 6. Architectural framework.

5.3. Comparison with Other Architectures

The 5GTN is an open innovation ecosystem for advancing research in 5G and beyond,
with a strong emphasis on AI and cybersecurity applications. For dataset creation, the
University of Oulu’s 5GTN site was employed [92]. The testbed incorporated several key
elements to optimize data collection, including two Nokia Flexi Zone Indoor Pico Base
Stations connected to both attacker nodes and benign traffic-generating devices. These
components were interconnected via a Dell N1524 switch and additional network interfaces,
including X2-C and S1-U. Attacker nodes were configured using Raspberry Pi 4 Model B
devices running Ubuntu, which connected to the base stations through Huawei 5G modems.
The comprehensive equipment setup for each base station, comprising attacker devices,
benign user equipment, and a data capture PC, is depicted in the diagram below. The
victim server was strategically positioned within a Multi-access Edge Computing (MEC)
environment to effectively isolate the target, ensuring separation from the attacker network.
Attack traffic was routed through the 5G network, thereby simulating realistic traffic
flows and underscoring the potential for adversaries to infiltrate various 5G subnetworks.
Figure 7 illustrates the deployment configuration of the NSA Option 3a, highlighting its
key features and operational design.

One of the distinguishing features of the 5G-NIDD dataset is its innovative approach
to generating benign traffic. Rather than relying on simulated traffic—which can often
misrepresent real network behavior—the 5G-NIDD dataset captures live traffic from actual
mobile devices within an operational network environment. This authentic traffic encom-
passes benign and attack flows, with benign traffic comprising protocols such as HTTP,
HTTPS, SSH, and Secure File Transfer Protocol (SFTP). For instance, HTTP and HTTPS
traffic are sourced from live streaming services and typical web browsing activities. At
the same time, SSH and SFTP traces are obtained by deploying genuine SSH clients and
servers on mobile devices, enhancing the dataset’s diversity. Finally, this research utilizes
the labeled 5G-NIDD dataset from a functional 5G test network comprising 85,112 samples,

Mathematics 2025, 13, x FOR PEER REVIEW 30 of 60   utilizes Prometheus and Loki for collecting and analyzing real-time metrics, ensuring pre-cise detection and comprehensive analysis of these threats. This robust platform provides researchers and security professionals with the necessary tools to evaluate the resilience of 5G networks against a diverse array of attack vectors. The overall system architecture is illustrated in Figure 6.  Figure 6. Architectural framework. 5.3. Comparison with Other Architectures The 5GTN is an open innovation ecosystem for advancing research in 5G and beyond, with a strong emphasis on AI and cybersecurity applications. For dataset creation, the Univer-sity of Oulu’s 5GTN site was employed [92]. The testbed incorporated several key elements to optimize data collection, including two Nokia Flexi Zone Indoor Pico Base Stations connected to both attacker nodes and benign traffic-generating devices. These components were inter-connected via a Dell N1524 switch and additional network interfaces, including X2-C and S1-U. Attacker nodes were configured using Raspberry Pi 4 Model B devices running Ubuntu, which connected to the base stations through Huawei 5G modems. The comprehensive equip-ment setup for each base station, comprising attacker devices, benign user equipment, and a data capture PC, is depicted in the diagram below. The victim server was strategically posi-tioned within a Multi-access Edge Computing (MEC) environment to effectively isolate the target, ensuring separation from the attacker network. Attack traffic was routed through the 5G network, thereby simulating realistic traffic flows and underscoring the potential for ad-versaries to infiltrate various 5G subnetworks. Figure 7 illustrates the deployment configura-tion of the NSA Option 3a, highlighting its key features and operational design. One of the distinguishing features of the 5G-NIDD dataset is its innovative approach to generating benign traffic. Rather than relying on simulated traffic—which can often mis-represent real network behavior—the 5G-NIDD dataset captures live traffic from actual mo-bile devices within an operational network environment. This authentic traffic encompasses benign and attack flows, with benign traffic comprising protocols such as HTTP, HTTPS, SSH, and Secure File Transfer Protocol (SFTP). For instance, HTTP and HTTPS traffic are sourced from live streaming services and typical web browsing activities. At the same time, SSH and SFTP traces are obtained by deploying genuine SSH clients and servers on mobile Mathematics 2025, 13, 1088

32 of 63

including Port Scans, DoS attacks, and regular traffic. With 60.72% normal and 39.28%
attack traffic, this dataset supports binary classification for security model development
through comprehensive data preprocessing and rigorous evaluation methods. These stud-
ies underscore the vital importance of robust NIDS, high-quality datasets, and adaptive
machine learning (ML) models in enhancing next-generation communication networks.

Figure 7. Testbed network architecture.

Research [93] leverages an extensive dataset generated within a controlled 5G testbed
environment, which serves as the source domain. This dataset comprises six million
flow-based instances from two network slices, encompassing both benign and malicious
traffic types. The testbed, implemented using Free5GC and UERANSIM, simulates 5G
network slicing with high fidelity. Benign traffic is generated through an automated
Python-based headless browser script, which replicates various online activities, including
streaming, file transfers, and general internet usage. In contrast, malicious traffic comprises
eight DDoS attacks (e.g., UDP flood, TCP SYN flood), generated using the hping3 tool
and converted into CSV format with CICFlowMeter for detailed flow-based analysis [94].
Figure 8 illustrates the network architecture of the 5G testbed environment, highlighting its
key structural components and integration framework.

The 5G-NIDD dataset is the target domain, derived from a real-world 5G testing
environment that captures benign and malicious traffic [95]. Benign traffic includes pro-
tocols such as HTTP, HTTPS, SSH, and SFTP, while malicious traffic comprises various
DoS attacks (e.g., ICMP flood, HTTP flood) and scan attacks (e.g., SYN scan). Notably,
5G-NIDD contains fewer flow samples than the source dataset, making it particularly
suitable for DTL applications. Initially, the dataset is processed with CICFlowMeter to
extract 84 traffic features. Subsequent feature selection and optimization focus on attributes
highly indicative of DDoS attacks. For example, Principal Component Analysis (PCA)
identifies key features—flow duration and forward packet length—and reduces the dataset
to eight significant features [96]. Any irregularities, such as infinite values, are removed or
converted to NaN to ensure a clean and consistent dataset for model training and evalua-
tion. The data are then reshaped to match the one-dimensional input DL models require.
To address the class imbalance, the number of attack traffic samples is downsampled to
roughly one-eighth that of benign samples, and random shuffling is employed to create rep-
resentative and unbiased subsets. Feature normalization is performed using StandardScaler

Mathematics 2025, 13, x FOR PEER REVIEW 31 of 60   devices, enhancing the dataset’s diversity. Finally, this research utilizes the labeled 5G-NIDD dataset from a functional 5G test network comprising 85,112 samples, including Port Scans, DoS attacks, and regular traffic. With 60.72% normal and 39.28% attack traffic, this dataset supports binary classification for security model development through comprehen-sive data preprocessing and rigorous evaluation methods. These studies underscore the vi-tal importance of robust NIDS, high-quality datasets, and adaptive machine learning (ML) models in enhancing next-generation communication networks.  Figure 7. Testbed network architecture. Research [93] leverages an extensive dataset generated within a controlled 5G testbed environment, which serves as the source domain. This dataset comprises six million flow-based instances from two network slices, encompassing both benign and malicious traffic types. The testbed, implemented using Free5GC and UERANSIM, simulates 5G network slicing with high fidelity. Benign traffic is generated through an automated Python-based headless browser script, which replicates various online activities, including streaming, file transfers, and general internet usage. In contrast, malicious traffic comprises eight DDoS attacks (e.g., UDP flood, TCP SYN flood), generated using the hping3 tool and converted into CSV format with CICFlowMeter for detailed flow-based analysis [94]. Figure 8 illus-trates the network architecture of the 5G testbed environment, highlighting its key structural components and integration framework. The 5G-NIDD dataset is the target domain, derived from a real-world 5G testing envi-ronment that captures benign and malicious traffic [95]. Benign traffic includes protocols such as HTTP, HTTPS, SSH, and SFTP, while malicious traffic comprises various DoS attacks (e.g., ICMP flood, HTTP flood) and scan attacks (e.g., SYN scan). Notably, 5G-NIDD contains fewer flow samples than the source dataset, making it particularly suitable for DTL applications. Initially, the dataset is processed with CICFlowMeter to extract 84 traffic features. Subsequent feature selection and optimization focus on attributes highly indicative of DDoS attacks. For example, Principal Component Analysis (PCA) identifies key features—flow duration and forward packet length—and reduces the dataset to eight significant features [96]. Any irregu-larities, such as infinite values, are removed or converted to NaN to ensure a clean and con-sistent dataset for model training and evaluation. The data are then reshaped to match the one-dimensional input DL models require. To address the class imbalance, the number of at-tack traffic samples is downsampled to roughly one-eighth that of benign samples, and Mathematics 2025, 13, 1088

33 of 63

from the sklearn library, standardizing features to have a mean of zero and a standard
deviation of one, eliminating bias toward larger-scale features, and maintaining uniformity
across training and testing sets.

Figure 8. Schematic of testbed network architecture.

Finally, the dataset is split into 80% training and 20% testing subsets, with stratification
to ensure balanced class distributions across splits. This strategy facilitates the evaluation
of realistic models and preserves consistency between the source and target datasets. The
Maximum Mean Discrepancy (MMD) metric is used to quantify the difference between
these datasets, with a calculated MMD score of 0.2605 indicating a moderate level of
dissimilarity—an essential consideration for DTL scenarios.

The proposed framework [97] is specifically designed for the 5G V2X network, op-
timizing intrusion detection in connected and autonomous vehicles (CAVs). Figure 7
illustrates the network architecture considered for deploying a NIDS. In this architecture,
CAVs communicate with gNodeB via the Uu interface, while the PC5 sidelink interface
facilitates direct vehicle-to-vehicle data exchange. Given the dynamic nature of V2X net-
works, each vehicle can participate in multiple network slices based on application-specific
requirements. To enhance security, deploying an IDS within each slice as a Network Virtu-
alization Function (NVF) on edge devices is proposed. The IDS training process follows
a two-stage approach: first, SSL pre-training is conducted independently on each edge
device, followed by model aggregation on a centralized cloud server. Subsequently, a
task-specific fine-tuning phase utilizing a small set of labeled data is performed on the
cloud. Upon completion of both training phases, the optimized IDS models are deployed as
NVFs on edge devices to continuously monitor network traffic. Furthermore, deploying an
IDS within a MEC environment significantly reduces the computational and energy burden

Mathematics 2025, 13, x FOR PEER REVIEW 32 of 60   random shuffling is employed to create representative and unbiased subsets. Feature normal-ization is performed using StandardScaler from the sklearn library, standardizing features to have a mean of zero and a standard deviation of one, eliminating bias toward larger-scale features, and maintaining uniformity across training and testing sets.  Figure 8. Schematic of testbed network architecture. Finally, the dataset is split into 80% training and 20% testing subsets, with stratification to ensure balanced class distributions across splits. This strategy facilitates the evaluation of realistic models and preserves consistency between the source and target datasets. The Max-imum Mean Discrepancy (MMD) metric is used to quantify the difference between these datasets, with a calculated MMD score of 0.2605 indicating a moderate level of dissimilar-ity—an essential consideration for DTL scenarios. The proposed framework [97] is specifically designed for the 5G V2X network, opti-mizing intrusion detection in connected and autonomous vehicles (CAVs). Figure 7 illus-trates the network architecture considered for deploying a NIDS. In this architecture, CAVs communicate with gNodeB via the 𝑈𝑢 interface, while the 𝑃𝐶5 sidelink interface facilitates direct vehicle-to-vehicle data exchange. Given the dynamic nature of V2X networks, each vehicle can participate in multiple network slices based on application-specific require-ments. To enhance security, deploying an IDS within each slice as a Network Virtualization Function (NVF) on edge devices is proposed. The IDS training process follows a two-stage approach: first, SSL pre-training is conducted independently on each edge device, followed by model aggregation on a centralized cloud server. Subsequently, a task-specific fine-tun-ing phase utilizing a small set of labeled data is performed on the cloud. Upon completion of both training phases, the optimized IDS models are deployed as NVFs on edge devices to continuously monitor network traffic. Furthermore, deploying an IDS within a MEC en-vironment significantly reduces the computational and energy burden on CAVs, enabling Mathematics 2025, 13, 1088

34 of 63

on CAVs, enabling them to allocate resources to other critical functions while ensuring
robust network security. Figure 9 presents the NIDS deployment in MEC environments.

Figure 9. Fifth-generation network architecture for NIDS deployment in MEC environments.

The Trust-Aware Intrusion Detection and Prevention System (TA-IDPS) [98] is de-
signed to enhance security in 5G cloud-enabled Mobile Ad Hoc Networks (MANETs). This
architecture comprises mobile devices, cloudlets, cloud servers, and a Trusted Authority
(TA). Mobile devices are organized into clusters, each managed by a Cluster Head (CH)
selected through the Moth Flame Optimization (MFO) algorithm, which considers factors
such as node degree, residual energy, distance, RSS Indicator, trust value, and mobility.
The TA authenticates nodes using an ultra-lightweight symmetric cryptographic technique
suitable for resource-constrained environments. Data packets are classified by the CH using
a Deep Belief Network (DBN) into categories: normal, malicious, or suspicious. Normal
packets are routed using an Adaptive Bayesian Estimator for next-best forwarder selection,
suspicious packets undergo further analysis by a Peek monitor employing Awad’s Infor-
mation Entropy, and malicious packets are discarded, with the originating node excluded
from the network. Cloudlets aggregate and verify packets from CHs before forwarding
them to cloud servers. Performance evaluations using the NS3 simulator indicate that
TA-IDPS outperforms traditional methods across various metrics, including detection
rate, FPR, detection delay, packet delivery ratio, energy consumption, throughput, rout-
ing overhead, and end-to-end delay. Figure 10 shows the TA-IDPS architecture for 5G
cloud-enabled MANETs.

The simulations [99] were conducted in a VM (Virtual Machine) running Ubuntu
20.04.5 LTS with an Intel Core i5-8300H (four cores, 2.3 GHz) and 8 GB RAM, utilizing NS-3
with the 5G-LENA module to model 4G, 5G, and V2X communication. The nr module, a
“hard fork” of the mmWave simulator, enabled physical and medium access control layer
simulations. SUMO was used to model traffic, generating four maps based on Lisbon,
Portugal, with varying vehicle counts and attack scenarios. The maps included 45, 45, 70,
and 100 vehicles, with attacker counts of 2, 4, 7, and 9, respectively. Simulations lasted
230 s, with packet exchanges beginning at second 170 via multicast address 225.0.0.0.

Mathematics 2025, 13, x FOR PEER REVIEW 33 of 60   them to allocate resources to other critical functions while ensuring robust network security. Figure 9 presents the NIDS deployment in MEC environments.  Figure 9. Fifth-generation network architecture for NIDS deployment in MEC environments. The Trust-Aware Intrusion Detection and Prevention System (TA-IDPS) [98] is de-signed to enhance security in 5G cloud-enabled Mobile Ad Hoc Networks (MANETs). This architecture comprises mobile devices, cloudlets, cloud servers, and a Trusted Authority (TA). Mobile devices are organized into clusters, each managed by a Cluster Head (CH) selected through the Moth Flame Optimization (MFO) algorithm, which considers factors such as node degree, residual energy, distance, RSS Indicator, trust value, and mobility. The TA authenticates nodes using an ultra-lightweight symmetric cryptographic technique suit-able for resource-constrained environments. Data packets are classified by the CH using a Deep Belief Network (DBN) into categories: normal, malicious, or suspicious. Normal pack-ets are routed using an Adaptive Bayesian Estimator for next-best forwarder selection, sus-picious packets undergo further analysis by a Peek monitor employing Awad’s Information Entropy, and malicious packets are discarded, with the originating node excluded from the network. Cloudlets aggregate and verify packets from CHs before forwarding them to cloud servers. Performance evaluations using the NS3 simulator indicate that TA-IDPS outper-forms traditional methods across various metrics, including detection rate, FPR, detection delay, packet delivery ratio, energy consumption, throughput, routing overhead, and end-to-end delay. Figure 10 shows the TA-IDPS architecture for 5G cloud-enabled MANETs. The simulations [99] were conducted in a VM (Virtual Machine) running Ubuntu 20.04.5 LTS with an Intel Core i5-8300H (four cores, 2.3 GHz) and 8 GB RAM, utilizing NS-3 with the 5G-LENA module to model 4G, 5G, and V2X communication. The nr module, a “hard fork” of the mmWave simulator, enabled physical and medium access control layer simulations. SUMO was used to model traffic, generating four maps based on Lisbon, Por-tugal, with varying vehicle counts and attack scenarios. The maps included 45, 45, 70, and 100 vehicles, with attacker counts of 2, 4, 7, and 9, respectively. Simulations lasted 230 s, with packet exchanges beginning at second 170 via multicast address 225.0.0.0. The dataset included features such as time (timestamp), txRx (transmission/reception), nodeId, IMSI, srcIp, dstIp, packetSizeBytes, srcPort, dstPort, pktSeqNum, delay, jitter, coord_x, coord_y, speed, and attack (class 0: benign, class 1: malicious). Mathematics 2025, 13, 1088

35 of 63

Figure 10. TA-IDPS system design for 5G cloud-connected MANETs.

The dataset included features such as time (timestamp), txRx (transmission/reception),
nodeId, IMSI, srcIp, dstIp, packetSizeBytes, srcPort, dstPort, pktSeqNum, delay, jitter,
coord_x, coord_y, speed, and attack (class 0: benign, class 1: malicious).

For intrusion detection, scikit-learn was used (v1.1.1). The first phase involved DTs,
trained on four datasets with different attacker numbers, evaluated using 10-fold cross-
validation across tree depths {2–55}, prioritizing F1 score due to class imbalance. The
second phase utilized RF and MLP, trained on three datasets and tested on the fourth,
using GroupKFold cross-validation. RFs were optimized over tree depths {2–10} and tree
counts {10–50}, while MLP was tuned for batch sizes {32, 64}, hidden layers {(10,2), (20,2)},
optimizers (“adam”, “sgd”), and activation functions (“tanh”, “relu”). This approach
enhances intrusion detection in 5G vehicular networks. Figure 11 describes the architecture
used for the simulation.

Table 12 compares and highlights the approaches to attack detection, network setups,
and methodologies used in the four architectures. Each architecture employs unique
techniques to simulate, detect, and evaluate attack scenarios in 5G-and-beyond networks.

Mathematics 2025, 13, x FOR PEER REVIEW 34 of 60    Figure 10. TA-IDPS system design for 5G cloud-connected MANETs. For intrusion detection, scikit-learn was used (v1.1.1). The first phase involved DTs, trained on four datasets with different attacker numbers, evaluated using 10-fold cross-vali-dation across tree depths {2–55}, prioritizing F1 score due to class imbalance. The second phase utilized RF and MLP, trained on three datasets and tested on the fourth, using GroupKFold cross-validation. RFs were optimized over tree depths {2–10} and tree counts {10–50}, while MLP was tuned for batch sizes {32, 64}, hidden layers {(10,2), (20,2)}, optimizers (“adam”, “sgd”), and activation functions (“tanh”, “relu”). This approach enhances intrusion detection in 5G vehicular networks. Figure 11 describes the architecture used for the simulation.  Figure 11. Basic framework. Mathematics 2025, 13, 1088

36 of 63

Table 12. Comparison of network architecture for 5G attack detection.

Network Architecture

[92]

[93]

[94]

[96]

[97]

[98]

[99]

Feature

Objective

Creating a dataset for
intrusion detection in
5G networks

DDoS attack
detection using DTL

Real-time detection
of DoS and DDoS
attacks using ML
algorithms

Analyzing 5G
network traffic and
detecting attack
anomalies

Network setup

5G network testbed
with Free5GC and
UERANSIM for
simulation

5G testbed with
Free5GC and
UERANSIM,
focusing on attack
scenarios

Real-time system
with ML algorithms
for DoS/DDoS
detection

5G-CN using
Open5GS,
UERANSIM, and
Prometheus for
monitoring

Key attacks
detected

Various DoS attacks,
including UDP flood,
TCP SYN flood

DDoS attacks like
UDP flood, TCP SYN
flood, and more

DoS and DDoS
attacks (e.g., SYN
flood, ICMP flood)

GTP-U DoS, attach
request flooding,
PFCP-based attacks

Traffic source

Automated Python
scripts simulating
web browsing,
streaming, etc.

Simulated and
real-world traffic
flows with attack
scenarios

Traffic from mobile
devices using
headless browsers

Live traffic from
mobile devices and
Raspberry Pi-based
attackers

Dataset

5G-NIDD dataset
with benign and
malicious traffic

5G-NIDD with
labeled benign and
malicious traffic
types

ML-based dataset
from real-time
attacks in 5G
networks

5GTN dataset (live
traffic from actual
mobile devices)

Develop a
privacy-preserving
IDS using SSL with
minimal labeled data
to protect 5G-V2X
networks

5G-V2X environment
with CAVs connected
via Uu and PC5
interfaces; IDS
deployed as NVFs in
MEC-enabled
network slices

Detects various cyber
threats, including
DDoS and other
intrusion attacks in
vehicular networks

Combines extensive
unlabeled vehicular
network traffic with a
small set of expert-
labeled samples

A large-scale 5G-V2X
traffic dataset refined
with minimal expert
labeling for SSL
pre-training and
fine-tuning

Secure 5G
MANET–Cloud
against
masquerading,
MITM, and
black/gray-hole
attacks; reduce
energy and delay

Develop an IDS to detect
flooding attacks in
5G-enabled vehicular
(IoV) scenarios

5G-based MANET
integrated with
cloudlets and a cloud
service layer

5G vehicular network
simulated via NS-3
(5G-LENA) and SUMO
with senders and
receivers

Masquerading,
MITM, black-hole,
gray-hole attacks

Flooding attacks

Mobile node traffic
aggregated at cluster
heads and forwarded
via cloudlets

Simulated vehicular
traffic with diverse
mobility and node
densities

Simulated 5G
MANET–Cloud
traffic (NS3.26)

Four datasets with 45, 45,
70, and 100 vehicles (2, 4,
7, and 9 attackers);
features include time,
nodeId, imsi, packet
details, delay, jitter,
coordinates, speed, and
attack class

Mathematics 2025, 13, 1088

37 of 63

Feature

Table 12. Cont.

Network Architecture

[92]

[93]

[94]

[96]

[97]

[98]

[99]

Testbed
environment

Testbed with
Free5GC and
UERANSIM

5G testbed with
Free5GC and
UERANSIM

Real-time 5G
network with attack
simulation

Nokia Flexi Zone
Indoor Pico Base
Stations, Dell N1524
Switch

Training on edge
devices with model
aggregation on a
cloud server; final
IDS deployed as an
NVF in the MEC
environment

NS3.26 simulation

Ubuntu VM (i5-8300H,
four cores, 8 GB RAM)
using NS-3 and SUMO

Key methodology

Feature extraction
from live traffic using
CICFlowMeter

DTL for DDoS attack
detection in 5G
networks

ML algorithms for
classification and
real-time detection

PCA, t-SNE, and
UMAP for attack
classification and
visualization

Uses SSL pre-training
on edge devices with-
in a FL framework,
followed by super-
vised fine-tuning

UL symmetric crypto,
MFO clustering,
DBN classification,
adaptive Bayesian
routing

ML-based IDS using DTs,
RFs, and MLP;
optimized via grid
search and
cross-validation

Traffic type for
DDoS detection

HTTP, HTTPS, SSH,
SFTP, DDoS traffic

HTTP, HTTPS, SSH,
SFTP, and malicious
DDoS traffic

HTTP, HTTPS, SSH,
SFTP, DoS, and DDoS
traffic

HTTP, HTTPS, SSH,
SFTP, attack traffic

Not exclusively for
DDoS; designed to
detect a range of
intrusion types in
vehicular networks

Feature
extraction

CICFlowMeter for
flow-based analysis

CICFlowMeter for
flow-based analysis

Traffic features used
for ML-based
detection

PCA for feature
reduction, focus on
flow duration and
packet length

Automatic DL-based
extraction during SSL
pre-training

Data
preprocessing

Downsampling for
class imbalance,
normalization

Downsampling for
class imbalance,
feature selection

Data scaling and
balancing for
ML-based models

Removal of infinite
values, NaN
conversion

Leverages SSL to
minimize manual
labeling; supports
data augmentation
for enhanced
robustness

Focuses on intrusion
attacks rather than
DDoS

Vehicular traffic under
simulated flooding
attacks

Extracts node
direction, position,
distance, RSSI, trust
value, and residual
energy via DBN

nodeId, imsi, packet size,
dstPort, delay, jitter,
coordinates, and speed
from simulation logs

Includes node
registra-
tion/authentication
(U-LSCT) and
clustering (MFO)

Removal of features
causing overfitting and
hyperparameter tuning
to optimize classifier
performance

Mathematics 2025, 13, 1088

38 of 63

Feature

Table 12. Cont.

Network Architecture

[92]

[93]

[94]

[96]

[97]

[98]

[99]

Evaluation
metrics

Metrics for traffic
classification
accuracy

Evaluation using
DTL-based models

Evaluation using
real-time detection
metrics

MMD between
source and target
datasets

Use case

Intrusion detection in
5G networks

DDoS attack
detection in 5G
networks

Real-time attack
detection using ML

Anomaly detection in
5G network traffic

Assessed using
accuracy, precision,
recall, F1-score, and
efficiency; up to 9%
improvement even
with limited labeled
data

Real-time intrusion
detection for 5G-V2X
networks, ensuring
robust automotive
cybersecurity while
preserving data
privacy

DR, FPR, energy
consumption, PDR,
throughput, routing
overhead, and delay

Primary metric: F1 score;
also reports accuracy,
precision, and recall

Enhance security and
QoS in 5G
MANET–Cloud
environments

Enhance security in
5G-enabled IoV by
accurately detecting
flooding attacks in
vehicular networks

Mathematics 2025, 13, 1088

39 of 63

Figure 11. Basic framework.

6. Knowledge Discovery in Databases (KDD)

The KDD process is a systematic and iterative approach used to identify meaningful
patterns in data, enabling insightful event interpretation and predictive analysis [100]. This
process consists of several key phases: data collection, preprocessing, reduction, mining,
and interpretation. Data collection involves acquiring raw data from various sources to
address specific analytical challenges. Preprocessing ensures data quality by cleaning,
organizing, and addressing noise, outliers, and missing values while extracting relevant
features to enhance data structuring. Data reduction improves computational efficiency
through feature selection and dimensionality reduction. In the data mining phase, ML
and analytical techniques are applied to extract patterns, with algorithm selection tailored
to the problem requirements and dataset characteristics. Finally, the interpretation and
evaluation phase assesses the performance of data mining models, visualizing results to
facilitate informed decision-making.

In NIDS, the KDD process begins with the acquisition of network data, followed
by preprocessing, which structures raw network traffic into organized records. Data
reduction techniques may be applied to enhance computational efficiency. The data mining
phase then identifies intrusion patterns, while system performance is evaluated using
key metrics such as detection and false alarm rates. The hierarchical nature of network
infrastructure influences data visibility in NIDS. CN nodes offer a broad detection scope
but require high processing throughput due to the large data volumes they handle. In
contrast, lower-level network devices capture localized traffic with reduced noise but offer
limited coverage [101].

Network data collection can be performed through various methods. Raw packet
capture retrieves complete network packets, including headers and payloads, providing
comprehensive insights into traffic. However, this approach requires significant storage
and processing resources. Tools such as Tcpdump and Gulp facilitate this method, although

Mathematics 2025, 13, x FOR PEER REVIEW 34 of 60    Figure 10. TA-IDPS system design for 5G cloud-connected MANETs. For intrusion detection, scikit-learn was used (v1.1.1). The first phase involved DTs, trained on four datasets with different attacker numbers, evaluated using 10-fold cross-vali-dation across tree depths {2–55}, prioritizing F1 score due to class imbalance. The second phase utilized RF and MLP, trained on three datasets and tested on the fourth, using GroupKFold cross-validation. RFs were optimized over tree depths {2–10} and tree counts {10–50}, while MLP was tuned for batch sizes {32, 64}, hidden layers {(10,2), (20,2)}, optimizers (“adam”, “sgd”), and activation functions (“tanh”, “relu”). This approach enhances intrusion detection in 5G vehicular networks. Figure 11 describes the architecture used for the simulation.  Figure 11. Basic framework. Mathematics 2025, 13, 1088

40 of 63

challenges arise from encrypted traffic and protocol variability [102]. An alternative is
flow-level data collection, where related packets are aggregated into flows based on shared
attributes, such as IP addresses and ports. This method significantly reduces the data
volume while preserving the essential characteristics of network activity. However, it may
exclude payload details, and flow sampling is often used in high-traffic environments to
manage computational constraints. To overcome the limitations of single-point data collec-
tion, distributed data collection aggregates traffic data from multiple network points before
central analysis. This strategy enhances visibility and detection accuracy by integrating
diverse traffic sources.

By systematically applying the KDD process and leveraging a combination of packet-
level and flow-level data collection, NIDSs can enhance cybersecurity outcomes through
advanced traffic analysis and adaptive analytics [103]. Figure 12 illustrates the iterative
KDD framework in NIDS, highlighting stage-wise outcomes and adaptive feedback mecha-
nisms for continuous refinement.

Figure 12. Illustration of the iterative KDD process for NIDS.

6.1. Data Cleaning and Feature Transformation in NIDS Preprocessing

The preprocessing stage in NIDS is critical for handling data heterogeneity, reducing
noise, and enhancing detection accuracy. The feature vectors used in intrusion detection
contain discrete and continuous attributes, necessitating transformation to ensure compat-
ibility with data mining algorithms. The presence of noise, such as missing values and
outliers, can significantly degrade the performance of learning models. To mitigate these
effects, techniques like Mahalanobis distance-based outlier detection and feature thresh-
olding are commonly employed [104]. Additionally, categorical data must be converted
into numerical representations to facilitate processing by ML models. Various encod-
ing strategies, including number encoding, one-hot encoding, and frequency encoding,

Mathematics 2025, 13, x FOR PEER REVIEW 38 of 60    Figure 12. Illustration of the iterative KDD process for NIDS. 6.1. Data Cleaning and Feature Transformation in NIDS Preprocessing The preprocessing stage in NIDS is critical for handling data heterogeneity, reducing noise, and enhancing detection accuracy. The feature vectors used in intrusion detection contain discrete and continuous attributes, necessitating transformation to ensure compati-bility with data mining algorithms. The presence of noise, such as missing values and outli-ers, can significantly degrade the performance of learning models. To mitigate these effects, techniques like Mahalanobis distance-based outlier detection and feature thresholding are commonly employed [104]. Additionally, categorical data must be converted into numerical representations to facilitate processing by ML models. Various encoding strategies, includ-ing number encoding, one-hot encoding, and frequency encoding, are utilized—each pre-senting distinct trade-offs between dimensionality and predictive accuracy. Furthermore, ensuring numerical feature comparability is crucial for model stability, as techniques such as min-max normalization, standardization, and logarithmic transformation are applied to enhance convergence and minimize biases. Discretization is pivotal in simplifying continuous attributes and improving algorithm efficiency and interpretability. Traditional approaches, such as equal-width discretization, equal-frequency discretization, K-means clustering, and entropy minimization discretiza-tion, segment continuous data into discrete intervals to optimize computational efficiency. However, more advanced methodologies, including dynamic K-means clustering, remain underexplored despite their adaptability to dynamic network environments [105]. Given the vast and often redundant feature sets in NIDS, dimensionality reduction is crucial for eliminating irrelevant variables, reducing computational costs, and enhancing detection accuracy. Feature reduction techniques can be categorized into three primary ap-proaches: manual feature removal, feature subset selection, and feature projection. Feature subset selection methods, including wrapper, filter, and embedded techniques, enable the identification of the most relevant attributes. Meanwhile, feature projection methods such as Principal Component Analysis (PCA), Independent Component Analysis (ICA), and Au-toencoder (AE) transform data into a more compact and informative representation. Despite Mathematics 2025, 13, 1088

41 of 63

are utilized—each presenting distinct trade-offs between dimensionality and predictive
accuracy. Furthermore, ensuring numerical feature comparability is crucial for model
stability, as techniques such as min-max normalization, standardization, and logarithmic
transformation are applied to enhance convergence and minimize biases.

Discretization is pivotal in simplifying continuous attributes and improving algorithm
efficiency and interpretability. Traditional approaches, such as equal-width discretization,
equal-frequency discretization, K-means clustering, and entropy minimization discretiza-
tion, segment continuous data into discrete intervals to optimize computational efficiency.
However, more advanced methodologies, including dynamic K-means clustering, remain
underexplored despite their adaptability to dynamic network environments [105].

Given the vast and often redundant feature sets in NIDS, dimensionality reduction is
crucial for eliminating irrelevant variables, reducing computational costs, and enhancing
detection accuracy. Feature reduction techniques can be categorized into three primary
approaches: manual feature removal, feature subset selection, and feature projection. Fea-
ture subset selection methods, including wrapper, filter, and embedded techniques, enable
the identification of the most relevant attributes. Meanwhile, feature projection methods
such as Principal Component Analysis (PCA), Independent Component Analysis (ICA),
and Autoencoder (AE) transform data into a more compact and informative representation.
Despite the effectiveness of these techniques, challenges remain, including selecting optimal
feature subsets and addressing concept drift in evolving network environments. Further
research is required to refine these methodologies and ensure robust feature engineering
in NIDS.

6.2. Data Acquisition and Processing

The research [106] outlined a comprehensive multi-stage data collection and processing
process aimed at developing a high-quality dataset for ML analysis in the context of IDS.
The data collection took place over two days, focusing on both attack and benign traffic
at two separate base stations. Attack sessions were captured in pcap format, with port
scan activities lasting 10 min and DoS attacks spanning 30 min. Notably, there was a 5 min
overlap during which both attacks were executed simultaneously. Concurrently, benign
traffic was continuously captured throughout the data collection period.

The initial postprocessing step involved removing the GTP-U layer using Tracewran-
gler, as it is specific to LTE and 5G networks. Subsequently, the Argus tool [107] was
employed to convert packet-based data into a flow-based format, aggregating packets with
common attributes to streamline the data, reduce its size, and facilitate subsequent analysis.
This process resulted in the extraction of 112 features, including key parameters such as
packet size and source/destination IP addresses.

Appropriate labels were assigned to distinguish between benign and attack traffic,
with unique labels for each attack type [108]. The data from both base stations were then
merged, yielding 1,215,890 flows. The feature selection phase utilized Pearson correlation
and ANOVA F-score techniques to identify the top 25 most relevant features, eliminating
redundant and highly correlated variables. Z-score normalization was applied to standard-
ize the dataset, ensuring that all features contributed equally by rescaling the data to have
a mean of zero and a variance of one [109].

The processed dataset was made available in multiple formats, including pcap, Argus,
and CSV, and was organized by session for ease of use. It was subsequently uploaded to the
IEEE DataPort platform [110] for broader access. To safeguard against data leakage during
preprocessing, the dataset was split into training and testing sets before any transformations
were applied. Crucially, all preprocessing steps, including feature scaling, encoding, and
selection, were independently applied to the training and test datasets to ensure unbiased

Mathematics 2025, 13, 1088

42 of 63

model evaluation and to promote the generalizability of the results. This meticulous
methodology resulted in creating a dataset well suited for machine learning-based Intrusion
Detection Systems (IDSs), providing robust and high-quality data for model training
and testing.

6.2.1. Data Transformation

Within the 112 extracted network flow features, some were categorical, which are not
inherently interpretable by ML algorithms. To facilitate their numerical representation, one-
hot encoding—a standard preprocessing technique—is applied. This approach transforms
a categorical variable with distinct categories into binary variables, where each observation
is assigned either 0 or 1, signifying the absence or presence of a specific category. To
implement this transformation efficiently, a custom Python script is utilized, ensuring
seamless integration with the ML pipeline.

6.2.2. Correlation Coefficient

To enhance the quality of the dataset, redundant features—those exhibiting the same
relationship with the output label—were identified and removed. Retaining multiple highly
correlated features introduces redundancy without contributing additional information.
Therefore, only one representative feature from each redundant group was maintained.
This selection process was based on the Pearson correlation coefficient, a statistical measure
that quantifies the linear dependence between two variables. The correlation coefficient
ranges from −1 to +1, where +1 indicates a strong positive correlation, −1 signifies a strong
negative correlation, and 0 represents no correlation. The mathematical representation of
Pearson correlation is given in Equation (1):

rxy =

COV (X, Y)
σxσy

=

E(cid:2)(X − µx)(cid:0)Y − µy
σxσy

(cid:1)(cid:3)

(1)

To assess relationships between features, pairwise correlation coefficients were com-
puted. Since both strong positive and strong negative correlations indicate a significant
relationship, absolute correlation values were considered. A threshold of 0.90 was set to
identify highly correlated feature pairs. Within each correlated pair, the feature with the
weaker association with the target variable was removed to optimize the feature set.

Following the elimination of redundant features, the ANOVA F-value was utilized to
determine the most influential features for classification. This approach was consistently
applied to both binary and multiclass classification tasks, ensuring that the feature selection
process effectively reduced dimensionality while preserving the predictive quality of
the dataset.

6.2.3. ANOVA Statistical Evaluation

Given that the dataset [111] consists of numerical input variables and the objective is
to classify the final output, the ANOVA F-score was selected as an effective method for
feature ranking. This statistical technique determines the importance of each feature by
analyzing the ratio of inter-group variance to intra-group variance, thereby identifying the
most significant attributes for classification.

To compute the necessary variances, the following expressions were used:
The variance across different groups was estimated as given in Equation (2):

Sbetween =

1
K − 1

∑K

i=1 ni(Mi − M)2

(2)

Mathematics 2025, 13, 1088

Similarly, the variance within each group was calculated using Equation (3):

Swithin =

1
N − K

∑K

i=1

∑ni

j=1 ni

(cid:0)Xij − Mi

(cid:1)2

43 of 63

(3)

Using these computed variances, the ANOVA F-score was determined as follows in

Equation (4):

F =

Sbetween
Swithin

(4)

where Mi represents the mean of the ith group, ni is the number of samples in that group, M
denotes the overall mean of the dataset, K indicates the number of groups, N is the total
number of observations, and Xij corresponds to the jth observation in the ith group.

The ANOVA test was conducted using the scikit-learn library to compute the F-scores
for each feature relative to the target variable. Features with lower F-scores were considered
less relevant and removed. The top 25 features were then selected for both binary and
multiclass classification, ensuring that only the most significant attributes were retained to
enhance model performance.

6.2.4. Data Standardization

Minimizing training time is a critical factor in evaluating the efficiency of an IDS. Stan-
dardizing data before training the ML model plays a key role in optimizing performance
by ensuring that no single feature disproportionately influences the learning process. This
process involves scaling values to a predefined range, thereby preventing larger values
from dominating smaller ones during model training. Z-score normalization is employed,
as given in Equation (5), which transforms the data using its mean and standard deviation,
resulting in a standardized dataset with a mean of zero and a variance of one. This ensures
that all features contribute equally to the model training process, improving convergence
speed and overall performance.

ˆxi,n =

xi,n − µi
σi

(5)

6.3. Data Mining Stage

The data mining phase in the KDD process plays a pivotal role in detecting pat-
terns and identifying intrusions within network traffic. NIDSs are broadly classified into
misuse-based, anomaly-based, and hybrid detection systems. Misuse-based NIDSs lever-
age predefined attack signatures and are highly effective against known threats, employing
SL algorithms such as NNs, SVM, KNN, and DTs. However, these systems require contin-
uous updates to detect emerging variants of attacks. In contrast, anomaly-based NIDSs
model normal network behavior and flag deviations as potential threats, typically rely-
ing on USL techniques. While they exhibit superior capabilities in identifying zero-day
attacks, their susceptibility to high false alarm rates remains a critical challenge. Hybrid
NIDSs integrate both approaches to achieve an optimal balance between detection accuracy
and adaptability.

Detection methodologies can be categorized into SL, USL, and SSL approaches. SL
relies on fully labeled datasets, ensuring high detection accuracy, but this requires extensive
data annotation efforts. USL is particularly valuable for detecting novel and unknown
attacks, as it does not depend on labeled datasets but often suffers from lower accuracy. SSL
offers a compromise by utilizing a limited amount of labeled data to enhance the robust-
ness of detection models. While misuse-based NIDSs require frequent signature updates,
anomaly-based systems must address the challenge of high FPRs [112]. The effectiveness
of NIDSs is significantly influenced by the selection of learning paradigms, where full
SL provides high precision but entails substantial annotation costs. In contrast, USL and

Mathematics 2025, 13, 1088

44 of 63

SSL methods introduce adaptability, making them particularly suited for evolving cyber
threats. Future research directions should focus on enhancing adaptability, minimizing
dependency on labeled data, and optimizing the trade-off between detection accuracy and
computational efficiency.

Another critical consideration in NIDS learning models is the distinction between
batch and incremental learning approaches. BLM is well suited for static network envi-
ronments but struggles with evolving attack landscapes, necessitating frequent retraining.
Conversely, incremental learning continuously updates models, enabling real-time adap-
tation to dynamic network traffic. While incremental learning enhances responsiveness,
it is inherently more noise-sensitive and may require additional computational resources.
A promising direction for scalable and adaptive NIDSs is the development of hybrid
learning frameworks that integrate both batch and incremental strategies—leveraging
BLM for initial training and incremental updates to ensure ongoing adaptation. Such
hybrid paradigms can significantly improve the scalability, adaptability, and real-time
threat detection capabilities of NIDSs in highly dynamic environments [113].

6.4. Performance Assessment

The evaluation stage of NIDSs plays a pivotal role in assessing the effectiveness of
the preceding stages by considering various factors such as detection accuracy, complexity,
adaptability, interpretability, and security. Among these, predictive performance metrics
are the most frequently reported, providing quantifiable measures for comparing different
NIDS approaches. While computational complexity has a significant impact on real-
world deployment, few studies offer comprehensive evaluations. Most assessments focus
solely on the computational cost of the data mining algorithms, often overlooking other
critical factors such as data preprocessing, implementation choices (e.g., programming
languages, libraries, compilers), and runtime environments. A holistic evaluation of these
elements necessitates a controlled experimental setup, which remains an open challenge in
NIDS research.

6.4.1. Evaluation Technique

NIDS performance is often assessed using cross-validation (CV) in BLM settings,
where the dataset is randomly divided into multiple folds, and the performance is averaged
across multiple iterations. However, CV is less suitable for incremental learning due to
its assumption of independent and identically distributed (i.i.d.) samples, disregarding
temporal dependencies and concept drift in network traffic. Alternative frameworks, such
as incremental cross-validation and prequential evaluation, address these limitations by
progressively evaluating models as new data become available. The prequential evalua-
tion compares predictions against actual labels before updating the model, ensuring that
performance estimates remain dynamic. A k-fold prequential evaluation approach further
enhances robustness by combining both methods.

Several performance metrics, summarized in Table 13, are used to evaluate the predic-
tive capability of NIDS, which is often referred to by different names in various studies. Key
metrics, such as Detection Rate (DR), True Negative Rate (TNR), Accuracy, and Precision,
should ideally approach 1, indicating strong performance. Conversely, the FPR and False
Negative Rate (FNR) should be minimized, as they represent failure rates.

Mathematics 2025, 13, 1088

45 of 63

Table 13. Key performance metrics extracted from the confusion matrix [113].

Metric

Formula

Description

DR, TPR, Recall, Sensitivity

FAR, FPR, Specificity

TNR

FNR, Miss Rate

Precision

Accuracy

Error Rate

F-measure

TP
TP+FN

FP
FP+TN

TN

FP+TN = 1 − FPR

FN

TP+FN = 1 − TPR

TP
TP+FP

TP+TN
TP+TN+FP+FN

FP+FN

TP+TN+FP+FN = 1 − Acc

2 ∗ Precision∗Recall
Precision+Recall

Latency in IDS Detection (ms)

(Time of Detection–Time of Attack Start)

Measures the percentage of actual attacks
correctly detected.

Measures the rate of normal traffic being
falsely classified as an attack.

Measures the percentage of normal traffic
correctly classified as benign.

Measures the rate of attacks incorrectly
classified as benign.

Measures the accuracy of attack detection
when an attack is classified.

Measures the overall performance of IDS
in detecting attacks correctly.

Measures the percentage of misclassified
instances.

Harmonic mean of precision and recall,
useful when data are imbalanced.

Measures the time taken by an IDS to
detect an attack, crucial for 5G real-time
security.

Energy Efficiency (Joules/Detection)

Total Power Consumption/Number of
Detections

Measures the power consumption of IDS,
crucial for 5G edge computing.

False Alarm Rate (FAR) for 5G Slices

FP in Slice X/(FP in Slice X + TN in Slice
X)

Evaluates misclassification of benign
traffic in specific 5G network slices.

Transferability Score (TS)

(Performance on Target
Domain)/(Performance on Source
Domain)

Adaptive Learning Rate in TL-based IDS

∆ Performance/∆ Training Time

Evaluates how well a model trained in
one environment adapts to a new
dataset/domain.

Tracks how efficiently a TL-based IDS
adapts to new threats.

Model Drift Sensitivity

1-(Performance on Old
Data/Performance on New Data)

Measures the impact of changing attack
patterns on IDS performance over time.

Packet Inspection Overhead (%)

(Computation Time for IDS/Total
Network Processing Time) * 100

Evaluates the computational burden
introduced by an IDS during deep packet
inspection.

6.4.2. Challenges in Incremental Evaluation Technique

The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between
the True Positive Rate (TPR) and the False Positive Rate (FPR) across different decision
thresholds, with the Area Under the Curve (AUC) used to assess overall performance. In
NIDS research, DR and FPR are key metrics, where an effective IDS aims to maximize
DR while minimizing FPR. A significant challenge in incremental evaluation is the high
initial error rates resulting from model immaturity and performance degradation due to
concept drift. To mitigate this, forgetting mechanisms, such as sliding windows and fading
factors, help models adapt by prioritizing recent data. A study [114] emphasizes the need
for robust evaluation methodologies to ensure the effectiveness of NIDSs in dynamic and
imbalanced network conditions.

6.4.3. Attacks and Impact Analysis

The NAS SMC Replay Attack in the 5Greplay system is implemented through two con-
figuration files, 5greplayudp.conf and 5greplaysctp.conf, which facilitate testing under various

Mathematics 2025, 13, 1088

46 of 63

network protocols. A pre-captured pcap file (ue_authentication.pcapng), which records the
UE-AMF authentication dialogue, is processed for analysis. This enables the examination
of the NAS SMC packet and verifying whether it is appropriately filtered and forwarded.
Tools like Wireshark or tcpdump can be used further to evaluate the effectiveness of filtering
and forwarding rules.

Analysis of Open5GS AMF logs, after injecting high-bandwidth traffic through the UE,
revealed that the AMF crashed due to its failure to decode Next-Generation Application
Protocol (NGAP) messages, disrupting communication with other network components.
Warnings regarding Abstract Syntax Notation and NGAP–Protocol Data Unit decoding fail-
ures indicated potential data corruption, which can lead to network instability. The attack
also caused errors in all UEs connected to the AMF, resulting in connectivity disruptions
and delays. UEs attempting to reconnect encountered public land mobile network selection
failures and struggled to find a cell, resulting in repeated connection attempts and further
instability. This highlights the urgent need for enhanced mechanisms in 5G CNs to handle
traffic surges and ensure reliability.

DDoS attacks launched by an attacker UE had two significant impacts on the network:
network connection disruptions for the victim UEs, impairing their communication ability,
and delays in connectivity, which affected data transmission speeds and overall network
performance. Log analysis revealed critical issues, including cell selection failures, which
indicated the network’s inability to establish connections, and a lack of response from
Radio Resource Control (RRC), signaling a communication breakdown between devices
and control units. These disruptions underscore the vulnerabilities of 5G networks to
cyberattacks, compromising normal operations and leading to failed connections and
communication failures.

6.4.4. ML Models and Performance Evaluation

The study by [115] trained 13 ML models, including linear classifiers (LR, LDA), nonlin-
ear models (SVM, NNs), ensemble methods (RF, GB), and a Voting Classifier. These models
were tested on a preprocessed dataset using default parameter settings. Performance was
evaluated using accuracy, precision, recall, F1-score, ROC, AUC, and execution time, each
providing insights into the model’s effectiveness. Execution time was significant for real-
time applications. Table 14 describes various machine learning (ML) algorithms, including
decision trees (DTs), random forests (RFs), K-Nearest Neighbors (KNN), Naive Bayes (NB),
and multilayer perceptrons (MLPs), which were optimized through hyperparameter tuning
using a grid search.

Table 14. Key ML algorithms for IDS.

Reference

Model

Description

[116]

[116]

[117]

[118]

[119]

DTs

RF

KNN

NB

MLP

Classifies data using specific rules with decision nodes, branches, and leaves;
implemented using scikit-learn in Python

An ensemble method that combines multiple DTs, aggregates predictions through voting,
handles overfitting, and is noise-tolerant

Assigns labels based on the nearest neighbors to a given data point; effective for IDSs

Based on Bayesian theory, assumes conditional independence of attributes; performs well
with high-dimensional data

Fully connected feedforward NNs are widely used for IDSs due to their high performance

For binary classification, a 70-30 train–test split was used to distinguish between
malicious and benign traffic. The experiments tested different feature sets (5, 10, 15, 20, 25),
and the top 10 features delivered optimal accuracy and computational time performance.

Mathematics 2025, 13, 1088

47 of 63

A 70-30 train–test split was used for binary classification, distinguishing malicious from
benign traffic. The top 10 features provided optimal accuracy and computational efficiency.
For multiclass classification, detecting nine attack types (e.g., HTTP flood, ICMP flood,
SYN flood), the top 10 features also yielded the best performance.

RF detected malicious traffic with fewer FNs, while DT minimized FPs. NB was
the fastest in training but performed poorly in classification. MLP showed slightly better
accuracy but had longer training times. KNN had longer prediction times than training.
Confusion matrix analysis revealed that UDP flood and UDP scan were misclassified as
benign, whereas HTTP flood and slow-rate DoS attacks were often misidentified. The
results suggest that RF and MLP performed best, while NB was the least suitable for this
classification task.

6.4.5. Evaluation of DL Models

This research examines the performance of several deep learning (DL) models, specif-
ically BiLSTM- and CNN-based architectures (including CNN, ResNet, and Inception),
in DDoS detection for 5G networks. The models were developed using Keras, a Python
library built on TensorFlow. They were trained on various platforms, including Google Co-
laboratory, Compute Canada, and a local laptop with an Intel Core i7 processor and 16 GB
of RAM. The source and target datasets used in the experiments consisted of eight features
and binary labels for the attack and benign categories. The data were split into 80% for
training and 20% for testing, with a 0.2 validation split. Hyperparameter optimization was
conducted through trial and error, focusing on the learning rate, number of convolutional
layers, L2 regularization strength, and dropout rate, significantly impacting model perfor-
mance. The Adam optimizer was employed with a learning rate of 1e-5, and the models
were trained for 200 epochs, with early stopping implemented to prevent overfitting.

The empirical evaluation demonstrated that the BiLSTM model consistently outper-
formed CNN-based models (including ResNet and Inception) across most metrics. In
particular, the BiLSTM model achieved superior results on the custom dataset, attaining
an accuracy of 98.74%, a recall of 97.90%, a precision of 99.62%, and an F1-score of 98.75%.
This was accompanied by relatively short inference times on both CPU and GPU. Inception,
although slightly slower, also demonstrated strong performance, particularly in terms of
accuracy and F1 score. ResNet and CNN performed well but were somewhat less effective
in comparison. This study further evaluated model performance on the 5G-NIDD dataset,
with the BiLSTM model showing substantial improvements over the baseline across various
DTL scenarios.

BiLSTM models consistently demonstrated superior performance across various met-
rics, particularly in terms of recall and F1-score. Inception excelled in precision. These
findings guide model selection based on specific task requirements and emphasize the
importance of DTL strategies in enhancing model performance for practical applications in
5G network security.

The Consumer-Centric Internet of Things (CIoT) is a key enabler of Industry 5.0, yet
its widespread adoption introduces significant security challenges. Recent studies have
explored federated learning (FL) for privacy-preserving intrusion detection in IoT; however,
many existing models rely on outdated datasets, fail to cover diverse attack scenarios,
and exhibit limitations in classification performance. To address these gaps, a study [54]
developed Federated Deep Learning (FDL) models using three recent datasets, enabling
both binary and multiclass classification. The findings demonstrate that FDL achieves high
accuracy (99.60 ± 0.46%), precision (92.50 ± 8.40%), recall (95.42 ± 6.24%), and F1-score
(93.51 ± 7.76%), performing on par with centralized deep learning models. Moreover, FDL

Mathematics 2025, 13, 1088

48 of 63

significantly improves computational efficiency, reducing training time by 30.52% to 75.87%
through distributed local training.

To evaluate model robustness, three distinct threat models relevant to CIoT are con-
sidered. Using the X-IIoTID dataset, FDL effectively detects nine major cyber threats,
including reconnaissance, weaponization, exploitation, lateral movement, command and
control, exfiltration, tampering, crypto-ransomware, and ransom denial of service. Attack-
ers exploit vulnerabilities through port scanning, brute-force attacks, MITM intrusions,
and unauthorized data exfiltration. The Edge-IIoTset dataset further validates the model’s
ability to detect information gathering, DoS/DDoS attacks, MITM attacks, injection attacks,
and malware threats, with adversaries leveraging TCP/UDP flooding, SQL injections, and
malware propagation techniques. Similarly, the WUSTL-IIoT-2021 dataset confirms the
effectiveness of FDL in identifying reconnaissance attempts, command injection attacks,
DoS attacks, and backdoor intrusions in IIoT environments.

By integrating DL with decentralized training, the proposed FDL framework offers
a scalable, privacy-preserving, and computationally efficient solution for securing CIoT
ecosystems. The results highlight its potential in proactively identifying cyber threats,
reducing computational burdens, and enabling real-time threat mitigation. Future work
will focus on enhancing model interpretability, strengthening adversarial robustness, and
adapting FDL to emerging attack vectors to ensure resilient security in Industry 5.0.

6.4.6. Comparative Model Analysis

This study highlights the overall robustness of BiLSTM in performance across several
metrics. Figure 13 illustrates the performance metrics of a model during training and
validation over 200 epochs. Through key indicators such as accuracy, loss, and conver-
gence trends, the model demonstrates its learning development and detection strength in
identifying intrusions.

Figure 13. Training and validation accuracy (%) using the 5G-NIDD dataset.

7. Challenges and Limitations of the NIDS

Cybercriminals have developed increasingly sophisticated techniques to evade de-
tection by NIDSs. Among these, evasion strategies such as fragmentation, flooding, ob-
fuscation, and encryption present significant challenges to existing detection mechanisms.
In fragmentation attacks, packets are deliberately split into smaller fragments that are
reassembled at the IP layer before reaching the application layer. However, attackers can
manipulate this process using fragmentation overlap, overwriting, and exploiting time-

Mathematics 2025, 13, x FOR PEER REVIEW 45 of 60   importance of DTL strategies in enhancing model performance for practical applications in 5G network security. The Consumer-Centric Internet of Things (CIoT) is a key enabler of Industry 5.0, yet its widespread adoption introduces significant security challenges. Recent studies have ex-plored federated learning (FL) for privacy-preserving intrusion detection in IoT; however, many existing models rely on outdated datasets, fail to cover diverse attack scenarios, and exhibit limitations in classification performance. To address these gaps, a study [54] devel-oped Federated Deep Learning (FDL) models using three recent datasets, enabling both bi-nary and multiclass classification. The findings demonstrate that FDL achieves high accu-racy (99.60 ± 0.46%), precision (92.50 ± 8.40%), recall (95.42 ± 6.24%), and F1-score (93.51 ± 7.76%), performing on par with centralized deep learning models. Moreover, FDL signifi-cantly improves computational efficiency, reducing training time by 30.52% to 75.87% through distributed local training. To evaluate model robustness, three distinct threat models relevant to CIoT are consid-ered. Using the X-IIoTID dataset, FDL effectively detects nine major cyber threats, including reconnaissance, weaponization, exploitation, lateral movement, command and control, ex-filtration, tampering, crypto-ransomware, and ransom denial of service. Attackers exploit vulnerabilities through port scanning, brute-force attacks, MITM intrusions, and unauthor-ized data exfiltration. The Edge-IIoTset dataset further validates the model’s ability to detect information gathering, DoS/DDoS attacks, MITM attacks, injection attacks, and malware threats, with adversaries leveraging TCP/UDP flooding, SQL injections, and malware prop-agation techniques. Similarly, the WUSTL-IIoT-2021 dataset confirms the effectiveness of FDL in identifying reconnaissance attempts, command injection attacks, DoS attacks, and backdoor intrusions in IIoT environments. By integrating DL with decentralized training, the proposed FDL framework offers a scalable, privacy-preserving, and computationally efficient solution for securing CIoT eco-systems. The results highlight its potential in proactively identifying cyber threats, reducing computational burdens, and enabling real-time threat mitigation. Future work will focus on enhancing model interpretability, strengthening adversarial robustness, and adapting FDL to emerging attack vectors to ensure resilient security in Industry 5.0. 6.4.6. Comparative Model Analysis This study highlights the overall robustness of BiLSTM in performance across several metrics. Figure 13 illustrates the performance metrics of a model during training and vali-dation over 200 epochs. Through key indicators such as accuracy, loss, and convergence trends, the model demonstrates its learning development and detection strength in identi-fying intrusions.  Figure 13. Training and validation accuracy (%) using the 5G-NIDD dataset. Mathematics 2025, 13, 1088

49 of 63

outs to modify or replace information within the reassembled packet, thereby embedding
malicious payloads that escape detection. On the other hand, flooding attacks overwhelm
NIDSs by saturating the network with excessive traffic. By generating massive amounts
of spoofed User Datagram Protocol (UDP) and Internet Control Message Protocol (ICMP)
traffic, attackers not only mask their malicious activities within the flood but potentially
degrade the performance or cause outright failure of the NIDS. Obfuscation techniques
further complicate detection by modifying the appearance of malicious code—through
methods such as hexadecimal encoding, unicode manipulation, or double encoding—while
preserving its underlying functionality. This alteration makes it considerably harder for
detection systems to analyze or reverse-engineer the attack signatures.

Encryption represents another formidable evasion strategy, enabling malware authors
to conceal malicious activities within encrypted traffic, such as HTTPS communications.
Because encrypted content cannot be quickly inspected against known signatures, tra-
ditional content-based malware detection methods become ineffective, allowing attacks
to remain hidden. These advanced evasion techniques underscore the urgent need for
next-generation NIDSs that are robust against the evolving tactics employed by threat
actors [120]. Despite extensive research efforts, NIDSs continue to grapple with several
challenges. Achieving high detection accuracy while minimizing false alarms remains a
persistent challenge, particularly in environments such as industrial control systems, where
the combination of supervisory control and data acquisition hardware, along with control
software, makes them prime targets for state-sponsored actors, competitors, malicious
insiders, and hacktivists. Intrusion evasion detection remains a critical challenge; attackers
frequently employ sophisticated methods to conceal their activities, complicating both
signature-based and anomaly-based detection approaches. The effectiveness of an IDS
ultimately depends on its ability to reconstruct original attack signatures or to generate
new ones that can counter these modified attack patterns—a capability that many current
systems lack, especially when confronted with encryption-based attacks.

DL techniques have emerged as a promising solution for NIDS, particularly when
handling large and complex datasets. However, the application of deep learning (DL) in
this context presents its own set of challenges. The high cost and extensive effort required
for data labeling, as well as the significant computational resources needed for training
deep models, pose substantial barriers. Many existing DL-based solutions are trained on
outdated datasets, such as KDD Cup ‘’99 and NSL-KDD, which, although yielding strong
results in controlled settings, struggle to generalize to modern, refined datasets. Moreover,
detecting attacks with few training samples is further hindered by class imbalances, which
reduce accuracy. A critical trade-off exists between model complexity and computational
efficiency, as deeper models demand more time and resources. Emerging strategies, such as
TL, FL, and edge computing, have been proposed to address these challenges; however, no
single approach has proven universally effective. Despite these hurdles, research into DL-
based IDSs remains vibrant, with ongoing efforts to enhance detection accuracy, mitigate
computational demands, and improve the overall robustness of NIDSs against evolving
cyber threats.

The IIoT connects vast networks of heterogeneous smart devices, enhancing automa-
tion with minimal human intervention. However, device vulnerabilities and network
heterogeneity introduce significant security risks, making IIoT systems susceptible to
various cyber threats. Security measures such as encryption, authorization control, and
verification help protect network nodes. ML-based intrusion detection faces challenges due
to diverse network traffic patterns.

To improve detection efficiency, the study [121] highlights the effectiveness of ensemble
models combined with feature selection techniques for IIoT intrusion detection. The Chi-

Mathematics 2025, 13, 1088

50 of 63

Square Statistical method enhances feature selection, while classifiers such as XGB, Bagging,
Extra Trees, RF, and AdaBoost improve detection accuracy. Evaluating these models on the
TON_IoT dataset demonstrates their potential to enhance IIoT security, highlighting the
need for optimized feature selection and ensemble learning approaches to achieve more
efficient and accurate intrusion detection.

8. Recent Trends, Future Research Directions, and Lessons Learned

Recent advancements in 5G NIDSs have highlighted both opportunities and chal-
lenges in securing next-generation communication networks. Analyzing existing 5G NIDS
datasets has revealed critical insights into attack patterns, detection efficacy, and limita-
tions of the datasets. This section highlights the strengths and shortcomings of current
approaches, synthesizing key lessons learned from these analyses. Additionally, it ex-
plores emerging research directions aimed at enhancing the robustness, scalability, and
real-time responsiveness of intrusion detection systems in dynamic 5G environments. By
addressing these challenges, future research can drive the development of more adaptive
and intelligent security solutions to safeguard 5G networks against evolving cyber threats.
This section presents key lessons learned from analyzing 5G NIDS datasets and outlines
future research directions to enhance intrusion detection systems for secure and efficient
5G networks.

8.1. Recent Trends

The evolution of NIDSs has been significantly influenced by advancements in AI and
ML, leading to improved threat detection accuracy, adaptability, and response efficiency.
While effective for known threats, traditional signature-based and rule-based detection
methods often struggle with zero-day attacks and evolving cyber threats. In response,
AI-driven approaches, mainly SL and USL-ML techniques, have demonstrated superior
performance in anomaly detection and threat classification. SL models have shown high
precision in identifying known attack signatures. At the same time, USL methods, such as
clustering and PCA, excel at detecting previously unknown threats by analyzing deviations
from normal network behavior.

A major challenge in traditional NIDSs has been the high rate of FPs, which can over-
whelm security teams with unnecessary alerts. Recent research has focused on reducing
false alarms by refining ML models to better distinguish between benign anomalies and
actual threats. Hybrid approaches that combine SL and USL have proven particularly
effective in addressing this issue, thereby enhancing the system’s ability to adapt to evolv-
ing attack patterns. Additionally, the integration of RL has enabled NIDSs to improve
response times by automating threat mitigation strategies and reducing reliance on manual
intervention. Real-time data processing frameworks, such as Apache Kafka and Apache
Spark Streaming, have further enhanced the efficiency of NIDSs by enabling the rapid
analysis of high-volume network traffic, thereby minimizing detection and response delays.
Privacy-preserving techniques are also gaining traction in modern NIDSs to address
concerns about data security and regulatory compliance. FL has emerged as a promising
approach, allowing multiple organizations to collaboratively train AI models without
sharing sensitive data. This decentralized learning framework enhances cybersecurity
while ensuring compliance with regulations, such as the General Data Protection Regulation.
However, implementing privacy-preserving AI techniques often comes with a trade-off in
detection accuracy, as noise introduced to protect data privacy can obscure critical threat
indicators. Researchers are actively exploring methods to optimize this balance, ensuring
security and high detection performance.

Mathematics 2025, 13, 1088

51 of 63

Another emerging trend in NIDSs is the push for XAI to improve the interpretability
of AI-driven detection mechanisms. While DL models have demonstrated remarkable
accuracy in intrusion detection, their black-box nature presents challenges in understanding
and validating detection decisions. XAI techniques aim to provide transparency, enabling
security analysts to interpret and trust AI-generated alerts. This is particularly important
in mission-critical environments where explainability is crucial for decision-making and
regulatory compliance [122].

8.2. Future Research Directions

Ensuring evaluation consistency in NIDSs is crucial for reproducible results, where
variations in performance can be attributed to model effectiveness rather than differences
in data selection. Maintaining fixed proportions for training and test sets across datasets
ensures uniformity in evaluating model performance. However, introducing dynamic
variability enables researchers to evaluate how changes in training conditions affect model
adaptability and detection accuracy. Unlike traditional approaches that rely on static
training sets, dynamic variability provides insights into how models learn and generalize
under evolving network conditions. Additionally, adjusting test set sizes to fit the 5G-
NIDD dataset facilitates a comprehensive analysis of adaptability and robustness, offering
a more nuanced understanding of model performance in real-world scenarios. The use
of multiple models further enhances benchmarking efforts, enabling the identification
of the most effective approaches for intrusion detection. Performance improvements are
expected as TL domains become more similar, addressing data limitations and optimizing
training efficiency.

Despite the robustness of the 5G-NIDD dataset, its controlled nature presents certain
limitations in assessing model generalizability. Future research should focus on integrating
real-world datasets to validate model performance under diverse conditions, ensuring
adaptability to dynamic network environments. Further optimization through advanced
techniques such as parameter tuning, feature engineering, and hyperparameter selection
will enhance detection capabilities. Exploring deep learning (DL) architectures, such as
MLP, DenseNet, and UNet, holds promise for improving classification accuracy and com-
putational efficiency. Extending these methodologies to broader cybersecurity challenges,
including intrusion detection across different 5G network slices, will facilitate the devel-
opment of more adaptive and resilient security mechanisms. Future research should also
investigate the integration of NIDSs with emerging technologies such as FL, XAI, and
blockchain to strengthen security frameworks. By advancing these areas, next-generation
NIDSs can be designed to effectively mitigate evolving cyber threats in 5G and beyond.

As cyber threats become more sophisticated, the future of NIDS research is expected
to focus on enhancing adversarial robustness, improving model interpretability, and inte-
grating AI with other emerging technologies such as blockchain and quantum computing.
AI-powered intrusion detection is increasingly seen as a vital component of modern cy-
bersecurity frameworks, offering proactive defense mechanisms against evolving threats.
However, balancing performance, interpretability, and privacy remains a critical chal-
lenge. Ongoing research and innovation in AI-driven Network Intrusion Detection Systems
(NIDSs) will be essential for developing more resilient, scalable, and efficient intrusion
detection systems that safeguard next-generation networks, including 5G and beyond.

8.3. Lesson Learned

Lesson 1: Reflections on IDS Technologies

This review presents a comprehensive analysis of IDS methodologies, approaches, and
technologies, emphasizing their strengths and inherent limitations in guiding the selection

Mathematics 2025, 13, 1088

52 of 63

of optimal techniques. Pattern-based IDSs, although simple to implement and highly
effective in detecting known attacks, struggle to identify zero-day threats, evasion-based
intrusions, and polymorphic attack variants. Rule-based approaches offer the potential for
detecting unknown attacks but suffer from the complexity of rule creation and continu-
ous updates. Heuristic-based techniques, though independent of prior attack knowledge,
impose high computational costs, making real-time deployment challenging. A holistic
understanding of IDS mechanisms and application-specific requirements is essential for ef-
fective implementation. To facilitate this, an in-depth review of IDS frameworks is provided,
supplemented by systematically structured tables and figures for a clear representation of
key insights. Additionally, two prominent open-source tools are introduced to support IDS
research and experimentation. With the increasing reliance on virtualization technologies,
particularly in cloud computing, IDS considerations in virtualized environments have
become crucial. As VMs serve as the primary interface for users, security vulnerabilities
targeting VMs pose significant risks, necessitating advanced IDS strategies to address
emerging challenges in virtualized infrastructures.

Lesson 2: Evolving AI Strategies in Cybersecurity

AI-powered threat detection and cybersecurity research highlight the evolving role
of artificial intelligence in addressing modern security challenges. ML and DL techniques
are increasingly applied to detect network intrusions, identify anomalies, and counter
adversarial attacks. A key focus in this field is improving the explainability and resilience
of AI models, ensuring transparency and trustworthiness in security decisions, which
is essential for user confidence and regulatory compliance. Additionally, enhancing the
robustness of AI models against zero-day threats and sophisticated adversarial attacks
remains a critical research priority. Applications of AI span various domains, including
Industry 5.0, IoT, 5G networks, and autonomous systems, each presenting unique security
challenges. Innovative detection techniques, such as transformer-based models for social
media threat analysis and blockchain-integrated FL, demonstrate the potential for real-time
threat detection and response. Furthermore, the adoption of collaborative and federated
security approaches enables multiple entities to work together in strengthening cyberse-
curity across distributed networks and IoT environments. Despite these advancements,
challenges persist, particularly in real-time data processing, large-scale data management,
and ensuring privacy and security in AI models. Addressing these issues will be essential
for the continued effectiveness and reliability of AI-driven cybersecurity solutions.

Lesson 3: Enhancing 5G IoT Security

The rapid expansion of the IoT and the advent of 5G mobile networks present signif-
icant opportunities for advancing network cybersecurity research. This study addresses
the comprehensive security of modern networks, emphasizing the necessity for IDSs to
undergo extensive validation using big data from diverse devices. The findings under-
score that ML techniques effectively automate intrusion detection processes, suggesting
potential for further development. Future research directions include creating new datasets
to capture contemporary network traffic patterns, as existing datasets may be outdated.
Implementing real-time traffic monitoring systems is also crucial for promptly detecting
threats. Applying ML approaches tailored explicitly to IoT environments can enhance
security measures. Additionally, safeguarding ML systems against potential exploits is im-
perative, as attackers might leverage vulnerabilities to access sensitive information within
5G databases. Exploring semi-supervised ML models is recommended, given the scarcity
of labeled data in many datasets. Investigating methods to detect adversarial attacks that
cause misclassification of unknown attack types is essential for robust IDS development.

Mathematics 2025, 13, 1088

53 of 63

Finally, adhering to communication and security standards in mobile applications is vital
to ensure comprehensive protection across all network layers.

Lesson 4: Ensuring Consistency in Evaluation

The first goal is to validate the reliability of the evaluation process by employing
standardized sampling methods and predetermined training and test set sizes. This begins
with gathering equal samples from both cyberattack types and benign cases to ensure a
balanced sample distribution. This approach prevents bias toward a particular category
during model training and evaluation. To ensure consistency, a fixed sampling technique is
applied across all iterations, which are repeated five times using the same “random_state”
values (42, 142, 12, 4, and 80).

Lesson 5: Incorporating Dynamic Variability

In contrast, the second objective acknowledges the inherent unpredictability of cy-
berattacks by introducing dynamic elements into the experimental design. This involves
using variable resampling techniques, unfixed training datasets, and evaluating model
performance as input data changes. The primary aim is to simulate real-world conditions.
IDSs must adapt continuously to emerging threats, assessing the models’ flexibility and
resilience. While maintaining an equal sample distribution between attack and benign
cases, the sampling method in each iteration is dynamic. Intrusion detection in cloud
environments is becoming increasingly complex due to the vast amount of data generated
and the evolving nature of threats. Key takeaways include the urgent need for IDS solutions
to address zero-day vulnerabilities, the importance of adaptive architectures for dynamic
cloud computations, and the potential of blockchain integration for enhanced security.
Future IDSs must be adaptable to changing environments, scalable with cloud expansion,
and capable of deploying additional VMs as needed. Advancing IDS technology is crucial
for ensuring resilience and efficiency in modern cloud security [123].

Lesson 6: Development of base models

BiLSTM- and CNN-based models, such as CNN, ResNet, and Inception, were applied
to one-dimensional data to build foundational models using the proposed dataset as the
source domain [124]. These models were selected for their proven success in similar
research. This research investigates the performance of LSTM and CNN layers in feature
generation, aiming to determine the most effective approach for the dataset.

Lesson 7: Domain Transfer Learning (DTL) Strategies

The source and target domains involve similar categories (benign and attack) but vary
in flow characteristics. Two main strategies for DTL are employed: (1) Strategy 1 focuses on
freezing most layers while retraining only a subset, and (2) Strategy 2 involves removing
the last layer, adding new layers, and retraining the model.

Lesson 8: Freezing Layers and Selective Retraining

•

• DTL0: In this scenario, all layers except the last one are frozen, and only the final
decision-making layer is retrained. This allows the model to retain the features learned
during the initial training while adjusting the final output layer to the specific task.
DTL1: This method freezes all layers except for the last 33%, which are then retrained.
This enables deeper layers to adapt to the new task while maintaining the integrity of
the initial feature extraction layers.
DTL2: Similar to DTL1, the last 66% of layers are unfrozen and retrained. This scenario
provides greater flexibility in adapting the model, allowing more parameters to be
updated compared to DTL1.

•

Mathematics 2025, 13, 1088

54 of 63

These scenarios are based on prior empirical research using the ResNet architecture,

which guided the selection of the 33% and 66% freezing percentages [55].

Lesson 9: Removing Layers, Adding New Layers, and Retraining

•

DTL3: In this scenario, all layers are frozen except for the last one, which is removed
and replaced by a new layer initialized randomly. This modification ensures the model
can adapt to the new task if the output structure differs from the original.

• DTL4: This is the same approach as DTL3, but two new layers are added instead of
one. This increases the model’s capacity to learn more complex patterns and provides
greater adaptability to the new task.
DTL5: In this scenario, three new layers are added after removing the last layer. This
offers the highest flexibility for adapting the model to tasks requiring significant
modifications in output behavior.

•

Each strategy explores how varying the number of frozen and added layers impacts
the model’s ability to adjust to new tasks, ensuring that the best possible approach is
applied to the specific scenario.

Lesson 10: Advancing Anomaly Detection with LSTM for Improved Network Security

Effective anomaly detection in computer networks requires robust feature selection
and DL techniques. While RNNs have been widely used for time-dependent data, their reli-
ability and speed limitations necessitate the development of alternative approaches. LSTM
networks address these challenges by introducing memory blocks with self-connecting cells,
mitigating the vanishing gradient problem and improving long-term dependency learning.
Unlike RNNs, which rely on shared parameters over sequential time steps, LSTM utilizes
three gating mechanisms to regulate data flow, enhancing its ability to capture both short-
and long-term correlations in time-series data. The study [125] highlights the superiority
of LSTM over traditional RNNs for intrusion detection, emphasizing its effectiveness in
handling sequential dependencies and improving anomaly detection accuracy.

Lesson 11: Choice of ML model

It depends on the ease of deployment. Complex models, such as neural networks
(NNs), require extensive training, fine-tuning, and ongoing maintenance to adapt to evolv-
ing threats, demanding significant resources and expertise. Simpler models are easier to
deploy and maintain but may require regular updates to remain effective, especially in
dynamic threat environments. This study highlights that selecting the most suitable ML
model involves balancing accuracy, computational efficiency, and ease of deployment based
on the application’s specific requirements, dataset characteristics, and available resources.
No single model performs optimally across all metrics; therefore, a careful and informed
approach is necessary to ensure the effectiveness and reliability of NIDSs in 5G networks.

Lesson 12: Integrating IDS, SDN Security, and ML

The study [126] highlights the critical interplay between IDSs, ML-based security mod-
els, and evolving SDN security challenges. The findings confirm that ZEEK and SNORT
can operate synergistically, with SNORT actively detecting threats and ZEEK passively
monitoring anomalies, enhancing network resilience against cyber threats. While deep
packet inspection of SNORT enables proactive threat identification, extensive traffic logging
of ZEEK strengthens forensic analysis. However, scalability concerns arise, as concurrent
execution on multiple VMs induces system overheating, emphasizing the need for op-
timized resource allocation. The increasing adoption of SDN introduces vulnerabilities,
particularly in the separation of control and data planes, making it susceptible to DDoS
attacks. To counteract this, the proposed framework integrates an intelligent intrusion

Mathematics 2025, 13, 1088

55 of 63

detection and prevention system that dynamically monitors TCP SYN requests, adapts
detection thresholds, and employs real-time mitigation strategies through Open vSwitch
flow validation. Additionally, an advanced ML-based model leveraging XGB demonstrates
superior accuracy in identifying malicious traffic patterns, provided robust data preprocess-
ing is performed to mitigate overfitting. Despite these advancements, limitations remain,
as the framework primarily addresses flooding-based DDoS attacks and lacks coverage
for low-rate, non-volumetric threats. Future research should focus on real-time training
with live network traffic, exploring multi-feature authentication techniques, and refining
ML-based threat classification to ensure adaptive, scalable, and intelligent cyber defense
mechanisms. Ultimately, the fusion of IDS, adaptive detection frameworks, and AI-driven
automation holds immense potential for fortifying modern networks against sophisticated
cyber threats, necessitating continued innovation and refinement.

9. Conclusions

This review offers an in-depth analysis of 5G network security challenges and the
critical role of Intrusion Detection Systems (IDSs) in mitigating these threats. It highlights
the limitations of existing IDS frameworks and datasets, emphasizing the need for more
reliable and representative data to enhance NIDSs. A robust testbed architecture integrated
with the 5GTN platform has been discussed, enabling the generation of realistic datasets
for improved evaluation of security solutions. Furthermore, this paper examines the
effectiveness of various machine learning (ML) models and deep learning (DL) approaches
for intrusion detection using the 5G-NIDD dataset, which provides a realistic representation
of 5G network traffic, including both benign and malicious scenarios. This study assesses
model performance under 5G network conditions, revealing that different models excel
in specific metrics. BiLSTM significantly improved accuracy, recall, and F1-score, while
Inception performed best in precision. These findings emphasize the importance of selecting
models based on specific security requirements to enhance network intrusion detection.
This study also highlights key evasion techniques employed by cybercriminals, including
fragmentation, flooding, obfuscation, and encryption, which pose significant challenges to
traditional Network Intrusion Detection System (NIDS) mechanisms.

Advancements in DL- and AI-driven approaches have significantly improved intru-
sion detection accuracy, adaptability, and response efficiency. The integration of SL and USL
has reduced FP and enhanced threat classification, while hybrid models and real-time data
processing frameworks have strengthened NIDS capabilities. AI-driven methods, including
DL and hybrid models, have also proven effective in countering evasion techniques such
as fragmentation, flooding, obfuscation, and encryption, which undermine traditional IDS
mechanisms. Additionally, privacy-preserving techniques such as federated learning (FL)
and explainable artificial intelligence (XAI) further enhance network intrusion detection in
5G and beyond, ensuring improved security and compliance with regulations.

Author Contributions: The manuscript was written through the contributions of all authors. Con-
ceptualization, K.N.; methodology, K.N., A.L.I., C.-T.L. and C.-Y.W.; software, C.-T.L. and C.-Y.W.;
validation, A.L.I., C.-T.L. and C.-Y.W.; formal analysis, A.L.I.; investigation, K.N. and A.L.I.; re-
sources, C.-T.L. and C.-Y.W.; data curation, K.N.; writing—original draft preparation, K.N. and A.L.I.;
writing—review and editing, K.N., A.L.I., C.-T.L. and C.-Y.W.; visualization, C.-T.L. and C.-Y.W.;
supervision, A.L.I.; project administration, C.-T.L. and C.-Y.W.; funding acquisition, A.L.I. All authors
have read and agreed to the published version of the manuscript.

Funding: This work was supported in part by the National Science and Technology Council in
Taiwan under contract no.: NSTC 113-2410-H-030-077-MY2.

Data Availability Statement: No new data were created or analyzed in this study.

Mathematics 2025, 13, 1088

56 of 63

Acknowledgments: The authors acknowledge the anonymous reviewers for their valuable comments.

Conflicts of Interest: The authors declare no conflicts of interest.

List of Acronyms

Abbreviation
ML
IDS
KNN
ROC
AUC
DDoS
DoS
DL
DTL
BiLSTM
CNN
NNs
DTs
NB
LR
GB
DNNs
ResNet
NIDSs
AI
IoT
TL
B5G
5GTN
CN
IIoT
DDPG
QoS
EDoS
ANNs
SVMs
SL
USL
SSL
RL
CG-GRU
XGB
eMBB
uRLLC
mMTC
NSA
SA
SBA
AMF
UPF
SDN
MITM
NFV

Full Meaning
Machine Learning
Intrusion Detection System
K-Nearest Neighbors
Receiver Operating Characteristic
Area Under the Curve
Distributed Denial of Service
Denial of Service
Deep Learning
Deep Transfer Learning
Bidirectional Long Short-Term Memory
Convolutional Neural Network
Neural Networks
Decision Trees
Naive Bayes
Logistic Regression
Gradient Boosting
Deep Neural Networks
Residual Network
Network Intrusion Detection Systems
Artificial Intelligence
Internet of Things
Transfer Learning
Beyond-5G
5G Test Network
Core Network
Industrial Internet of Things
Deep Deterministic Policy Gradient
Quality of Service
Economical Denial of Sustainability
Artificial Neural Networks
Support Vector Machines
Supervised Learning
Unsupervised Learning
Semi-Supervised Learning
Reinforcement Learning
Control Gated–Gated Recurrent Unit
Extreme Gradient Boost
Enhanced Mobile Broadband
Ultra-Reliable Low-Latency Communication
Massive Machine-Type Communications
Non-Standalone
Standalone
Service-Based Architecture
Access and Mobility Management Function
User Plane Function
Software-Defined Networking
Man in the Middle
Network Function Virtualization

Mathematics 2025, 13, 1088

57 of 63

RAN
SMF
SIDSs
AIDSs
HIDS
WIDS
NBA
MIDS
SNs
FPs
FNs
R-NIDS
GUI
ANOVA
UE
PFCP
MEC
SFTP
PCA
MMD
NVF
KDD
GTP-U
CV
TNR
FPR
FNR
TPR
NGAP
RRC
UDP
ICMP
XAI
VMs
RSS
CAVs
AE
SAE
RBM
RNN
LSTM
DBN
GRU
TA-IDPS
MANETs
TA
CH
MFO
CIoT
FL
FDL
BLMs
DSMs
HT

Radio Access Network
Session Management Function
Signature-Based Intrusion Detection Systems
Anomaly-Based Intrusion Detection Systems
Host-based IDS
Wireless-based IDS
Network Behavior Analysis
Mixed IDS
Standard Networks
False Positives
False Negatives
Reliable-NIDS
Graphical User Interface
Analysis of Variance
User Equipment
Packet Forwarding Control Protocol
Multi-access Edge Computing
Secure File Transfer Protocol
Principal Component Analysis
Maximum Mean Discrepancy
Network Virtualization Function
Knowledge Discovery in Databases
General Packet Radio Service Tunneling Protocol
Cross-Validation
True Negative Rate
False Positive Rate
False Negative Rate
True Positive Rate
Next-Generation Application Protocol
Radio Resource Control
User Datagram Protocol
Internet Control Message Protocol
Explainable AI
Virtual Machines
Received Signal Strength
Autonomous Vehicles
Autoencoder
Stacked Autoencoder
Restricted Boltzmann Machine
Recurrent Neural Network
Long Short-Term Memory
Deep Belief Network
Gated Recurrent Unit
Trust-Aware Intrusion Detection and Prevention System
Mobile Ad Hoc Networks
Trusted Authority
Cluster Head
Moth Flame Optimization
Consumer-Centric Internet of Things
Federated Learning
Federated Deep Learning
Batch Learning Models
Data Streaming Models
Hoeffding Tree

Mathematics 2025, 13, 1088

58 of 63

OBA
R2L
U2R
RF
QNNs

OzaBagAdwin
Remote-to-Local
User-to-Root
Random Forest
Quantum Neural Networks

References

1. Moubayed, A.; Manias, D.M.; Javadtalab, A.; Hemmati, M.; You, Y.; Shami, A. OTN-over-WDM optimization in 5G networks:

2.

3.

4.

5.

6.

Key challenges and innovation opportunities. Photonic Netw. Commun. 2023, 45, 49–66. [CrossRef]
Aoki, S.; Yonezawa, T.; Kawaguchi, N. RobotNEST: Toward a Viable Testbed for IoT-Enabled Environments and Connected and
Autonomous Robots. IEEE Sensors Lett. 2022, 6, 6000304. [CrossRef]
Siriwardhana, Y.; Porambage, P.; Liyanage, M.; Ylianttila, M. AI and 6G Security: Opportunities and Challenges. In Proceedings
of the 2021 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit), Porto, Portugal,
8–11 June 2021; pp. 616–621.
Dini, P.; Elhanashi, A.; Begni, A.; Saponara, S.; Zheng, Q.; Gasmi, K. Overview on Intrusion Detection Systems Design Exploiting
Machine Learning for Networking Cybersecurity. Appl. Sci. 2023, 13, 7507. [CrossRef]
Sadhwani, S.; Mathur, A.; Muthalagu, R.; Pawar, P.M. 5G-SIID: An intelligent hybrid DDoS intrusion detector for 5G IoT networks.
Int. J. Mach. Learn. Cybern. 2024, 16, 1243–1263. [CrossRef]
Ahuja, N.; Mukhopadhyay, D.; Singal, G. DDoS attack traffic classification in SDN using deep learning. Pers. Ubiquitous Comput.
2024, 28, 417–429. [CrossRef]

7. Nguyen, C.T.; Van Huynh, N.; Chu, N.H.; Saputra, Y.M.; Hoang, D.T.; Nguyen, D.N.; Pham, Q.-V.; Niyato, D.; Dutkiewicz, E.;

8.

9.

10.

Hwang, W.-J. Transfer Learning for Wireless Networks: A Comprehensive Survey. Proc. IEEE 2022, 110, 1073–1115. [CrossRef]
Bouke, M.A.; Abdullah, A. An empirical study of pattern leakage impact during data preprocessing on machine learning-based
intrusion detection models reliability. Expert Syst. Appl. 2023, 230, 120715. [CrossRef]
Bouke, M.A.; Abdullah, A. An empirical assessment of ML models for 5G network intrusion detection: A data leakage-free
approach. e-Prime-Adv. Electr. Eng. Electron. Energy 2024, 8, 100590. [CrossRef]
Jayasinghe, S.; Siriwardhana, Y.; Porambage, P.; Liyanage, M.; Ylianttila, M. Federated Learning based Anomaly Detection as an
Enabler for Securing Network and Service Management Automation in Beyond 5G Networks. In Proceedings of the 2022 Joint
European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit), Grenoble, France, 7–10 June 2022;
pp. 345–350.

12.

11. Nait-Abdesselam, F.; Darwaish, A.; Titouna, C. Malware forensics: Legacy solutions, recent advances, and future challenges.
In Advances in Computing, Informatics, Networking and Cybersecurity: A Book Honoring Professor Mohammad S. Obaidat’s Significant
Scientific Contributions; Springer: Cham, Switzerland, 2022; pp. 685–710.
Saranya, T.; Sridevi, S.; Deisy, C.; Chung, T.D.; Khan, M. Performance Analysis of Machine Learning Algorithms in Intrusion
Detection System: A Review. Procedia Comput. Sci. 2020, 171, 1251–1260. [CrossRef]
Sarker, I.H. Machine Learning: Algorithms, Real-World Applications and Research Directions. SN Comput. Sci. 2021, 2, 160.
[CrossRef]

13.

14. Khan, M.S.; Farzaneh, B.; Shahriar, N.; Hasan, M.M. DoS/DDoS Attack Dataset of 5G Network Slicing; IEEE Dataport: Piscataway,

15.

16.

NJ, USA, 2023.
Imanbayev, A.; Tynymbayev, S.; Odarchenko, R.; Gnatyuk, S.; Berdibayev, R.; Baikenov, A.; Kaniyeva, N. Research of Machine
Learning Algorithms for the Development of Intrusion Detection Systems in 5G Mobile Networks and Beyond. Sensors 2022, 22,
9957. [CrossRef] [PubMed]
Joseph, L.P.; Deo, R.C.; Prasad, R.; Salcedo-Sanz, S.; Raj, N.; Soar, J. Near real-time wind speed forecast model with bidirectional
LSTM networks. Renew. Energy 2023, 204, 39–58. [CrossRef]

17. Li, Z.; Liu, F.; Yang, W.; Peng, S.; Zhou, J. A survey of convolutional neural networks: Analysis, applications, and prospects. IEEE

18.

Trans. Neural Netw. Learn. Syst. 2021, 33, 6999–7019.
Stahlke, M.; Feigl, T.; García MH, C.; Stirling-Gallacher, R.A.; Seitz, J.; Mutschler, C. Transfer learning to adapt 5G AI-based
fingerprint localization across environments. In Proceedings of the 2022 IEEE 95th Vehicular Technology Conference (VTC2022-
Spring), Helsinki, Finland, 19–22 June 2022; pp. 1–5.

19. Yang, B.; Fagbohungbe, O.; Cao, X.; Yuen, C.; Qian, L.; Niyato, D.; Zhang, Y. A Joint Energy and Latency Framework for Transfer

Learning Over 5G Industrial Edge Networks. IEEE Trans. Ind. Inform. 2021, 18, 531–541. [CrossRef]

20. Lv, Z.; Lou, R.; Singh, A.K.; Wang, Q. Transfer Learning-powered Resource Optimization for Green Computing in 5G-Aided

Industrial Internet of Things. ACM Trans. Internet Technol. 2021, 22, 1–16. [CrossRef]

Mathematics 2025, 13, 1088

59 of 63

21. Guan, J.; Cai, J.; Bai, H.; You, I. Deep transfer learning-based network traffic classification for scarce dataset in 5G IoT systems. Int.

J. Mach. Learn. Cybern. 2021, 12, 3351–3365. [CrossRef]

22. Coutinho, R.W.L.; Boukerche, A. Transfer Learning for Disruptive 5G-Enabled Industrial Internet of Things. IEEE Trans. Ind.

Inform. 2021, 18, 4000–4007. [CrossRef]

23. Mai, T.; Yao, H.; Zhang, N.; He, W.; Guo, D.; Guizani, M. Transfer Reinforcement Learning Aided Distributed Network Slicing

Optimization in Industrial IoT. IEEE Trans. Ind. Inform. 2021, 18, 4308–4316. [CrossRef]

24. Benzaïd, C.; Taleb, T.; Sami, A.; Hireche, O. A Deep Transfer Learning-Powered EDoS Detection Mechanism for 5G and Beyond
Network Slicing. In Proceedings of the GLOBECOM 2023—2023 IEEE Global Communications Conference, Kuala Lumpur,
Malaysia, 4–8 December 2023; pp. 4747–4753.

25. Benzaïd, C.; Taleb, T.; Sami, A.; Hireche, O. FortisEDoS: A Deep Transfer Learning-Empowered Economical Denial of Sustainability
Detection Framework for Cloud-Native Network Slicing. IEEE Trans. Dependable Secur. Comput. 2023, 21, 2818–2835. [CrossRef]
26. Kasongo, S.M.; Sun, Y. Performance Analysis of Intrusion Detection Systems Using a Feature Selection Method on the UNSW-NB15

Dataset. J. Big Data 2020, 7, 105. [CrossRef]

27. Thakkar, A.; Lohiya, R. Fusion of statistical importance for feature selection in Deep Neural Network-based Intrusion Detection

28.

29.

System. Inf. Fusion 2023, 90, 353–363. [CrossRef]
Shaukat, K.; Luo, S.; Varadharajan, V.; Hameed, I.A.; Xu, M. A Survey on Machine Learning Techniques for Cyber Security in the
Last Decade. IEEE Access 2020, 8, 222310–222354. [CrossRef]
Su, T.; Sun, H.; Zhu, J.; Wang, S.; Li, Y. BAT: Deep Learning Methods on Network Intrusion Detection Using NSL-KDD Dataset.
IEEE Access 2020, 8, 29575–29585. [CrossRef]

30. Rodríguez, M.; Alesanco, Á.; Mehavilla, L.; García, J. Evaluation of Machine Learning Techniques for Traffic Flow-Based Intrusion

Detection. Sensors 2022, 22, 9326. [CrossRef]

31. Ayantayo, A.; Kaur, A.; Kour, A.; Schmoor, X.; Shah, F.; Vickers, I.; Kearney, P.; Abdelsamea, M.M. Network intrusion detection

using feature fusion with deep learning. J. Big Data 2023, 10, 167. [CrossRef]

32. Vashishtha, L.K.; Chatterjee, K. Strengthening cybersecurity: TestCloudIDS dataset and SparkShield algorithm for robust threat

detection. Comput. Secur. 2025, 151, 104308. [CrossRef]

33. Bekkouche, R.; Omar, M.; Langar, R.; Hamdaoui, B. A Dynamic Predictive Maintenance Approach for Resilient Service Orches-
tration in Large-Scale 5G Infrastructures. SSRN 5123374. 2025. Available online: https://papers.ssrn.com/sol3/papers.cfm?
abstract_id=5123374 (accessed on 1 February 2025).

34. Kheddar, H.; Himeur, Y.; Awad, A.I. Deep transfer learning for intrusion detection in industrial control networks: A comprehen-

sive review. J. Netw. Comput. Appl. 2023, 220, 103760. [CrossRef]

35. Kheddar, H.; Dawoud, D.W.; Awad, A.I.; Himeur, Y.; Khan, M.K. Reinforcement-Learning-Based Intrusion Detection in Commu-

nication Networks: A Review. IEEE Commun. Surv. Tutor. 2024. [CrossRef]

36. Hancke, G.P.; Hossain, M.A.; Imran, M.A. 5G beyond 3GPP Release 15 for connected automated mobility in cross-border corridors.

Sensors 2020, 20, 6622.

37. Rischke, J.; Sossalla, P.; Itting, S.; Fitzek, F.H.P.; Reisslein, M. 5G Campus Networks: A First Measurement Study. IEEE Access 2021,

38.

9, 121786–121803. [CrossRef]
Singh, V.P.; Singh, M.P.; Hegde, S.; Gupta, M. Security in 5G Network Slices: Concerns and Opportunities. IEEE Access 2024, 12,
52727–52743. [CrossRef]

39. Granata, D.; Rak, M.; Mallouli, W. Automated Generation of 5G Fine-Grained Threat Models: A Systematic Approach. IEEE

Access 2023, 11, 129788–129804. [CrossRef]

40. Bao, S.; Liang, Y.; Xu, H. Blockchain for Network Slicing in 5G and Beyond: Survey and Challenges. J. Commun. Inf. Netw. 2022, 7,

41.

349–359. [CrossRef]
Iashvili, G.; Iavich, M.; Bocu, R.; Odarchenko, R.; Gnatyuk, S. Intrusion detection system for 5G with a focus on DOS/DDOS
attacks. In Proceedings of the 2021 11th IEEE International Conference on Intelligent Data Acquisition and Advanced Computing
Systems: Technology and Applications (IDAACS), Krakow, Poland, 22–25 September 2021; Volume 2, pp. 861–864.
Silva, R.S.; Meixner, C.C.; Guimaraes, R.S.; Diallo, T.; Garcia, B.O.; de Moraes, L.F.M.; Martinello, M. REPEL: A Strategic Approach
for Defending 5G Control Plane from DDoS Signalling Attacks. IEEE Trans. Netw. Serv. Manag. 2020, 18, 3231–3243. [CrossRef]
43. Nencioni, G.; Garroppo, R.G.; Olimid, R.F. 5G Multi-Access Edge Computing: A Survey on Security, Dependability, and

42.

44.

45.

Performance. IEEE Access 2023, 11, 63496–63533. [CrossRef]
Fakhouri, H.N.; Alawadi, S.; Awaysheh, F.M.; Hani, I.B.; Alkhalaileh, M.; Hamad, F. A Comprehensive Study on the Role of
Machine Learning in 5G Security: Challenges, Technologies, and Solutions. Electronics 2023, 12, 4604. [CrossRef]
Iavich, M.; Gnatyuk, S.O.; Odarchenko, R.; Bocu, R.; Simonov, S. The novel system of attacks detection in 5G. In Proceedings of
the 35th International Conference on Advanced Information Networking and Applications (AINA-2021), Toronto, ON, Canada,
12–14 May 2021; pp. 580–591.

Mathematics 2025, 13, 1088

60 of 63

47.

46. Yang, L.; Shami, A. A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles. In
Proceedings of the ICC 2022—IEEE International Conference on Communications, Seoul, Republic of Korea, 16–20 May 2022;
pp. 2774–2779.
Sullivan, S.; Brighente, A.; Kumar, S.A.P.; Conti, M. 5G Security Challenges and Solutions: A Review by OSI Layers. IEEE Access
2021, 9, 116294–116314. [CrossRef]
Salazar, Z.; Nguyen, H.N.; Mallouli, W.; Cavalli, A.R.; de Oca, E.M. 5Greplay: A 5G Network Traffic Fuzzer—Application to
Attack Injection. In Proceedings of the ARES 2021: The 16th International Conference on Availability, Reliability and Security,
Virtual, 17–20 August 2021; pp. 1–8.

48.

49. Amponis, G.; Radoglou-Grammatikis, P.; Nakas, G.; Goudos, S.; Argyriou, V.; Lagkas, T.; Sarigiannidis, P. 5G core PFCP intrusion
detection dataset. In Proceedings of the 2023 12th International Conference on Modern Circuits and Systems Technologies
(MOCAST), Athens, Greece, 28–30 June 2023; pp. 1–4.

50. Coldwell, C.; Conger, D.; Goodell, E.; Jacobson, B.; Petersen, B.; Spencer, D.; Anderson, M.; Sgambati, M. Machine learning 5G
attack detection in programmable logic. In Proceedings of the 2022 IEEE Globecom Workshops (GC Wkshps), Rio de Janeiro,
Brazil, 4–8 December 2022; pp. 1365–1370.

51. Alkasassbeh, M.; Baddar, S.A.-H. Intrusion Detection Systems: A State-of-the-Art Taxonomy and Survey. Arab. J. Sci. Eng. 2023,

48, 10021–10064. [CrossRef]

52. Kannari, P.R.; Chowdary, N.S.; Biradar, R.L. An anomaly-based intrusion detection system using recursive feature elimination

53.

technique for improved attack detection. Theor. Comput. Sci. 2022, 931, 56–64. [CrossRef]
Sharma, B.; Sharma, L.; Lal, C.; Roy, S. Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning
based approach. Expert Syst. Appl. 2024, 238, 121751. [CrossRef]

54. Popoola, S.I.; Imoize, A.L.; Hammoudeh, M.; Adebisi, B.; Jogunola, O.; Aibinu, A.M. Federated Deep Learning for Intrusion

Detection in Consumer-Centric Internet of Things. IEEE Trans. Consum. Electron. 2023, 70, 1610–1622. [CrossRef]

55. Meng, R.; Gao, S.; Fan, D.; Gao, H.; Wang, Y.; Xu, X.; Wang, B.; Lv, S.; Zhang, Z.; Sun, M.; et al. A survey of secure semantic

56.

communications. arXiv 2025, arXiv:2501.00001.
Jiang, S.; Zhao, J.; Xu, X. SLGBM: An Intrusion Detection Mechanism for Wireless Sensor Networks in Smart Environments. IEEE
Access 2020, 8, 169548–169558. [CrossRef]

57. Ozkan-Okay, M.; Samet, R.; Aslan, O.; Gupta, D. A Comprehensive Systematic Literature Review on Intrusion Detection Systems.

IEEE Access 2021, 9, 157727–157760. [CrossRef]

58. Di Mauro, M.; Galatro, G.; Fortino, G.; Liotta, A. Supervised feature selection techniques in network intrusion detection: A critical

review. Eng. Appl. Artif. Intell. 2021, 101, 104216. [CrossRef]

59. Qaddoura, R.; Al-Zoubi, A.M.; Faris, H.; Almomani, I. A Multi-Layer Classification Approach for Intrusion Detection in IoT

Networks Based on Deep Learning. Sensors 2021, 21, 2987. [CrossRef]

60. Ali, M.H.; Jaber, M.M.; Abd, S.K.; Rehman, A.; Awan, M.J.; Damaševiˇcius, R.; Bahaj, S.A. Threat Analysis and Distributed Denial

61.

of Service (DDoS) Attack Recognition in the Internet of Things (IoT). Electronics 2022, 11, 494. [CrossRef]
Santos, L.; Gonçalves, R.; Rabadão, C.; Martins, J. A flow-based intrusion detection framework for internet of things networks.
Clust. Comput. 2023, 26, 37–57. [CrossRef]

62. de Souza, C.A.; Westphall, C.B.; Machado, R.B.; Sobral, J.B.M.; Vieira, G.d.S. Hybrid approach to intrusion detection in fog-based

IoT environments. Comput. Netw. 2020, 180, 107417. [CrossRef]

63. Keserwani, P.K.; Govil, M.C.; Pilli, E.S.; Govil, P. A smart anomaly-based intrusion detection system for the Internet of Things

64.

(IoT) network using GWO–PSO–RF model. J. Reliab. Intell. Environ. 2021, 7, 3–21. [CrossRef]
Sharma, H.S.; Singh, K.J. Intrusion detection system: A deep neural network-based concatenated approach. J. Supercomput. 2024,
80, 13918–13948. [CrossRef]

65. Parra, G.D.L.T.; Rad, P.; Choo, K.-K.R.; Beebe, N. Detecting Internet of Things attacks using distributed deep learning. J. Netw.

Comput. Appl. 2020, 163, 102662. [CrossRef]

66. Mohamed, D.; Ismael, O. Enhancement of an IoT hybrid intrusion detection system based on fog-to-cloud computing. J. Cloud

Comput. 2023, 12, 41. [CrossRef]

67. Anwer, M.; Khan, S.M.; Farooq, M.U.; Waseemullah. Attack Detection in IoT using Machine Learning. Eng. Technol. Appl. Sci. Res.

2021, 11, 7273–7278. [CrossRef]

68. Ullah, I.; Mahmoud, Q.H. A Two-Level Flow-Based Anomalous Activity Detection System for IoT Networks. Electronics 2020, 9,

69.

70.

530. [CrossRef]
Farrukh, Y.A.; Wali, S.; Khan, I.; Bastian, N.D. AIS-NIDS: An intelligent and self-sustaining network intrusion detection system.
Comput. Secur. 2024, 144, 103982. [CrossRef]
Singh, G.; Khare, N. A survey of intrusion detection from the perspective of intrusion datasets and machine learning techniques.
Int. J. Comput. Appl. 2021, 44, 659–669. [CrossRef]

Mathematics 2025, 13, 1088

61 of 63

71. Magán-Carrión, R.; Urda, D.; Diaz-Cano, I.; Dorronsoro, B. Improving the Reliability of Network Intrusion Detection Systems

Through Dataset Integration. IEEE Trans. Emerg. Top. Comput. 2022, 10, 1717–1732. [CrossRef]

72. Keserwani, P.K.; Govil, M.C.; Pilli, E.S. An effective NIDS framework based on a comprehensive survey of feature optimization

and classification techniques. Neural Comput. Appl. 2023, 35, 4993–5013. [CrossRef]

73. Layeghy, S.; Gallagher, M.; Portmann, M. Benchmarking the benchmark—Comparing synthetic and real-world Network IDS

datasets. J. Inf. Secur. Appl. 2024, 80, 103689. [CrossRef]

74. Engelen, G.; Rimmer, V.; Joosen, W. Troubleshooting an Intrusion Detection Dataset: The CICIDS2017 Case Study. In Proceedings

75.

of the 2021 IEEE Security and Privacy Workshops (SPW), San Francisco, CA, USA, 27 May 2021; pp. 7–12.
Stamatis, C.; Barsanti, K.C. Development and application of a supervised pattern recognition algorithm for identification of
fuel-specific emissions profiles. Atmos. Meas. Tech. 2022, 15, 2591–2606. [CrossRef]

76. Tsourdinis, T.; Makris, N.; Korakis, T.; Fdida, S. AI-driven network intrusion detection and resource allocation in real-world
O-RAN 5G networks. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking
(MobiCom ’24), Washington, DC, USA, 18–22 November 2024; pp. 1842–1849.

77. Dhanushkodi, K.; Thejas, S. AI Enabled Threat Detection: Leveraging Artificial Intelligence for Advanced Security and Cyber

78.

Threat Mitigation. IEEE Access 2024, 12, 173127–173136. [CrossRef]
Soliman, H.M.; Sovilj, D.; Salmon, G.; Rao, M.; Mayya, N. RANK: AI-Assisted End-to-End Architecture for Detecting Persistent
Attacks in Enterprise Networks. IEEE Trans. Dependable Secur. Comput. 2023, 21, 3834–3850. [CrossRef]

79. Yadav, N.; Pande, S.; Khamparia, A.; Gupta, D. Intrusion Detection System on IoT with 5G Network Using Deep Learning. Wirel.

80.

Commun. Mob. Comput. 2022, 2022, 9304689. [CrossRef]
Farzaneh, B.; Shahriar, N.; Al Muktadir, A.H.; Towhid, M.S. DTL-IDS: Deep transfer learning-based intrusion detection system in
5G networks. In Proceedings of the 2023 19th International Conference on Network and Service Management (CNSM), Niagara
Falls, ON, Canada, 30 October–2 November 2023; pp. 1–5.

81. Zhou, M.-G.; Liu, Z.-P.; Yin, H.-L.; Li, C.-L.; Xu, T.-K.; Chen, Z.-B. Quantum neural network for quantum neural computing.

Research 2023, 6, 0134. [CrossRef] [PubMed]

82. Zhou, M.-G.; Cao, X.-Y.; Lu, Y.-S.; Wang, Y.; Bao, Y.; Jia, Z.-Y.; Fu, Y.; Yin, H.-L.; Chen, Z.-B. Experimental Quantum Advantage

with Quantum Coupon Collector. Research 2022, 2022, 9798679. [CrossRef]

83. Adewole, K.S.; Salau-Ibrahim, T.T.; Imoize, A.L.; Oladipo, I.D.; AbdulRaheem, M.; Awotunde, J.B.; Balogun, A.O.; Isiaka, R.M.;
Aro, T.O. Empirical Analysis of Data Streaming and Batch Learning Models for Network Intrusion Detection. Electronics 2022, 11,
3109. [CrossRef]

84. Umar, M.A.; Chen, Z.; Shuaib, K.; Liu, Y. Effects of feature selection and normalization on network intrusion detection. J. Inf.

Technol. Data Manag. 2025, 8, 23–39. [CrossRef]

85. Algan, G.; Ulusoy, I. Image classification with deep learning in the presence of noisy labels: A survey. Knowl.-Based Syst. 2021,

215, 106771. [CrossRef]

86. Gupta, A.R.; Agrawal, J. The multi-demeanor fusion based robust intrusion detection system for anomaly and misuse detection

in computer networks. J. Ambient Intell. Humaniz. Comput. 2021, 12, 303–319. [CrossRef]

87. Nabi, F.; Zhou, X. Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine

learning techniques for cyber security. Cyber Secur. Appl. 2024, 2, 100033. [CrossRef]

88. Dogan, A.; Birant, D. Machine learning and data mining in manufacturing. Expert Syst. Appl. 2021, 166, 114060. [CrossRef]
89. Belouadah, E.; Popescu, A.; Kanellos, I. A comprehensive study of class incremental learning algorithms for visual tasks. Neural

Netw. 2021, 135, 38–54. [CrossRef] [PubMed]

90. Amponis, G.; Radoglou-Grammatikis, P.; Lagkas, T.; Mallouli, W.; Cavalli, A.; Klonidis, D.; Markakis, E.; Sarigiannidis, P.
Threatening the 5G core via PFCP DoS attacks: The case of blocking UAV communications. EURASIP J. Wirel. Commun. Netw.
2022, 2022, 124. [CrossRef]

91. Mazarbhuiya, F.A.; Alzahrani, M.Y.; Mahanta, A.K. Detecting anomaly using partitioning clustering with merging. ICIC Express

92.

93.

Lett. 2020, 14, 951–960.
Samarakoon, S.; Siriwardhana, Y.; Porambage, P.; Liyanage, M.; Chang, S.-Y.; Kim, J.; Kim, J.; Ylianttila, M. 5G-NIDD: A
comprehensive network intrusion detection dataset generated over 5G wireless network. arXiv 2022, arXiv:2212.01298.
Farzaneh, B.; Shahriar, N.; Al Muktadir, A.H.; Towhid, S.; Khosravani, M.S. DTL-5G: Deep transfer learning-based DDoS attack
detection in 5G and beyond networks. Comput. Commun. 2024, 228, 107927. [CrossRef]

94. Berei, E.; Khan, M.A.; Oun, A. Machine Learning Algorithms for DoS and DDoS Cyberattacks Detection in Real-Time Environment.
In Proceedings of the 2024 IEEE 21st Consumer Communications & Networking Conference (CCNC), Las Vegas, NV, USA, 6–9
January 2024; pp. 1048–1049.

95. Ghani, H.; Salekzamankhani, S.; Virdee, B. Critical analysis of 5G networks’ traffic intrusion using PCA, t-SNE, and UMAP

visualization and classifying attacks. In Proceedings of Data Analytics and Management; Springer: Singapore, 2024; pp. 375–389.

Mathematics 2025, 13, 1088

62 of 63

96. Vu, L.; Nguyen, Q.U.; Nguyen, D.N.; Hoang, D.T.; Dutkiewicz, E. Deep Transfer Learning for IoT Attack Detection. IEEE Access

2020, 8, 107335–107344. [CrossRef]

97. Hossain, S.; Senouci, S.-M.; Brik, B.; Boualouache, A. A privacy-preserving Self-Supervised Learning-based intrusion detection

system for 5G-V2X networks. Ad Hoc Netw. 2024, 166, 103674. [CrossRef]

98. Alghamdi, S.A. Novel trust-aware intrusion detection and prevention system for 5G MANET–Cloud. Int. J. Inf. Secur. 2022, 21,

99.

469–488. [CrossRef]
Sousa, B.; Magaia, N.; Silva, S. An Intelligent Intrusion Detection System for 5G-Enabled Internet of Vehicles. Electronics 2023, 12,
1757. [CrossRef]

100. Ciaburro, G.; Iannace, G. Machine Learning-Based Algorithms to Knowledge Extraction from Time Series Data: A Review. Data

2021, 6, 55. [CrossRef]

101. Jurkiewicz, P. Flow-models: A framework for analysis and modeling of IP network flows. SoftwareX 2022, 17, 100929. [CrossRef]
102. Larin, D.V.; Get’man, A.I. Tools for Capturing and Processing High-Speed Network Traffic. Program. Comput. Softw. 2022, 48,

756–769. [CrossRef]

103. Sheikhi, S.; Kostakos, P. DDoS attack detection using unsupervised federated learning for 5G networks and beyond. In Proceedings
of the 2023 Joint European Conference on Networks and Communications and 6G Summit (EuCNC/6G Summit), Gothenburg,
Sweden, 6–9 June 2023; pp. 442–447.

104. Reddy, R.; Gundall, M.; Lipps, C.; Schotten, H.D. Open source 5G core network implementations: A qualitative and quantitative
analysis. In Proceedings of the 2023 IEEE International Black Sea Conference on Communications and Networking (BlackSeaCom),
Istanbul, Turkey, 4–7 July 2023; pp. 253–258.

105. Rouili, M.; Saha, N.; Golkarifard, M.; Zangooei, M.; Boutaba, R.; Onur, E.; Saleh, A. Evaluating Open-Source 5G SA Testbeds:
Unveiling Performance Disparities in RAN Scenarios. In Proceedings of the NOMS 2024—2024 IEEE Network Operations and
Management Symposium, Seoul, Republic of Korea, 6–10 May 2024; pp. 1–6.

106. Liu, X.; Wang, X.; Jia, J.; Huang, M. A distributed deployment algorithm for communication coverage in wireless robotic networks.

J. Netw. Comput. Appl. 2021, 180, 103019. [CrossRef]

107. Nguyen, L.G.; Watabe, K. A Method for Network Intrusion Detection Using Flow Sequence and BERT Framework. In Proceedings

of the ICC 2023—IEEE International Conference on Communications, Rome, Italy, 28 May–1 June 2023; pp. 3006–3011.

108. Büyükkeçeci, M.; Okur, M.C. A Comprehensive Review of Feature Selection and Feature Selection Stability in Machine Learning.

Gazi Univ. J. Sci. 2023, 36, 1506–1520. [CrossRef]

109. Dissanayake, K.; Md Johar, M.G. Comparative study on heart disease prediction using feature selection techniques on classification

algorithms. Appl. Comput. Intell. Soft Comput. 2021, 2021, 5581806. [CrossRef]

110. Singh, D.; Singh, B. Investigating the impact of data normalization on classification performance. Appl. Soft Comput. 2020, 97,

105524. [CrossRef]

111. Yuliana, H.; Iskandar; Hendrawan. Comparative Analysis of Machine Learning Algorithms for 5G Coverage Prediction:

Identification of Dominant Feature Parameters and Prediction Accuracy. IEEE Access 2024, 12, 18939–18956. [CrossRef]

112. Nahum, C.V.; Pinto, L.D.N.M.; Tavares, V.B.; Batista, P.; Lins, S.; Linder, N.; Klautau, A. Testbed for 5G Connected Artificial

Intelligence on Virtualized Networks. IEEE Access 2020, 8, 223202–223213. [CrossRef]

113. Mehdi, A.; Bali, M.K.; Abbas, S.I. Unleashing the Potential of Grafana: A Comprehensive Study on Real-Time Monitoring
and Visualization. In Proceedings of the 2023 14th International Conference on Computing Communication and Networking
Technologies (ICCCNT), Delhi, India, 6–8 July 2023; pp. 1–5.

114. Abbasi, A.; Javed, A.R.; Chakraborty, C.; Nebhen, J.; Zehra, W.; Jalil, Z. ElStream: An Ensemble Learning Approach for Concept

Drift Detection in Dynamic Social Big Data Stream Learning. IEEE Access 2021, 9, 66408–66419. [CrossRef]

115. Hamza, M.A.; Ejaz, U.; Kim, H.-C. Cyber5Gym: An Integrated Framework for 5G Cybersecurity Training. Electronics 2024, 13,

888. [CrossRef]

116. Al-Fuhaidi, B.; Farae, Z.; Al-Fahaidy, F.; Nagi, G.; Ghallab, A.; Alameri, A. Anomaly-Based Intrusion Detection System in Wireless
Sensor Networks Using Machine Learning Algorithms. Appl. Comput. Intell. Soft Comput. 2024, 2024, 2625922. [CrossRef]
117. Pérez-Hernández, F.; Tabik, S.; Lamas, A.; Olmos, R.; Fujita, H.; Herrera, F. Object Detection Binary Classifiers methodology
based on deep learning to identify small objects handled similarly: Application in video surveillance. Knowl.-Based Syst. 2020,
194, 105590. [CrossRef]

118. Dhanke, J.; Patil, R.N.; Kumari, I.; Gupta, S.; Hans, S.; Kumar, K. Comparative study of machine learning algorithms for intrusion

detection. Int. J. Intell. Syst. Appl. Eng. 2023, 12, 647–653.

119. Gupta, S.; Kumar, S.; Singh, A. A hybrid intrusion detection system based on decision tree and support vector machine. In
Proceedings of the 2020 IEEE 5th International Conference on Computing Communication and Automation (ICCCA 2020),
Greater Noida, India, 30–31 October 2020; pp. 510–515.

120. Lansky, J.; Ali, S.; Mohammadi, M.; Majeed, M.K.; Karim, S.H.T.; Rashidi, S.; Hosseinzadeh, M.; Rahmani, A.M. Deep Learning-

Based Intrusion Detection Systems: A Systematic Review. IEEE Access 2021, 9, 101574–101599. [CrossRef]

Mathematics 2025, 13, 1088

63 of 63

121. Awotunde, J.B.; Folorunso, S.O.; Imoize, A.L.; Odunuga, J.O.; Lee, C.-C.; Li, C.-T.; Do, D.-T. An Ensemble Tree-Based Model for

Intrusion Detection in Industrial Internet of Things Networks. Appl. Sci. 2023, 13, 2479. [CrossRef]

122. Akbar, R.; Zafer, A. Next-Gen Information Security: AI-Driven Solutions for Real-Time Cyber Threat Detection in Cloud and

Network Environments. J. Cybersecur. Res. 2024, 12, 123–145.

123. Rana, P.; Batra, I.; Malik, A.; Imoize, A.L.; Kim, Y.; Pani, S.K.; Goyal, N.; Kumar, A.; Rho, S. Intrusion detection systems in cloud

computing paradigm: Analysis and overview. Complexity 2022, 2022, 3999039. [CrossRef]

124. Teuwen, K.T.; Mulders, T.; Zambon, E.; Allodi, L. Ruling the Unruly: Designing Effective, Low-Noise Network Intrusion Detection

Rules for Security Operations Centers. arXiv 2025, arXiv:2501.09808.

125. Jimoh, R.G.; Imoize, A.L.; Awotunde, J.B.; Ojo, S.; Akanbi, M.B.; Bamigbaye, J.A.; Faruk, N. An Enhanced Deep Neural
Network Enabled with Cuckoo Search Algorithm for Intrusion Detection in Wide Area Networks. In Proceedings of the 2022 5th
Information Technology for Education and Development (ITED), Abuja, Nigeria, 1–3 November 2022; pp. 1–5.

126. AbdulRaheem, M.; Oladipo, I.D.; Imoize, A.L.; Awotunde, J.B.; Lee, C.-C.; Balogun, G.B.; Adeoti, J.O. Machine learning assisted

snort and zeek in detecting DDoS attacks in software-defined networking. Int. J. Inf. Technol. 2024, 16, 1627–1643. [CrossRef]

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
people or property resulting from any ideas, methods, instructions or products referred to in the content.

