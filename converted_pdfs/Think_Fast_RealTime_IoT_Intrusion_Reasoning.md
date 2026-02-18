# Think Fast: Real-Time IoT Intrusion Reasoning

Think Fast: Real-Time IoT Intrusion Reasoning
Using IDS and LLMs at the Edge Gateway

Saeid Jamshidi Amin Nikanjam Negar Shahabi Kawser Wazed Nafi Foutse Khomh Samira Keivanpour Rolando
Herrero Omar Abdul Wahab Martine Bella¨ıche

1

5
2
0
2

v
o
N
3
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
0
3
2
8
1
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

Abstract—As the number of connected IoT devices continues
to grow, securing these systems against cyber threats remains a
pressing challenge, especially within environments constrained by
limited computational and energy resources. This paper presents
an edge-centric Intrusion Detection System (IDS) framework
that seamlessly integrates lightweight Machine Learning (ML)-
based IDS models with pre-trained Large Language Models
(LLMs) to enhance detection accuracy, semantic interpretability,
and operational efficiency at the network edge. The system
evaluates six ML-based IDS models: Decision Tree (DT), K-
Nearest Neighbors (KNN), Random Forest (RF), Convolutional
Neural Network (CNN), Long Short-Term Memory (LSTM), and
a hybrid model of CNN and LSTM, on low-power edge gateways,
achieving accuracy up to 98% under real-world cyberattacks.
Furthermore, for anomaly detection, the system transmits a
compact, secure telemetry snapshot (e.g., CPU usage, memory
usage, latency, and energy consumption) via low-bandwidth API
calls to LLMs, including GPT-4-turbo, DeepSeek V2, and LLaMA
3.5. These models employ zero-shot, few-shot, and Chain-of-
Thought
(CoT) reasoning to deliver human-readable threat
analyses and actionable mitigation recommendations. Extensive
evaluations across diverse attacks such as DoS, DDoS, brute force
and port scanning, demonstrate the system’s ability to enhance
interpretability while maintaining low latency (<1.5s), minimal
bandwidth usage (<1.2 kB per prompt), and energy efficiency
(<75 J), establishing it as a practical and scalable IDS solution
for the edge gateway.

Index Terms—Cybersecurity, Intrusion Detection, Machine

Learning, DoS Attacks

I. INTRODUCTION

The Internet of Things (IoT) has become a fundamental
layer of modern digital infrastructure[1], connecting billions
of physical devices that range from industrial sensors to
home automation systems [2] [3]. This massive integration
enables real-time control, predictive analytics, and pervasive
automation in various domains, including healthcare, energy,

Saeid Jamshidi, Omar Abdul Wahab, and Martine Bella¨ıche are with the
Department of Computer and Software Engineering, Polytechnique Montr´eal,
Quebec, H3T 1J4, Canada. Contact emails:
jamshidi.saeid@polymtl.ca,
omar.abdul-wahab@polymtl.ca, martine.bellaiche@polymtl.ca.

Amin Nikanjam was with the Huawei Distributed Scheduling and Data
Engine Lab (work done while at Polytechnique Montr´eal), Montreal, Quebec,
Canada. Email: amin.nikanjam@huawei.com.

Negar Shahabi is with the Concordia Institute for Information Systems En-
gineering (CIISE), Concordia University, Montr´eal, Quebec, Canada. Email:
negar.shahabi@mail.concordia.ca.

Kawser Wazed Nafi and Foutse Khomh are with the SWAT Laboratory,
Polytechnique Montr´eal, Quebec, H3T 1J4, Canada. Emails: kawser.wazed-
nafi@polymtl.ca, foutse.khomh@polymtl.ca.

Samira Keivanpour is with Poly Circle X.O, Polytechnique Montr´eal,

Quebec, H3T 1J4, Canada. Email: samira.keivanpour@polymtl.ca.

Rolando Herrero is with the College of Engineering, Northeastern Univer-

sity, Boston, MA, USA. Email: r.herrero@northeastern.edu.

agriculture, and transportation [4]. However, as this con-
nectivity becomes more ubiquitous,
it also brings with it
a rapidly expanding and deeply fragmented attack surface
[5]. Most IoT devices operate with limited processing power,
minimal memory, and very modest energy budgets [6]. They
are often deployed with weak default credentials, lack regular
firmware updates, and run on lightweight operating systems
with minimal security controls [7] [8]. As a result, they are
particularly vulnerable to attacks, e.g., Distributed Denial-of-
Service (DDoS), brute-force attacks, and port scanning [9].
These threats are not just technical nuisances; they can dis-
rupt safety-critical processes or silently compromise sensitive
infrastructure.
Furthermore, what makes securing IoT networks challeng-
ing is the tight coupling between performance and resource
constraints. Many devices must operate with strict limitations
on energy consumption and CPU usage [10] [11]. A slight
increase in computational load and energy consumption can
cause an entire sensor cluster to fail prematurely, missing
critical timing windows [12]. Therefore, any effective security
solution for the IoT must strike a balance between detection
effectiveness and computational efficiency. It must be able to
accurately identify threats while preserving the functional in-
tegrity of the device and respecting its operational boundaries
[13].
One approach that has shown promise is the use of ML-based
IDS at the edge gateway [14], [15]. These systems learn the
expected behavior of a device and network and flag deviations
without relying on static signatures—predefined patterns of
known threats stored in a database that traditional signature-
based IDS use for detection. By avoiding dependence on such
fixed patterns, ML-based IDS can identify both known and
previously unseen attacks. When properly tuned, these models
can achieve high detection accuracy with minimal supervision
[16]. Moreover, their lightweight nature makes them suitable
for on-device deployment, reducing reliance on external infras-
tructure and enabling real-time detection [17]. However, ML-
based IDS models also face persistent challenges. They tend to
produce a high rate of false positives, which undermines trust
and usability. They often return binary and numeric results
without contextual explanation, leaving humans in the loop to
infer the meaning of each alert [18] [19]. Additionally, when
these models are deployed on resource-limited platforms, e.g.,
single-board computers, they must be simplified, which can
result in a loss of precision and reliability [20].
Concurrently, large language models (LLMs), e.g., GPT-4-
turbo, DeepSeek-V2, and Llama3.5, have emerged as robust
mechanisms for semantic reasoning and pattern interpretation.

They can identify complex relationships in data and generate
coherent explanations, even from limited examples, a capabil-
ity often referred to as few-shot learning [21]. These models
demonstrate significant strength in interpreting intricate inputs
and producing human-readable insights [22][23]. In partic-
ular, their ability to generalize from limited examples and
perform reasoning tasks across structured and unstructured
datasets provides promising avenues for enhancing security
applications. Nevertheless, the integration of LLMs within
ML-based IDS pipelines introduces considerable practical
challenges [24]. Key concerns include evaluating whether the
LLM can effectively augment ML-based IDS outputs without
substantially increasing computational
load and breaching
latency and energy consumption constraints. Additionally, it
remains crucial to determine whether pre-trained LLMs can
accurately interpret compact telemetry from the edge gateway
and whether such interpretations can significantly enhance
detection accuracy under realistic attack conditions.
To address these challenges, this paper proposes a novel edge-
centric IDS architecture, where six ML-based IDS models
are deployed exclusively on a low-power edge gateway. Upon
detecting an anomaly, the gateway compiles a concise snapshot
of key system metrics, e.g., CPU usage, memory usage,
latency, bandwidth, and energy consumption. This aggregated
telemetry is then transmitted to large LLMs via encrypted,
low-overhead API calls. Each LLM performs a structured,
multi-stage analysis encompassing zero-shot classification to
hypothesize the nature of the attack, few-shot reasoning to
assess pattern similarity, and Chain-of-Thought (CoT) reason-
ing to produce contextually relevant explanations and potential
countermeasures. The resulting alert is semantically enhanced,
significantly improving interpretability and facilitating more
informed decision-making. Importantly, all processes, e.g.,
orchestration, metric aggregation, and integration, are executed
directly on the edge gateway. The significant contributions of
this paper can be summarized as follows:

• Edge-Initiated Semantic Defence: A fully autonomous
ML-based IDS architecture where detection and coordi-
nation occur entirely on the edge gateway, leveraging
external LLMs solely through minimal API calls to
enhance reasoning capabilities.

• Multi-Stage LLM Reasoning Integration: Integration
of advanced LLM reasoning methods, zero-shot clas-
sification, few-shot reasoning, and CoT processing, to
convert raw anomaly alerts into structured, interpretable,
and actionable security intelligence.

• Resource-Conscious Evaluation Framework: An eval-
uation approach that simultaneously assesses detection
accuracy and precisely measures system resource im-
pacts,e.g, latency, CPU usage, memory usage, and energy
consumption, thus balancing effectiveness with practical
feasibility.

The remainder of this paper is organized as follows. Sec-
tion II introduces the foundational ML-based IDS. Section III
reviews prior research on ML-based IDS and LLM integration
for IoT security. Section IV details the proposed system,
covering the inference pipeline, telemetry, context augmen-

2

tation, prompt construction, reasoning strategies, and evalua-
tion framework. Section V describes the research questions,
datasets, and experimental setup. Section VI presents the
experimental outcomes across various attack scenarios and
resource metrics. Section X discusses key findings, model
performance, and interpretability improvements. Section XI
outlines threats to validity that impact the study’s generaliz-
ability. Section XII suggests future enhancements, including
retraining, LLM optimization, and real-world deployment.
Section XIII concludes the paper by summarizing the con-
tributions and emphasizing the trade-off between detection
accuracy, interpretability, and efficiency.

II. BACKGROUND

This section introduces the baseline ML-based models un-
derpinning our research and their relevance to IDS. Each
model was selected for its strengths in resource efficiency,
detection accuracy, and interpretability. Together, they form
the foundation upon which LLM-based semantic reasoning is
integrated.
Decision Tree (DT): DTs are widely deployed in IDS due
to their interpretable, rule-based structure. By recursively
partitioning traffic into binary splits, they generate hierarchical
paths that clearly map input features to attack decisions. Their
transparency and low computational overhead make them well-
suited for resource-constrained IoT environments and real-time
contexts [25], [26], [27].
Random Forest (RF): RF improves upon DT by combining
multiple decision trees into an ensemble. By leveraging ma-
jority voting across trees, RF reduces variance and minimizes
overfitting, resulting in robust classification even with high-
dimensional or imbalanced IoT traffic [28], [29], [30].
K-Nearest Neighbor (KNN): KNN performs instance-based
classification by comparing feature vectors with labeled ex-
amples using distance metrics. Its non-parametric nature and
adaptability make it effective for anomaly detection tasks in
IDS, especially when training resources are limited [31], [32],
[33].
Long Short-Term Memory (LSTM): LSTMs, a variant of
recurrent neural networks (RNNs), are designed to model
sequential dependencies. Their gated memory cells capture
long-range temporal patterns, making them highly effective for
detecting evolving attacks such as slow brute force attempts
and distributed DoS floods [34], [35].
Convolutional Neural Network (CNN): CNNs extract spatial
correlations in traffic by applying convolutional filters to fea-
ture maps. In IDS, CNNs excel at detecting both well-known
and novel attack vectors by capturing structural regularities in
traffic flow, including packet bursts or scanning patterns [36],
[37].
Hybrid of model CNN and LSTM: The hybrid approach
integrates CNN’s spatial extraction with LSTM’s temporal
reasoning, enabling richer multi-perspective analysis. Such
hybrids are particularly effective for complex, multi-stage
attacks that require recognition of both packet-level bursts and
long-term behavioral deviations [38], [36].

A. The ML-based IDS

We benchmarked six ML-based IDS models( e.g., DT,
RF, KNN, LSTM, CNN, and a hybrid model of CNN and
LSTM) under realistic edge gateway conditions to establish
a baseline for efficiency and effectiveness. Each model was
optimized for edge deployment and evaluated against system-
level performance metrics,
including CPU usage, memory
usage, latency, bandwidth, and energy consumption. Building
on these results, this study integrates LLM-based semantic
reasoning to enhance interpretability and resilience in IDS.
A comparative summary of neural network–based IDS config-
urations is presented in Table I. The results highlight the trade-
off between structural complexity and accuracy. For example,
the hybrid model of CNN and LSTM achieved the highest
testing accuracy (95.75%) while requiring 12,795 parameters,
compared to the CNN model, which achieved 94.74% testing
accuracy with only 3,497 parameters. This demonstrates that
although hybrid architectures provide stronger adaptability to
complex traffic patterns, they impose a higher computational
burden that may limit suitability for resource-constrained IoT
gateways.

Table II further compares classical ML-based IDS mod-
els (DT, KNN, RF) against neural networks (LSTM, CNN,
a hybrid model of CNN and LSTM). Classical ML-based
IDS models consistently achieved near-perfect scores across
accuracy, precision, recall, and F1 (99%), confirming their
efficiency and suitability for lightweight IoT deployments.
Neural models, while slightly less efficient, offered improved
generalization for complex and evolving attack patterns, but
at the cost of increased latency and higher energy demand.
This trade-off underscores the motivation for augmenting IDS
with LLM-based reasoning, where semantic interpretability
can alleviate computational constraints and improve robustness
against sophisticated attack scenarios.

B. Evaluation Metrics

To assess feasibility in edge-deployed environments, we
evaluate the proposed ML-based IDS and LLM hybrids using
six system-level performance metrics: CPU usage, memory
usage, energy consumption, bandwidth, latency, and detection
accuracy. These metrics together capture computational de-
mand, communication overhead, and energy efficiency, all of
which are critical in resource-constrained edge gateways.
CPU usage and memory usage quantify the computational
complexity of each ML-based IDS model, directly reflecting
its scalability for low-power hardware. Energy consumption
is measured through fine-grained integration of instantaneous
power over inference cycles:

Et =

N
(cid:88)

i=1

Pi · ∆τ, ∆τ = 10 ms, N =

Tt
∆τ

(1)

where Pi denotes instantaneous power, ensuring resilience
evaluation against adversarial energy-drain scenarios. Latency
is modeled as the total detection delay:

Tt = TIDS + Ttx + TLLM

(2)

3

where TIDS represents local inference time, Ttx network delay,
and TLLM reasoning time. This guarantees the IDS can be
validated against real-time operational constraints.
Bandwidth reflects the communication overhead of transmit-
ting telemetry and prompts, a crucial factor in IoT networks
with strict
throughput budgets. Accuracy-related measures
(precision, recall, F1-score), as reported in Table II, quantify
detection effectiveness, balancing false positives and false
negatives in practical deployments. To ensure statistical ro-
bustness, we apply one-way ANOVA followed by Tukey’s
to determine whether differences across LLMs
HSD test
and IDS configurations are significant. This controls Type-
I error inflation in multi-comparison settings and provides
effect size estimates to evaluate practical impact. Evaluation
was conducted under four representative attack scenarios(
e.g., DoS, DDoS, brute force, and port scanning), which
collectively span availability and reconnaissance threats in IoT
deployments. Across 62 independent trials, both standalone
ML-based IDS and LLM-augmented hybrids were systemati-
cally tested. This design enables a quantification of detection
accuracy, interpretability, and operational cost, establishing a
strong foundation for cross-attack analysis.

III. RELATED WORK

IoT

security

capabilities within

Recent advances in ML-based IDS have significantly
enhanced
systems.
researchers have explored both lightweight
Specifically,
detection models, appropriate for resource-limited devices,
and more robust
inference systems to effectively manage
increasingly sophisticated threats. Concurrently, LLMs have
for enhancing the
begun to demonstrate their potential
reasoning and interpretability of alerts generated by IDS.
Therefore, this section provides a thorough review of relevant
research across ML-based IDS techniques and LLM-enhanced
cybersecurity.
Otoum et al. [39] present an LLM-driven threat detection
and prevention framework specifically designed for
IoT.
The proposed system integrates lightweight BERT variants,
namely TinyBERT, BERT-Mini, and BERT-Small, fine-tuned
on domain-specific datasets, including IoT-23 and TON-IoT.
Detection outputs are further linked to a rule-based decision
engine, enabling real-time mitigation actions. The entire
architecture is containerized and deployed in a simulated
IoT using Docker, allowing for detailed evaluation under
constrained computational conditions. Experimental results
indicate that BERT-Small offers the best trade-off between
accuracy and efficiency, achieving 99.75%, detection accuracy
with low latency and minimal energy consumption, thereby
demonstrating the practical viability of compact LLMs for
edge-based intrusion prevention.
Diaf et al. [40] propose a novel intrusion prediction framework
that proactively identifies cyber threats in IoT networks by
leveraging a combination of fine-tuned LLMs and LSTM.
Their architecture integrates GPT-2 for next-packet prediction
and BERT for evaluating the coherence of predicted packet
sequences, forming a feedback loop that enhances predictive
robustness. A final LSTM layer is employed to classify these

4

TABLE I: Comparison of Neural Network Models

Metric

Dataset

LSTM

CNN + LSTM

CNN

CICIDS2017

CICIDS2017

CICIDS2017

Number of Categories

Number of Layers

Number of Parameters

Training Accuracy

Testing Accuracy

15

10

56,386

97.72%

93.86%

15

11

12,795

98.77%

95.75%

15

8

3,497

97.92%

94.74%

TABLE II: Performance Comparison of ML-based IDS Models

Metric

Accuracy

Precision

Recall

F1-Score

DT

0.9985

0.9985

0.9985

0.9985

KNN

0.9967

0.9966

0.9967

0.9966

RF

LSTM LSTM + CNN

CNN

0.9981

0.9980

0.9981

0.9980

0.9386

0.9771

0.9524

0.9646

0.9575

0.9877

0.9645

0.9760

0.9474

0.9792

0.9611

0.9701

sequences as either benign or malicious. The framework is
trained and tested using the CICIoT2023 dataset, achieving
an overall IDS accuracy of 98%, and demonstrating strong
generalization to multistage and unseen attacks.
introduce security BERT, a privacy-
Ferrag et al.
[41]
tailored for IoT. The
preserving and efficient IDS model
framework leverages a modified BERT architecture with a
novel BBPE tokenization strategy and employs a BERT-
based encoder-decoder structure to analyze network traffic.
is evaluated on the IoTID20 and Edge-IIoT datasets,
It
demonstrating robust detection across multiple types of
attacks. SecurityBERT achieves a classification accuracy of
98.2%, while maintaining extremely low inference time (0.67
ms) and a compact model size (14.3 MB), confirming its
suitability for deployment at the edge.
Abdelhamid et al. [42] present an attention-driven transfer
learning framework for IDS in IoT by transforming tabular
data into image representations. Their method applies a CNN
pre-trained on ImageNet, enhanced with Convolutional Block
Attention Modules (CBAM), to extract deep spatial features
from visualized network data. The final classification is
performed through an ensemble of CNN, RF, and XGBoost
models. Evaluated on the CICIoT2023 dataset, the proposed
system achieves impressive detection accuracy (up to 99.98%)
while maintaining low false positive rates across various
attack types.
Adjewa et al. [43] propose a federated anomaly-based IDS for
5G-enabled IoT networks using an optimized BERT model
tailored for edge deployment. Their architecture reduces the
base BERT model to four encoder layers, incorporating linear
quantization to compress the model without a significant
loss in performance. The system is evaluated using both
centralized and federated setups with the Edge-IIoTset
dataset. In centralized learning, the model achieves 97.79%
it reaches up
accuracy, while in federated configurations,
to 97.12% with IID data and 96.66% with non-IID data
using ten clients. The model also performs effectively on
constrained hardware, e.g, Raspberry Pi, with an inference
time of 0.45 seconds, underscoring its practical viability for
decentralized and privacy-preserving IoT security.

in

efficiency

deployment

Alshamrani et al.
[44] propose a two-phase ensemble
IDS designed to balance high detection accuracy with
computational
resource-
for
constrained IoT. The framework employs an Extra Trees
classifier and a Deep Neural Network (DNN) for initial binary
classification, followed by the RF model to identify specific
attack categories. Experiments conducted on the TON-IoT
dataset demonstrate that
the system achieves competitive
accuracy across multiple attacks (e.g., DoS, DDoS), while
maintaining a lightweight computational footprint.
Aldaej et al.[45] propose a two-stage ensemble IDS framework
tailored for IoT-edge platforms, addressing the limitations
imposed by resource-constrained devices. In the first stage,
an Extra Tree (E-Tree) is used to detect whether incoming
IoT traffic is benign and anomalous. In the second stage, an
ensemble model comprising E-Tree, DNN, and RF refines the
classification by identifying the specific nature of the IDS.
The system is evaluated on multiple datasets, including Bot-
IoT, CICIDS2018, NSL-KDD, and IoTID20, demonstrating
superior performance compared to existing ML techniques.

The literature synthesis demonstrates that recent research
has achieved notable progress in ML-based IDS, transformer-
driven feature extraction, and energy-efficient computing at
the edge. Despite these advances, most approaches remain
fragmented, lacking integration between detection models and
semantic reasoning tools. Few studies have explored the
practical combination of lightweight IDS with external LLMs
in a deployable architecture. Moreover, the impact of LLMs
on enhancing IDS interpretability and decision-making in
constrained edge gateways remains largely unexamined. To ad-
dress this gap, we benchmark six ML-based IDS models under
various cyberattacks at the edge. Their alerts are augmented
using three external LLMs (GPT-4-turbo, DeepSeek-V2, and
LLaMA 3.5) via real-time API calls. We also evaluate key real-
world deployment metrics, such as CPU usage, bandwidth,
memory usage, latency, and energy consumption, at the edge.
Additionally, we assess the quality of LLM-generated seman-
tic reasoning to determine its practical value for situational
awareness.

IV. PROPOSED METHOD

This section presents the design and methodology of the
proposed semantic ML-based IDS deployed at
the edge.
The approach integrates ML-based IDS with semantically
enhanced reasoning via LLMs, forming a hybrid detection
pipeline that is both interpretable and resource-efficient. Fur-
thermore, each component is formalized mathematically, fol-
lowed by detailed explanations that emphasize system perfor-
mance, security implications, and the practical feasibility of
deployment.

A. Traffic Monitoring and Feature Extraction

Incoming edge traffic at time t is represented as a structured

feature vector:

xt = [f1(pt), f2(pt), . . . , fn(pt)]⊤, xt ∈ Rn

(3)

where fi are feature extraction functions applied to packets
pt. Each fi represents: statistical descriptors (mean packet
size, variance), temporal metrics (flow duration, inter-arrival
time), and entropy-based measures (protocol diversity, TCP
flag entropy). The extracted vector is normalized as:

˜xt =

xt − µ
σ

, µ =

1
T

T
(cid:88)

t=1

xt,

σ =

(cid:118)
(cid:117)
(cid:117)
(cid:116)

1
T

T
(cid:88)

(xt − µ)2

t=1

(4)
where µ and σ are the empirical mean and standard deviation
vectors estimated over a baseline benign dataset of size T .
This normalization ensures strict comparability across het-
erogeneous edge gateways, regardless of hardware baseline
and traffic scale. From a security perspective, normalization
protects against adversarial evasion strategies such as scaling
attacks, in which malicious traffic is deliberately embedded
within extreme magnitudes to mislead classifiers. Furthermore,
by transforming all features into a standardized space, the IDS
prioritizes relative deviations (e.g., sudden connection surges
or latency spikes) over absolute values, making it significantly
harder for attackers to camouflage anomalies inside large
flows. Formally, the mapping is expressed as:

N : Rn → Rn, N (xt) = ˜xt

(5)

where N is the normalization operator. This projects xt into a
standardized feature domain where classification boundaries
are invariant
to absolute scale. Entropy-based features are
included as:

H(f ) = −

k
(cid:88)

i=1

pi log pi,

(6)

where pi denotes the empirical probability of observing pro-
tocol or flag category i. These measures capture protocol
improving resilience
diversity and TCP flag irregularities,
against polymorphic and obfuscation-based attacks.
A major benefit of this formulation is device-agnostic invari-
ance. For example, let x(s)
CPU and x(g)
CPU denote CPU usage for a
sensor node and a gateway server, respectively. Although their
absolute baselines differ (µ(s) ̸= µ(g)), normalization ensures

5

that an 80% CPU spike is mapped to the same standardized
representation:

CPU ≈ ˜x(g)
˜x(s)
CPU.
This alignment enables a unified IDS policy across heteroge-
neous edge gateways. From a computational standpoint, the
pipeline is bounded by:

(7)

C(N ) = O(n),

(8)

large volumes of traffic,

Ensuring predictable runtime even for high-throughput gate-
ways. Linear complexity provides resilience against adver-
sarial attempts at computational flooding: even if attackers
inject
the system maintains real-
time processing without bottlenecks. Additionally, this stage
establishes the semantic bridge to the LLM. The normalized
feature vectors ˜xt form the structured input layer for telemetry
encoding and prompt construction:

πt = ENCODE(˜xt, R(yt)),

(9)

Where R(yt) maps preliminary ML-based IDS labels into
semantic descriptors (see Section IV-D). Instead of raw packet
traces, the LLM receives compact, security-aware representa-
tions that are invariant to device heterogeneity and adversarial
scaling, ensuring semantically stable reasoning at the edge.

B. Anomaly Detection using ML-based IDS

Each normalized vector ˜xt is evaluated by a trained classi-

fier fθ:

yt = fθ(˜xt),

st ∈ [0, 1]

(10)

Where yt is the predicted traffic class label and st is the
calibrated anomaly score. The anomaly score is defined as:

st = 1 − P (yt = benign | ˜xt, θ),

(11)

Representing the probability mass assigned to non-benign hy-
potheses. This formulation can be interpreted as a generalized
likelihood-ratio test:

Λ(˜xt) =

P (˜xt | H1)
P (˜xt | H0)

,

trigger alert if Λ(˜xt) ≥ η

(12)

Where H0 and H1 correspond to benign and attack hy-
potheses, respectively. For classifiers with probabilistic outputs
(e.g., logistic regression, random forest, neural networks with
softmax layers), the anomaly score is reformulated as:

st = max

y∈Y\{benign}

P (yt = y | ˜xt, θ),

(13)

This reflects the most probable malicious hypothesis against
the benign baseline. In practice, this acts as a security margin:
higher st indicates more substantial evidence of malicious
behavior. For margin-based classifiers, anomaly scores are
defined as:

st = σ(w⊤ ˜xt + b),

(14)

where w is the separating hyperplane, b is the bias, and σ(·)
is a sigmoid calibration function. This ensures anomaly scores
are smoothly mapped into [0, 1], making them compatible with
threshold-based decision policies. From a security perspective,
this stage represents the system’s first line of defense. The

generalized likelihood-ratio test formulation enforces bounded
false-alarm rates (PF A ≤ α for threshold η), ensuring that
benign background noise does not overwhelm downstream
reasoning layers. This statistical grounding is critical in ad-
versarial environments where attackers may attempt to inject
camouflage traffic to mimic benign distributions. From a
computational perspective, anomaly detection operates with
complexity:

O(n · d),

(15)

Where n is the number of input features and d represents
model depth (tree depth for decision trees and number of
hidden layers for neural models). This guarantees scalabil-
ity for edge-deployed ML-based IDS under high-frequency
feature extraction. In addition, this design inherently supports
adversarial robustness. Since st is derived from probabilistic
margins rather than raw thresholds, adversaries manipulate
multiple correlated features simultaneously to evade detection.
This raises the cost of evasion and enhances resilience against
poisoning and adversarial bypass strategies, ensuring reliable
anomaly detection at the edge.

C. Telemetry Collection and Normalization

Once the anomaly score st exceeds the predefined alert
threshold, the system transitions into telemetry collection. At
this stage, a compact system-state vector is constructed:

mt = [ct, mt, lt, et, st]⊤

(16)

where ct represents CPU usage, mt denotes memory usage,
lt captures latency, et indicates energy consumption, and st
reflects the anomaly score inherited from the detection stage.
To ensure device independence, the raw telemetry vector is
normalized against a baseline capacity vector ¯m:

˜mt = mt ⊘ ¯m,

¯m = [100, 2048, 50, 300, 1]⊤

(17)

This guarantees consistency across heterogeneous platforms.
For example, a Raspberry Pi reporting 80% CPU usage
maps to a normalized value of 0.80. In effect, normalization
eliminates bias arising from absolute hardware differences,
thereby enabling fair semantic reasoning across devices.
Formally, each component of the telemetry vector is scaled as:

˜mt,i =

mt,i
¯mi

,

i ∈ {1, . . . , 5}

(18)

Ensuring that all dimensions are bounded within [0, 1]. This
introduces scale invariance, which shifts detection logic from
raw resource magnitudes toward relative deviations. Con-
sequently, reasoning becomes device-agnostic, focusing on
anomalous utilization rather than absolute specifications. From
a statistical perspective, the normalized telemetry vector can
be modeled as a point in a five-dimensional unit hypercube:

˜mt ∈ [0, 1]5

(19)

This compact embedding enables efficient similarity opera-
tions between telemetry snapshots across devices, allowing for
comparative anomaly tracking at scale. From a security per-
spective, normalization thwarts evasion attempts that exploit
device heterogeneity. Without normalization, an attacker might

6

saturate a weaker IoT sensor while remaining undetected on a
stronger edge server. Moreover, by mapping all observations
into the same normalized scale, such tactics are neutralized,
since anomaly detection now depends only on proportional de-
viations. From a computational perspective, telemetry collec-
tion is lightweight, with complexity O(k) where k = 5 metrics
are monitored. The normalization itself requires only vector
division, incurring insignificant cost and latency well below
a millisecond, even across thousands of devices. Additionally,
the normalized telemetry vector ˜mt serves as the core evidence
for subsequent LLM-based reasoning. It provides a mathemat-
ically bounded, semantically interpretable summary of system
health that is fused with contextual attack descriptors during
prompt construction. In this way, telemetry normalization not
only harmonizes cross-device monitoring but also establishes
the foundation for robust semantic reasoning at the edge.

D. Prompt Construction

Following telemetry normalization, the system transitions
into prompt construction, where the evidence is transformed
into a semantically interpretable input for the LLM. The
encoded prompt is defined as:

πt = ENCODE( ˜mt, R(yt))

(20)

Here, ˜mt represents the normalized telemetry vector, while
R(yt) maps each predicted attack class to a textual description
drawn from a fixed, domain-specific knowledge base. This
ensures that quantitative anomalies are seamlessly aligned with
qualitative semantic context.
To maintain interpretability and efficiency, the final prompt is
structured into three ordered blocks: (i) a snapshot of telemetry
metrics, (ii) the contextualized description of the predicted
attack, and (iii) explicit reasoning instructions guiding the
LLM toward actionable diagnostics. Formally, the prompt is
bounded in size by:

|πt| ≤ Bmax, Bmax = 1.2 kB

(21)

This upper bound ensures that prompts remain lightweight
enough for low-bandwidth communication channels. From
an optimization perspective, prompt construction can be ex-
pressed as a constrained concatenation problem:

πt = arg min

π

(cid:16)

Lsemantic(π)

(cid:17)

,

s.t. |π| ≤ Bmax

(22)

Where Lsemantic(π) penalizes information loss due to trunca-
tion, ensuring that critical attack context is always preserved
under bandwidth limitations.
From a security perspective, prompt construction serves as
a safeguard against ambiguous or adversarial signals. For
example, a raw metric such as “CPU = 0.92” is semantically
tied to its associated threat domain (e.g., “brute force with
repeated failed logins”) through R(yt). This mapping prevents
misinterpretation and constrains the LLM to reason strictly
within the appropriate attack context. Furthermore, since πt is
both compact and normalized, it reduces the attack surface for
prompt injection or adversarial tampering during transmission.
Additionally, prompts can be treated as structured embeddings

7

in a semantic vector space, enabling retrieval-based reasoning.
Specifically, similarity between two prompts is defined as:

sim(πt, πt′) =

⟨πt, πt′⟩
∥πt∥∥πt′∥

(23)

This formulation allows the system to retrieve historical
prompts most similar to the current one, thereby support-
ing few-shot reasoning without maintaining heavy persistent
databases. As a result, the IDS inherits a form of lightweight
semantic memory, improving adaptability to recurring attack
patterns. From a computational standpoint, prompt construc-
tion is efficient: it requires O(k) operations for telemetry
encoding, where k is the number of monitored metrics, and
O(1) for context mapping. This makes its overhead negligible
compared to IDS inference, ensuring that the pipeline can
operate seamlessly on constrained edge hardware. Besides,
prompt construction acts as the semantic bridge between quan-
titative system telemetry and qualitative reasoning. It com-
presses heterogeneous signals into a bandwidth-constrained
yet context-rich representation, enabling the LLM to generate
interpretable, attack-specific insights that extend far beyond
raw anomaly scores.

From a security perspective, remote reasoning adds multi-step
causal inference that traditional IDS lacks. By synthesizing
the LLM
telemetry, context, and prior attack knowledge,
generates adaptive and context-aware defense strategies. At
the same time, new risks emerge: adversaries may attempt
adversarial prompt engineering or resource-exhaustion attacks.
To mitigate this, integrity hashing, semantic thresholds, and
rate-limiting policies are embedded into the pipeline, ensuring
that only validated and high-confidence requests consume
LLM resources. From a computational perspective, LLM calls
represent the dominant contributor to both latency and energy
consumption. To balance scalability with security, these calls
are activated selectively: only when the anomaly score st
surpasses the alert threshold τalert. This conditional invocation
ensures that semantic reasoning is reserved for suspicious
traffic,
thereby maintaining efficiency across hundreds of
concurrent edge gateways. Additionally, this stage transforms
raw statistical detections into semantically grounded, context-
aware security responses. It secures the integrity of transmitted
prompts, quantifies reliability via calibrated confidence, and
integrates LLM-driven reasoning into a resource-aware IDS
pipeline suitable for the edge gateway.

E. Remote Reasoning via LLM API

Once the prompt is constructed, it is transmitted to the
remote LLM for semantic reasoning. The inference process
is defined as:

L⋆(πt) (cid:55)→ ⟨ˆy⋆, γ⋆, µ⋆, ρ⋆⟩

(24)

Where ˆy⋆ denotes the refined attack classification, γ⋆ ∈ [0, 1]
is the calibrated semantic confidence, µ⋆ encodes mitigation
recommendations, and ρ⋆ ∈ {Normal, Warning, Critical} spec-
ifies the severity level.
In this stage, the raw anomaly score st is elevated into action-
able cybersecurity intelligence. For instance, instead of a low-
level alert such as “anomaly score = 0.95,” the LLM output a
structured message: “Brute force attack detected, confidence
= 92%, severity = Critical, mitigation = lockout + MFA.”
This semantic enrichment bridges the gap between statistical
detection and operator-ready defense policies. To preserve
trustworthiness, each prompt is signed before transmission:

h(πt) = HASH(πt)

And acceptance is conditioned on integrity verification:

h(πt) = h(πreceived

t

)

(25)

(26)

Otherwise, the request is discarded. This mechanism ensures
resilience against prompt tampering, replay attacks, and at-
tempts at adversarial injection.
From a probabilistic viewpoint,
interpreted as:

the confidence γ⋆ can be

γ⋆ = P (ˆy⋆ | πt, ΘLLM)

(27)

where ΘLLM are the model parameters. This posterior formu-
lation provides a principled measure of reliability, allowing
operators to enforce strict acceptance thresholds (e.g., γ⋆ ≥
γmin). Consequently, uncertain inferences are either escalated
for human review or filtered out to avoid false mitigations.

F. Inference Pipeline

the start,

The whole system pipeline is summarized in Algorithm 1.
This algorithm provides a structured workflow that integrates
traditional ML-based IDS with semantic reasoning through
LLMs. At
incoming traffic is transformed into
normalized feature vectors to ensure consistency across het-
erogeneous edge gateways and to prevent adversaries from
exploiting statistical biases. Once features are prepared, the
ML-based IDS performs local classification and assigns an
anomaly score st, which serves as a primary defense filter by
preventing benign traffic from triggering unnecessary semantic
analysis. When suspicious activity is detected, runtime teleme-
try is collected and normalized, ensuring that CPU, memory,
latency, energy, and anomaly values remain comparable across
devices of different capacities. This guarantees fairness in
interpretation and prevents attackers from exploiting hardware-
specific weaknesses. The prompt construction stage then em-
beds both the telemetry and contextual information about the
predicted attack type, producing a semantically rich query
for the LLM. Importantly, before transmission, the system
applies a hashing function h(πt) to the prompt, ensuring
integrity verification and safeguarding against tampering or
adversarial prompt injection during communication. Once the
LLM returns its enriched response,
the system applies a
validation mechanism that discards outputs with low semantic
confidence values γ⋆ < γmin. This security check ensures
that unreliable or adversarially manipulated responses are not
automatically acted upon. The final findings, including refined
attack labels, severity ratings, and recommended mitigations,
are logged and forwarded both to human analysts and au-
tomated defense actuators. Additionally, Algorithm 1 is not
only a pipeline for semantic enrichment but also a layered
defense mechanism. It combines anomaly filtering, telemetry
normalization, cryptographic hashing, and semantic validation,

thereby enhancing robustness against adversarial noise, data
poisoning, and injection attacks.

Algorithm 1 LLM-Integrated ML-based IDS Pipeline

Require: Feature stream {xt}, trained IDS fθ, threshold τalert,

context map R(·), LLM API L⋆

where adversaries deliberately trigger costly LLM invocations
to exhaust device resources. By continuously monitoring Et
relative to expected baselines, the IDS gains an additional
detection dimension that complements anomaly scoring. This
ensures that resilience is not limited to traffic-level anomalies
but extends to system-level sustainability.

8

1: for each time step t do
2:
3:
4:
5:
6:
7:

Extract features xt ← Extract(pt)
Normalize ˜xt ← (xt − µ)/σ
Predict class yt ← fθ(˜xt)
Compute anomaly score st
if st ≥ τalert then

Collect telemetry mt ← [ct, mt, lt, et, st]
Normalize ˜mt ← mt ⊘ ¯m
Retrieve context ctxt ← R(yt)
Construct πt ← ENCODE( ˜mt, ctxt)
Sign h(πt) ← HASH(πt)
Query LLM ⟨ˆy⋆, γ⋆, µ⋆, ρ⋆⟩ ← L⋆(πt)
Validate: reject if γ⋆ < γmin
Log outputs, trigger mitigations

Continue benign logging

8:
9:
10:
11:
12:

else

13:
14:
15:
16:
17:
18: end for

end if

G. Latency and Energy Modeling

The end-to-end detection delay is formalized as:

Tt = TIDS + Ttx + TLLM

(28)

Where TIDS denotes the local ML-based inference time (typi-
cally measured in milliseconds), Ttx represents the commu-
nication delay between the edge gateway and the remote
reasoning service, and TLLM accounts for the dominant latency
introduced by semantic reasoning This decomposition enables
precise identification of bottlenecks within the pipeline. For
instance, lightweight classifiers reduce TIDS, whereas limited-
bandwidth edge networks often dominate Ttx. In contrast,
complex reasoning over large prompts primarily increases
TLLM. From a security perspective, bounding Tt is essential:
adversaries exploit delayed responses to saturate resources
before mitigations are applied, thereby amplifying attack im-
pact. Consequently, latency modeling serves not only as a
performance metric but also as a defensive safeguard against
timing-based adversarial strategies. The energy consumption
per detection cycle is expressed as:

Et =

N
(cid:88)

i=1

Pi · ∆τ, ∆τ = 10 ms, N =

Tt
∆τ

(29)

Where Pi denotes instantaneous power at sub-interval i, this
discrete-time integration offers fine-grained profiling of de-
vice energy usage during the inference and reasoning cycle.
Crucially, the model highlights that latency and energy are
tightly coupled; higher TLLM simultaneously increases Tt and
amplifies Et. For edge devices with constrained batteries, this
trade-off determines long-term sustainability. From a defensive
perspective, deviations in Et can signal energy-drain attacks,

H. Operational Constraints

To ensure resilience under adversarial and resource-limited
conditions, the IDS is governed by strict operational con-
straints:

Tt ≤ Tmax = 1.5 s
Et ≤ Ebudget = 100 J
γ⋆ ≥ γmin = 0.60

(30)

(31)

(32)

Constraint
(1) enforces near-real-time responsiveness by
bounding the end-to-end detection latency. This ensures that
attack mitigation actions are applied quickly enough to prevent
escalation, even under high traffic loads or adversarial attempts
to delay system responses. Constraint (2) restricts the per-cycle
energy consumption of the IDS, thereby protecting battery-
powered and resource-constrained edge gateways against
energy-drain attacks. By maintaining Et ≤ 100 J, the system
guarantees sustainable operation without sacrificing detection
capability. Constraint (3) introduces a semantic confidence
floor on LLM-based reasoning. Any refined prediction with
γ⋆ < γmin is discarded, ensuring that low-certainty outputs
do not trigger misleading and unsafe mitigation actions. This
guards against adversarial prompt manipulation that attempts
to generate ambiguous or low-confidence outputs. Moreover,
these bounds define a constrained optimization problem:

max Detection Accuracy,

s.t. (1) ∧ (2) ∧ (3)

(33)

Where the IDS strives to maximize detection performance
while simultaneously respecting latency, energy, and reliability
budgets, from a system-level perspective, these constraints
operate as runtime defenses. They prevent adversaries from
bypassing security by overwhelming the IDS with timing or
resource-based attacks and guarantee stable operation across
heterogeneous edge gateways. By embedding these safeguards
directly into the optimization framework, the IDS balances
accuracy, efficiency, and robustness, ensuring practicality for
deployment at the edge.

I. Evaluation Framework

We evaluate six ML-based IDS configurations:

fθ ∈ {DT, KNN, RF, CNN, LSTM, CNN+LSTM}

(34)

Across five cyber attack classes:

Y′ = {DoS, DDoS, brute force, port scanning}

(35)

Evaluation is based on four dimensions:

• Accuracy: Standard classification metrics (precision, re-
call, F1 score), measuring raw ML-based IDS effective-
ness.

9

• Semantic gain: Improvement ∆F1 after LLM reasoning,
quantifying interpretability and contextual awareness.
• Runtime: Average latency Tt and energy Et, validating

efficiency under attack stress.

both benign behavior and diverse cyberattacks. Its comprehen-
sive coverage makes it ideal for evaluating the performance of
ML-based IDS in IoT systems. The distribution of attacks and
benign entries is detailed in Table III.

• Interpretability: Human expert rating ι ∈ [1, 5], as-
sessing how actionable and understandable the enriched
outputs are.

This multi-metric framework goes beyond accuracy to capture
interpretability, efficiency, and resilience. Moreover, by ana-
lyzing ∆F1, it quantifies the semantic contribution of LLMs
in reducing false positives and improving operator trust. Run-
time and energy measurements verify that constraints (1)–(3)
are met even under resource-intensive attacks. Interpretabil-
ity scores confirm that the enriched outputs bridge the gap
between raw anomaly signals and practical security decision-
making. Collectively, this framework validates the feasibility
of deploying the hybrid IDS at the IoT edge, ensuring both
technical and security robustness.

V. EXPERIMENTAL DESIGN

This section describes our methodology for evaluating the
impact of integrating ML-based IDSs with LLMs on the
edge gateway. We first present our Research Questions (RQs),
followed by an explanation of the study design, which includes
the deployment environment, traffic scenarios, and evaluation
metrics used to analyze detection performance.

A. Research questions(RQs)

Our research aims to address the following RQs:
• RQ1: How does the integration of LLMs impact the
detection accuracy of ML-based IDSs under different
cyberattacks?
This question examines whether LLM-enhanced ML-
based IDS can achieve measurable improvements in clas-
sification accuracy across various types of cyberattacks.
• RQ2: To what extent can external LLMs enhance the
semantic reasoning and interpretability of ML-based
IDS outputs without compromising real-time detection
performance?
This question examines how effectively LLMs improve
the clarity and contextual relevance of ML-based IDS
alerts while maintaining operational responsiveness for
real-world applications.

• RQ3: What

is the impact of LLM-assisted ML-
based IDS on system-level performance metrics, e.g,
CPU usage, energy consumption, and latency when
deployed at the edge?
This question assesses the resource overhead introduced
by incorporating LLMs and evaluates whether such in-
tegration respects the real-time constraints of low-power
platforms( e.g., Raspberry Pi).

B. Dataset

We used the CICIDS2017 dataset [46], a widely accepted
benchmark that simulates realistic network traffic, including

TABLE III: Distribution of labeled IoT attacks in the dataset

IoT Attack Labels
BENIGN
DoS Hulk
Port Scan
DDoS
DoS GoldenEye
FTP-Patator
SSH-Patator
DoS Slowloris
DoS Slowhttptest
Bot
Web Attack & Brute Force
Web Attack & XSS
Infiltration
Web Attack & SQL Injection
Heartbleed

No of Labeled Entries
2,271,320
230,124
158,804
128,025
10,293
7,935
5,897
5,796
5,499
1,956
1,507
652
36
21
11

C. Implementation details

To assess the practicality of deploying a semantically ML-
based IDS at the edge, we designed and implemented an
experimental testbed using two Raspberry Pi 4 Model B units,
each equipped with 8 GB of RAM and a 1.5 GHz 64-bit quad-
core processor. This configuration reflects the constraints of a
real-world edge gateway, allowing for a thorough evaluation of
resource consumption and operational performance. On these
devices, we deployed six widely adopted ML-based IDS for
real-time IDS. When an anomaly is identified, the edge device
collects a telemetry snapshot comprising CPU and memory
usage, latency, energy consumption, and the model’s anomaly
score. This information is normalized and transmitted via
secure, low-bandwidth API calls to external LLMs to generate
interpretable threat explanations and recommended mitigation
actions. Furthermore, the system was evaluated using real-time
cyber threats by Kali Linux. The overall architecture of the IoT
testbed is illustrated in Figure 1.

VI. EXPERIMENTAL RESULTS
To address RQ1 and RQ2, we designed a controlled evalua-
tion framework that tests the effectiveness of our composable
prompt architecture. Specifically, four representative cyberat-
tacks were simulated and applied across three LLMs(GPT-
4-turbo, DeepSeek-V2, and LLaMA 3.5), ensuring fairness
by providing each model with identical
telemetry inputs,
static RAG-based context, and standardized instruction blocks.
In addition, every LLM was evaluated under three distinct
reasoning modes (zero-shot, few-shot, and chain-of-thought),
enabling a systematic analysis of how prompting strategies
influence detection accuracy,
interpretability, and resource
efficiency.

A. Brute Force Attack Evaluation

This section evaluates the detection of brute-force attacks
using ML-based IDS integrated with three LLMs (GPT-
the edge. The
4-turbo, DeepSeek-V2, and LLaMA3.5) at

10

Fig. 1: IoT-edge testbed topology.

analysis focuses on three main aspects: (i) the reasoning
quality and interpretability of each LLM, (ii) the resource
consumption patterns under brute-force attack, and (iii) the
statistical validity of observed variations across IDS and LLM
configurations. Moreover, by presenting a holistic evaluation,
we highlight both the strengths and weaknesses of hybrid
IDS-LLM frameworks in handling high-frequency brute-force
intrusions.

1) Security and Reasoning Quality: Brute-force attacks are
characterized by repeated authentication failures, which makes
reasoning quality and interpretability of IDS responses partic-
ularly critical. As illustrated in Figure 2, across all IDS mod-
els, GPT-4-turbo exhibited the most coherent and actionable
reasoning. In few-shot mode, it consistently aligned anomaly
scores (93-97%) and detected connection attempts (300-370)
with brute-force behavioral signatures, thereby yielding simi-
larity scores of 91-94% while maintaining a false positive rate
of 5%. Moreover, its CoT reasoning was both structured and
multi-modal, as it integrated anomaly scores, CPU usage (70-
82%), and latency values (9-11 ms) to produce confidence
levels of up to 84%. In addition, GPT-4-turbo generated
semantically meaningful mitigation strategies, including multi-
factor authentication (MFA), lockout policies, and IP blocking.
By contrast, DeepSeek-V2 also demonstrated substantial nu-
merical precision, with similarity levels between 86-89% and
confidence scores often exceeding 90%. However, its rea-
soning remained formulaic, as the system heavily empha-
sized CPU usage (70-92%) and anomaly scores (91-97%),
while underweighting secondary telemetry, such as bandwidth
fluctuations (<95 MB/s). Consequently, this one-dimensional
reasoning approach occasionally limited its interpretability,
as secondary indicators such as connection diversity were
underrepresented. As a result, false positive rates were higher
(8-12%), suggesting that while DeepSeek-V2 is reliable in
detecting brute-force attempts, its context awareness and cross-
metric integration remain limited compared to GPT-4-turbo.

On the other hand, LLaMA3.5 consistently ranked lowest in
reasoning quality. Its outputs often isolated single features such
as latency (10-13 ms) and memory usage (>1150 MB) without
contextual integration. Although structurally compliant with
reasoning formats, it nevertheless failed to establish causal
relationships across metrics. Consequently, similarity scores
remained lower (80-85%), confidence frequently dropped to
65-78%, and in some instances, severe brute-force attacks were
downgraded to “warnings.” This conservative bias. However,
reducing false negatives in some scenarios poses serious risks
in security-critical scenarios at the edge, where underestimat-
ing an attack can directly compromise device integrity and
system safety.

2) Resource Consumption : Resource utilization during
brute-force detection revealed essential insights into the ef-
ficiency of ML-based IDS-LLM integration. As shown in
Figure 3, classical ML-based IDS models such as DT, KNN,
and RF consistently demonstrated resource efficiency, since
they required lower memory, CPU, and energy demands.
Specifically, these models consumed 1020-1320 MB of mem-
ory, maintained CPU usage between 76-89%, and produced
latency values in the range of 8-11 ms, thereby making them
suitable for resource-constrained edge gateways.
By contrast, deeper ML-based IDS models (CNN, LSTM, and
a hybrid model of CNN and LSTM) imposed significantly
higher overhead. Memory usage peaked at 1340 MB, CPU
usage ranged from 79-93%, and energy consumption reached
up to 79 J. Moreover, latency also increased (10-13 ms),
reflecting the computational complexity of DL architectures.
Interestingly,
the integration of LLMs did not drastically
amplify resource consumption beyond the base ML-based IDS
requirements. Instead, the LLM component primarily impacted
interpretability and reasoning quality, while imposing insignif-
icant additional overhead on memory, CPU, and bandwidth.
Consequently, the resource trade-off highlights a fundamental
deployment challenge; while CNN and LSTM-based IDS

11

Fig. 2: Security reasoning performance under brute-force.

improve raw detection accuracy, they also increase operational
costs. For edge, this implies that selecting the ML-based IDS
backbone should depend not only on accuracy but also on the
available device capabilities and energy budgets. Furthermore,
the LLM choice (GPT-4-turbo, DeepSeek-V2, or LLaMA3.5)
had only a marginal impact on system-level resource foot-
prints.

3) Statistical Analysis: Statistical evaluation using one-way
ANOVA and Tukey HSD confirmed that differences in brute-
force detection overhead across LLMs were not statistically
significant. Specifically, as summarized in Table IV, for all
five resource metrics (memory, bandwidth, CPU, energy, and
latency), p-values exceeded 0.05 and effect sizes remained
insignificant (η2 < 0.003). Consequently, this indicates that
LLM selection does not materially impact resource consump-
tion, even under sustained brute-force scenarios. Moreover,
since LLM choice does not introduce significant computational
costs, deployment decisions can instead be guided by reason-
ing quality, interpretability, and security robustness rather than
raw performance efficiency. Thus, the emphasis shifts toward
evaluating how effectively each LLM translates raw telemetry
into semantically actionable insights.

Finding

GPT-4-turbo delivered the strongest balance of rea-
soning depth, interpretability, and actionable mitiga-
tion strategies, making it the most security-relevant
choice for brute-force detection. DeepSeek-V2 of-
fered reliable anomaly detection with strong numerical
accuracy but remained formulaic and less context-
aware. LLaMA3.5, while lightweight, provided shal-
low reasoning and, in some cases, downgraded critical
intrusions, creating risks for a mission-critical edge
gateway. Since resource utilization did not significantly
differ across LLMs, the decisive factor in ML-based
IDS-LLM integration lies in the quality of semantic
reasoning rather than computational overhead.

B. DoS Attack Evaluation

This section evaluates DoS detection under LLM-integrated
ML-based IDS, focusing on three significant aspects: (i) the

reasoning quality and interpretability of LLM outputs, (ii)
the resource consumption patterns when subjected to DoS,
and (iii) the statistical significance of observed differences
across models. Since DoS attacks aim to saturate system
resources from a single source, both the semantic reasoning
ability of LLMs and the efficiency of underlying ML-based
IDS backbones play a crucial role in sustaining detection
performance under stress.

1) Security and Reasoning Quality: GPT-4-turbo consis-
tently outperformed the other LLMs in reasoning quality;
specifically,
it demonstrated strong multi-metric reasoning
by aligning anomalies (94-96%), bandwidth usage (92-110
MB/s), and latency (up to 13 ms) with canonical DoS signa-
tures. Consequently, this produced similarity scores of 90-92%
while maintaining the lowest false positive risk of 4-5%. More-
over, GPT-4-turbo was able to cross-link telemetry, including
CPU, bandwidth, and latency, and generate contextualized mit-
igation strategies such as SYN cookies, WAF rate-limiting, and
IP blocking. Furthermore, these recommendations demonstrate
a layered defense and highlight its ability to transition from
anomaly detection to actionable cyber defense, as shown in
Figure 4.
By contrast, DeepSeek-V2 performed reliably in terms of
numeric accuracy, achieving similarity levels of 85-88% and
confidence scores of 85-89%. However, its reasoning was more
rigid and template-driven, since the model frequently empha-
sized CPU usage (85-89%) and anomaly scores (92-95%)
while neglecting secondary telemetry such as bandwidth and
memory. As a result, this narrow focus reduced interpretability
and limited causal reasoning. Although detection itself re-
mained correct, its explanations were repetitive and lacked
cross-metric integration, which is important for diagnosing
complex DoS variants that exploit multiple system bottlenecks.
Consequently, false positive risk was higher (6-8%).
On the other hand, LLaMA3.5 consistently lagged, achieving
only 82-86% similarity and 69-84% confidence, with false
positive risk reaching 8-9%. In particular, its reasoning was
shallow, as it often focused on single indicators such as latency
spikes and memory usage above 1300 MB. This single-feature
dependency made its outputs less robust and occasionally
misleading, since it failed to establish cause-and-effect links
between anomalous metrics. In some instances, attacks were
flagged only as “warnings,” which could delay mitigation.

12

Fig. 3: Resource usage under brute-force.

TABLE IV: Statistical analysis under brute-force.

Metric
Memory
Bandwidth
CPU
Energy
Latency

ANOVA F
0.868
0.106
1.483
0.106
1.182

p-value
0.420
0.899
0.227
0.899
0.307

Effect Size (η2)
0.0016
0.0002
0.0027
0.0002
0.0022

Tukey Result
No difference
No difference
No difference
No difference
No difference

For a mission-critical edge gateway,
LLaMA3.5 unsuitable as a primary detection engine.

this weakness makes

2) Resource Consumption: Resource consumption analysis
revealed a clear trend: deeper IDS models (CNN, LSTM, and
a hybrid model of CNN and LSTM) demanded significantly
higher resources under DoS, whereas classical ML-based IDS
models (DT, KNN, RF) remained lightweight and efficient.
Specifically, memory usage spanned from 980 MB to 1380
MB, with LLaMA3.5-based configurations tending toward the
higher end. CPU usage ranged from 72-94%, thereby reflecting
the strain of sustained DoS. Moreover, bandwidth usage varied
between 50 and 110 MB/s, highlighting the network stress
imposed by flooding attempts. Energy demand ranged from 61
J to 80 J, and latency values spanned 8-14 ms. These patterns
are summarized in Figure 5.
Classical ML-based IDS models, e.g., DT, KNN, and RF,
have proven suitable for edge gateway, as efficiency is crit-
ical in constrained environments. Their lower memory and
CPU requirements allowed them to sustain operations while

still enabling semantic reasoning through LLM integration.
Moreover, by contrast, CNN and LSTM-based architectures
consumed significantly more resources, a trade-off that could
limit their applicability at the edge. Consequently, these find-
ings suggest that LLM choice has little impact on raw resource
usage, whereas ML-based IDS model complexity remains the
primary determinant of operational cost.

3) Statistical Analysis: To determine the statistical sig-
nificance of these observations, ANOVA and Tukey HSD
tests were applied across all LLM-integrated ML-based IDS
models. As summarized in Table V, the results confirmed
that memory usage was the only metric with statistically
significant differences (p < 0.0001), with LLaMA3.5 consum-
ing substantially more than GPT-4-turbo and DeepSeek-V2.
Moreover, the effect size for memory was large (η2 = 0.949),
underscoring its practical impact. By contrast, for all other
metrics, including CPU usage, bandwidth, latency, and energy,
p-values exceeded 0.05 and effect sizes remained insignificant
(η2 < 0.005). Consequently, this indicates that LLM integra-

13

Fig. 4: Security and reasoning quality under DoS.

Fig. 5: Resource usage under DoS.

tion does not introduce variability in these dimensions. From a
security standpoint, this consistency is advantageous, since it
ensures predictable resource footprints under DoS conditions
regardless of the chosen LLM.

Finding

GPT-4-turbo demonstrated the strongest balance of
reasoning quality, accuracy, and actionable mitigation
under DoS. Its ability to integrate multiple telemetry
features into coherent narratives makes it highly effec-
tive for real-time defense. DeepSeek-V2 provided nu-
merically sound but repetitive and less insightful out-

puts, making it appropriate for scenarios where raw de-
tection is prioritized over interpretability. LLaMA3.5,
while functional, was limited in reasoning quality and
imposed the heaviest memory burden, making it less
suited for constrained deployments. Resource analysis
confirmed that the choice of ML-based IDS backbone,
rather than LLM, remains the dominant factor in
resource usage. Thus, prioritizing semantic reasoning
performance (favoring GPT-4-turbo) while deploying
lightweight ML-based IDS models offers the most
balanced and practical defense strategy against DoS
at the edge gateway.

TABLE V: Statistical analysis under DoS.

Metric
Memory
Bandwidth
CPU
Energy
Latency

ANOVA F
9963.87
0.045
0.386
0.385
1.765

p-value
<0.0001
0.956
0.680
0.680
0.172

Effect Size (η2)
0.949
0.00008
0.0007
0.0007
0.0033

Tukey Result
LLaMA > GPT-4, DeepSeek
No difference
No difference
No difference
No difference

14

C. DDoS Attack Evaluation

This section evaluates DDoS detection with LLM and ML-
based IDS integration, focusing on three core aspects: (i)
security reasoning quality and interpretability of LLM outputs,
(ii) system-level resource usage under large-scale distributed
flooding traffic, and (iii) statistical validation of differences
across models. Since DDoS attacks combine distributed ori-
gins with sustained volume, they represent one of the most
resource-intensive and challenging scenarios for both ML-
based IDS and LLM integration.

1) Security and Reasoning Quality: GPT-4-turbo consis-
tently provided the strongest reasoning depth, achieving 90-
95% similarity with known DDoS patterns and confidence
levels of 85-92%. In addition, its false positive rate remained
at 5-6%, which was the lowest among all LLMs. Importantly,
GPT-4-turbo employed multi-step causal reasoning that linked
anomaly scores, bandwidth saturation (often above 100 MB/s),
and latency increases (10-13 ms) into coherent explanations.
the recommended mitigations-e.g., rate-limiting,
Moreover,
SYN-proxying, and blackholing of distributed IP sources-
highlighted its ability to generate actionable security, as shown
in Figure 6.
By contrast, DeepSeek-V2 achieved 85-90% similarity and
88-92% confidence, thereby showing solid recognition accu-
racy. However, its reasoning style was more descriptive than
analytical, since it primarily focused on CPU and memory
indicators without connecting these with network-layer anoma-
lies. Consequently, explanations often appeared formulaic and
less adaptable to evolving multi-vector DDoS campaigns, even
though raw classification remained correct. Furthermore, the
false positives ranged between 7-12%.
On the other hand, LLaMA3.5 lagged, with similarity re-
stricted to 74-83% and confidence levels between 70-80%. Ad-
ditionally, its false positive risk ranged from 9% to 12%. The
reasoning was shallow, as it frequently highlighted isolated
features, e.g., latency and CPU, without integrating them into a
causal attack narrative. As a result, this fragmented reasoning
increased the chance of mislabeling severe DDoS floods as
moderate anomalies, thereby undermining its reliability for a
mission-critical edge gateway.

2) Resource Consumption Summary: Resource consump-
tion under DDoS was elevated compared to DoS because of
the distributed traffic load. Specifically, memory usage ranged
from 1000-1350 MB across IDS models, with CNN and
LSTM variants consuming the most. Moreover, CPU usage
reached 75-92%, thereby highlighting the heavy computational
demand of processing distributed flooding. In addition, energy
consumption was between 65 and 80 J, and latency spanned
8-13 ms, which nevertheless remained acceptable for near

real-time detection. Bandwidth stress was particularly high;
however, DT, KNN, and RF managed it more efficiently,
whereas CNN and LSTM models approached upper thresholds
of 110 MB/s. These results are illustrated in Figure 7.
Taken together, these findings underscore that classical ML-
based IDS models are resource-efficient, as they sustain op-
eration with minimal degradation under large-scale traffic.
By contrast, CNN and LSTM models, while offering slightly
stronger detection accuracy, impose higher operational costs.
Furthermore, LLM integration did not drastically alter CPU,
energy, and latency footprints; nevertheless, bandwidth han-
dling varied, reflecting GPT-4’s tendency to include richer
multi-metric reasoning in its semantic outputs.

3) Statistical Analysis: ANOVA and Tukey HSD were
applied to validate the statistical relevance of differences
across metrics. The results, summarized in Table VI, indicated
that bandwidth was the only significantly different metric
(p < 0.001, η2 = 0.9725). Moreover, Tukey analysis con-
firmed that GPT-4-turbo introduced slightly higher bandwidth
demands compared to DeepSeek-V2 and LLaMA3.5, which is
consistent with its multi-metric reasoning style that required
processing larger contextual prompts. In contrast, for memory,
CPU, energy, and latency, p-values exceeded 0.05, and effect
sizes remained minimal (η2 < 0.005). Consequently,
this
suggests that LLM choice does not impact these metrics in a
statistically significant way. From a security perspective, this
stability is desirable, since it guarantees predictable resource
costs regardless of which LLM is deployed.

Finding

GPT-4-turbo demonstrated the most effective reason-
ing, as it integrated multiple features into causal attack
narratives and proposed layered mitigations. By con-
trast, DeepSeek-V2, while accurate, relied on descrip-
tive and less adaptable outputs. Moreover, LLaMA3.5
showed fragmented reasoning and higher false pos-
itive risks, thereby limiting its use in critical envi-
ronments. From a resource standpoint, classical ML-
based IDS backbones proved efficient under DDoS
stress, whereas deeper architectures added cost without
providing significant statistical benefit. The only ex-
ception was bandwidth, since GPT-4-turbo’s reasoning
demanded more resources; however, this cost is offset
by its superior interpretability and actionable insights.

D. Port Scanning Evaluation

This section evaluates port scanning detection with ML-
based IDS and LLM integration. Moreover, port scanning is

15

Fig. 6: Security reasoning under DDoS.

Fig. 7: Resource usage under DDoS.

TABLE VI: Statistical analysis under DDoS.

Metric
Memory
Bandwidth
CPU
Energy
Latency

ANOVA F
0.612
19062.1
0.100
1.066
2.704

p-value
0.542
<0.001
0.905
0.345
0.067

Effect Size (η2)
0.0011
0.9725
0.0011
0.0020
0.0050

Tukey Result
No difference
Significant difference
No difference
No difference
No difference

a reconnaissance activity that generates abnormal connection
surges and sequential port access, making it critical to detect
early before exploitation attempts can occur. The evaluation
highlights the quality of reasoning, efficiency under resource
constraints, and statistical validation of observed differences

across ML-based IDS and LLM settings.

1) Security and Reasoning Quality: GPT-4-turbo demon-
strated the strongest interpretability, as it correlated anomaly
levels (92-98%) with connection surges and bandwidth pat-
terns, while also factoring in latency (∼11 ms). Consequently,

this holistic analysis yielded similarity scores of 89-93%,
confidence of 85-92%, and false positives around 5%. More-
over, its causal reasoning was robust, since it narrated the
relationship between rapid connection bursts and port-based
probing. Recommended mitigations, therefore, included dy-
namic firewall rules and scan-throttling strategies, as illustrated
in Figure 8.
By contrast, DeepSeek-V2 achieved 83-88% similarity and 84-
90% confidence, with false positives between 6-8%. Although
accurate, its reasoning was formulaic, as it repeatedly high-
lighted CPU and anomaly alignment without deeper causal
connections. As a result, outputs were less useful for analysts
seeking rich situational awareness.
On the other hand, LLaMA3.5 remained the weakest, achiev-
ing 81-86% similarity and 70-79% confidence, with false
positives at 8-9%. In particular, its reasoning often isolated
individual features, e.g., latency spikes, without linking them
to broader scan patterns. Consequently, this reduced inter-
pretability and occasionally led to misclassification of stealth
scans as benign.

2) Resource Consumption: Resource consumption during
port scanning remained consistent with other attack classes.
Specifically, memory usage spanned 1000-1350 MB, CPU
ranged from 75-92%, and bandwidth demands were between
50-110 MB/s. In addition, energy costs varied from 63-77
J, while latency remained at 8-13 ms. Moreover, classical
ML-based IDS backbones (DT, KNN, RF) sustained efficient
operation, whereas CNN and LSTM consumed more re-
sources, particularly memory and bandwidth. These results are
summarized in Figure 9, thereby highlighting that despite the
additional complexity of sequential scan detection, overhead
nevertheless remained manageable across all ML-based IDS
and LLM configurations.

3) Statistical Analysis: Statistical testing confirmed that no
significant differences emerged across LLMs for port scanning.
Specifically, the results summarized in Table VII show that
ANOVA returned p > 0.05 for all metrics, and effect sizes
were insignificant (η2 < 0.003). Moreover, Tukey HSD further
confirmed this uniformity, suggesting that the resource over-
head of integrating LLMs into ML-based IDS does not vary
meaningfully under scan detection. Consequently, this stability
is valuable for deployment, since it guarantees predictable
performance across heterogeneous edge gateways.

Finding

GPT-4-turbo delivered the most
interpretable port
scanning detection, as it generated coherent, causal
explanations linking anomaly scores, latency, and con-
nection surges. Furthermore, by contrast, DeepSeek-
V2 offered accurate detection but relied on repetitive
and narrow reasoning, whereas LLaMA3.5 was shal-
low and inconsistent, often isolating single features
without integration. Moreover, since resource usage
patterns were stable across all models, the decisive
factor remains reasoning quality, thereby making GPT-

16

4-turbo the most effective choice for early reconnais-
sance threat detection.

VII. CROSS-ATTACKS ANALYSIS

This section addresses RQ3, as it synthesizes the results
across four attack categories( Brute Force, DoS, DDoS, and
Port Scanning) by comparing mL-based IDS and LLM hybrid
performance. Specifically, the comparative discussion empha-
sizes three axes: (i) reasoning quality and interpretability
of LLM outputs, (ii) resource consumption patterns across
ML-based IDS and LLM configurations, and (iii) statisti-
cal validation of observed differences. Taken together, these
findings provide a holistic understanding of how semantic
reasoning impacts the robustness of ML-based IDS in a real
edge gateway, and consequently, they offer direct evidence in
support of answering RQ3.

A. Security and Reasoning Quality Across Attacks

GPT-4-turbo consistently demonstrated the most coherent
and actionable reasoning across all four attack categories.
Specifically, it achieved anomaly alignment of 92-98%, sim-
ilarity scores above 90%, semantic confidence in the 85-
92% range, and maintained the lowest false positive risk
(4-6%). Moreover, its reasoning combined multiple teleme-
try dimensions, e.g., anomaly levels, CPU usage, bandwidth
surges, memory usage, and latency variations, into coherent
causal narratives. Consequently, security recommendations,
i.e., MFA, IP blocking, SYN cookies, and WAF throttling,
were context-specific,
thereby making GPT-4 outputs both
accurate and operationally useful, as shown in Figure 10.
By contrast, DeepSeek-V2 also provided high anomaly accu-
racy (91-97%), with similarity in the 85-90% range and seman-
tic confidence between 85-92%. However, its interpretability
lagged because of formulaic reasoning, as explanations often
emphasized a single metric (e.g., CPU or anomaly rate) while
neglecting secondary features such as bandwidth and energy
consumption. As a result, while DeepSeek-V2 was reliable in
raw detection, its explanatory depth remained limited, which
hindered analysts during forensic investigations or layered
defense planning.
On the other hand, LLaMA3.5 consistently lagged in seman-
tic integration. Its anomaly scores ranged between 80-94%,
similarity dropped to 74-86%, confidence fell to 70-80%, and
false positives were higher (8-12%). In particular, its reasoning
often isolated one or two signals (e.g., memory overhead in
DoS, latency spikes in DDoS) without integrating them into a
coherent story. Consequently, this caused critical intrusions to
be occasionally downgraded to “warnings,” thereby represent-
ing a potential blind spot in the mission-critical edge gateway.

B. Resource Consumption Comparison

Resource costs were broadly stable across all attack types,
thereby confirming that IDS-LLM integration introduces min-
imal additional overhead. Specifically, the typical resource

17

Fig. 8: Security reasoning under port scanning.

Fig. 9: Resource usage under port scanning.

Fig. 10: Comparative security reasoning performance of LLMs across attack categories.

18

TABLE VII: Statistical analysis under port scanning.

Metric
Memory
Bandwidth
CPU
Energy
Latency

ANOVA F
0.256
0.037
0.901
0.901
1.228

p-value
0.774
0.964
0.407
0.407
0.293

Effect Size (η2)
0.0005
0.0004
0.0017
0.0017
0.0065

Tukey Result
No difference
No difference
No difference
No difference
No difference

ranges observed included 950–1380 MB of memory, 72-
95% CPU utilization, bandwidth consumption of less than
120 MB/s, energy draw between 61-80 J, and latency of 8-
14 ms. Moreover, these values remained consistent across
Brute Force, DoS, DDoS, and Port Scanning scenarios, as
summarized in Figure 11.
In terms of model efficiency, classical IDS models (DT, KNN,
RF) were consistently more efficient, as they consumed fewer
resources while still maintaining adequate detection accuracy.
By contrast, CNN, LSTM, and the hybrid model of CNN
and LSTM incurred higher memory and CPU costs, yet
they achieved stronger raw detection capability. Consequently,
this trade-off reinforces the suitability of lightweight models
for constrained IoT gateways, whereas deep models remain
valuable in deployments where stronger detection is prioritized
over efficiency.
Among the LLMs, GPT-4-turbo and DeepSeek-V2 showed
similar resource profiles. However, LLaMA3.5 consumed sig-
nificantly more memory under DoS conditions. A second
notable trend emerged under DDoS, where GPT-4-turbo intro-
duced higher bandwidth peaks due to larger reasoning outputs;
nevertheless, these remained within operational IoT thresholds.

C. Statistical Analysis

One-way ANOVA with Tukey HSD post-hoc testing re-
vealed that for most metrics, LLM differences were statis-
tically insignificant (p > 0.05, η2 < 0.005), as summarized
in Table VIII. This result supports the conclusion that LLM
integration does not substantially alter system efficiency under
most conditions. However, two significant exceptions emerged.
First, under DoS conditions, LLaMA3.5 consumed substan-
tially more memory, with results highly significant (p <
0.0001, η2 ≈ 0.95). This makes LLaMA3.5 unsuitable for
constrained IoT environments where memory overhead is a
critical concern. Second, during DDoS scenarios, GPT-4-turbo
exhibited consistently higher bandwidth demands, significant
at p < 0.001, η2 ≈ 0.97. Although these demands remained
within edge tolerances, they suggest that bandwidth-aware
policies require stable deployment.

Finding

GPT-4-turbo emerged as the most reliable LLM for
ML-based IDS integration. Specifically, it consistently
achieved anomaly alignment above 92%, similarity
greater than 90%, minimized false positives, and de-
livered actionable, multi-metric reasoning. By contrast,

DeepSeek-V2 struck a balance between accuracy and
efficiency but was less interpretable, thereby making it
more suitable in contexts where numeric precision out-
weighs explainability. On the other hand, LLaMA3.5,
while functional, was resource-heavy under DoS and
semantically shallow across all attacks, which makes it
better suited as a backup option rather than a primary
detection engine. From the ML-based IDS backbone
perspective, DT, KNN, and RF remain optimal for con-
strained devices. In contrast, CNN, LSTM, and a hy-
brid of CNN and LSTM enhance detection robustness
but at a higher computational cost. Ultimately, these
insights confirm that the quality of security reasoning,
rather than raw computational overhead, should guide
the selection of LLMs at the edge gateway.

VIII. RUNTIME LOG EXAMPLE

To illustrate the entire operation of the proposed edge-
based semantic ML-based IDS, Figure 12 presents a structured
runtime log generated during a brute-force attack. It demon-
strates the multi-model anomaly detection process, real-time
system telemetry, context-based prompt creation, LLM-based
enhanced, and mitigation feedback.

IX. ANALYSIS OF LLM-INTEGRATED ML-BASED IDS
PERFORMANCE

This section presents a detailed analysis of the perfor-
mance, resource overhead, and qualitative benefits of inte-
grating LLMs and ML-based IDS. Through quantitative met-
rics, we evaluate the trade-offs introduced by LLM semantic
improvement across latency, energy consumption, detection
performance, interpretability, and reasoning modes.

A. Latency and Energy Impact

Figure 13 illustrates the latency and energy consumption
before and after incorporating LLMs across various ML-based
IDS. The latency increment remains under 35% across all
models, ensuring suitability for real-time operations. Espe-
cially, simpler models such as DT and RF experienced lower
latency and energy increases compared to deeper models,
e.g, CNN and LSTM, which showed the highest resource
demand (28.8% latency increase, 33% energy rise). Despite
these increases, all models stay within acceptable thresholds
(≤1.5s latency and ≤100J energy), validating the deployment
feasibility of our framework in the edge gateway.

19

Fig. 11: Resource consumption across ML-based IDS and LLM hybrids under all attack categories.

TABLE VIII: Aggregate statistical analysis across attack categories.

Metric (by Attack)
DoS Memory
DDoS Bandwidth
CPU (all)
Energy (all)
Latency (all)

ANOVA F
9963.87
19062.1
0.386
0.901
1.228

p-value
<0.0001
<0.001
0.680
0.407
0.293

Effect Size (η2)
0.949
0.9725
0.0007
0.0017
0.0065

Tukey Result
LLaMA > GPT-4, DeepSeek
GPT-4 > DeepSeek, LLaMA
No difference
No difference
No difference

B. Detection Performance Improvement

D. Model Performance Evaluation

Figure 14 demonstrates the improvement in F1-Score (∆F1)
for each ML-based IDS post-LLM. Although all models
demonstrate performance improvements, DL-based IDS, such
as the hybrid model of CNN and LSTM, benefit the most, with
F1 improvements of 0.0114 and 0.0109, respectively. These
results confirm the effectiveness of semantic reasoning in
improving detection accuracy, particularly for complex models
that process intricate data patterns.

C. Interpretability and Reasoning Mode Latency

Figure 15 summarizes the trade-offs between interpretabil-
ity and latency across GPT-4-turbo, DeepSeek-V2, and
LLaMA 3.5. GPT-4-turbo achieves the highest interpretability
median > 4.5, essential for scenarios requiring human-in-
the-loop validation. Conversely, latency varies with reasoning
modes: zero-shot achieves the fastest response ( 1.15s), while
CoT incurs the highest latency ( 1.45s) due to deeper reason-
ing.

We benchmark our proposed LLM-enhanced IDS frame-
work against several recent state-of-the-art ML-based IDS
developed for the edge gateway. As summarized in Table IX,
SecurityBERT [42] uses a 15-layer BERT with privacy-
preserving encoding and achieves 98.2% accuracy with low la-
tency (¡0.15s) and compact size (16.7MB), though it lacks se-
mantic reasoning capabilities. Fed-BERT [43] reaches 97.79%
accuracy in centralized and federated modes with latency
around 0.45s, supporting edge deployment via quantization.
The TL-CBAM-EL model [44] employs attention and transfer
learning, achieving 99.93% accuracy; however,
it requires
image transformation, which limits its real-time applicabil-
ity. ETree, RF and DNN [45] applies a two-stage ensemble
achieving 98.5% accuracy, though lacking latency and energy
benchmarks. Lastly, the GPT embedding-based approach [47]
achieves competitive accuracy but is computationally inten-
sive and unsuitable for the edge gateway. In contrast, our
framework delivers balanced performance across latency, in-

20

[Edge Node: RPi-Gateway-01]
==================================================================
-> New traffic window detected [session_id: 7492]
-> Feature vector extracted:

Timestamp: 2025-06-25 14:08:46

x_t = [0.88, 16, 4.2, 0, 3389, 22, 0.61, 0.97, 230, 18, 0.07, 0.54]

-> Running predictions across ML-based IDS models...

[*] DT
[*] KNN
[*] RF
[*] LSTM
[*] CNN
[*] Hybrid CNN and LSTM

-> Label: brute force
-> Label: brute force
-> Label: brute force
-> Label: brute force
-> Label: brute force
-> Label: brute force

| Score: 0.91
| Score: 0.89
| Score: 0.94
| Score: 0.96
| Score: 0.92
| Score: 0.97

-> Consensus: 6/6 models classify as "brute force"
-> Aggregated anomaly score s_t = 0.93 >= tau_alert = 0.70 -> ALERT triggered

-> System metrics captured:

CPU = 47.6\%, Memory = 372 MB, Latency = 48.2 ms, Energy = 21.7 J
Normalized telemetry vector:
m_t = [0.476, 0.182, 0.964, 0.072, 0.930]

-> Retrieved context for class "brute force":

"Repeated login attempts over network protocols such as SSH or RDP,
Typically using dictionary-based or credential-stuffing attacks."

-> Constructing LLM prompt with telemetry and context...
-> Sending prompt to external LLM: GPT-4 Turbo

=> LLM Response [elapsed: 0.84 sec]
{

"revised_label": "brute force",
"confidence": 0.95,
"severity": "Critical",
"mitigation": [

"Block the offending IP address",
"Apply rate limiting on authentication endpoints",
"Enforce multi-factor authentication."

]

}

-> Final enriched classification:

: brute force

- Class
- Confidence : 95\%
- Severity : Critical
- Mitigation : IP block, login throttling, and enforce multi-factor authentication

-> Total round-trip latency: 1.32 s
-> Total energy consumption: 23.9 J
-> Mitigation instructions have been dispatched to the local firewall.
==================================================================

Fig. 12: Log showing ML-based IDS, LLM, and automated mitigation pipeline on the edge gateway.

terpretability, and energy, making it well-suited for the edge
gateway.

X. DISCUSSION

This study aimed to design a resource-aware ML-based
IDS framework tailored explicitly for edge gateways. By com-
bining fast, ML-based anomaly detection with modular edge
deployment, we aimed to achieve real-time responsiveness
with minimal system burden. The results strongly suggest that
this approach is both technically sound and practically viable.
One of the most significant findings is that the tested ML-based
IDS demonstrated consistent and stable performance across
key resource metrics. During port scanning, statistical analyses
showed no significant differences in bandwidth, latency, CPU

usage, energy consumption, and memory usage across deploy-
ments using different LLM integrations. In other words, de-
spite differences in ML-based IDS, the impact on edge system
performance remains insignificant. This provides system de-
signers with the confidence to implement such models without
compromising real-time efficiency and resource availability.
Furthermore, when benchmarked against state-of-the-art IDS
architectures, e.g., SecurityBERT [42], Fed-BERT [43], and
ensemble-based approaches like ETree+RF+DNN [45], the
proposed system performs competitively. Although some alter-
natives demonstrate marginal gains in detection accuracy, they
often require heavier on-device computation, increased energy
demands, and lack the lightweight adaptability necessary at
the edge.

21

Fig. 13: Latency and energy consumption comparison before and after LLM integration across ML-based IDS

Fig. 14: Impact of semantics on F1-Score improvement across ML-based IDS

In contrast, our method offers a balanced trade-off: accurate
detection, rapid inference, and minimal overhead. Some chal-
lenges remain. The current implementation utilizes static ML
pipelines and predefined detection parameters, which perform
well against known attack types but require adaptation in
the presence of zero-day exploits and evolving threat signa-
tures. Future improvements include dynamic model retraining,
online learning modules, and integration with decentralized
learning strategies (e.g., federated learning) to accommodate
concept drift. Furthermore, while the inclusion of LLMs
significantly enhances interpretability and actionable threat
intelligence,
this benefit comes with trade-offs in energy
consumption and reliance on external API services. This
dependency is unsuitable for highly constrained and offline

environments. Exploring on-device distilled LLMs and effi-
cient quantized models addresses these concerns. Additionally,
while we demonstrated minimal latency overhead (<1.5 s),
time-critical healthcare systems require even faster semantic
reasoning. Optimizing the LLM prompt construction process
and integrating retrieval-augmented mechanisms with low-
latency knowledge bases further improve performance.

XI. THREATS TO VALIDITY

Empirical research in cyber-physical systems, particularly
in IDS for IoT, faces inherent threats to validity. Following
this section outlines
the guidelines of Wohlin et al. [48],
limitations and the mitigation steps undertaken.

22

Fig. 15: Comparison of interpretability and latency across GPT-4-turbo, DeepSeek-V2, and LLaMA 3.5.

TABLE IX: Comparison of Our Framework with State-of-the-Art IDS Models

Metric
Accuracy (Top Model)
Latency
Energy Consumption
Model Size
Interpretability
Edge Suitability

Our Framework
98.9% (CNN+LSTM)
<1.5s (LLM+ML)
<100J
∼20MB
High (LLM-CoT)
High

SecurityBERT [42] Fed-BERT [43] TL-CBAM-EL [44] ETree+RF+DNN [45] GPT Embed [47]

98.2%
<0.15s
Low
16.7MB
Moderate
High

97.79%
0.45s
Medium
∼25MB
Limited
High

99.93%
High
High
>50MB
Low
Low

98.5% (avg)
Not reported
Medium
Medium
None
High

∼98%
Not reported
High
>300MB
Moderate
Low

A. Internal Validity

The internal validity of our results depends on the ac-
curacy of resource measurements (e.g., CPU usage, mem-
ory utilization, latency, and energy consumption) as well as
the correctness of the ML-based IDS process. To minimize
measurement errors, all experiments were conducted under
controlled and repeatable testbed conditions on Raspberry Pi
4 edge gateways. Each configuration (LLM and ML-based
IDS) was evaluated across 62 independent trials to reduce the
influence of transient system or network variability. System
telemetry was captured using validated logging tools, and
energy consumption was measured with standardized watt-
meter instrumentation. LLM prompts were programmatically
generated in a fixed format with structured reasoning in-
structions to ensure response consistency. Results were then
averaged across trials and subjected to robust statistical anal-
ysis (ANOVA and Tukey HSD), which allowed us to control
random variation and improve confidence in the observed
effects.

B. External Validity

Although the results demonstrate promise for real-time
LLM-enhanced ML-based IDS on the edge gateways, gen-
eralization beyond our testbed is subject to external threats.
These include hardware variability, network topology differ-
ences, and the diversity of LLMs available for integration.

Experiments targeted representative cyberattacks (brute force,
port scanning, DoS, and DDoS) using the CICIDS2017 dataset
and simulated traffic. Moreover, CICIDS2017 is not explic-
itly IoT-centric; it remains one of the most comprehensive,
publicly available, and widely adopted datasets for intrusion
detection research. It captures modern attack behaviors at
the network flow level, which are directly applicable to
IoT networks, as edge gateways also observe traffic at this
same granularity. Moreover, the dataset has been extensively
used in IoT security studies as a benchmark for proxying
IDS approaches, enabling reproducibility and comparability
across various strategies. Nonetheless, it does not capture all
emerging IoT-specific attacks (e.g., protocol-level exploits or
sensor spoofing), and this represents a limitation of the study.
Reliance on externally hosted LLMs also introduces opera-
tional risks related to network availability, latency variability,
and privacy, since prompts must be transmitted outside the
local edge environment. However, our modular architecture
allows for the substitution of different LLM providers or the
deployment of lightweight on-device models with minimal
reconfiguration. Similarly, while evaluations were performed
on Raspberry Pi 4 platforms, performance, latency, and energy
profiles can vary across other IoT hardware classes, includ-
ing ARM-based gateways, industrial controllers, or battery-
powered devices. External APIs further impose operational
constraints in environments with limited connectivity or strict

privacy requirements. These external factors must therefore be
considered when extending the results of this study to broader
IoT deployments.

XII. FUTURE WORK
The proposed framework demonstrates strong performance
in terms of detection accuracy, resource efficiency, and se-
mantic interpretability; however, several avenues for future
research remain to be explored. First, the current implemen-
tation relies on static prompt templates and a limited set of
manually curated few-shot examples to guide LLM reason-
ing. This approach constrains adaptability when encountering
novel and evolving attack types. Future work will explore
retrieval-augmented generation (RAG) techniques to dynam-
ically improve LLM prompts with relevant context drawn
from threat databases, historical attack logs, and real-time
event streams. The architecture currently assumes cloud-based
LLM inference, introducing vulnerabilities such as latency
spikes, service interruptions, and data privacy risks. Research
into on-device or edge-deployable LLMs, using quantization,
pruning, and knowledge distillation, enables local fallback
capabilities in bandwidth-constrained environments. Addition-
ally, the system operates with a fixed pipeline for telemetry
extraction and scoring. Integrating adaptive or self-learning
mechanisms, such as reinforcement learning and feedback-
aware prompt tuning, enables the system to evolve based on
operator feedback, changes in network behavior, and emerging
attack patterns. Another direction is the incorporation of cross-
device collaborative detection. While experiments focused on
individual edge nodes, federated and swarm-based reasoning
across multiple gateways can enhance situational awareness
and detection robustness in large-scale IoT deployments. Al-
though the evaluation utilized a representative attack dataset
and real-time benchmarks, deploying the system in live pro-
duction environments is essential to validate generalization,
operational reliability, and user interpretability under realistic
conditions. The proposed framework establishes a foundation
for an extensible and intelligent IDS paradigm, supporting
ongoing innovation in LLM integration, adaptive reasoning,
and collaborative edge security systems.

XIII. CONCLUSION
This research introduces a practical and resource-conscious
approach to IoT security by integrating ML-based IDS with
external LLMs for real-time IDS and reasoning at the net-
work edge. The proposed system achieves a delicate balance
between detection accuracy, semantic interpretability, and op-
erational efficiency, even under the constraints typical of edge
gateways. By employing structured LLM reasoning strategies,
zero-shot, few-shot, and CoT, the framework transforms raw
anomaly detections into actionable intelligence, enhancing
both automated and human-in-the-loop responses. Experimen-
the system operates within strict
tal results validate that
latency, bandwidth, and energy budgets while substantially
improving interpretability and decision-making precision. This
work not only advances the state of IoT security but also
lays a scalable foundation for future research in deploying
AI-augmented security solutions in edge gateways.

23

REFERENCES

[1] O. Vermesan and P. Friess, Digitising the industry Internet of Things
connecting the physical, digital and VirtualWorlds. CRC Press, 2022.
[2] G. Lampropoulos, K. Siakas, and T. Anastasiadis, “Internet of things
in the context of industry 4.0: An overview,” International Journal of
Entrepreneurial Knowledge, vol. 7, no. 1, 2019.

[3] Y. Perwej, K. Haq, F. Parwej, M. Mumdouh, and M. Hassan, “The
internet of things (iot) and its application domains,” International
Journal of Computer Applications, vol. 975, no. 8887, p. 182, 2019.
[4] J. Holler, V. Tsiatsis, C. Mulligan, S. Karnouskos, S. Avesand, and

D. Boyle, Internet of things. Academic Press, 2014.

[5] I. Butun, P. ¨Osterberg, and H. Song, “Security of the internet of things:
Vulnerabilities, attacks, and countermeasures,” IEEE Communications
Surveys & Tutorials, vol. 22, no. 1, pp. 616–644, 2019.

[6] M. Capra, R. Peloso, G. Masera, M. Ruo Roch, and M. Martina, “Edge
computing: A survey on the hardware requirements in the internet of
things world,” Future Internet, vol. 11, no. 4, p. 100, 2019.

[7] K. Yang, D. Blaauw, and D. Sylvester, “Hardware designs for security
in ultra-low-power iot systems: An overview and survey,” IEEE Micro,
vol. 37, no. 6, pp. 72–89, 2017.

[8] A. O. Akmandor, Y. Hongxu, and N. K. Jha, “Smart, secure, yet energy-
efficient, internet-of-things sensors,” IEEE Transactions on Multi-Scale
Computing Systems, vol. 4, no. 4, pp. 914–930, 2018.

[9] S. Jamshidi, K. W. Nafi, A. Nikanjam, and F. Khomh, “Evaluating
machine learning-driven intrusion detection systems in iot: Performance
and energy consumption,” Computers & Industrial Engineering, vol.
204, p. 111103, 2025.

[10] S. I. Nilima, M. K. Bhuyan, M. Kamruzzaman, J. Akter, R. Hasan,
and F. T. Johora, “Optimizing resource management for iot devices in
constrained environments,” Journal of Computer and Communications,
vol. 12, no. 8, pp. 81–98, 2024.

[11] M. Aziz Al Kabir, W. Elmedany, and M. S. Sharif, “Securing iot devices
against emerging security threats: Challenges and mitigation techniques,”
Journal of Cyber Security Technology, vol. 7, no. 4, pp. 199–223, 2023.
[12] K. A. Darabkh and M. Al-Akhras, “Evolutionary cost analysis and
computational intelligence for energy efficiency in internet of things-
enabled smart cities: Multi-sensor data fusion and resilience to link and
device failures,” Smart Cities, vol. 8, no. 2, p. 64, 2025.

[13] A. Heidari and M. A. Jabraeil Jamali, “Internet of things intrusion de-
tection systems: a comprehensive review and future directions,” Cluster
Computing, vol. 26, no. 6, pp. 3753–3780, 2023.

[14] E. Gyamfi and A. Jurcut, “Intrusion detection in internet of things
systems: a review on design approaches leveraging multi-access edge
computing, machine learning, and datasets,” Sensors, vol. 22, no. 10, p.
3744, 2022.

[15] D. Manivannan, “Recent endeavors in machine learning-powered intru-
sion detection systems for the internet of things,” Journal of Network
and Computer Applications, p. 103925, 2024.

[16] N. Sahani, R. Zhu, J.-H. Cho, and C.-C. Liu, “Machine learning-
based intrusion detection for smart grid computing: A survey,” ACM
Transactions on Cyber-Physical Systems, vol. 7, no. 2, pp. 1–31, 2023.
I. Amgbara, C. Akwiwu-Uzoma, and O. David, “Exploring
lightweight machine learning models for personal internet of things (iot)
device security,” 2024.

[17] S.

[18] R. Duraz, “Trustable machine learning for intrusion detection systems,”
Ph.D. dissertation, Ecole nationale sup´erieure Mines-T´el´ecom Atlan-
tique, 2024.

[19] H. Afifi, S. Pochaba, A. Boltres, D. Laniewski, J. Haberer, L. Paeleke,
R. Poorzare, D. Stolpmann, N. Wehner, A. Redder et al., “Machine
learning with computer networks: techniques, datasets, and models,”
IEEE access, vol. 12, pp. 54 673–54 720, 2024.

[20] A. S. Dina and D. Manivannan, “Intrusion detection based on machine
learning techniques in computer networks,” Internet of Things, vol. 16,
p. 100462, 2021.

[21] C. Luca, “Future trends in llm adaptability: The evolving impact of

zero-shot and few-shot learning.”

[22] A. Bhat, A human-centered approach to designing effective large lan-
guage model (llm) based tools for writing software tutorials. McGill
University (Canada), 2023.

[23] N. S. Agarwal and S. K. Sonbhadra, “A review on large language models

for visual analytics,” arXiv preprint arXiv:2503.15176, 2025.

[24] F. Sarhaddi, N. T. Nguyen, A. Zuniga, P. Hui, S. Tarkoma, H. Flores,
and P. Nurmi, “Llms and iot: A comprehensive survey on large language
models and the internet of things,” Authorea Preprints, 2025.

24

[46] D. Stiawan, M. Y. B. Idris, A. M. Bamhdi, R. Budiarto et al., “Cicids-
2017 dataset feature analysis with information gain for anomaly detec-
tion,” IEEE Access, vol. 8, pp. 132 911–132 921, 2020.

[47] E. Nwafor, U. Baskota, M. S. Parwez, J. Blackstone, and H. Olufowobi,
“Evaluating large language models for enhanced intrusion detection in
internet of things networks.”

[48] C. Wohlin, P. Runeson, M. H¨ost, M. C. Ohlsson, B. Regnell, and
A. Wessl´en, Experimentation in software engineering. Springer Science
& Business Media, 2012.

[25] B. Ingre, A. Yadav, and A. K. Soni, “Decision tree based intrusion
detection system for nsl-kdd dataset,” in International conference on
information and communication technology for intelligent systems.
Springer, 2017, pp. 207–218.

[26] Z. Azam, M. M. Islam, and M. N. Huda, “Comparative analysis of
intrusion detection systems and machine learning-based model analysis
through decision tree,” IEEE Access, vol. 11, pp. 80 348–80 391, 2023.
[27] M. A. Bouke, A. Abdullah, S. H. ALshatebi, and M. T. Abdullah, “E2ids:
an enhanced intelligent intrusion detection system based on decision tree
algorithm,” Journal of Applied Artificial Intelligence, vol. 3, no. 1, pp.
1–16, 2022.

[28] S. Wali, Y. A. Farrukh, and I. Khan, “Explainable ai and random forest
based reliable intrusion detection system,” Computers & Security, p.
104542, 2025.

[29] M. Choubisa, R. Doshi, N. Khatri, and K. K. Hiran, “A simple and
robust approach of random forest for intrusion detection system in
cyber security,” in 2022 International conference on IoT and blockchain
technology (ICIBT).

IEEE, 2022, pp. 1–5.
[30] T. Markovic, M. Leon, D. Buffoni, and S. Punnekkat, “Random forest
based on federated learning for intrusion detection,” in IFIP interna-
tional conference on artificial intelligence applications and Innovations.
Springer, 2022, pp. 132–144.

[31] M. Mohy-Eddine, A. Guezzaz, S. Benkirane, and M. Azrour, “An
efficient network intrusion detection model for iot security using k-
nn classifier and feature selection,” Multimedia Tools and Applications,
vol. 82, no. 15, pp. 23 615–23 633, 2023.

[32] E. Ozturk Kiyak, B. Ghasemkhani, and D. Birant, “High-level k-
nearest neighbors (hlknn): a supervised machine learning model for
classification analysis,” Electronics, vol. 12, no. 18, p. 3828, 2023.
[33] M. Mohy-eddine, A. Guezzaz, S. Benkirane, and M. Azrour, “An
intrusion detection model using election-based feature selection and k-
nn,” Microprocessors and Microsystems, p. 104966, 2023.

[34] F. Laghrissi, S. Douzi, K. Douzi, and B. Hssina, “Intrusion detection
systems using long short-term memory (lstm),” Journal of Big Data,
vol. 8, no. 1, p. 65, 2021.

[35] A. A. Awad, A. F. Ali, and T. Gaber, “An improved long short term
memory network for intrusion detection,” Plos one, vol. 18, no. 8, p.
e0284795, 2023.

[36] A. Halbouni, T. S. Gunawan, M. H. Habaebi, M. Halbouni, M. Kartiwi,
and R. Ahmad, “Cnn-lstm: hybrid deep neural network for network
intrusion detection system,” IEEE Access, vol. 10, pp. 99 837–99 849,
2022.

[37] L. Mohammadpour, T. C. Ling, C. S. Liew, and A. Aryanfar, “A survey
of cnn-based network intrusion detection,” Applied Sciences, vol. 12,
no. 16, p. 8162, 2022.

[38] I. A. Abdulmajeed and I. M. Husien, “Mlids22-ids design by applying
hybrid cnn-lstm model on mixed-datasets,” Informatica, vol. 46, no. 8,
2022.

[39] Y. Otoum, A. Asad, and A. Nayak, “Llm-based threat detec-
tion and prevention framework for iot ecosystems,” arXiv preprint
arXiv:2505.00240, 2025.

[40] A. Diaf, A. A. Korba, N. E. Karabadji, and Y. Ghamri-Doudane,
“Beyond detection: Leveraging large language models for cyber attack
prediction in iot networks,” in 2024 20th International Conference on
Distributed Computing in Smart Systems and the Internet of Things
(DCOSS-IoT).

IEEE, 2024, pp. 117–123.

[41] M. Zong, A. Hekmati, M. Guastalla, Y. Li, and B. Krishnamachari,
“Integrating large language models with internet of things: applications,”
Discover Internet of Things, vol. 5, no. 1, p. 2, 2025.

[42] M. A. Ferrag, M. Ndhlovu, N. Tihanyi, L. C. Cordeiro, M. Debbah,
T. Lestable, and N. S. Thandi, “Revolutionizing cyber threat detection
with large language models: A privacy-preserving bert-based lightweight
model for iot/iiot devices,” IEEe Access, vol. 12, pp. 23 733–23 750,
2024.

[43] F. Adjewa, M. Esseghir, and L. Merghem-Boulahia, “Efficient federated
intrusion detection in 5g ecosystem using optimized bert-based model,”
in 2024 20th International Conference on Wireless and Mobile Com-
puting, Networking and Communications (WiMob).
IEEE, 2024, pp.
62–67.

[44] S. Abdelhamid, I. Hegazy, M. Aref, and M. Roushdy, “Attention-driven
transfer learning model for improved iot intrusion detection,” Big Data
and Cognitive Computing, vol. 8, no. 9, p. 116, 2024.

[45] A. Aldaej, I. Ullah, T. A. Ahanger, and M. Atiquzzaman, “Ensemble
technique of intrusion detection for iot-edge platform,” Scientific Re-
ports, vol. 14, no. 1, p. 11703, 2024.

