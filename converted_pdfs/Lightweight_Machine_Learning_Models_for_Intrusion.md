# Lightweight Machine Learning Models for Intrusion

Lightweight Machine Learning Models for Intrusion
Detection

Mafalda Barbosa de Brito

Final Report of

2nd Cycle Integrated Project in Eletrical and Computer
Engineering

Supervisor(s): Prof. Paulo Rogerio Barreiros d’Almeida Pereira

Prof. Naercio David Pedro Magaia

January 2026

Contents

List of Tables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

List of Figures

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Glossary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1 Introduction

1.1 Problem Context and Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1.2 Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1.3 Report Outline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2 Background

2.1 Intrusion Detection Systems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.1.1 Detection Systems Methodologies . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.1.2 Detection Systems Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.1.3 Technologies and Classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.2 IDS for IoT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.2.1 Types of Intrusions in IoT Networks

. . . . . . . . . . . . . . . . . . . . . . . . . .

2.2.2 Evaluation Criteria for IoT IDS . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.3 Machine Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

iii

iv

v

2

3

3

4

5

5

5

6

6

8

8

9

9

2.3.1 Training and Inference in Machine Learning . . . . . . . . . . . . . . . . . . . . . . 11

2.4 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

2.4.1 Supervised Learning Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

2.5 Lightweight Deep Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

2.5.1 Deep Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

2.5.2 Lightweight DL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

2.5.3 TinyML . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

3 State of Art

19

3.1 ML Methods Applied to IDS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

3.2 Deep Learning Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

3.2.1 Convolutional Neural Networks in IDS . . . . . . . . . . . . . . . . . . . . . . . . . 20

3.2.2 Recurrent Neural Networks in IDS . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

3.2.3 Hybrid Deep Learning Models in IDS . . . . . . . . . . . . . . . . . . . . . . . . . . 20

ii

3.2.4 Feature Selection for High-Dimensional IoT Data . . . . . . . . . . . . . . . . . . . 21

4 Frameworks for TinyML

23

4.1 Lite Runtime (LiteRT)

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

4.2 ExecuTorch (PyTorch)

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

4.3 Embedded Learning Library (ELL)

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

4.4 Framework Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

5 Methodology

25

5.1 Dataset, Feature Selection Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

5.2 Proposed Model

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

5.3 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

5.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

5.5 Timeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

5.6 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

6 Preliminary Experiment

27

6.1 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

6.2 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

6.2.1 Model Quantization Efficacy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

6.2.2 Latency Trade-offs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

6.2.3 Resource Efficiency on Edge Hardware . . . . . . . . . . . . . . . . . . . . . . . . 28

6.2.4 Memory Considerations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

6.2.5 Practical Implications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

Bibliography

29

iii

List of Tables

2.1 Evaluation Criteria for Generic and IoT IDS [19] . . . . . . . . . . . . . . . . . . . . . . . . 10

6.1 Comparison of model performance and resource utilization on Google Colab (Keras),

Raspberry Pi 4 (LiteRT), and Raspberry Pi 4 (Keras). Raw metrics are from 1000 infer-

ence runs on the full MNIST test set. CPU and RAM added load metrics are modal values

from samples taken every 0.01 seconds. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

iv

List of Figures

2.1 Biological neuron and computational perceptron [37]. . . . . . . . . . . . . . . . . . . . . . 14

2.2 Artificial deep neural network with a feedforward neural network with eight input variables

(x1, . . .

,x8), four output variables (y1, y2, y3, y4), and two hidden layers with three

neurons each [38].

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

5.1 Gantt chart of the work plan highlighting the crucial tasks (bars) and their durations . . . . 26

v

Glossary

ACK

AD

AI

AIDS

ANN

Acknowledgment

Anomaly-based Detection

Artificial Intelligence

Anomaly-based intrusion detection system

Artificial Neural Network

BiLSTM

Bidirectional Long Short-Term Memory

BPTT

CFS

CIA

CNN

Backpropagation Through Time

Correlation-based Feature Selection

Confidentiality, Integrity, Availability (security triad)

Convolutional Neural Network

CNN-BiLSTM Convolutional Neural Network-Bidirectional LSTM (hybrid model)

CPU

DL

DL-IID

DNN

Central Processing Unit

Deep Learning

Deep Learning-based Intelligent Intrusion Detection

Deep Neural Network

DNN-BiLSTM Deep Neural Network-Bidirectional LSTM (hybrid model)

DoS

DS

DSRC

FN

FP

GA

HIDS

HTTP

IDES

IDS

IETF

IoT

IoV

IPCA

Denial-of-Service

Database Server

Dedicated Short-Range Communications

False Negative

False Positive

Genetic Algorithm

Host-based IDS

HyperText Transfer Protocol

Intrusion Detection Expert System

Intrusion Detection System

Internet Engineering Task Force

Internet of Things

Internet of Vehicles

Incremental Principal Component Analysis

vi

IPS

kNN

L1

L2

LIME

LiteRT

LSTM

MIDS

ML

MLPs

MN

MS

NBA

NIDES

NIDS

OBU

Intrusion Prevention System

K-Nearest Neighbors

LASSO

Ridge regression

Local Interpretable Model-agnostic Explanations

Lite Runtime

Long Short-Term Memory

Mixed IDS

Machine Learning

Multi-Layer Perceptrons

Managed Network

Management Server

Network Behavior Analysis

Next-generation Intrusion Detection Expert System

Network-based IDS

On-Board Unit

OCSVM

One-Class SVMs

PCA

PTQ

QAT

Principal Component Analysis

Post-Training Quantization

Quantization-Aware Training

RAL-MIFS

Redundancy-Adjusted Logistic Mutual Information Feature Selection

RAM

RF

RL

RNN

SD

SIDS

SIEM

SN

SPA

SQL

SVM

SYN

TCP

TIM

Random Access Memory

Radio Frequency

Reinforcement Learning

Recurrent Neural Networks

Signature-based Detection

Signature-based intrusion detection system

Security information and event management

Standard Network

Stateful Protocol Analysis

Structured Query Language

Support Vector Machine

Synchronize

Transmission Control Protocol

Time-based Inductive Machine

TinyML

Tiny Machine Learning

TinyMLaaS

TinyML as-a-Service

TinyOL

UDP

Tiny Machine Learning with Online-Learning

User Datagram Protocol

vii

V2V

VANET

WIDS

Vehicle-to-Vehicle

Vehicular Ad Hoc Network

Wireless-based IDS

1

Chapter 1

Introduction

The Internet of Things (IoT) represents a vast network of interconnected devices that exchange

data to enable automation, efficiency, and intelligence across multiple domains, including smart homes,

industrial systems, and autonomous vehicles. While IoT has brought remarkable technological advance-

ments, its growing interconnectivity has also expanded the attack surface, exposing systems to numer-

ous cybersecurity vulnerabilities. To address these challenges, Intrusion Detection System (IDS) has

emerged as a fundamental component of IoT security frameworks, continuously monitoring network

traffic, detecting anomalies, and identifying potential intrusions in real time.

Within the broader IoT landscape, Vehicle-to-Vehicle (V2V) communication has emerged as a criti-

cal research domain for enhancing traffic efficiency, road safety, and the advancement of autonomous

driving technologies [1]. V2V systems form a core component of the evolving Internet of Vehicles (IoV)

ecosystem, enabling vehicles to exchange real-time information about their position, speed, and intent.

However, the complexity of vehicular applications necessitates sophisticated network and security pro-

tocols to maintain reliability and trustworthiness, particularly against targeted cyberattacks that exploit

communication vulnerabilities.

Traditional IDS and Deep Learning architectures, while powerful, are computationally expensive and

unsuitable for deployment on resource-constrained vehicular devices. These environments demand so-

lutions that balance detection accuracy with computational efficiency and real-time responsiveness. The

Lightweight Deep Learning paradigm [2] addresses this challenge by enabling the design of compact,

optimized neural networks capable of operating under tight hardware constraints. Building upon this

foundation, Tiny Machine Learning (TinyML) [3] enables efficient on-device inference while maintaining

high detection performance. In the vehicular context, deploying lightweight IDS models directly on On-

Board Unit (OBU) reduces latency, enhances privacy, and minimizes dependency on cloud resources,

crucial requirements for real-time detection of fast-evolving cyberattacks.

2

1.1 Problem Context and Motivation

Ensuring information availability is essential for the reliable operation of any network [1].

In the

context of vehicular networks, disruptions in availability can severely impact emerging vehicular services

and applications.

In V2V communication, timely and reliable data exchange enables vehicles to make critical, rapid

safety decisions, such as avoidance of collisions or route optimization. Flooding attacks, however,

overwhelm the network with a massive number of fake or redundant information, leading to congestion,

communication delays, or complete service disruption. Such failures can degrade the perfor-

mance of essential vehicular applications and, in extreme cases, endanger human lives.

That is why this study will employ datasets from the realistic simulations of UDP and SYN flood-

ing attacks (described in Section 2.2.1) developed by Sousa et al.

[1] to train the ML model. Unlike

typical simulated datasets, these were obtained using three MK5 On-Board Units (OBUs) operating

with the IEEE 802.11p standard protocol, the core communication protocol of Dedicated Short-Range

Communications (DSRC) systems. Consequently, the data more accurately reflects realistic vehicular

communication conditions.

The datasets encompass multiple experimental scenarios and diverse traffic patterns, including

both normal and malicious packets, to enhance heterogeneity and realism. Feature extraction using

Tranalyzer [4] and Wireshark [5] produced 89 detailed features, including time-based metrics essential

for effective attack detection. Baseline machine learning results indicate a realistic level of task difficulty

(F1-scores between 0.771 and 0.829). Since the F1-score reflects model performance, especially on

imbalanced data where one class is more frequent than others, these results confirm the dataset’s

complexity and its suitability for evaluating robust IDS models.

These devices, like other IoT elements, operate in resource-constrained environments, possess-

ing limited processing power, memory capacity, and energy availability. Deploying traditional, com-

plex IDS models in these settings is often inadequate. The existing baseline results used a Decision

Tree algorithm. The future work explicitly calls for building novel ML models for intrusion detection using

the collected real V2V data and comparing their performance with state-of-the-art solutions. Since the

environment where this solution would be deployed is resource-limited, these new models must adhere

to the principles of Lightweight Deep Learning.

1.2 Objectives

The main objective of this study is to design, implement, and evaluate a lightweight Deep Learn-

ing (DL) model for intrusion detection against V2V flooding attacks (SYN flood and UDP flood),

utilizing the realistic datasets generated from MK5 OBU devices, with a focus on optimizing the model

for deployment on resource-constrained hardware.

3

1.3 Report Outline

The remainder of this thesis is organized as follows:

Chapter 2 provides theoretical foundations for intrusion detection systems and lightweight machine

learning, covering IDS methodologies and architectures, IoT/V2V security threats, ML paradigms, and

Deep Learning fundamentals (CNN, RNN, LSTM). It concludes with Lightweight DL techniques (pruning,

quantization, knowledge distillation) and TinyML.

Chapter 3 surveys ML-based intrusion detection approaches, reviewing traditional methods (decision

trees, random forests, SVMs), deep learning architectures, and feature selection for high-dimensional

data.

Chapter 4 evaluates TinyML frameworks (LiteRT, ExecuTorch, ELL) across optimization support,

hardware compatibility, conversion workflows, and performance to guide framework selection for resource-

constrained deployment.

Chapter 5 describes dataset characteristics, feature selection strategies, the proposed model archi-

tecture, quantization approaches, and evaluation metrics.

Chapter 6 validates quantized DL deployment on edge devices using MNIST, demonstrating model

size reduction and latency improvement with LiteRT on Raspberry Pi 4.

4

Chapter 2

Background

2.1 Intrusion Detection Systems

Intrusion can be understood as any form of unauthorized activity that disrupts, damages, or ex-

ploits an information system. Such activities are considered intrusions because they endanger one

or more of the system’s core security principles: confidentiality, integrity, and availability.

Intrusion detection systems (IDSs) are the ‘burglar alarms’ (or rather ‘intrusion alarms’) of the com-

puter security field [6]. They can be implemented as software or hardware solutions that automate the

intrusion detection process in a computer system or network. The goal of an IDS is to identify different

kinds of malicious network traffic and computer usage, which cannot be identified by a traditional firewall

[7]. Traditional firewalls function as a barrier that blocks unauthorized access, while an IDS monitors

the network to detect and alert on suspicious or malicious activity. In other words, a firewall acts like a

locked door, and an IDS works like a security camera.

2.1.1 Detection Systems Methodologies

Intrusion detection methodologies are generally divided into three main categories: Signature-based

intrusion detection system (SIDS), Anomaly-based intrusion detection system (AIDS), and Stateful Pro-

tocol Analysis (SPA).

Stateful Protocol Analysis monitors the state of network protocols - meaning it can track how a

protocol operates over time (e.g., matching a request with its reply). SPA relies on vendor-developed

profiles based on official protocol standards (like those from the Internet Engineering Task Force (IETF)).

Because of this, SPA is also called Specification-based Detection.

A SIDS relies on pattern-matching techniques to identify known attacks by comparing activity

against a database of intrusion signatures. These systems achieve high accuracy for previously iden-

tified threats but struggle with zero-day attacks (exploiting previously unknown vulnerabilities) and

polymorphic attacks (malware that continually changes its signature) since no matching signature ex-

ists until added [7]. Traditional SIDS tools (e.g., Snort, NetSTAT) analyze individual network packets,

but modern threats often span multiple packets, requiring more advanced methods. Due to the rise of

5

sophisticated and zero-day attacks [8], SIDS are becoming less effective, and alternative approaches,

such as anomaly-based IDS, are being explored.

An AIDS detects attacks by modeling normal system behavior and flagging significant deviations

as anomalies. Unlike SIDS, AIDS can identify zero-day attacks and internal malicious activities be-

cause they do not rely on a signature database. They use statistical, knowledge-based, or machine

learning methods, typically involving a training phase to learn normal behavior and a testing phase

to evaluate detection of unseen intrusions. While AIDS provides stronger protection against novel

threats, its main drawback is a high false positive rate, since unusual but legitimate behaviors may be

misclassified as attacks.

Most IDSs use multiple methodologies to maximize detection coverage. Signature-based and

anomaly-based detection are complementary approaches: signature-based systems excel at identifying

known threats using predefined attack patterns, while anomaly-based systems detect novel attacks

by identifying deviations from normal behavior [9].

2.1.2 Detection Systems Approaches

Although detailed analyses of detection approaches are generally scarce, a few articles [7, 9] have

proposed a classification into five subclasses - Statistics-based, Pattern-based, Rule-based, State-

based, and Heuristic-based - offering a comprehensive perspective on their distinctive characteristics.

Statistics-based approaches analyze network traffic using statistical algorithms, thresholds, mean,

standard deviation, and probabilities to detect anomalies. They are simple but may require substantial

statistical knowledge and can be less accurate in real-time scenarios.

Pattern-based detection identifies known attacks through string matching, character patterns, or

forms in the data, often using hash functions for identification. It is easy to implement and focuses on

known threats.

Rule-based techniques employ If–Then or If–Then–Else rules or attack signatures to model and de-

tect intrusions. While they offer high detection rates and low false positives, they can be computationally

expensive and require extensive rule sets.

State-based methods examine streams of events and exploit finite state machines derived from

network behavior to detect attacks. They are probabilistic, self-trained, and achieve low false-positive

rates.

Finally, Heuristic-based approaches draw on artificial intelligence or biological inspiration to identify

abnormal activity. These methods rely on experience, experimental learning, and evolutionary strate-

gies.

2.1.3 Technologies and Classification

There are many types of IDSs technologies that may be systematically classified, according to where

they are deployed to inspect suspicious activities, and what event types they can recognize [10–13], as

6

follows: Host-based IDS (HIDS), Network-based IDS (NIDS), Wireless-based IDS (WIDS), Network

Behavior Analysis (NBA), and Mixed IDS (MIDS).

Host-based IDSs monitor and analyze activity on individual hosts, including sensitive systems and

servers, by inspecting logs from the operating system, applications, firewalls, and databases. They are

particularly effective at detecting insider attacks that do not generate network traffic [14].

Network-based IDSs monitor network traffic using sensors and data sources such as packet cap-

tures and NetFlow - a network protocol developed by Cisco that collects and records information about

IP traffic flowing through a network - analyzing application and protocol activity to detect suspicious

incidents. NIDS can oversee multiple hosts and identify external threats early, but may struggle with

high-bandwidth networks [15]. When deployed alongside HIDS and firewalls, NIDS contributes to a

multi-layered defense against both external and insider attacks.

Wireless-based IDS functions similarly to NIDS but monitors wireless traffic, including ad hoc, sen-

sor, and mesh networks. Network Behavior Analysis detects attacks by analyzing unusual traffic flows.

And Mixed IDS or Hybrid IDS combines multiple technologies for more comprehensive detection.

Components and Architecture

IDS systems use sensors (for NIDS, WIDS, NBA) or agents (software) (for HIDS) to collect data.

Both the sensor and agent can deliver data to a Management Server (MS) for analysis and a Database

Server (DS) for storage.

Networks may be deployed as a Managed Network (MN), an isolated network deployed for security

software management to conceal the IDS information from intruders, or a Standard Network (SN), a

public network without protection, secured by virtual isolation [9].

Core Capabilities

Most intrusion detection systems commonly integrate four essential security mechanisms: informa-

tion gathering, logging, anomaly detection, and intrusion prevention [9]. Information gathering collects

information on hosts/networks from observed activities. Logging, where the related logging data for

detected events can be used to validate the alerts and investigate incidents. Detection methodologies

that in most IDSs usually need sophisticated tuning to receive a higher accuracy. And some intrusion

detection systems include prevention capabilities and are called Intrusion Prevention Systems (IPS).

These systems possess all IDS functionalities while also being able to actively block threats in real-time

[11].

Accuracy Challenges

A notable limitation of IDS technologies is that they cannot achieve perfect detection. The accuracy

of an IDS is typically assessed using false positives (FP) - when benign activity is mistakenly classified

as malicious - and false negatives (FN) - when malicious activity goes undetected.

7

Because the consequences of missing an attack are often more severe than dealing with extra alerts,

security administrators usually prioritize reducing FNs, even if it results in more FPs. This means the

IDS is tuned to be more sensitive, catching more potential threats at the cost of generating additional

false alarms. Ho et al.

(2012) [16] analyzed FP and FN cases from real-world traffic and reported

three key findings: (1) most false cases are FNs, as many application behaviors and data formats

are custom-defined rather than adhering to RFC standards; (2) the majority of FP alerts arise from

management policies rather than actual security issues; and (3) older, well-known types of cyber-

attacks, such as buffer overflows, Structured Query Language (SQL) server exploits, and the Worm

Slammer attack, tend to be missed more often by IDSs, resulting in a high rate of FNs.

2.2 IDS for IoT

Unlike traditional enterprise IDS, IoT environments demand lightweight detection methods operat-

ing under strict computational, memory, and energy constraints. This section examines IoT-specific

attack vectors, IDS design requirements, and evaluation criteria appropriate for resource-constrained

deployments.

2.2.1 Types of Intrusions in IoT Networks

Intrusions in IoT environments can target the physical infrastructure, communication protocols,

software components, or cryptographic mechanisms. Following the taxonomy of Thakkar and Lo-

hiya [17], IoT intrusions are categorized into four primary types:

1. Physical Intrusions - Attacks directed at IoT hardware or physical devices that disrupt normal

operations, damage components, or alter stored data.

2. Network Intrusions - Exploitation of weaknesses in data routing processes, enabling attackers

to intercept, drop, or reroute network packets, potentially compromising multiple nodes within the

network.

3. Software Intrusions - Malicious programs such as viruses or worms that exploit vulnerabilities in

system hardware or software, leading to data theft, corruption, or deletion.

4. Encryption Intrusions - Attacks that undermine the confidentiality of encrypted communications

by observing and decoding side-channel information transmitted through the network.

These intrusion categories collectively threaten the confidentiality, integrity, and availability (CIA

triad) of IoT systems.

Flooding Attacks

Flooding Attacks represent a class of Denial-of-Service (DoS) attacks that exhaust device resources

by overwhelming targets with malicious traffic. Common variants include:

8

• SYN Flooding: Attackers send numerous Transmission Control Protocol (TCP) Synchronize (SYN)

packets (TCP connection initiation flag), leaving the server with unfinished half-open connections

that exhaust connection tables and Central Processing Unit (CPU) resources [18].

• UDP Flooding: Attackers transmit large volumes of User Datagram Protocol (UDP) packets to

consume bandwidth and computational resources, particularly effective against IoT devices with

limited processing capability.

• ACK Flooding: Invalid Acknowledgment (ACK) packets (TCP flag indicating acknowledgment of

received packets) are sent to overwhelm the target, forcing it to expend resources on processing

malformed acknowledgments [18].

• HTTP Flooding: Application-layer attacks (targeting where users interact directly with web ser-

vices), sending legitimate-appearing HyperText Transfer Protocol (HTTP) requests to exhaust web

service resources.

These flooding attacks are particularly dangerous in resource-constrained IoT environments, where even

moderate attack volumes can deplete battery reserves, exhaust memory, or saturate communication

channels.

2.2.2 Evaluation Criteria for IoT IDS

Table 2.1 compares evaluation criteria for generic and IoT-specific IDS, highlighting the distinct prior-

ities and constraints of each deployment model.

2.3 Machine Learning

Machine learning is fundamentally about enabling systems to learn and improve from experience

without being explicitly programmed. More formally, Machine Learning (ML) is a branch of Artificial

Intelligence (AI) focused on making AI systems automatically learn from data rather than relying on

explicit programming rules [20][21].

The IDS methodologies and technologies discussed above provide the framework for intrusion de-

tection. However, their effectiveness depends heavily on the underlying algorithms that power them.

While signature-based and specification-based approaches have proven effective for known threats, the

emergence of zero-day attacks and the resource constraints of IoT environments necessitate more in-

telligent, adaptive detection methods. Machine learning has emerged as a transformative approach

to address these limitations, enabling IDS to learn attack patterns from data rather than relying solely on

handcrafted rules or signatures.

Before exploring deep learning architectures and their lightweight variants, it is essential to under-

stand the foundational machine learning algorithms that form the basis of modern intrusion detection

systems. Traditional ML methods include both supervised algorithms (Decision Trees, Random Forest,

9

Table 2.1: Evaluation Criteria for Generic and IoT IDS [19]

Evaluation Factor

Generic IDS

IoT IDS

Detection Accuracy

Measures the system’s ability to cor-
rectly identify intrusions and minimize
false positives/negatives.

Equally important, but detection meth-
ods must be lightweight to avoid over-
loading limited IoT resources.

False Positive Rate

Low false positives reduce unnecessary
alerts and prevent security team over-
load.

False positives can disrupt normal IoT
operations and waste energy.

False Negative Rate

Avoiding missed attacks is critical
maintain security.

to

Real-time
mance

Perfor-

Quick response is desirable, but slight
delays may be acceptable in non-critical
systems.

Scalability

Must handle large enterprise networks
using powerful infrastructure.

in IoT, as missed detec-
More critical
tions can interrupt real-time opera-
tions (e.g., healthcare or industrial sys-
tems).

IoT environ-
Extremely important;
ments often require immediate threat
detection (e.g., autonomous vehicles,
industrial control).

Must efficiently manage thousands of
resource-limited IoT devices in dis-
tributed setups.

Computational Over-
head

Can be high, especially for AI-based
IDS, as enterprise devices have ample
computing power.

Must remain low due to the limited CPU,
memory, and energy of IoT devices.

Network Overhead

Moderate overhead is acceptable for
monitoring and logging.

Must be minimized, as IoT networks
often have low bandwidth.

Adaptability to New
Attacks

Requires regular updates to detect
evolving threats.

Energy Efficiency

Not a major concern in traditional IDS.

Needs lightweight, adaptive models
capable of
learning new threats with
minimal retraining. Frequent updates
may be impractical.

IoT devices often
Highly important;
run on batteries and cannot support
energy-intensive monitoring.

Privacy & Data Sen-
sitivity

Operates within secure enterprise in-
frastructure, so privacy is less of a con-
cern.

in healthcare, smart homes,
Critical
and industrial IoT, where sensitive data
must be protected.

Deployment Model

Typically centralized with a dedicated
security team.

Robustness Against
Adversarial Attacks

Must handle sophisticated attacks like
polymorphic malware.

Often decentralized, using edge or fog
computing to bring detection closer to
devices.

More vulnerable to attacks like ad-
versarial ML, sensor spoofing, and
firmware exploits.

Integration with Se-
curity Frameworks

Works with firewalls, Security informa-
tion and event management
(SIEM)
systems, and endpoint security tools.

Requires lightweight integration due to
resource constraints;
traditional fire-
walls may be unavailable.

10

Support Vector Machine (SVM), Na¨ıve Bayes, K-Nearest Neighbors (kNN), Logistic Regression) and

unsupervised methods (K-Means Clustering, Semi-Supervised Learning).

2.3.1 Training and Inference in Machine Learning

All machine learning models follow two fundamental phases: training and inference. Training is the

process by which a model learns patterns from data by adjusting its parameters to optimize an objective

function.

In supervised learning, this involves minimizing prediction errors on labeled data; in unsu-

pervised learning, it involves optimizing objectives like clustering quality or reconstruction accuracy; in

reinforcement learning, it involves maximizing cumulative rewards. The goal is to capture underlying

patterns or structure in the data so the model can generalize to unseen examples. Training typically in-

volves multiple passes (epochs) through the dataset, with parameter updates based on the optimization

objective.

To monitor how well a model generalizes, practitioners use a separate validation dataset and track

performance metrics across epochs in learning curves. These curves reveal two common problems:

underfitting, where the model is too simple to capture patterns and performs poorly on both training and

validation data, and overfitting, where the model memorizes training-specific patterns (including noise)

and performs well on training but poorly on validation data. By observing these patterns, practitioners

can adjust hyperparameters, modify architecture, or apply regularization techniques to improve

generalization.

Inference (or testing) occurs after training is complete. During inference, the model uses its learned

parameters to make predictions, generate outputs, or assign data to learned structures (such as clus-

ters). Unlike training, inference does not involve parameter updates. This phase is typically faster and

represents what is deployed in real-world applications like IDS for detecting network intrusions.

2.4 Evaluation Metrics

Model evaluation is the process of assessing how well a machine learning model performs on unseen

data using different metrics and techniques. It ensures that the model not only memorizes training data

but also generalizes to new situations. Common evaluation metrics for classification tasks include:

• Accuracy: The ratio of correctly predicted instances to the total instances.

• Precision: The ratio of true positive predictions to the total predicted positives. It indicates how

many of the predicted positive instances are actually positive, reflecting the model’s ability to avoid

false positives.

• Recall (Sensitivity): The ratio of true positive predictions to the total actual positives. It measures

the model’s ability to identify all relevant instances, reflecting its capacity to avoid false negatives.

• F1-Score: The harmonic mean of precision and recall. It provides a balanced measure that con-

siders both false positives and false negatives, especially useful in imbalanced datasets.

11

• Confusion Matrix: A table that summarizes the performance of a classification model by display-

ing true positives, true negatives, false positives, and false negatives. It helps visualize the types

of errors made by the model.

2.4.1 Supervised Learning Methods

Supervised learning methods rely on labeled datasets where each example consists of input vari-

ables called features and corresponding output variables called labels. Features are the measurable

characteristics or attributes extracted from raw data that the model uses to make predictions, while la-

bels are the target outputs that the model aims to predict [22]. This dependence on labeled data is the

key distinction between supervised learning and unsupervised methods, which operate on unlabeled

data without predefined outcomes.

Decision Trees are popular in IDS due to their simplicity, interpretability, and efficiency in clas-

sifying network traffic. They are non-parametric supervised learning algorithms [23] that recursively

partition data, with nodes representing features, branches representing decision rules, and leaves repre-

senting class labels. They handle both categorical (e.g., protocol type, connection state) and numerical

data (e.g., packet size, duration) to distinguish normal from malicious activity [19].

Random Forests improve performance by combining multiple decision trees to produce a single

result, enhancing accuracy and controlling overfitting [19, 24]. They handle large, high-dimensional

datasets and class imbalance effectively, reducing false positives [19, 25]. Feature selection techniques

further improve efficiency, though Random Forests can be time-consuming and resource-intensive com-

pared to a single tree.

Support Vector Machines (SVM) are supervised machine learning algorithms that classify data by

finding an optimal hyperplane that maximizes the distance between classes in an N-dimensional space

[26]. They excel in binary classification and high-dimensional datasets. One-Class SVMs (OCSVM)

are useful when only normal data is available, making them suitable for scenarios where labeled attack

data is scarce [27].

Na¨ıve Bayes is a probabilistic supervised classifier based on Bayes’ theorem. It efficiently classifies

network traffic using labeled datasets and can be combined with feature selection to improve detection

performance [28, 29]. Its simplicity and effectiveness in processing labeled data make it suitable for

real-time network security applications [30].

K-Nearest Neighbors (kNN) is a non-parametric, supervised learning classifier that uses proximity

to make classifications. It works by finding the k nearest labeled training examples to a new data point

and assigning the class label based on the majority vote among neighbors [31]. kNN adapts well to

dynamic IDS environments and different types of network attacks.

Logistic Regression is a classification algorithm that predicts the probability that a given input be-

longs to a particular class [32]. It models this probability by applying a logistic function to a linear combi-

nation of input features. Logistic regression is employed to analyze network traffic features and predict

the likelihood of an intrusion [19], with the ability to identify various types of network attacks effec-

12

tively [33] and, combined with a multinominal regression model, can enhance detection performance

and reduce misclassification [34].

2.5 Lightweight Deep Learning

Within the broader ML landscape, Deep Learning (DL) represents a specialized subset that uses

multi-layered neural networks to automatically extract hierarchical representations from raw data.

2.5.1 Deep Learning

The primary goal of Deep Learning is to automatically learn useful representations of data to solve

complex tasks such as classification, prediction, or generation. Unlike traditional machine learning meth-

ods, which rely heavily on handcrafted features - individual measurable properties or characteristics of

the data used by the model to make predictions - deep learning architectures are capable of extracting

and refining hierarchical features directly from raw data, such as pixels, text, or audio signals.

Formally, Deep Learning can be narrowly defined as the optimization of Artificial Neural Network

(ANN) with many layers, enabling the model to capture increasingly abstract representations.

In a

broader sense, Deep Learning encompasses all methods, architectures, and applications that involve

multi-layered ANN representations. [35]

An Artificial Neural Network is a structure containing simple elements that are interconnected in

many ways with hierarchical organization. It tries to interact with objects in the real world in the same

way as the biological nervous system [36].

It is composed of layers of interconnected units called

perceptrons (also called artificial neurons or nodes), which transform input data into output predictions

through a series of weighted connections and nonlinear activation functions.

Just as a biological neuron receives input signals from other neurons through its dendrites, a per-

ceptron receives data from preceding units through its input nodes. Each connection between an input

node and the perceptron is associated with a weight, which represents the relative importance of that

input. In biological systems, the dendrites transmit signals to the nucleus, where they are processed to

generate an output. Analogously, the perceptron’s processing unit performs a weighted computation

on its inputs and produces an output value.

In the brain, this output is carried away by the axon;

in artificial neural networks, the perceptron’s output is propagated forward as input to subsequent

perceptrons (Fig. 2.1).

In ANNs, the connections between neurons are represented mathematically using vectors and ma-

trices. A vector can be used to describe the set of inputs received by a perceptron, while a matrix

organizes the weights that connect one layer of perceptrons to the next. For example, if a layer with m

neurons is connected to a layer with n neurons, the weights of these connections can be stored in an m

× n matrix. During the forward pass, the input vector is multiplied by the weight matrix, and a bias vector

is added before applying the activation function. This matrix representation is fundamental because it al-

lows neural networks to efficiently perform large-scale computations using well-established linear

13

Figure 2.1: Biological neuron and computational perceptron [37].

algebra operations.

During training, the neural network learns from a dataset by adjusting its parameters (weights and

biases) to minimize a loss function, which measures the difference between the predicted outputs and

the true labels. This process involves forward propagation, where inputs pass through the network

to generate predictions, and backpropagation, where gradients of the loss function are computed and

used to update the weights via an optimization algorithm. The goal of training is to capture the under-

lying patterns in the data so that the network can generalize to unseen examples.

Once the network is trained, it enters the inference phase, also called testing or prediction. During

inference, the model uses the learned weights to process new input data and generate outputs. Unlike

training, inference does not involve weight updates; it only performs forward propagation to produce

predictions. This phase is typically faster and is what is deployed in real-world applications.

Layers

A unilayer ANN like that in Figure 2.1 has a low processing capacity by itself, and its level of applica-

bility is low; its true power lies in the interconnection of many ANNs, as happens in the human brain. This

has motivated different researchers to propose various topologies (architectures) to connect neurons to

each other in the context of an ANN.

From Figure 2.2 we can see that a multi-layer Artificial Neural Network is a directed graph whose

nodes correspond to perceptrons and whose edges correspond to links between them. The model

given is organized as several interconnected layers: the input layer - the set of neurons that directly

receives the information coming from the external sources of the network - hidden layers - sets of

internal neurons of the network that do not have direct contact with the outside - and output layer - set

of neurons that transfers the information that the network has processed to the outside [38].

An ANN with multiple hidden layers is called a Deep Neural Network (DNN). The adjective “deep”

14

Figure 2.2: Artificial deep neural network with a feedforward neural network with eight input variables
(x1, . . . ,x8), four output variables (y1, y2, y3, y4), and two hidden layers with three neurons each [38].

applies not to the acquired knowledge, but to the way in which the knowledge is acquired [39], since

it stands for the idea of successive layers of representations. The “depth” of the model refers to the

number of layers that contribute to the model. [38]

Convolutional Neural Network (CNN) is a specialized form of DNN for analyzing input data that

contains some form of spatial structure [40]. They introduce the concept of filters (also called kernels). A

filter is a small matrix of trainable weights that is applied across the input data to detect local patterns

such as edges, textures, or more complex features. Just as the learning algorithm in fully connected

networks adjusts synaptic weights, in CNN it adjusts the values of these filters during training. The effect

is that the network automatically learns which features are most relevant for the task, allowing it to

capture spatial hierarchies of patterns with far fewer parameters than a fully connected design.

Recurrent Neural Networks (RNN) is a deep neural network trained on sequential or time series

data to make sequential predictions or conclusions based on sequential inputs [41]. They maintain

a hidden state that retains information from previous time steps, enabling the network to process se-

quences effectively [19]. RNNs are trained using forward propagation and backpropagation through

time (BPTT), which calculates errors across all time steps to adjust model parameters. Unlike traditional

backpropagation in feedforward networks, BPTT accounts for shared parameters across the sequence,

enabling the model to learn from temporal relationships within the data [41].

15

2.5.2 Lightweight DL

Lightweight deep learning indicates the procedures of compressing DNN models into more compact

ones, which are suitable to be executed on edge devices due to their limited resources and computa-

tional capabilities while maintaining comparable performance to the original. Currently, the approaches

of model compression include but are not limited to network pruning, quantization, knowledge distil-

lation, neural architecture search, low-rank factorization, and huffman coding.

Pruning is a common compression technique that aims to eliminate redundant parameters in DNNs.

It can be used for minimizing the cost of computation by pruning parameters or filters from the con-

volutional layers. When it came to pruning filters, it could decrease the number of operations in the

convolutional stage and thus improve the inference time [42]. Meanwhile, since the model parameters

or filters could be pruned according to their importance [43, 44], it could thus strike a balance of ex-

ecution speed and performance. Among the different types of pruning, the two main ones are weight

pruning and filter pruning.

Weight pruning consists of setting certain weights to zero. Among all the recent research works,

the most popular approach is by means of sparsity - the property of a model where most elements are

zero (or near zero). Sparsity could be carried out by LASSO (L1) regularization, a way of “punishing”

the model by setting the unnecessary weights to zero if they do not meet a predefined threshold. While

weight pruning is powerful, it faces unstructured or structured problems. Since unstructured sparsity

requires additional overhead to record indices, this makes it less efficient on hardware. On the other

hand, structured sparsity limits the sparsity patterns in the weights so that they can be described in low-

overhead representations, such as strides or blocks. Although this reduces index storage overhead and

simplifies processing, structured sparsity might result in a worse model because it limits the freedom of

weight updates during the sparsification process [2].

Filter pruning is a technique that attempts to determine the importance of each filter and eliminate

unnecessary ones. The most prevalent criterion is adopting the L1-norm or Ridge regression (L2)-norm

based on the influence on the error function to verify unimportant filters. It can be done by ranking feature

maps [45], or by search algorithms, such as PruningNet [46], which uses meta-learning to automatically

determine which filters to prune.

Quantization consists of representing weights, gradients, and activations - outputs from each layer

- using fewer bits. This method is used to speed up both training and inference time for the model.

However, it is challenging to maintain high accuracy while reducing bit precision. Recent research aims

to solve this problem by reducing quantization error or mimicking the results of networks with full

precision [2].

Knowledge Distillation trains a lightweight (student) model from the original (teacher) model. And

because of that, knowledge distillation is not a compression technique but rather a methodology used to

train a lightweight model.

16

One of the techniques used for knowledge distillation is the transfer of “soft logits” - a smoothed

probability distribution that carries richer information - generated by the teacher model to guide the

optimization of the student model. Another technique, aimed at reducing the difference between the

student model and the teacher model, uses an assistant teacher of intermediate size to divide knowl-

edge distillation into different phases (intermediate layer knowledge). We can also categorize knowledge

distillation methods into two schemes: offline distillation, in which the teacher is trained until high ac-

curacy is achieved and only then is the knowledge transferred, and online distillation, in which the

teacher and student are trained simultaneously. Online distillation, although more complex, has a more

efficient training process and lower computational cost, in addition to obtaining not only a compact

student but also a more powerful teacher model. [2]

2.5.3 TinyML

Now that we have talked about what Lightweight machine learning models are, we delve into Tiny

Machine Learning (TinyML) [3] that refers to the deployment of these models on ultra-low-power,

resource-constrained devices, such as microcontrollers, embedded systems, or IoT sensors. TinyML’s

key idea is to bring intelligence directly to the device, without relying on cloud processing.

TinyML is successfully applied in areas such as healthcare, agriculture, industrial IoT, and the

environment, driven by the need to integrate intelligence into applications where it was previously un-

feasible due to the high energy and resource consumption of conventional ML models.

Advantages of TinyML

• Reduced Latency

TinyML models run on the device itself, so response time is much faster than when information

needs to be sent to the cloud for processing. This is critical for applications that require real-time

decision-making. Deployment of models on TinyML systems significantly reduces latency, with a

range of 0 to 5 ms, as compared to cloud-based machine learning models [3]. In addition, TinyML

systems maintain high accuracy, with a slight reduction from 95% to 85% due to compression

and optimization for devices with limited resources [3].

• Offline Capability

TinyML models work without an internet connection, unlike cloud models, making them ideal for

areas with limited or no internet access.

• Improving Privacy and Security

TinyML keeps data on the device itself, avoiding sending it to the cloud, which protects user

privacy and complies with data protection regulations.

• Low Energy Consumption

An important part of reducing energy consumption in TinyML is to decrease the amount of data

to be transported and processed. The algorithms are designed to be efficient, helping to reduce

17

device consumption. Other strategies include using low-power components, operating at low

voltage or battery power, and adopting sleep or idle modes.

• Reducing Cost

TinyML models save on the costs of sending data to the cloud, such as bandwidth and storage,

in addition to having low energy consumption. They allow IoT devices to perform data analysis

locally, speeding up decision-making, and to provide independent ML services.

Deployment of TinyML

Currently, models cannot be trained directly on embedded devices due to limited resources. They are

typically trained in the cloud or on more powerful devices before being transferred. Deployment can be

done by hand coding (low-level optimization, but time-consuming), code generation (optimized code,

but with portability issues), or ML interpreters, which allow machine learning algorithms to be executed

on Micro Controller Units within frameworks with predefined libraries and kernels.

A TinyML framework includes, in addition to the interpreter, libraries and tools for data processing,

as well as a Tiny Inference Engine, which efficiently performs the calculations necessary for inference.

These frameworks also offer development tools for data preparation, training, validation, and perfor-

mance profiling, facilitating the creation and execution of models on low-power devices. [3]

Challenges Facing TinyML

TinyML faces several challenges. Environmental adaptation is limited since most models are

trained offline and cannot adjust to changing conditions (concept drift), although solutions like TinyOL

(TinyML with Online-Learning) enable real-time online learning on microcontrollers. Memory limita-

tions force trade-offs between model size, accuracy, and energy use, addressed by techniques such

as compression and quantization. Hardware and software heterogeneity across devices, operating

systems, and sensors complicates integration, while diverse data formats and noise hinder model gen-

eralization. Accuracy drop often occurs when deploying models on low-power devices due to memory

and energy constraints, with reported losses of 1-2% in sensitive applications like healthcare. Privacy

risks arise from sensors (cameras, microphones) capturing sensitive data, with mitigation strategies

including on-device learning and TinyML as-a-Service (TinyMLaaS) to avoid cloud sharing. Finally, reli-

ability and robustness are concerns, as TinyML devices may suffer from hardware errors, degradation,

or interference in safety-sensitive domains. [3]

18

Chapter 3

State of Art

Around 1990, IDS prototypes used for the first time “inductive learning / sequential user patterns”

(e.g., Time-based Inductive Machine (TIM) [47]). In the early 90s, systems like IDES (Intelligent Detec-

tion Expert System) and NIDES (Next-generation Intrusion Detection Expert System) were developed.

NIDES used a combination of statistical metrics and profiles [48]. And eventually, research into neural

networks for intrusion detection started around 1992 [49] [50].

By incorporating machine learning, IDS products became more accurate, learning from past data to

adapt to new and evolving threats. The following sections review research applications and empirical

findings on ML and AI methods in IDS, focusing on system performance, comparative studies, and

domain-specific implementations.

3.1 ML Methods Applied to IDS

Decision trees are effective for binary and multi-class classification and provide an interpretable

framework for IDS, though they may overfit, exhibit high variance, and be computationally expensive.

On traditional benchmark datasets, decision trees achieve accuracies starting at 84.10% depending on

the complexity of the attack patterns [51]. Random Forests mitigate overfitting issues by aggregating

multiple decision trees, and when combined with feature selection techniques such as Boruta [52], their

performance improves substantially.

Na¨ıve Bayes, when combined with feature selection, improves detection performance [28, 29].

Moreover, ensemble implementations using Gaussian Na¨ıve Bayes with other lightweight models have

demonstrated 93% accuracy on the ToN-IoT dataset [53].

Logistic regression combined with multinomial models demonstrates enhanced detection perfor-

mance and reduced misclassification [33, 34]. On the UNSW-NB15 dataset, logistic regression achieves

91.83% accuracy, 94.31% precision, and 92.80% F1-score [51], making it a computationally efficient

baseline for resource-constrained deployments, though deep learning approaches consistently outper-

form it on complex attack patterns.

19

3.2 Deep Learning Approaches

While machine learning methods provide effective data-driven approaches, Deep Learning in IDS en-

compasses broader techniques including expert and rule-based systems that rely on predefined rules

or signatures to detect known attacks, as well as hybrid systems that integrate machine learning

with rule-based logic to enhance detection accuracy and reduce false positives. Deep learning models

extend traditional ML by analyzing complex, high-dimensional, or sequential data. This section re-

views research on CNN and RNN applications in IDS, as these architectures are discussed theoretically

in Section 2.5.1 of the Background chapter.

3.2.1 Convolutional Neural Networks in IDS

CNN models have demonstrated superior performance compared to traditional ML methods in de-

tecting diverse network intrusions. The capability to automatically learn and extract discriminative fea-

tures directly from raw data makes CNNs particularly effective [54]. Vinayakumar et al. (2017) applied

CNNs to analyze time-series network traffic data and demonstrated that the model accurately detected

anomalies, reduced false alarms, and improved overall IDS performance [55]. Research in CNN-

based IDS has shown that these architectures excel at capturing complex patterns in network traffic for

effective detection of sophisticated intrusions [56, 57].

3.2.2 Recurrent Neural Networks in IDS

RNNs are well-suited for IDS because they can model the temporal dynamics of network traffic,

capturing patterns that may indicate an ongoing intrusion [19]. Long Short-Term Memory (LSTM)

networks, a variant of RNNs, have been emphasized in IDS research to enhance performance [58].

LSTMs improve upon standard RNNs by using memory cells that retain information over extended se-

quences, enabling the model to capture long-term dependencies in network traffic. Both RNNs and

their variants are powerful tools for sequential data analysis, effectively learning temporal patterns

critical for accurate intrusion detection [19].

3.2.3 Hybrid Deep Learning Models in IDS

Recent advances combine CNNs with Bidirectional LSTM (BiLSTM) layers to leverage both spatial

feature extraction and temporal sequence modeling. Jouhari et al.

(2024) proposed a lightweight

CNN-BiLSTM architecture for IoT intrusion detection on the UNSW-NB15 dataset [59]. By combining

CNN’s spatial feature extraction with BiLSTM’s capability to capture temporal dependencies bidirection-

ally, their model achieved 97.28% accuracy for binary classification and 96.91% accuracy for multi-

class classification. Notably, prediction time was reduced from 22.4 and 3.2 seconds, on the next best

performing model (LSTM - 96.49%), to 1.1 seconds (binary) and 2.10 seconds (multi-class), demon-

strating the lightweight nature suitable for resource-constrained IoT devices. In a later study, the same

author applied Chi-square feature selection to further enhance performance [51], and achieved 97.90%

20

binary classification accuracy and 97.09% multi-class accuracy with prediction times of 1.1-2.10 sec-

onds.

Misrak and Melaku (2025) combined DNN with BiLSTM (DNN-BiLSTM) on the CIC-IDS2017 and

CIC-IoT2023 datasets [53]. Their quantized model achieved 99.73% accuracy on CIC-IDS2017 with a

model size of only 25.6 KB, and 93.95% accuracy on CIC-IoT2023 with 31.3 KB.

Khan et al. (2025) proposed the Deep Learning-based Intelligent Intrusion Detection (DL-IID) frame-

work combining DNN with BiLSTM using genetic algorithm-based feature selection on multiple IoT

datasets, including Radio Frequency (RF) fingerprinting (450 IoT devices), CIC-IDS2017, CIC-IoMT2024,

and UNSW-NB15 [60]. Their model achieved 99.84% accuracy, 100% precision, 99.69% recall, and

99.84% F1-score while reducing model size to 108.42 KB through post-training dynamic quantization.

These results demonstrate that hybrid DNN/CNN-BiLSTM approaches, when combined with ad-

vanced feature engineering and quantization, achieve state-of-the-art performance while remaining de-

ployable on severely resource-constrained edge devices.

3.2.4 Feature Selection for High-Dimensional IoT Data

Network traffic datasets often contain hundreds of features capturing protocol-level

information,

packet statistics, and behavioral indicators. High-dimensional feature spaces create substantial chal-

lenges for resource-constrained IoT devices through increased memory consumption, elevated com-

putational burden, and amplified communication overhead. Feature selection techniques address

these constraints by identifying and retaining only the most discriminative features while discarding re-

dundant or irrelevant attributes, thereby reducing model complexity and deployment resource require-

ments.

Correlation-Based Feature Selection (CFS) and Principal Component Analysis (PCA). Hassan

et al. (2025) combined CFS and PCA for feature reduction on CIC-IDS2017 (79 features) [61]. CFS

identifies features correlated with attack classes while removing redundancy, achieving 33-44% reduc-

tion. PCA performs linear dimensionality reduction while preserving variance, achieving 64-71% reduc-

tion. Combined with optimized neural networks, this hybrid approach improved F1-score to 98.43%,

reduced model size by 28%, and achieved 9 KB post-training quantized size [61]. This demonstrates

that high-performing IDS can achieve low computational complexity suitable for Vehicular Ad Hoc Net-

works (VANETs).

Chi-Square Feature Selection.

Jouhari et al. (2024) applied Chi-square (χ2) statistical feature se-

lection to reduce the dimensionality of the UNSW-NB15 dataset (44 features) [51]. Chi-square testing

identifies the most relevant features for distinguishing between attack and normal classes by measuring

feature independence from class labels, significantly reducing model complexity. Combined with the

lightweight CNN-BiLSTM architecture, this feature selection approach contributed to achieving a higher

accuracy and lower prediction time.

21

Advanced Feature Selection: RAL-MIFS and Genetic Algorithm. Misrak and Melaku (2025) intro-

duced Redundancy-Adjusted Logistic Mutual Information Feature Selection (RAL-MIFS) combined with

two-stage Incremental Principal Component Analysis (IPCA) for feature engineering on IoT datasets

[53]. RAL-MIFS uses logistic functions to assess feature redundancy more precisely than traditional

mutual information methods, enabling identification of truly complementary features. This advanced fea-

ture engineering, when combined with DNN-BiLSTM and dynamic quantization techniques, achieved

99.73% accuracy on CIC-IDS2017 with a model size of only 25.6 KB.

Khan et al.

(2025) employed wrapper-based genetic algorithm (GA) feature selection for IoT in-

trusion detection, optimizing feature subsets to maximize performance while minimizing computational

burden [60]. GA-based feature selection proved effective across multiple datasets (RF fingerprinting,

CIC-IDS2017, CIC-IoMT2024, UNSW-NB15), contributing to their DL-IID model achieving 99.84% ac-

curacy with 108.42 KB model size.

Local Interpretable Model-Agnostic Explanations (LIME)

is an explanation technique that explains

the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable

model locally around the prediction [62].

In IDS, this local view clarifies which input features drive

specific alerts, helping operators validate whether the model reacts to attack-relevant signals rather than

spurious artifacts. Hassan et al. (2025) applied LIME to their optimized neural network on CIC-IDS2017,

revealing the feature contributions behind each decision and increasing analyst trust and deployability in

safety-critical VANET settings [61].

22

Chapter 4

Frameworks for TinyML

TinyML relies on a variety of software platforms, hardware requirements, and libraries to make pre-

dictions. In this chapter, we will go over some of the most well-known frameworks.

4.1 Lite Runtime (LiteRT)

LiteRT is a deep learning framework that is open source and supports edge-aware learning inference

[63]. Lite Runtime (LiteRT), formerly known as TensorFlow Lite, is Google’s high-performance runtime

for on-device AI. You can find ready-to-run LiteRT models for a wide range of ML/AI tasks, or convert

and run TensorFlow, PyTorch, and JAX models to the TFLite format using the AI Edge conversion

and optimization [64].

It provides tools to convert models from TensorFlow, PyTorch, and JAX models into the FlatBuffers

format (.tf lite), enabling the use of a wide range of state-of-the-art models on LiteRT.

4.2 ExecuTorch (PyTorch)

ExecuTorch is PyTorch’s solution for efficient AI inference on edge devices - from mobile phones

to embedded systems. Allows developers to deploy PyTorch-trained models directly on diverse plat-

forms, from high-end mobile to constrained microcontrollers. It is a lightweight runtime with full hardware

acceleration [65].

4.3 Embedded Learning Library (ELL)

The Embedded Learning Library (ELL) is an Open Source Library for Embedded AI and Ma-

chine Learning from Microsoft. It enables the design and deployment of machine learning models on

resource-constrained platforms and small single-board computers (e.g., Raspberry Pi, Arduino, micro:

bit). Models run locally without cloud dependencies. ELL is aimed at makers, students, and developers,

and its tools and code are freely available, though the library is still evolving.

23

4.4 Framework Comparison

For deploying machine learning models on resource-constrained hardware such as vehicular on-

board units (OBUs), several TinyML frameworks have been proposed. These frameworks differ in their

resource efficiency, inference performance, and optimization support.

TensorFlow Lite (LiteRT) is specifically designed for on-device inference with minimal memory and

computational overhead. LiteRT uses model conversion and optimization techniques to reduce

model size and runtime requirements, making it suitable for embedded scenarios where memory and

latency are critical [66].

Compared to ExecuTorch (PyTorch), LiteRT generally produces smaller binary sizes and faster

inference on embedded devices, due to its aggressive optimizations and custom runtime tailored for lim-

ited hardware resources [67]. While ExecuTorch offers flexibility and ease of use, its default deployments

often consume more CPU cycles and memory, which can be suboptimal for deeply resource-constrained

environments [67].

Libraries such as the Embedded Learning Library (ELL) can generate efficient C++ code for em-

bedded inference, yet they are generally less mature and lack the extensive optimization tooling

found in LiteRT.

Taken together, these comparisons highlight that LiteRT offers a compelling combination of low

memory footprint, real-time inference performance, strong optimization support, and broad sup-

port for edge hardware targets. These characteristics make LiteRT especially suitable for the thesis’s

objective of deploying lightweight deep learning intrusion detection models on resource-limited vehicular

devices.

24

Chapter 5

Methodology

This methodology operationalizes the state-of-the-art findings toward a deployable, lightweight Intru-

sion Detection System (IDS) for V2V flooding on MK5 OBUs. Feature reduction is prioritized to curb

memory and latency overheads on edge hardware, leveraging correlation-based and χ2 selection con-

sistent with prior IoT/VANET studies that paired these filters with Deep Learning models [51, 53, 61]. The

chosen CNN-BiLSTM architecture aligns with evidence that convolutional plus bidirectional temporal

modeling delivers high accuracy with low inference time on intrusion datasets [59]. Given the tight mem-

ory envelope of on-board units, both post-training and quantization-aware strategies are included to

replicate the compact, kilobyte-scale models reported in recent IDS work [53, 60]. LIME is integrated

to expose feature attributions, following successful applications on VANET-oriented IDS [61, 62]. Deci-

sion Trees, validated on this dataset by Sousa et al. (2024) [1], serve as an interpretable baseline to

contextualize the gains of the lightweight deep model.

5.1 Dataset, Feature Selection Methods

Data Source: Sousa et al. (2024) [1]. Real V2V network traffic from MK5 OBUs over IEEE 802.11p,

89 statistical/temporal features capturing normal and attack (SYN/UDP flooding) conditions.

Feature Selection: Two approaches evaluated: (1) Correlation-based retains top-K features by

target correlation; (2) Chi-square (χ2) measures feature independence from class labels;

Feature Importance Analysis: LIME (Local Interpretable Model-agnostic Explanations) [62] to iden-

tify which features most influenced the model’s decisions.

5.2 Proposed Model

Architecture: CNN-BiLSTM processes temporal windows (spatial feature extraction) + Bidirectional

LSTM (temporal pattern capture) + Dense classifier layers.

Quantization Strategies: Post-Training Quantization (PTQ): 32-bit → 8-bit integers post-training

using LiteRT. Quantization-Aware Training (QAT): Simulates quantization during training via Tensor-

25

Flow Model Optimization Toolkit.

5.3 Evaluation Metrics

Classification Metrics: Accuracy, Precision, Recall, F1-Score (primary metric), Confusion Matrix

(TP/TN/FP/FN). Resource Metrics: Model size (.tflite, KB), Inference latency (Raspberry Pi 4, ms),

CPU load (%), Random Access Memory (RAM) peak (MB).

5.4 Implementation

Environment: Python 3.8+, TensorFlow 2.x/Keras, LiteRT, NumPy/Pandas, Scikit-learn. Training:

Google Colab (GPU/TPU). Testing: Raspberry Pi 4 (4GB RAM, ARM Cortex-A72). Monitoring: psutil

(CPU/RAM), Python time (latency).

5.5 Timeline

Data preparation, feature selection

CNN-BiLSTM training and optimization

PTQ and QAT Testing

Metrics collection and Results analysis

Dissertation and presentation

Feb

Mar

Apr

May

Jun

Jul

Aug

Sep

Figure 5.1: Gantt chart of the work plan highlighting the crucial tasks (bars) and their durations

5.6 Summary

Prior V2V intrusion detection work addresses either lightweight model design or V2V-specific at-

tacks, but rarely integrates aggressive optimization with real IEEE 802.11p OBU data. This methodology

bridges that gap by developing a lightweight CNN-BiLSTM on the Sousa et al. (2024) 89-feature dataset

[1], combining feature selection (χ2, correlation-based) with quantization strategies (PTQ/QAT) and re-

alistic Raspberry Pi 4 evaluation against a Decision Tree baseline, demonstrating practical deployment

feasibility for resource-constrained MK5 OBUs.

26

Chapter 6

Preliminary Experiment

6.1 Methodology

A convolutional neural network was trained in Google Colab (12 GB RAM, 2.20 GHz CPU with

2 cores [68]) using the MNIST dataset [69], a standard benchmark for handwritten digit classification

consisting of 70,000 28×28 grayscale images (60,000 training, 10,000 testing). The model architecture

comprised a 3×3 convolution with 12 filters, max-pooling for spatial reduction, a dense layer with 64 units,

and a final classification layer, totaling 130,626 trainable parameters predominantly in fully connected

layers.

Training required approximately 5 min 28 sec. The model was converted to LiteRT using dynamic

range quantization [70], statically converting weights to 8-bit integers while dynamically quantizing

activations during inference, achieving a 91.43% size reduction (1563 KB to 134 KB) in approximately

1.13 s.

Both the optimized LiteRT and original Keras models were deployed on a Raspberry Pi 4 (4 GB

RAM, 1.8 GHz CPU with 4 cores [71]).

Evaluation consisted of 1000 inference runs over the full MNIST test set, capturing inference la-

tency, throughput (images/s), and resource usage (CPU %, resident RAM bytes). Resource monitoring

sampled every 10 ms; each run was summarized by modal values from 50-bin histograms, providing

typical resource consumption while down-weighting transient spikes.

6.2 Conclusions

The comparative analysis of MNIST inference across Google Colab (Keras) and Raspberry Pi 4

(TensorFlow Lite and Keras) reveals critical insights into the effectiveness of model optimization for edge

deployment. See Table 6.1 for a summary of results.

27

6.2.1 Model Quantization Efficacy

Both platforms achieve virtually identical accuracy (≈ 98.5%), demonstrating that dynamic range

quantization preserves the model’s predictive capability while dramatically reducing computational bur-

den. The LiteRT model achieves a 91.4% size reduction (1563 KB to 134 KB), enabling deployment on

severely resource-constrained devices without accuracy degradation.

6.2.2 Latency Trade-offs

On the Raspberry Pi 4, LiteRT demonstrates superior latency performance compared to Keras,

achieving 0.209 ms per image versus 0.223 ms for the unoptimized Keras model, representing a 6.3%

latency reduction. This advantage stems from both the aggressive quantization applied to the model

and the highly optimized LiteRT runtime, which minimizes computational overhead. While both imple-

mentations exhibit approximately 1.55× higher latency than the Colab environment (0.135 ms). This

overhead is acceptable given the Pi’s limited computational capacity.

6.2.3 Resource Efficiency on Edge Hardware

The most striking result is CPU utilization: LiteRT on the Pi consumes only 24.3% of the CPU load

required by Keras on the same hardware (88.90% vs 365.57%), demonstrating that quantization and

the optimized LiteRT runtime dramatically improve computational efficiency. This efficiency advantage

enables other processes to run concurrently without starvation, a critical requirement for embedded

applications.

6.2.4 Memory Considerations

Quantization significantly reduces model size (91.4%), but does not necessarily reduce runtime

memory usage. The observation that Keras consumes less RAM (728.09 MB) than LiteRT (999.82

MB) on Raspberry Pi likely reflects differences in memory allocation strategies. LiteRT preallocates

a single contiguous memory arena before inference to accommodate all tensors, as documented in its

arena allocator [72]. In contrast, Keras’s eager execution mode, which executes operations immediately

rather than deferring them to a graph [73], may employ more flexible allocation strategies.

The arena must hold not only model weights but also all intermediate activation tensors, which

typically maintain full precision during inference and dominate memory consumption. Consequently,

LiteRT requires 999.82 MB (24.4% of Pi’s 4 GB RAM) despite its 134 KB quantized model file.

6.2.5 Practical Implications

This experiment validates LiteRT’s suitability for edge-based intrusion detection on vehicular net-

works. The results demonstrate a clear optimization trade-off: while LiteRT consumes 37.3% more

peak RAM than Keras (999.82 MB vs 728.09 MB), it delivers substantial benefits in deployment scenar-

ios where computational resources are constrained.

28

Metric

Hardware
CPU

RAM

Model

Format
Size (KB)

Raw Performance
Accuracy (%)
Latency (ms/image)

CPU Added Load

Process CPU (%)

RAM Added Load

Colab
(Keras)

Pi 4 (LiteRT)

Pi 4 (Keras)

Ratio Pi-
Keras/Colab

Ratio
Pi-LiteRT/Pi-
Keras

2.20 GHz (2
cores)
12 GB

1.80 GHz (4
cores)
4 GB

1.80 GHz (4
cores)
4 GB

Keras (.keras)
1563

LiteRT (.tflite)
134

Keras (.keras)
1563

—

—

—
—

98.53
0.135

98.51
0.209

98.53
0.223

∼ 1.000
1.652

—

—

—
0.086

∼ 1.000
0.937

194.47

88.90

365.57

—

0.243

Process RAM (MB)
Share of total RAM (%)

1085.68
8.85

999.82
24.40

728.09
17.80

0.671
—

—
1.371

Table 6.1: Comparison of model performance and resource utilization on Google Colab (Keras), Rasp-
berry Pi 4 (LiteRT), and Raspberry Pi 4 (Keras). Raw metrics are from 1000 inference runs on the full
MNIST test set. CPU and RAM added load metrics are modal values from samples taken every 0.01
seconds.

The 91.4% model size reduction enables deployment on storage-limited devices without sig-

nificant accuracy loss (98.51% vs 98.53%). More critically for embedded systems, LiteRT’s CPU

efficiency is exceptional: consuming only 24.3% of Keras’s CPU load (88.90% vs 365.57%) frees

computational resources for concurrent security monitoring tasks-essential for vehicular onboard units

handling network traffic inspection alongside other vehicle functions. The 6.3% latency improvement

(0.209 ms vs 0.223 ms) ensures real-time inference performance on constrained hardware.

For V2V intrusion detection systems operating on resource-limited OBUs, LiteRT’s trade-off pro-

file is favorable: manageable absolute RAM usage (24.4% of 4 GB) combined with dramatic CPU

efficiency gains justifies the memory overhead. This optimization enables deployment of multiple

detection models or concurrent monitoring tasks without computational starvation - a critical require-

ment for vehicular cybersecurity applications where detection must not compromise vehicle safety or

responsiveness.

29

Bibliography

[1] B. Sousa, N. Magaia, S. Silva, N. Thanh Hieu, and Y. Liang Guan. Vehicle-to-vehicle flooding

datasets using mk5 on-board unit devices. Scientific Data, 11(1), Dec. 2024. URL http://dx.doi.

org/10.1038/s41597-024-04173-4.

[2] C.-H. Wang, K.-Y. Huang, Y. Yao, J.-C. Chen, H.-H. Shuai, and W.-H. Cheng. Lightweight deep

learning: An overview.

IEEE Consumer Electronics Magazine, 13(4):51–64, July 2024. doi: 10.

1109/MCE.2022.3181759.

[3] Y. Abadade, A. Temouden, H. Bamoumen, N. Benamar, Y. Chtouki, and A. S. Hafid. A compre-

hensive survey on tinyml.

IEEE Access, 11:96892–96922, 2023. doi: 10.1109/ACCESS.2023.

3294111.

[4] Lightweight flow generator and packet analyzer - tranalyzer. https://tranalyzer.com/. Accessed:

2025-11-05.

[5] Wireshark • go deep. https://www.wireshark.org/. Accessed: 2025-11-05.

[6] S. Axelsson.

Intrusion detection systems: A survey and taxonomy. Technical Report 99–15,

Chalmers University of Technology, Gothenburg, Sweden, 2000.

[7] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman. Survey of intrusion detection sys-

tems:

techniques, datasets and challenges. Cybersecurity, 2(1), July 2019. doi: 10.1186/

s42400-019-0038-7.

[8] Symantec.

Internet security threat

report 2017.

Technical

report, Symantec Corpora-

tion, April 2017.

URL https://www.symantec.com/content/dam/symantec/docs/reports/

istr-22-2017-en.pdf. Accessed: 28-09-2025.

[9] H.-J. Liao, C.-H. Richard Lin, Y.-C. Lin, and K.-Y. Tung.

Intrusion detection system: A compre-

hensive review. Journal of Network and Computer Applications, 36(1):16–24, Jan. 2013. doi:

10.1016/j.jnca.2012.09.004.

[10] B. Mukherjee, L. Heberlein, and K. Levitt. Network intrusion detection. IEEE Network, 8(3):26–41,

May 1994. doi: 10.1109/65.283931.

[11] P. Stavroulakis and M. Stamp, editors. Handbook of Information and Communication Security.

Springer Berlin Heidelberg, 2010. doi: 10.1007/978-3-642-04117-4.

30

[12] F. Sabahi and A. Movaghar. Intrusion detection: A survey. 2008 Third International Conference on

Systems and Networks Communications, page 23–26, 2008. doi: 10.1109/icsnc.2008.44.

[13] C. Modi, D. Patel, B. Borisaniya, H. Patel, A. Patel, and M. Rajarajan. A survey of intrusion detection

techniques in cloud. Journal of Network and Computer Applications, 36(1):42–57, Jan. 2013. doi:

10.1016/j.jnca.2012.05.003.

[14] G. Creech. Developing a high-accuracy cross platform Host-Based Intrusion Detection System

capable of reliably detecting zero-day attacks. PhD thesis, UNSW Sydney, 2014. URL http:

//hdl.handle.net/1959.4/53218.

[15] M. H. Bhuyan, D. K. Bhattacharyya, and J. K. Kalita. Network anomaly detection: Methods, systems

and tools.

IEEE Communications Surveys & Tutorials, 16(1):303–336, 2014. doi: 10.1109/surv.

2013.052213.00046.

[16] C.-Y. Ho, Y.-C. Lai, I.-W. Chen, F.-Y. Wang, and W.-H. Tai. Statistical analysis of false positives and

false negatives from real traffic with intrusion detection/prevention systems. IEEE Communications

Magazine, 50(3):146–154, Mar. 2012. doi: 10.1109/mcom.2012.6163595.

[17] A. Thakkar and R. Lohiya. A review on machine learning and deep learning perspectives of ids

for iot: Recent updates, security issues, and challenges. Archives of Computational Methods in

Engineering, 28(4):3211–3243, Oct. 2020. doi: 10.1007/s11831-020-09496-0.

[18] D. Tymoshchuk, O. Yasniy, M. Mytnyk, N. Zagorodna, and V. Tymoshchuk. Detection and classifi-

cation of ddos flooding attacks by machine learning method. arXiv, 2024. doi: 10.48550/ARXIV.

2412.18990. URL https://arxiv.org/abs/2412.18990.

[19] O. Horny ´ak.

Intelligent intrusion detection systems – a comprehensive overview of applicable ai

methods with a focus on iot security. Infocommunications journal, 17(Special Issue):61–76, 2025.

doi: 10.36244/icj.2025.5.8.

[20] S. J. Russell and P. Norvig, editors. Artificial Intelligence A Modern Approach, Global Edition.

Pearson, 3rd edition, 2016. ISBN:9781292153964.

[21] T. M. Mitchell, editor. Machine learning. McGraw-Hill Science/Engineering/Math, 1997.

ISBN:0070428077.

[22] Features

and

labels

in

supervised

learning:

A

practical

approach

-

geeksforgeeks.

URL

https://www.geeksforgeeks.org/machine-learning/

features-and-labels-in-supervised-learning-a-practical-approach/. Accessed: 2026-

01-14.

[23] IBM. What is a decision tree? — ibm. https://www.ibm.com/think/topics/decision-trees, .

Accessed: 2025-10-06.

[24] IBM. What is random forest? — ibm. https://www.ibm.com/think/topics/random-forest, .

Accessed: 2025-10-06.

31

[25] W. Tao, F. Honghui, Z. HongJin, Y. CongZhe, Z. HongYan, and H. XianZhen.

Intrusion detection

system combined enhanced random forest with smote algorithm. Mar. 2021. doi: 10.21203/rs.3.

rs-270201/v1.

[26] IBM. What

is support vector machine? — ibm.

https://www.ibm.com/think/topics/

support-vector-machine, . Accessed: 2025-10-07.

[27] L. Mhamdi, D. McLernon, F. El-moussa, S. A. Raza Zaidi, M. Ghogho, and T. Tang. A deep learning

approach combining autoencoder with one-class svm for ddos attack detection in sdns.

In 2020

IEEE Eighth International Conference on Communications and Networking (ComNet), pages 1–6,

2020. doi: 10.1109/ComNet47917.2020.9306073.

[28] Z. Muda, W. Yassin, M. N. Sulaiman, and N. I. Udzir. Intrusion detection based on k-means cluster-

ing and na¨ıve bayes classification. 2011 7th International Conference on Information Technology in

Asia, page 1–6, 2011. doi: 10.1109/cita.2011.5999520.

[29] D. H. Deshmukh, T. Ghorpade, and P. Padiya.

Intrusion detection system by improved pre-

processing methods and na¨ıve bayes classifier using nsl-kdd 99 dataset. 2014 International

Conference on Electronics and Communication Systems (ICECS), page 1–7, Feb. 2014. doi:

10.1109/ecs.2014.6892542.

[30] M. Panda and M. R. Patra. Network intrusion detection using naive bayes. International journal of

computer science and network security, 7(12):258–263, 2007.

[31] IBM. What is the k-nearest neighbors algorithm? — ibm. https://www.ibm.com/think/topics/

knn, . Accessed: 2025-10-07.

[32] IBM.

What

is logistic regression?

— ibm.

https://www.ibm.com/think/topics/

logistic-regression, . Accessed: 2025-10-07.

[33] E. Besharati, M. Naderan, and E. Namjoo. Lr-hids: logistic regression host-based intrusion detec-

tion system for cloud environments. Journal of Ambient Intelligence and Humanized Computing,

10(9):3669–3692, Oct. 2018. doi: 10.1007/s12652-018-1093-8.

[34] Y. Wang. A multinomial logistic regression modeling approach for anomaly intrusion detection.

Computers & Security, 24(8):662–674, Nov. 2005. doi: 10.1016/j.cose.2005.05.003.

[35] I. Drori. The Science of Deep Learning. Cambridge University Press, 2022. ISBN:9781108835084.

[36] T. Kohonen. The self-organizing map. Proceedings of the IEEE, 78(9):1464–1480, 2002.

[37] B. J. Shiland. Chapter 8-Introduction to Anatomy and Physiology, page 204. Elsevier, 2nd edition,

2016.

[38] O. A. Montesinos L ´opez, A. Montesinos L ´opez, and J. Crossa. Fundamentals of Artificial Neural

Networks and Deep Learning. Springer International Publishing, Cham, 2022.

ISBN 978-3-030-

89010-0. doi: 10.1007/978-3-030-89010-0 10.

32

[39] N. Lewis. Deep learning made easy with r. A gentle introduction for data science. South Carolina:

CreateSpace Independent Publishing Platform, 2016.

[40] I. Goodfellow, Y. Bengio, A. Courville, and Y. Bengio. Deep learning, volume 1. MIT press Cam-

bridge, 2016.

[41] IBM. What is a recurrent neural network? — ibm.

https://www.ibm.com/think/topics/

recurrent-neural-networks, . Accessed: 2025-10-07.

[42] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf. Pruning filters for efficient convnets. arXiv

preprint arXiv:1608.08710, 2016.

[43] S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural networks with

pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015.

[44] T.-J. Yang, Y.-H. Chen, and V. Sze. Designing energy-efficient convolutional neural networks using

energy-aware pruning.

In Proceedings of the IEEE conference on computer vision and pattern

recognition, pages 5687–5695, 2017.

[45] M. Lin et al. Hrank: Filter pruning using high-rank feature map.

In Proceedings of the IEEE

Conference on Computer Vision and Pattern Recognition (CVPR), pages 1526–1535, 2020. doi:

10.48550/arXiv.2002.10179.

[46] Z. Liu et al. Rethinking the value of network pruning. arXiv preprint arXiv:1810.05270, 2018. doi:

10.48550/arXiv.1810.05270.

[47] H. S. Teng, K. Chen, and S. C. Lu. Adaptive real-time anomaly detection using inductively generated

sequential patterns. In Proceedings of the IEEE Symposium on Security and Privacy, pages 278–

284, Oakland, California, May 1990. IEEE. doi: 10.1109/RISP.1990.63857.

[48] R. Sommer and V. Paxson. Outside the closed world: On using machine learning for network

intrusion detection. 2010 IEEE Symposium on Security and Privacy, page 305–316, 2010. doi:

10.1109/sp.2010.25.

[49] H. Debar, M. Becker, and D. Siboni. A neural network component for an intrusion detection sys-

tem.

In Proceedings of the 1992 IEEE Computer Society Symposium on Research in Security

and Privacy, pages 240–250, Oackland, CA, May 1992. IEEE, IEEE Computer Society Press.

ISBN 0-8186-2825-1. doi: 10.1109/RISP.1992.213257. URL http://ieeexplore.ieee.org/xpl/

articleDetails.jsp?arnumber=213257.

[50] H. Debar and B. Dorizzi. An application of a recurrent network to an intrusion detection sys-

tem.

In Proceedings of the International Joint Conference on Neural Networks (IJCNN 1992),

volume 2, pages 478–483, Baltimore, MD, USA, June 1992. IEEE, IEEE Computer Society Press.

ISBN 0-7803-0559-0. doi: 10.1109/IJCNN.1992.226942. URL http://ieeexplore.ieee.org/

xpl/articleDetails.jsp?arnumber=226942.

33

[51] M. Jouhari, H. Benaddi, and K.

Ibrahimi. Efficient

intrusion detection: Combining x2 fea-

ture selection with cnn-bilstm on the unsw-nb15 dataset.

In 2024 11th International Confer-

ence on Wireless Networks and Mobile Communications (WINCOM), pages 1–6, 2024. doi:

10.1109/WINCOM62286.2024.10658099.

[52] S. Subbiah, K. S. M. Anbananthen, S. Thangaraj, S. Kannan, and D. Chelliah. Intrusion detection

technique in wireless sensor network using grid search random forest with boruta feature selection

algorithm. Journal of Communications and Networks, 24(2):264–273, Apr. 2022. doi: 10.23919/

jcn.2022.000002.

[53] S. F. Misrak and H. M. Melaku. Lightweight intrusion detection system for iot with improved feature

engineering and advanced dynamic quantization. Discover Internet of Things, 5(1), 2025. URL

http://dx.doi.org/10.1007/s43926-025-00203-8.

[54] Y. Ding and Y. Zhai. Intrusion detection system for nsl-kdd dataset using convolutional neural net-

works. Proceedings of the 2018 2nd International Conference on Computer Science and Artificial

Intelligence, page 81–85, Dec. 2018. doi: 10.1145/3297156.3297230.

[55] R. Vinayakumar, K. P. Soman, and P. Poornachandran. Applying convolutional neural network for

network intrusion detection. 2017 International Conference on Advances in Computing, Communi-

cations and Informatics (ICACCI), page 1222–1228, 2017. doi: 10.1109/icacci.2017.8126009.

[56] L. Mohammadpour, T. C. Ling, C. S. Liew, and A. Aryanfar. A survey of cnn-based network intrusion

detection. Applied Sciences, 12(16):8162, Aug. 2022. doi: 10.3390/app12168162.

[57] M. Azizjon, A. Jumabek, and W. Kim. 1d cnn based network intrusion detection with normalization

on imbalanced data. 2020 International Conference on Artificial Intelligence in Information and

Communication (ICAIIC), page 218–224, Feb. 2020. doi: 10.1109/icaiic48513.2020.9064976.

[58] S. M. Sohi, J.-P. Seifert, and F. Ganji. Rnnids: Enhancing network intrusion detection systems

through deep learning. Computers & Security, 102:102151, Mar. 2021. doi: 10.1016/j.cose.2020.

102151.

[59] M. Jouhari and M. Guizani. Lightweight cnn-bilstm based intrusion detection systems for resource-

constrained iot devices. pages 1558–1563, 2024. doi: 10.1109/IWCMC61514.2024.10592352.

[60] A. Khan, M. Asdaque Hussain, and F. Anwer. A hybrid lightweight deep learning-based intrusion

detection approach in iot utilizing feature selection & explainable artificial intelligence. IEEE Access,

13:192451–192466, 2025. doi: 10.1109/ACCESS.2025.3630753.

[61] F. Hassan, Z. S. Syed, A. A. Memon, S. S. Alqahtany, N. Ahmed, M. S. A. Reshan, Y. Asiri, and

A. Shaikh. A hybrid approach for intrusion detection in vehicular networks using feature selection

and dimensionality reduction with optimized deep learning. PLOS ONE, 20(2):e0312752, Feb.

2025. URL http://dx.doi.org/10.1371/journal.pone.0312752.

34

[62] M. T. Ribeiro, S. Singh, and C. Guestrin. “why should i trust you?”. Proceedings of the 22nd ACM

SIGKDD International Conference on Knowledge Discovery and Data Mining, page 1135–1144,

Aug. 2016. URL http://dx.doi.org/10.1145/2939672.2939778.

[63] N. Schizas, A. Karras, C. Karras, and S. Sioutas. Tinyml for ultra-low power ai and large scale iot

deployments: A systematic review. Future Internet, 14(12):363, Dec. 2022. URL http://dx.doi.

org/10.3390/fi14120363.

[64] Lite rt overview — google ai edge — google ai for developers. https://ai.google.dev/edge/

litert, . Accessed: 2025-12-02.

[65] Welcome to the executorch documentation. https://docs.pytorch.org/executorch/stable/

index.html#wins-success-stories. Accessed: 2026-01-03.

[66] On-device inference with litert — google ai edge — google ai for developers. https://ai.google.

dev/edge/litert/inference, . Accessed: 2026-01-03.

[67] P. D. Benchmarking tensorflow lite vs pytorch performance, 2025. URL http://bit.ly/3LwAuTL.

Accessed: 2026-01-03.

[68] colab.google. https://colab.google/. Accessed: 2025-11-26.

[69] Mnist database - wikipedia. https://en.wikipedia.org/wiki/MNIST_database. Accessed: 2025-

11-26.

[70] Post-training quantization — google ai edge — google ai for developers. https://ai.google.dev/

edge/litert/models/post_training_quantization#dynamic_range_quantization. Accessed:

2025-12-02.

[71] Raspberry pi 4 model b specifications – raspberry pi. https://www.raspberrypi.com/products/

raspberry-pi-4-model-b/specifications/. Accessed: 2025-11-26.

[72] Arena planner

implementation.

URL https://github.com/tensorflow/tensorflow/blob/

master/tensorflow/lite/arena_planner.h. TensorFlow Lite C++ API Reference, Accessed:

2026-01-10.

[73] Eager execution. URL https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_

eager_execution. TensorFlow API Documentation, Accessed: 2026-01-10.

35

