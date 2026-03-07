# OPeN

OPeN

DATA DeScRIPTOR

Vehicle-to-Vehicle Flooding
Datasets using MK5 On-board
Unit Devices

Breno Sousa

 1, Naercio Magaia2 ✉, Sara Silva1, Nguyen thanh Hieu3 & Yong Liang Guan3

The availability of information is a key requirement for the proper functioning of any network. When
the availability problem is brought to vehicular networks, it may hinder novel vehicular services and
applications and potentially put human lives at risk, as malicious users can send a massive number of
spurious packets to disrupt them. Although flooding attacks in vehicular contexts have been the focus
of attention of the research community, most proposed datasets are generated using simulated data
and only contain the modeled network’s behavior. In this work, we generated datasets of such attacks
using three realistic vehicular devices, i.e., MK5 On-board Unit (OBU). We applied a machine learning
algorithm to get the first insights into the complexity of the proposed datasets, reporting the achieved
Accuracy, F1-Score, Precision, and Recall.

Background & Summary
From Intelligent Transport Systems (ITS) to the Internet of Vehicles (IoV), vehicular application scenarios are
becoming more complex by requesting more elaborate network and security protocols (e.g., routing protocols
or mechanisms to avoid network packets being modified by malicious users while in transit). Therefore, to
understand the risks of network breaches and how harmful network attacks can be, the researcher community
has been using network simulators since deploying real test beds and gathering data from them is not trivial.
However, the network simulators’ behavior is predictable and expected. In realistic scenarios, the dynamics
of vehicles (e.g., the number of vehicles, their speed, and direction) and road obstacles (e.g., buildings) add
some degree of unpredictability, resulting in packet loss or reducing the communication range. For example,
Gonçalves et al.1 proposed some datasets considering the Denial-of-Service (DoS), Fake Heading, Fake Speed,
and Fake Acceleration attacks by simulating different scenarios in the Network Simulator 3 (NS-3), and they
reported very high metrics (e.g., almost 100% in classifying these attacks2). Here, we point out that the simulated
data extracted from the simulators may be biased since the proposed datasets only have characteristics from the
simulated environment, not real vehicle hardware. Hence, the results may not be accurate.

When vehicles share information among themselves (e.g., using Vehicle-to-Vehicle (V2V) communication),
they can have a “view” of the road by sharing information gathered by their sensors. Road congestion systems,
collision avoidance, weather conditions, access to Internet applications, and Basic Safety Messages (BSM) are
examples of applications and standardization that allow the vehicle to become more than a simple transpor-
tation machine and generate valuable driver data. In addition, BSMs are sent using Dedicated Short Range
Communication (DSRC), which contains vehicle data such as speed and Global Positioning System (GPS)
coordinates.

Despite the current debate about the de facto communication technology to be adopted worldwide for vehic-
ular networks, either Long-Term Evolution (LTE)/4G, 5G, or IEEE 802.11p, the latter standard is the oldest
proposed for information sharing in such networks. In addition, Roadside Units (RSUs) are also part of the
network, aiding in routing information via Vehicle-to-Infrastructure (V2I) communication so that messages
can reach their intended destination.

Due to the vehicles’ high mobility, the connection duration between the nodes tends to be short compared
to traditional networks, resulting in frequent changes in the network topology. The works by Cardote et al.3 and
Nagel et al.4 tried to find out the average connection duration for vehicles, where they considered some variables

1LASIGE, Faculty of Sciences, University of Lisbon, 1749-016, Lisbon, Portugal.  2School of engineering and
Informatics, University of Sussex, Brighton, BN1 9RH, UK. 3School of electrical and electronic engineering, nanyang
technological University, Singapore, Singapore. ✉e-mail: n.Magaia@sussex.ac.uk

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

1

www.nature.com/scientificdataFig. 1  Our test bed, consisting of three MK5 OBU devices with the wireless antennas placed on top of them.

to achieve the mean time, such as the vehicles’ direction and speed. The mean time reported in these two studies
was 31.57 seconds and 25 seconds, respectively.

Regarding cybersecurity, researchers have already used different techniques and tools to develop optimal
solutions that consider vehicular characteristics. For instance, an Intrusion Detection System (IDS) is a type of
security tool that can sniff network packets and try to classify them as legitimate or malicious. In our previous
work5, we simulated flooding attacks on a 5G-enabled vehicular network on the Network Simulator 3 (NS-3).
Then, we generated three datasets and proposed an IDS using machine learning (ML) algorithms, such as deci-
sion trees, random forests, and multilayer perceptron. We realized the importance of data diversity for building
robust models that can accurately identify an attack on unseen data; we also realized that more complex algo-
rithms, e.g., neural networks, do not necessarily produce better results than simple methods, e.g., decision trees.
ML-based IDS solutions need to train their models on network data. However, most works found in the lit-
erature use simulated or non-vehicular network data for this purpose. Although simulators are a prominent way
to gather network data, they lack realistic data, mostly considering simulated behaviour. In realistic scenarios,
vehicles may behave unexpectedly due to the dynamic nature of the vehicular environment, such as obstacles
that can cause non-line-of-sight (NLOS) problems, signal interference, and shadowing, among others. In this
sense, Nguyen et al.6 discuss the impact of the shadowing problem (e.g., large vehicles shadowing normal-sized
vehicles), where they mention a signal loss from 10 to 15 dB due to this.

In our quest for datasets with realistic network devices, we did not find any publicly available datasets of V2V
communication. To the best of our knowledge, the datasets we propose here are the first publicly available data-
sets of V2V communication. Rahal et al.7 proposed a realistic dataset for the flooding attacks, i.e., SYN Flood,
User Datagram Protocol (UDP) flood, and Slowloris attack. However, they connected two vehicles in V2V mode
using the IEEE 802.11g standard, which was not designed for vehicular communication. Their dataset is avail-
able only upon request.

The importance of publicly available realistic datasets can be highlighted by the study carried out by
Banafshehvaragh and Rahmani8. They investigated published papers from 2018 to 2022 related to smart vehicles
in reputable journals from publishers like Elsevier, IEEE, and Springer, among others. Besides other findings,
their study tries to answer five research questions, from which we highlight one of them: “What simulators and
datasets were used in smart vehicle intrusion, anomaly, and attack detection methods?”. To answer this, they
call attention to 27.58% of published papers that do not use datasets to learn detection models, instead relying
on known static attack signatures, and 15.38% generated datasets from controller area networks (CAN) that,
although realistic, are not representative of V2V communication. In addition, 26.91% do not use vehicular net-
work data (i.e., 11.53% from the NSL-KDD dataset9, 7.69% from the UNSW-NB15 dataset10, and 7.69% from
the CICIDS2017 dataset11). Many other works explore the use of datasets like CAN-intrusion12, Car-Hacking13,
ROAD14, and NAIST CAN15, among others.

Moreover, most mentioned datasets have considered different network features to analyze and build their ML
models. The Tranalyzer16 tool is an example of a feature extraction tool that can extract network data from stored
network files, such as a .pcap file. Burschka and Dupasquier17 introduced it to perform network traffic analysis,
and Azab et al.18 surveys techniques, datasets, and challenges for network traffic classification, where they also
highlight the use of the Tranalyzer tool.

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

2

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 2  Connections diagram of our test bed.

Fig. 3  Attacking scenarios.

Furthermore, reporting realistic results is an important key point. However, the lack of data collected from
real vehicular network devices is still a major challenge in reporting realistic results. For example, many works
claim to have achieved very near 100% performance in classifying vehicular data, but the datasets used are prone
to biases (e.g., the network will just have implemented behavior) related to the limitations of network simulators.
Information availability is a critical requirement in any connected scenario. If we consider information avail-
ability in vehicular scenarios, losing shared information can be a serious concern due to fatal accidents. For
example, Moradi-Pari et al.19 mention the quality of service (QoS) requirements in different vehicular appli-
cations, where the maximum latency platooning applications should be between 10 and 500 milliseconds. In
addition, Tangirala et al.20 show platooning problems when the information of the platoon leader is lost. In this
case, they simulated different platooning scenarios and reported vehicle crashes when platoon members did not
receive information for 1 second.

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

3

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 4  Activity diagram.

Feature’s name

Feature’s description

pktNo

flowInd

flowStat

time

pktIAT

pktTrip

Packet number

Flow index

Flow status and warnings

Time in milliseconds

Packet inter-arrival time

Packet round-trip time

flowDuration

Flow duration in seconds

numHdrs

hdrDesc

Number of headers (depth) in hdrDesc

Headers description

Table 1.  Extracted features using tranalyzer (part 1).

Methods
Our test bed consisted of three MK5 On-Board Unit (OBU)21 devices. These devices support Dedicated
Short-Range Communications (DSRC), hence coming equipped with an IEEE 802.11p radio. They support
Global Navigation Satellite System (GNSS) and come equipped with Ethernet 100 Base-T. Figure 1 shows the
MK5 OBU devices used in our test bed, where the wireless antennas are placed on top of them.

Since the UDP (User Datagram Protocol) and TCP (Transmission Control Protocol) protocols22 can be used
in vehicular applications, we considered the following attacks: SYN flood, where a malicious user sends TCP
packets containing only SYN flags to consume the victim’s resources, and UDP flood, where a malicious user
sends junk UDP packets to disrupt a network. Both attacks can help us to collect and extract vehicular network
data. To analyze their impacts, we built two scenarios. We used UDP sockets in each scenario, where the devices
simulate four clients. That is, each client is a thread, and all clients share the same antenna for sending and receiv-
ing network packets. In addition, device 1 has an extra client that performs the flooding attacks. Each client has
a transmission period ranging from 5 to 30 seconds and a stop period ranging from 10 to 20 seconds. The trans-
mission and stop periods are selected randomly for each run. We performed a total of 200 runs in each scenario.

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

4

www.nature.com/scientificdatawww.nature.com/scientificdata/Feature’s name

Feature’s description

vlanID

srcMac

dstMac

ethType

srcIP

srcIPCC

srcIPOrg

srcPort

dstIP

dstIPCC

dstIPOrg

dstPort

l4Proto

srcMacLbl

dstMacLbl

dstPortClassN

dstPortClass

pktLen

udpLen

snapL4Len

snapL7Len

l7Len

ipToS

ipID

ipIDDiff

ipFrag

ipTTL

VLAN number (inner VLAN)

Source MAC address

Destination MAC address

Ethernet type

Source IP address

Destination IP country

Source IP organization

Source port number

Destination IP address

Destination IP country

Destination IP organization

Destination port number

Layer 4 protocol

Source MAC label

Destination MAC label

Port-based classification of the destination
port number

Port-based classification of the destination
port name

Packet size on the wire

Length in UDP/UDP-Lite header

Snapped layer 4 length

Snapped layer 7 length

Layer 7 length

IP Type of Service (ToS)

IP identifier

IP identifier difference

IP fragment

IP Time-to-live (TTL)

ipHdrChkSum

IP header checksum

ipCalChkSum

IP header computed checksum

l4HdrChkSum

Layer 4 header checksum

l4CalChkSum

Layer 4 header computed checksum

ipFlags

IP aggregated flags

ip6HHOptLen

IPv6 Hop-by-Hop options length

ip6HHOpts

ip6DOptLen

ip6DOpts

ipOptLen

ipOpts

seq

ACK

SeqMax

seqDiff

ackDiff

seqLen

ackLen

seqFlowLen

ackFlowLen

tcpMLen

IPv6 Hop-by-Hop options

IPv6 Destination options length

IPv6 Destination options

IPv4 options length

IPv4 options

Sequence number

Acknowledgment number

Maximum sequence number

Sequence number difference

Acknowledgment number difference

Sequence length

Acknowledgment length

Sequence flow length

Acknowledgment flow length

Aggregated valid bytes transmitted so far

Table 2.  Extracted features using tranalyzer (part 2).

Figure 2 presents the connections diagram of our test bed. It also shows how we accessed each device. They
were wired to a switch and accessed via Secure Shell (SSH). In addition, although we have a wired connection to
access all MK5 devices for management purposes, they exchange information via the wireless network interface,
as shown in Fig. 1. Besides, in our test bed, we use an OBU to perform the attacks, but it is possible to use exter-
nal devices to carry attacks in vehicular communication. An example of this is the work by Seo et al.23, where

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

5

www.nature.com/scientificdatawww.nature.com/scientificdata/Feature’s name

Feature’s description

tcpBFlgt

tcpFStat

tcpFlags

Number of bytes in flight (not acknowledged)

TCP aggregated protocol flags + combinations (CWR, ACK, PSH, RST, SYN, FIN, …)

TCP aggregated protocol flags (FIN, SYN, RST, PSH, ACK, URG, ECE, CWR)

tcpAnomaly

TCP aggregated header anomaly flags

tcpWin

tcpWS

tcpMSS

tcpTmS

tcpTmER

tcpMPTyp

tcpMPF

cpMPAID

tcpMPDSSF

tcpOptLen

TCP window size

TCP window scale factor

TCP maximum segment size

TCP timestamp

TCP time echo reply

Multipath TCP (MPTCP) type

MPTCP flags

MPTCP address identifier

MPTCP data sequence signal (DSS) flags

TCP options

tcpStatesAFlags

TCP state machine anomalies

icmpStat

icmpType

icmpCode

icmpID

icmpSeq

ICMP status

ICMP type

ICMP code

ICMP identifier

ICMP sequence number

icmpPFindex

Parent flow index

ftpStat

7Content

FTP Status

Layer 7 Content

Table 3.  Extracted features using tranalyzer (part 3).

Scenario

Number of Rows Class 0 Number of Rows Class 1 Total Number of Rows

1a

1b

2

2,151,768 (75%)

2,795,001 (80%)

2,185,811 (58%)

708,106 (25%)

695,784 (20%)

1,572,289 (42%)

2,859,874

3,490,785

3,758,100

Table 4.  Distribution of classes in each dataset.

they performed some attacks (e.g., DoS, Fuzzy, among others) by connecting a Raspberry Pi to an On-Board
Diagnostic II (OBD-II) and generated the car-hacking dataset13.

Figure 3 presents our two scenarios. In both scenarios, each device communicates with surrounding devices.
These devices send information in broadcast mode, and the content of each packet is filled with data based on
gpspipe24, i.e., latitude, longitude, climb, and speed, among others. In scenario 1, device 1 simulates four legit-
imate clients and 1 malicious client that performs the flooding attacks. In scenario 2, device 1 does not send
legitimate packets to generate more heterogeneous data. Besides, as we carry out denial-of-service attacks, and
aim to gather as much network data as possible in our scenarios, we only have one attacker.

In both scenarios, device 3 is the only victim device. We used tcpdump25 to collect all received packets in the
“wave-data” network interface, which uses the IEEE 802.11p standard. After completing all runs, tcpdump saves
all network traffic into a .pcap file. Three datasets were generated, two for scenario 1 and one for scenario 2,
which will provide insight into whether legitimate packets sent by device 1 affect the ability to detect an attack.

Another important point of a realistic scenario is data diversity. Because we send packets in broadcast mode,
devices use UDP. To add more data to the network, devices 1 and 2 also run iPerf26 tool to send TCP packets
to device 3. Figure 4 shows a diagram of how the devices communicate with each other. The UDP traffic is
generated via UDP sockets. Before sending packets, all devices select random times (seconds) to start and stop
sending packets. In addition, as we need to simulate a realistic scenario, starting and stopping to send packets is
necessary, as it resembles real vehicle communication patterns.

Since the operating system of MK5 OBU is a Linux distribution, we were able to use the hping327 tool from
the Kali Linux distribution on device 1. Therefore, the SYN flood and UDP flood attacks send packet sizes of 344
bytes and 284 bytes, respectively. With it, one can send custom UDP/TCP packets for a victim device aiming
to disrupt the network. In scenario 1, when the malicious client starts the attack, it also selects a random time
following the same time range of normal communication. As the malicious client can perform two attack types
based on the UDP and TCP protocols (e.g., UDP flood and SYN flood), we set a likelihood of 50% for perform-
ing SYN Flood or UDP Flood attacks.

SYN flood attack.  Although the SYN flood attack is a well-known problem in networked systems, it is still
an open research problem to the scientific community. This type of attack affects the “TCP 3-way handshake”

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

6

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 5  All the steps followed in our work.

process necessary to establish a TCP connection. In summary, when a client tries to establish a connection, it first
sends a “SYN” flag to the server to allocate resources for this new connection. Then, the server responds to the
client with a “SYN-ACK” flag to inform the client that it knows this new connection. Finally, the client responds
to the server with an “ACK” flag to inform that a connection was established.

Malicious users perform a denial of service attack sending a high number of packets with only the "SYN”
flag active to a victim aiming to consume device resources such as network bandwidth, processing, and energy,
among others28.

UDP flood attack.  Similar to the SYN flood attack, the UDP flood attack is also a well-known problem in
computer networks. In a normal situation, when a server receives a UDP packet, it checks if any program/service
is listening to requests in that communication port. Otherwise, the server replies to the sender via ping (ICMP
packet) that the destination is unavailable.

To perform this attack, malicious users flood a network by sending a high number of junk UDP packets. The
problem gets worse when the server receives a large number of requests and has to inform each sender that the
desired destination is not available. Responding to each sender consumes the server’s resources and may make it
unavailable to respond to new requests. In the latter case, the denial of service was successful.

Data Records
As previously stated, all the packets received by device 3 were stored in a .pcap file. We use the Tranalyzer16 tool
to extract network data and build three datasets. As in each run the clients randomly select the time of sending
packets, the number of packets in each dataset differs. The tranalyzer enabled extracting 81 features from each
.pcap file. Tables 1, 2, and 3 present all 81 features. Moreover, we point out that since our datasets all have the
same features, other tests could be extended by combining them to induce more robust ML models. Our datasets
are hosted on Zenodo29.

In order to collect as much information as possibly needed to build good models, we also extracted the listed

features from Wireshark tool30:

•	 Time delta from the previously captured frame
•	 Time delta from the previously displayed frame
•	 Time since reference or first frame
•	 Time since the first frame
•	 Time since the previous frame
•	 Time since the first frame in this TCP stream
•	 Time since the previous frame in this TCP stream
•

iRTT - initial Round Trip Time

We show the distribution of classes in each dataset in Table 4.
Data exploration/validation is an important step when generating new data. We used the decision tree ML
algorithm to provide insight into the complexity of the generated data in the classification problem. We used

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

7

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 6  Plotted features using the dataset 1 from scenario 1.

scikit-learn (sklearn version 1.3.0). It is important to note that the rationale behind the utilization of a basic ML
algorithm is twofold: firstly, to demonstrate the potential of such an approach in terms of initial application,
and secondly, to highlight the versatility of simpler ML models in the context of attack classification. The results
presented will serve as a baseline for future tests.

We present our proposed datasets by plotting a pair graph of a set of features tested and considering the class
distribution (e.g., normal and attack). Due to the large number of features used (i.e., 23 features) and the total
number of rows, generating this type of figure for each dataset would be very time-consuming if we used all
the features. So, we used the following features: pktIAT, l4Proto, ipID, ack, tcpMLen, Time delta from previous
captured frame, Time delta from previous displayed frame, Time since first frame, and Time since previous frame.
Figure 6 presents how the data points are distributed in dataset 1 from scenario 1. Furthermore, the result test
of 0.829 can be explained as the challenge of distinguishing class 0 (e.g., green dots) and class 1 (e.g., red dots)
since both classes share similar regions. Figure 7 presents the data points of dataset 2 from scenario 1. Here, we
draw the reader’s attention: although we carried out the same scenario with different random sending times,
we got different classification results (i.e., as shown in Tables 6 and 8). The differing results can be explained by
the different number of attacking data points in each dataset. For example, dataset 1 from scenario 1 has more
attacking data points than dataset 2 from scenario 1. Finally, Fig. 8 presents the data points for scenario 2.

Baseline results.  To demonstrate the performance of applying ML algorithms in our datasets, we used a
decision tree. Therefore, we performed a preprocessing step before applying the decision tree algorithm. For this,
we transformed the output into tabular data, used only the features with numerical values, and replaced “NaN”

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

8

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 7  Plotted features using the dataset 2 from scenario 1.

and/or null values to 0 (zero). Figure 5 presents an overview of the steps we followed in our work. When con-
sidering binary classification, the classes 0 and 1 correspond to normal traffic and flooding attacks, respectively.

The features extracted from wireshark are important and relevant because they represent information about
the time of receiving the network packets. Trying to vary the type of used information, we used some merged
features from tranalyzer and wireshark, a total of 23 features: pktIAT, numHdrs, l4Proto, ipID, ack, seqDiff, ack-
Diff, seqLen, ackLen, seqFlowLen, ackFlowLen, tcpMLen, tcpMSS, tcpTmS, tcpTmER, tcpOptLen, Time delta from
previous captured frame, Time delta from previous displayed frame, Time since first frame, Time since previous
frame, Time since first frame in this TCP stream, Time since previous frame in this TCP stream, iRTT - initial
Round Trip Time.

First, we split the main dataset into training (70%) and test (30%) datasets, keeping the test only for report-
ing results in the end. Then, we searched for the best max_depth parameter using RandomizedSearchCV on the
training set with the default 10-fold cross-validation and the “f1” as the scoring parameter (as we are using an
unbalanced dataset, F1-Score better represents the performance of the classification). The tested tree depths
were 2, 5, 10, 15, 16, 17, 18, 19, 20, 21,22, 23, 24, 25. With the best tree depth returned, 20, we built a decision
tree on the training set. Figure 9 shows the confusion matrix of the achieved results on the test set, normalized
by row (i.e., inside each true class).

For evaluating this model, we used the following metrics, Accuracy, F1-Score, Precision, and Recall, for
insights into how the generated datasets perform when we apply a simple machine learning algorithm. Table 5
shows the feature’s importance when we apply the decision tree algorithm in dataset 1 from scenario 1. Here we
can see that not all features are relevant for classification, although they are related to the TCP connection (ack,

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

9

www.nature.com/scientificdatawww.nature.com/scientificdata/Feature’s name

pktIAT

numHdrs

l4Proto

ipID

ack

seqDiff

ackDiff

seqLen

ackLen

seqFlowLen

ackFlowLen

tcpMLen

tcpMSS

tcpTmS

tcpTmER

tcpOptLen

Feature’s importance

0.0178965

0

0

0.0964007

0

0

0

0

0

0.231537

0

0.374709

0

0

0

0

Time delta from previous captured frame

0.058257

Time delta from previous displayed frame

0.0180903

Time since first frame

Time since previous frame

0.179088

0.0240228

Time since first frame in this TCP stream

0

Time since previous frame in this TCP stream 0

iRTT

0

Table 5.  Feature’s importance of dataset 1 from scenario 1.

Accuracy

F1-Score

Precision

Training

Test

0.945

0.923

0.877

0.829

0.977

0.925

Recall

0.7963

0.751

Table 6.  Metrics from application of decision tree to dataset 1 from scenario 1.

seqDiff, ackDiff, seqLen, ackLen, seqFlowLen, ackFlowLen). In this decision tree model, the classifier learned
how to distinguish class 0 and class 1 by the amount of transmitted bytes. In a flooding scenario, it makes sense
to classify a malicious node that is sending a large number of packets. Table 6 shows the metrics achieved by
making a binary classification.

In addition, Fig. 9 shows that our decision tree model has a more realistic performance when compared to
other published works. Please recall that, in the investigated literature, most of the report metrics were near
100%. Here, we obtained a True-positive (TP) equal to 159,641 records, a False-positive (FP) equal to 12,947
records, a False-negative (FN) equal to 52,842, and a True-negative (TN) equal to 632,533 records. Those results
show us that the model misses 25% of the attacks and erroneously classifies as attack only 2% of the records.

We now present the feature importance of dataset 2 from scenario 1 in Table 7 and our results in Table 8. In
this run, GridSearchCV returned the max_depth 25. Unlike Table 5, using this dataset the model learned how to
distinguish class 0 and class 1 by the time of the first frame received and the IP identifier, where each IP address
has a different identification. Considering that our script generates a new IP address in the attack phase, this is
to be expected since legitimate clients always have the same IP identifier.

We highlight the importance of heterogeneous data. As nodes select a random time to send the packets, our
data is not the same in all datasets. For example, when vehicles are on the roads, they may be unreachable in
different parts of the trip. Figure 10 shows the confusion matrix of the results obtained in Table 6, where we have
a TP equal to 141,265 records, an FP equal to 16,319 records, an FN equal to 67,674records, and a TN equal to
821,978 records. In this scenario, the percentage of missed attacks is high (32%) and the percentage of records
erroneously classified as attacks is equal to dataset 1 from scenario 1 (2%).

Lastly, we applied the decision tree algorithm to the scenario 2 dataset. Using different ways of how the nodes
share information could give us relevant insights into how complex realistic scenarios are. Please note that by
“different ways”, we mean different starting and stopping times to send information. Table 9 shows the feature’s
importance of scenario 2 when we consider that device 1 only sends malicious packets. In this dataset, similarly
to Table 7, we can see that the model could learn to distinguish class 0 and class 1 based on the time of the first
frame received and also the IP identifier. Figure 11 shows the confusion matrix of the dataset from scenario 2,
where we report a TP equal to 364,784 records, an FP equal to 48,944 records, an FN equal to 107,470 records,

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

1 0

www.nature.com/scientificdatawww.nature.com/scientificdata/Feature’s name

pktIAT

numHdrs

l4Proto

ipID

ACK

seqDiff

ackDiff

seqLen

ackLen

seqFlowLen

ackFlowLen

tcpMLen

tcpMSS

tcpTmS

tcpTmER

tcpOptLen

Feature’s importance

0.0320586

0

0

0.112788

0

0

0

0

0

0

0

0.38158

0

0

0.192554

0

Time delta from previous captured frame

0.0174575

Time delta from previous displayed frame

0.0323714

Time since first frame

Time since previous frame

0.205388

0.0258014

Time since first frame in this TCP stream

0

Time since previous frame in this TCP stream 0

iRTT

0

Table 7.  Feature’s importance of dataset 2 from scenario 1.

Accuracy

F1-Score

Precision

Recall

Training

Test

0.942

0.920

0.835

0.771

0.968

0.896

0.734

0.676

Table 8.  Metrics from application of decision tree to dataset 2 from scenario 1.

and a TN equal to 364,784 records. Here, 23% of the attacks were missed, and 7% of the records were erroneously
classified as attacks. Table 10 shows the achieved metrics.

As  the  main  goal  of  our  work  is  to  present  our  datasets,  we  do  not  make  any  comparison  to  the
state-of-the-art. We also need to highlight that the presented data were gathered using real vehicular commu-
nication devices, which differs from other published papers as discussed by Rahal et al.7. We also can consider
our results more realistic than the ones reported in the literature using simulated or non-vehicular network data.
To further extend our experiments, we plan to use the collected data to build novel ML models for intrusion

detection that can classify malicious traffic and compare their performance with state-of-the-art solutions.

technical Validation
Data recorder validation.  We transmitted and captured network packets using three MK5 OBU devices.
Our application aimed to emulate the exchange of messages, taking into account the average communication time
between vehicles on the road, whose transmission time was chosen at random. Moreover, to add more traffic in
the proposed scenarios, we used two types of normal traffic applications: 1) network packets sent over sockets,
with an average packet size of 250 bytes - the average size is because it depends on the information contained in
the packet; and, 2) iperf tool, which sent packets with a size of 1,514 bytes. In addition, we used a consolidated
network tool (i.e., tcpdump) to store all network packets received by one of the devices used. We also presented
baseline results using a simple decision tree.

Attack validations.  Our DoS attacks were designed to disrupt network behavior by targeting a single device.
By analyzing the captured traffic from the victim device, we can validate the presence of malicious traffic on the
network. The malicious packets were sent using hping3 with the “--flood” parameter set to send packets as fast as
the network card supports, dramatically increasing the number of packets on the network and provoking packet
drops.

UDP and SYN flooding attacks.  Since no distinction is made between the two types of attacks, by exam-
ining class 1, the hdrDesc feature indicates the type of protocol (i.e., UDP or TCP). In other words, it is a binary
classification problem (e.g., class 0 represents normal traffic). Analyzing the sequence of messages received, it
becomes clear that device 3 received malicious packets as well as normal packets. This indicates that normal

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

1 1

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 8  Plotted features using the dataset from scenario 2.

Fig. 9  Normalized confusion matrix of dataset 1 from scenario 1.

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

1 2

www.nature.com/scientificdatawww.nature.com/scientificdata/Fig. 10  Normalized confusion matrix of dataset 2 from scenario 1.

Fig. 11  Normalized confusion matrix of dataset from scenario 2.

Feature’s name

Feature’s importance

pktIAT

numHdrs

l4Proto

ipID

ACK

seqDiff

ackDiff

seqLen

ackLen

seqFlowLen

ackFlowLen

tcpMLen

tcpMSS

tcpTmS

tcpTmER

tcpOptLen

Time delta from previous captured frame

Time delta from previous displayed frame

Time since first frame

Time since previous frame

Time since first frame in this TCP stream

0.0598621

1.00636e-06

0

0.153357

0

0

0

0

0

0

0

0.344791

0

0

0

1.0034e-01

0.0222851

0.0338902

0.248472

0.0502894

0.0870501

Time since previous frame in this TCP stream 0

iRTT

0

Table 9.  Feature’s importance of dataset from scenario 2.

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

13

www.nature.com/scientificdatawww.nature.com/scientificdata/Accuracy

F1-Score

Precision

Recall

Training

0.930

Test

0.861

0.910

0.823

0.972

0.882

0.854

0.772

Table 10.  Metrics from the application of decision tree to the dataset from scenario 2.

packets were either delayed, lost, or dropped. However, it is important to note that platooning scenarios can also
be compromised by delayed packets31.

Usage Notes
To the best of our knowledge, our proposed datasets are the first publicly available using vehicular communica-
tion technology. To this end, we cannot make a fair comparison to others. Although we can find a similar dataset
– but at the same time different – developed by Rahal et al.7, their dataset is only available upon request and does
not use IEEE 802.11p standard.

We need to highlight that we tried to simulate more clients (i.e., run more threads) in our experiments.
However, the attempts did not work properly due to MK5 OBU hardware constraints and the high amount of
packets generated by the malicious client. Furthermore, to perform the flooding attacks, we needed to install
the hping3 running the following command: “sudo apt install hping3”. Please take a note that hping3 has some
dependencies such as libc6, libpcap0.8, and libtcl8.6.

comparison to other datasets.  Comparing our datasets to other datasets is challenging. Therefore, we
have not yet identified a published paper or publicly available dataset that has extracted data from physical devices
using the IEEE 802.11p standard. As we mentioned earlier, a comparable dataset7 has been developed using
a physical device. However, the data they collected is not according to the established vehicle communication
standard. After analyzing the reported metrics using ML algorithms, it was observed that the results were signif-
icantly high. For instance, the support vector machine (SVM) achieved an f1-score of 0.9886, while the random
forest and decision tree models attained f1-scores of 0.99989 and 0.99994, respectively. Thus, we have already
proposed a dataset5 using a network simulator. Comparing their results with our previous work5, we can point out
a primary similarity: their dataset does not seem to be difficult for classification, since we obtained a very close
f1-score with simulated data: the decision tree obtained an f1-score of 0.99 and the random forest an f1-score of
0.98. A very good performance was also reported by Gonçalves et al.2 where they used a network simulator and
reported metrics that were very close to 100%. A comparison of our reported metrics with datasets generated
from network simulators raises three interesting questions: first, “how realistic are the reported metrics using
simulators?”; second, “how difficult is their7 data for ML discriminators to distinguish between normal and attack
traffic?”; third, based on the similarity of the results obtained amongst all these datasets, one may question if the
one proposed by Rahal et al.7 was not obtained via simulation since it is not publicly available.

code availability
We created a repository to make our source code and datasets available for those who want to explore our datasets
further. The codes we used were written in Python programming language and shell script, where they can be
found in our repository on GitHub.

Received: 10 April 2024; Accepted: 26 November 2024;
Published: xx xx xxxx

References
  1.  Goncalves, F. et al. Synthesizing datasets with security threats for vehicular ad-hoc networks. In GLOBECOM 2020-2020 IEEE

Global Communications Conference, 1–6 (IEEE, 2020).

  2.  Gonçalves, F., Macedo, J. & Santos, A. Intelligent hierarchical intrusion detection system for vanets. In 2021 13th International

Congress on Ultra Modern Telecommunications and Control Systems and Workshops (ICUMT), 50–59 (IEEE, 2021).

  3.  Cardote, A., Sargento, S. & Steenkiste, P. On the connection availability between relay nodes in a vanet. In 2010 IEEE Globecom

Workshops, 181–185 (IEEE, 2010).

  4.  Nagel, R. The effect of vehicular distance distributions and mobility on vanet communications. In 2010 IEEE intelligent vehicles

symposium, 1190–1194 (IEEE, 2010).

  5.  Sousa, B., Magaia, N. & Silva, S. An intelligent intrusion detection system for 5g-enabled internet of vehicles. Electronics 12, 1757

(2023).

  6.  Nguyen, H. et al. Impact of big vehicle shadowing on vehicle-to-vehicle communications. IEEE Transactions on Vehicular Technology

69, 6902–6915 (2020).

  7.  Rahal, R., Amara Korba, A. & Ghoualmi-Zine, N. Towards the development of realistic dos dataset for intelligent transportation

systems. Wireless Personal Communications 115, 1415–1444 (2020).

  8.  Banafshehvaragh, S. T. & Rahmani, A. M. Intrusion, anomaly, and attack detection in smart vehicles. Microprocessors and

Microsystems 96, 104726 (2023).

  9.  NSL-KDD dataset. https://www.unb.ca/cic/datasets/nsl.html. Accessed: 2024-01-18.
 10.  The UNSW-NB15 Dataset. https://research.unsw.edu.au/projects/unsw-nb15-dataset. Accessed: 2024-01-18.
 11.  Intrusion Detection Evaluation Dataset (CIC-IDS2017). https://www.unb.ca/cic/datasets/ids-2017.html. Accessed: 2024-01-18.
 12.  CAN Dataset for intrusion detection (OTIDS). https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset. Accessed: 2024-01-18.
 13.  Car-Hacking Dataset for the intrusion detection. https://ocslab.hksecurity.net/Datasets/car-hacking-dataset. Accessed: 2024-01-18.
 14.  ROAD (ROAD: The ROad event Awareness Dataset for Autonomous Driving). https://paperswithcode.com/dataset/road. Accessed:

2024-01-18.

 15.  Hossain, M. D., Inoue, H., Ochiai, H., Fall, D. & Kadobayashi, Y. Lstm-based intrusion detection system for in-vehicle can bus

communications. IEEE Access 8, 185489–185502 (2020).

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

1 4

www.nature.com/scientificdatawww.nature.com/scientificdata/ 16.  TRANALYZER - Lightweight flow generator and packet analyzer. https://tranalyzer.com/. Accessed: 2024-02-05.
 17.  Burschka, S. & Dupasquier, B. Tranalyzer: Versatile high performance network traffic analyser. In 2016 IEEE symposium series on

computational intelligence (SSCI), 1–8 (IEEE, 2016).

 18.  Azab, A., Khasawneh, M., Alrabaee, S., Choo, K.-K. R. & Sarsour, M. Network traffic classification: Techniques, datasets, and

challenges. Digital Communications and Networks 10, 676–692 (2024).

 19.  Moradi-Pari, E., Tian, D., Bahramgiri, M., Rajab, S. & Bai, S. Dsrc versus lte-v2x: Empirical performance analysis of direct vehicular

communication technologies. IEEE Transactions on Intelligent Transportation Systems 24, 4889–4903 (2023).

 20.  Tangirala, N. T. et al. Analysis of packet drops and channel crowding in vehicle platooning using v2x communication. In 2018 IEEE

Symposium Series on Computational Intelligence (SSCI), 281–286 (IEEE, 2018).

 21.  MK5 OBU. https://www.cohdawireless.com/solutions/hardware/mk5-obu/. Accessed: 2024-01-18.
 22.  Mu’azu, A. A., Jung, L. T., Hasbullah, H., Lawal, I. A. & Shah, P. A. Guaranteed qos for udp and tcp flows to measure throughput in

vanets. In Advances in Computer Science and its Applications: CSA 2013, 1137–1143 (Springer, 2014).

 23.  Seo, E., Song, H. M. & Kim, H. K. Gids: Gan based intrusion detection system for in-vehicle network. In 2018 16th Annual

Conference on Privacy, Security and Trust (PST), 1–6 (IEEE, 2018).

 24.  gpspipe(1). https://gpsd.gitlab.io/gpsd/gpspipe.html. Accessed: 2024-01-22.
 25.  TCPDUMP & Libpcap. https://www.tcpdump.org/. Accessed: 2024-01-22.
 26.  iPerf - The ultimate speed test tool for TCP, UDP and SCTP. https://iperf.fr/. Accessed: 2024-02-05.
 27.  Hping3. https://www.kali.org/tools/hping3/. Accessed: 2024-02-05.
 28.  Kepçeoğlu, B., Murzaeva, A. & Demirci, S. Performing energy consuming attacks on iot devices. In 2019 27th Telecommunications

Forum (TELFOR), 1–4 (IEEE, 2019).

 29.  Sousa, B., Magaia, N., Silva, S., Hieu Thanh, N. & Guan Yong, L. Vehicle-to-vehicle flooding datasets https://doi.org/10.5281/

zenodo.13220637 (2024).

 30.  Wireshark. https://www.wireshark.org/. Accessed: 2024-02-05.
 31.  Liu, X., Goldsmith, A., Mahal, S. S. & Hedrick, J. K. Effects of communication delay on string stability in vehicle platoons. In ITSC

2001. 2001 IEEE Intelligent Transportation Systems. Proceedings (Cat. No. 01TH8585), 625–630 (IEEE, 2001).

Acknowledgements
This work was supported by H2020-MSCA-RISE under grant No. 101006411; and by Fundação para a Ciência
e a Tecnologia (FCT), Portugal, through doctoral grant SFRH/BD/151413/2021 (https://doi.org/10.54499/
SFRH/BD/151413/2021) under the MIT Portugal Program, and through funding of the LASIGE Research
Unit, ref. UIDB/00408/2020 (https://doi.org/10.54499/UIDB/00408/2020) and ref. UIDP/00408/2020 (https://
doi.org/10.54499/UIDP/00408/2020). This work was supported by Temasek Laboratories@NTU seed research
project No. TLSP24-05.

Author contributions
B.S., N.T.H. and N.M. conceived the experiment, B.S. conducted the experiment, and B.S., N.M., and S.S. analysed
the results. All authors reviewed the manuscript.

competing interests
The authors declare no competing interests.

Additional information
Correspondence and requests for materials should be addressed to N.M.

Reprints and permissions information is available at www.nature.com/reprints.

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and
institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribu-
tion and reproduction in any medium or format, as long as you give appropriate credit to the original author(s)
and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed mate-
rial. You do not have permission under this licence to share adapted material derived from this article or parts of
it. The images or other third party material in this article are included in the article’s Creative Commons licence,
unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative
Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted
use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/.

© The Author(s) 2024

Scientific Data |         (2024) 11:1363  | https://doi.org/10.1038/s41597-024-04173-4

1 5

www.nature.com/scientificdatawww.nature.com/scientificdata/
