# A Machine Learning Framework for

A Machine Learning Framework for
Intrusion Detection in VANET
Communications

Nourhene Ben Rabah and Hanen Idoudi

1 Introduction

1

2

3

4

5

Vehicular ad hoc networks (VANET) stand for different communication schemas 6
that can be performed between connected vehicles and anything (V2X). This 7
includes vehicle-to-vehicle communications, vehicle-to-roadside infrastructure 8
components, or intra-vehicle communications.
9
A VANET system relies on two main components: Roadside Unit (RSU) and 10
On-Board Unit (OBU). RSU is the roadside communication equipment. It provides 11
Internet access to vehicles and ensures exchanging data between vehicles. The 12
OBU is the mobile treatment and communication unit embedded on the vehicle. It 13
allows communication with other vehicles or with the infrastructure’s equipment. 14
VANET communication can be deployed according to different architectures, 15
such as vehicle-to-vehicle (V2V), vehicle-to-infrastructure (V2I), infrastructure-to- 16
vehicle (I2V), infrastructure-to-infrastructure (I2I), and hybrid [1]. Furthermore, 17
a VANET system is composed of three planes: vehicular plane, RSU plane, and 18
services plane. In the vehicular plane, each vehicle is equipped with OBU. The 19
latter allows V2V communication. The RSU plane facilitates V2I, I2V, and I2I 20
communications. In the service plane, different types of services can be deployed 21
such as safety, infotainment, payment, Internet, and cloud-based services. A VANET 22
has some similar features of MANET (mobile ad hoc networks) such as omnidirec- 23

N. Ben Rabah
Centre de Recherche en Informatique, Université Paris 1 Panthéon Sorbonne, Paris, France

ESIEE-IT, Pontoise, France
e-mail: nbenrabah@esiee-it.fr

H. Idoudi ((cid:2))
National School of Computer Science, University of Manouba, Manouba, Tunisia
e-mail: hanen.idoudi@ensi-uma.tn

© The Author(s), under exclusive license to Springer Nature Switzerland AG 2022
K. Daimi et al. (eds.), Emerging Trends in Cybersecurity Applications,
https://doi.org/10.1007/978-3-031-09640-2_10

N. Ben Rabah and H. Idoudi

tional broadcast, short transmission range, and low bandwidth. In contrast, it has 24
particular characteristics. First, a VANET has a highly dynamic topology due to 25
the high mobility of vehicles. This leads also to frequent disconnections. Secondly, 26
target vehicles can be reached upon their geographical location. Thirdly, signal 27
propagation is affected by the environment such as buildings, trees, etc. [1]. Finally, 28
energy, storage failure, and computing capacity are less critical for VANETs as for 29
MANET. Despite that, the serious challenge for VANET is processing huge amount 30
of data in a real-time manner.
31
This diversity of communication schemas and the inherent characteristics of 32
wireless communications make VANETs vulnerable to many security attacks and 33
vulnerabilities. This is emphasized by the critical aspect of some exchanged 34
information that is used for road safety purposes. Security breaches are several 35
and can affect all network layers and all communication aspects in VANET. 36
Moreover, VANETs suffer from traditional vulnerabilities that affect any wireless 37
environment but are also subject to new and speciﬁc attacks exploiting inherent 38
vehicular characteristics [1]. Most of the security solutions deﬁned for traditional 39
networks are not suitable for vehicular networks. Subsequently, researchers are 40
looking for appropriate systems that support vehicular network characteristics and 41
provide robust security mechanisms.
42
Different security countermeasures have been proposed such as key manage- 43
ment systems, anonymity, traceability techniques, cryptographic algorithms, trust 44
management methods, etc. [2]. Recently, many researchers showed that integrating 45
artiﬁcial intelligent (AI) methods in intrusion detection systems increases their 46
effectiveness in detecting attacks on V2X networks. IDS are a widely used approach 47
that analyzes the trafﬁc for indicators of security breaches and creates an alert 48
for any observed security anomaly. Moreover, machine learning (ML) can realize 49
anomaly-based detection systems capable of detecting unknown and zero-day 50
attacks, learning, and training itself by analyzing network activity and increasing 51
its detection accuracy over time.
52
Applying ML techniques for intrusion detection in VANET is of particular inter- 53
est due to the huge amount of exchanged data and the diversity of attacks that can 54
occur. In recent years, many published datasets, describing real traces of VANET 55
communication, have allowed the assessment of ML techniques performances for 56
intrusion detection.
57
This work intends to deﬁne a novel comprehensive framework to design an 58
IDS for V2X communications. Furthermore, unlike most existing works, we use a 59
very recent dataset to evaluate and compare both ensemble and standalone learning 60
techniques to detect various types of DOS and DDOS attacks in VANET.
61
We deﬁne ﬁrst a novel framework for applying ML techniques to detect 62
anomalies in VANET communication. Then, we use a very recent dataset, VDOS- 63
LRS dataset, that describes urban vehicular trafﬁc to assess and compare the 64
performances of well-known standalone ML methods and ensemble ML methods 65
to detect DOS and DDOS attacks in urban environment.
66

The rest of this chapter is structured as follows.

67

A Machine Learning Framework for Intrusion Detection in VANET Communications

In Sect. 2, we review related works related to security issues in VANET, and we 68
review most important works on ML-based IDS for VANETs. In Sect. 3, we 69
expose our framework for designing ML-based IDS for VANET communication. 70
Main results are discussed in Sect. 4 where we study the performances of several 71
ML techniques, both standalone and ensemble learning techniques, on detecting 72
DOS and DDOS attacks in urban trafﬁc using a very recent VANET dataset, 73
namely, VDOS-LRS dataset.
74

Finally, Sect. 5 gives the conclusion of the study.

2 Security of VANET Communications

75

76

In this section, we discuss the security issue in VANET communication; then, we 77
focus on the most important works that considered the use of machine learning- 78
based intrusion detection systems for VANET.
79

2.1 Security Attacks and Vulnerabilities in VANET

80

In-vehicle communications involve embedded units mainly interacting via CAN- 81
Bus, Ethernet, or WiFi standards whereas inter-vehicle networks refer to different 82
kind of interactions between vehicles and other components of the ITS system. 83
These latter can be vehicle-to-infrastructure (V2I), vehicle-to-cloud (V2C), vehicle- 84
to-vehicle (V2V), and vehicle-to-device (V2D) communications [1]. This diversity 85
of architectures and communication schemes led to the inception of the vehicle-to- 86
anything or V2X paradigm.
87
Many security attacks are targeting VANET communications taking proﬁt from 88
the highly heterogeneity of such environments, the highly dynamic topology 89
induced by mobility, and the lack of standard security so far [1, 2]. Security 90
requirements such as availability, data integrity, conﬁdentiality, authenticity, and 91
non-repudiation can be compromised.
92
Denial of Service (DoS) and Distributed DoS (DDOS) attacks aim to disrupt 93
network service’s availability by ﬂooding the OBU (On-Board Unit) and/or RSU 94
(Roadside Unit) communication channels with an unhandled huge amount of 95
requests, resulting in network out of service [3]. In black hole and gray hole 96
attacks, an attacker can capture illegitimate trafﬁc, then drops, retains, or forwards 97
them to erroneous destinations [4]. In Sybil attacks, malicious nodes may create 98
several virtual cars with the same identity to mislead some functionalities. Node 99
impersonation attack tries to impersonate legitimate node’s identity. Additionally, 100
GPS spooﬁng or position faking attacks, also known as hidden vehicle attacks, 101
generate fake position alarms [5].
102
Different attacks can also threaten the integrity and/or the conﬁdentiality of data 103
104

such as tampering attacks and spooﬁng [1, 2].

N. Ben Rabah and H. Idoudi

In-vehicle communications are equally vulnerable as the inter-vehicle commu- 105
nications and can also suffer from all kinds of attacks following the illegitimate 106
intrusion of malicious data [6].
107
We compare the characteristics of some notable security attacks in Table 1 with 108
109

regard to the targeted environment and the compromised security requirement.

2.2 Security Countermeasures

110

Many security mechanisms are considered to secure vehicular communication 111
while taking into account their inherent characteristics. Most important cover the 112
following categories.
113

– Cryptography

114

Its aim is to ensure conﬁdentiality of data while being transmitted from one 115
source node to a destination node. Moreover, they involve encryption algorithms, 116
hash functions, and digital signature algorithm and can provide solutions for 117
diverse types of threats at different levels in VANET. New lightweight solutions for 118
data encryption are more considered to tackle the limited computation capacities 119
of different VANET equipment. The Elliptic Curve Digital Signature Algorithm 120
(ECDSA) is one of the most widely used digital signatures algorithms in IoT in 121
general and in securing VANET communications [7] [8].
122

– Key Management Systems

123

PKI are core ITS component for identity and key management and can be 124
implemented as centralized, decentralized, or distributed systems. Many enhanced 125
solutions based on PKI are proposed to secure authentication and revocation 126
[9]. For instance, in [33], authors deﬁne Enhanced Secure Authentication and 127
Revocation (ESAR) scheme for VANETs which is responsible for revocation 128
checking, processing, and PKI key pair updating.
129

– Anonymity, Unlinkability, and Traceability Techniques

130

These strategies intend to ensure the privacy of users’ data by means of data 131
suppression, randomization, or cloaking to prevent unauthorized access. They offer 132
a countermeasure against several attacks such as eavesdropping, trajectory tracking, 133
or location disclosure.
134
For instance, anonymity techniques are based on the use of pseudonyms by 135
Group Signature and Pseudonymous Authentication schemes. In a group signature 136
approach, a group private key will be used by all vehicles, whereas in pseudonymous 137
authentication schemes, each vehicle is assigned a set of identities that it stores 138
locally. Hybrid approaches that combine both group signature and pseudonymous 139
authentication schemes are also considered [10, 11].
140
To achieve traceability, unique electronic license plate (ELP) should be used. 141
Pseudonyms could be linked with a speciﬁc ELP identity. This would allow 142

A Machine Learning Framework for Intrusion Detection in VANET Communications

n
o
i
t
a
i
d
u
p
e
r
-
n
o
N

y
t
i
c
i
t
n
e
h
t
u
A

y
t
i
l
a
i
t
n
e
d
ﬁ
n
o
C

y
t
i
r
g
e
t
n
I

y
t
i
l
i
b
a
l
i
a
v
A

.
h
e
v
-
n
I

.
h
e
v
-
r
e
t
n
I

l
a
n
r
e
t
x
E

l
a
n
r
e
t
n
I

t
n
e
m
e
r
i
u
q
e
r

y
t
i
r
u
c
e
s

d
e
t
e
g
r
a
T

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

X

e
l
o
h

y
e
r
g
/
e
l
o
h

k
c
a
l
B

n
o
i
t
a
n
o
s
r
e
p
m

i

e
d
o
N

e
l
c
i
h
e
v

n
e
d
d
i
H

n
o
i
t
a
c
ﬁ
i
s
l
a
f

n
o
i
t
i
s
o
P

g
n
ﬁ
o
o
p
S

S
O
D
D
/
S
O
D

k
c
a
t
t

A

y
a
l
p
e
R

y
z
z
u
F

l
i
b
y
S

1
.
3
t

2
.
3
t

3
.
3
t

4
.
3
t

5
.
3
t

6
.
3
t

7
.
3
t

8
.
3
t

9
.
3
t

s
k
c
a
t
t
a

y
t
i
r
u
c
e
s
T
E
N
A
V
e
m
o
s

f
o

s
c
i
t
s
i
r
e
t
c
a
r
a
h
C

1

e
l
b
a
T

N. Ben Rabah and H. Idoudi

authorities to trace a misbehaved user whenever it is needed. Moreover, in group 143
signatures, a tracing manager can revoke the malicious vehicles by analyzing their 144
signatures [12].
145

– Security Protocols

146

Standard communication and routing protocols need to be secured, hence the 147
need for integrating with security protocols at network, transport, or application 148
level. Several security protocols are proposed or adapted to the context of IoT 149
communications in general such as TLS and DTLS [13].
150

– Intrusion Detection Systems

151

Intrusion detection systems (IDS) are an efﬁcient way to detect and prevent 152
malicious or abnormal activities. A typical IDS relies on three main components: 153154

Information collector: It relies on sensors commonly deployed at different 155
sensitive locations.
156
(cid:129) Analysis engine: Its main purpose is to analyze information collected via sensors. 157
(cid:129) Reporting engine: This component is responsible for logging and raising alarms 158
159

when a malicious node or an abnormal event is detected.

In VANET networks, IDS sensors are generally located at RSU and on vehicles. 160
First, these sensors collect nodes’ communication information. Second, the data 161
collected is sent to the analysis engine. Third, the analysis engine analyzes the 162
received data using different methods which depend on the IDS type. If an abnormal 163
event or a malicious node is detected, a report is sent to the reporting engine. Finally, 164
the reporting engine informs the appropriate nodes about the attack.
165
IDS for VANET are mainly classiﬁed into four categories. This classiﬁcation is 166
based on the techniques used to detect threats. These classes are signature based, 167
watchdog based, behavior based, and hybrid IDS [2].
168

2.3 ML-Based Intrusion Detection Systems for VANETs

169

Behavior-based IDS, also known as anomaly-based, use AI and ML as well as 170
other statistical methods to analyze data on a network to detect malicious behavior 171
patterns as well as speciﬁc behaviors that may be linked to an attack.
172
ML-based IDS are part of behavior-based IDS. This approach assumes that 173
intrusive activities are a subclass of abnormal activities. In ML-based IDS, different 174
ML techniques can be used to recognize the misbehavior pattern. In fact, it extracts 175
relations between different attributes and builds attack models [7]. This mechanism 176
allows the RSU or OBU to detect any misbehavior in the network by analyzing 177
received messages and network information. The main advantage of this approach 178
is its ability to detect zero-day attacks and anomalies.
179

So far, many works adopted ML techniques to build efﬁcient IDS.

180

(cid:129)
A Machine Learning Framework for Intrusion Detection in VANET Communications

In [14], Fuad A. Ghaleb et al. proposed a misbehavior detection model based on 181
ML techniques. Authors used real-world trafﬁc dataset, namely, Next Generation 182
Simulation (NGSIM) to train and evaluate the model. They used artiﬁcial neural 183
network.
184
In [3], authors aimed at detecting wormhole attacks in VANET using ML- 185
based IDS. Firstly, they generated a dataset by using both the trafﬁc simulator 186
Simulation of Urban Mobility Model (SUMO) and NS3. Secondly, two different 187
ML algorithms were applied on the generated dataset to train the model, namely, 188
k-nearest neighbors (kNN) and support vector machines (SVM). Finally, to evaluate 189
the different models, the authors used the accuracy rate and four different alarms 190
which are true positive (TP), false positive (FP), true negative (TN), and false 191
negative (FN). As a result, authors pointed out that both the SVM and kNN 192
performed well on detecting wormhole attacks.
193
In [15], authors proposed a ML-based IDS to detect position falsiﬁcation attack 194
in VANET. To train and evaluate ML models, the authors used Vehicular Reference 195
Misbehavior Dataset (VeReMi dataset). Authors used logistic regression (LR) and 196
SVM models. To evaluate the work, they used F-measure. As a result, they proved 197
that SVM performed better than LR.
198
In [16], authors developed an intrusion detection system based on gradient 199
boosting decision tree (GBDT) for CAN-Bus and proposed a new feature based on 200
entropy as the feature construction of GBDT and used a dataset from a real domestic 201
car to evaluate the model.
202
Authors in [17] showed that tree-based and ensemble learning models show more 203
performance in detection compared to other models. Random forest, bagging, and 204
AdaBoosting methods are trained and tested on the Can-hacking dataset, and the 205
DT-based model results in yield performance.
206
Vuong et al. [18] proposed a decision tree-based method for detecting DoS and 207
command injection attacks on robotic vehicles using cyber and physical features 208
to show the importance of incorporating the physical features in improving the 209
performance of the model. They tested their model in a collected dataset. In 210
addition to DoS and command injection attack detection, they also provide in [19] a 211
lightweight intrusion detection mechanism that can detect malware against network 212
and malware against CPU using both cyber and physical input features using the 213
decision tree model.
214
A tree-based intelligent IDS for the internet of vehicles (IoV) that detects 215
DoS and fuzzy attacks is proposed by Li Yang et al. [20]. Firstly, they tested 216
the performance of decision tree (DT), random forest (RF), extra trees (ET), 217
and XGradient Boost (XGB) methods and applied multi-threading to get a lower 218
execution time. Then, they selected three models that generate the lowest execution 219
time as a meta-classiﬁer in the second layer of the stacking ensemble model. 220
Besides, they used an ensemble feature selection (FS) technique to improve the 221
conﬁdence of the selected features. Finally, the authors tested the model on the car- 222
hacking dataset.
223

N. Ben Rabah and H. Idoudi

In [34], authors deﬁne a novel machine learning model using random forest and 224
a posterior detection based on coresets. Their model showed high accuracy for DOS 225
attack detection.
226
The use of ML techniques is undoubtedly efﬁcient, but due to the numerous 227
opportunities that ML techniques offer, more works are still needed to investigate 228
the design of the best ML framework for VANET IDS.
229
In our work, we intend to deﬁne a comprehensive framework to design VANET 230
IDS. Furthermore, unlike most existing works, we use a very recent dataset to 231
evaluate and compare both ensemble learning and standalone learning techniques 232
to detect various types of DOS attack.
233

Our contribution is exposed in the next section.

3 Proposed ML Framework

234

235

In this section, we introduce a novel machine learning framework for intrusion 236
detection in V2X communications. The elaboration process comprises three major 237
phases: dataset description, data preprocessing, and the application of standalone 238
and ensemble learning methods, as shown in Fig. 1.
239

3.1 First Phase: Dataset Description

240

One of the challenges of building efﬁcient V2X ML-based IDS is the lack of public 241
datasets with a big collection of network trafﬁc logs depicting both normal and 242
abnormal activities. More recent works that tried to tackle IDS design using ML 243
or DL (deep learning) techniques to mitigate more complex or new attacks have 244
pointed out this problem, and some tried to build simulated datasets at that end [21– 245

Fig. 1 Proposed ML framework

A Machine Learning Framework for Intrusion Detection in VANET Communications

23]. A survey of the most important and most recent datasets dedicated to VANET 246
communication and involving some well-known security attacks is given in [24].
247
To evaluate the proposed framework, we used the Vehicular Denial of Service 248
Networks and Systems Laboratory (VDOS-LRS) dataset [25]. It is one of the most 249
recently published datasets that incorporate real network trafﬁc collected in different 250
environments (urban, rural, and highway). This dataset involves traces of three DoS 251
attacks:
252

– SYN ﬂood attack is based on sending a huge number of TCP-SYN requests to a 253
254

vehicle to make it out of service.

– UDP ﬂood overloads random ports on the targeted host with UDP datagrams.
255
– Slowloris attack is an application layer DDoS attack that uses partial HTTP 256
257

requests to open multiple connections towards a target.

For this study, we focused on the urban environment. It is initially presented as a 258
PCAP ﬁle. For this purpose, we used the network trafﬁc ﬂow generator and analyzer, 259
CICFlowMeter [26], which allowed us to generate bidirectional ﬂows described 260
through 84 statistical features such as duration, number of packets, number of bytes, 261
packet length, etc. These ﬂows are then saved as a csv ﬁle, representing our dataset. 262
It includes 26,334 normal instances, 124,968 SYN ﬂood attack instances, 122,457 263
UDP ﬂood attack instances, and 650 Slowloris attack instances.
264

3.2 Second Phase: Data Preprocessing

265

These different steps are used to improve the data quality and, consequently, 266
the performance of the machine learning models. It includes data cleaning, data 267
normalization, data transformation, and class balancing.
268

1. Data Cleaning

269

It is used to handle erroneous, inaccurate, and irrelevant data to improve the 270
dataset quality. Indeed, we do not consider source and destination IP addresses and 271
ports, as attackers can easily modify them [22]. Therefore, we removed these ﬁve 272
features: “Flow ID,” “Src IP,” “Src Port,” “DST IP,” and “DST Port.” Thus, we 273
replaced the missing values of some features with the mean values of these features. 274
275

2. Data Normalization

276

It is performed to avoid bias when feature values belong to very different scales. 277
Some features in our dataset vary between 0 and 1, while others can reach inﬁnite 278
values. Therefore, we normalized these features according to Eq. 1, deﬁned as 279
follows:
280

Xnormalized =

X − Xmin
Xmax − Xmin

(1)

N. Ben Rabah and H. Idoudi

where Xnormalized is the normalization result and X is the initial value. Here, Xmax and 281
Xmin represent the maximum and the minimum values of each feature, respectively. 282
283

3. Data Transformation

284

It is used to modify data to ﬁt the input of any ML model. Indeed, some 285
ML models can work with qualitative data (i.e., non-numerical data) such as k- 286
nearest neighbors (kNN), naive Bayes (NB), and decision trees (DT). However, 287
most of them require numerical inputs and outputs to work properly. Therefore, it is 288
important to convert qualitative data to numerical data. In our dataset, each instance 289
is represented by 77 numerical features and one object feature (“Timestamp”) 290
that represents the date and time values of the ﬂow. In this step, we propose to 291
replace this feature by six features of numerical type: “Timestamp_year,” “Times- 292
tamp_month,” “Timestamp_day,” “Timestamp_hour,” “Timestamp_minute,” and 293
“Timestamp_second.”
294

4. Class Balancing

295

Class imbalance is a major problem in supervised ML methods. It usually 296
occurs when the dataset traces are collected from a real environment. Indeed, 297
in such an environment, the data is usually unbalanced, and the models learned 298
from the data may have better accuracy on the majority class but very poor 299
accuracy on the other classes. There are three main ways to deal with this problem: 300
modifying the ML algorithm, introducing a misclassiﬁcation cost, and data sampling 301
[27]. Data sampling is the only solution that can be done independently of the 302
classiﬁcation algorithm, since the other two require direct or indirect modiﬁcations 303
to the algorithm. Data sampling is performed using two methods: undersampling 304
the majority class or oversampling the minority class.
305
Since the classes in our dataset are unbalanced (see Fig. 2), we use the Synthetic 306
Minority Oversampling Technique (SMOTE) [28, 29] to solve this problem. 307
SMOTE involves synthesizing new examples of the minority classes so that the 308
number of examples of the minority class gets closer to or matches the number of 309
examples of the majority class. After performing it, we get 124,968 instances of 310
each class (see Fig. 3).
311

5. Filter-Based Feature Selection

312

Feature selection [30] is a very important step that consists in selecting from the 313
initial dataset the most relevant features. Indeed, if there are too many features, or 314
if most of them are not relevant, the models will consume more resources and be 315
difﬁcult to train. On the other hand, if there are not enough informative features, the 316
models will not be able to perform their ultimate tasks.
317
To achieve such a goal, we propose to use a ﬁlter-based feature selection method 318
that consists of selecting the most relevant subsets of features according to their 319
relationship with the target variable. We, therefore, use statistical tests that consist in 320
(a) evaluating the relationship between each input feature and the output feature and 321
(b) discarding input variables that have a weak relationship with the target variable. 322

A Machine Learning Framework for Intrusion Detection in VANET Communications

150000

100000

50000

0

s
e
c
n
a
t
s
n
i

f
o
r
e
b
m
u
N

SynFlood
attack

UDP
Flood
attack

Benign Slowloris

attack

Classes

Fig. 2 Number of instances of each class before SMOTE: 124,968 instances of “SYN ﬂood
attack” (majority class), 122,457 instances of “UDP ﬂood attack,” 26,334 instances of “Benign,”
and 650 instances of “Slowloris attack” (minority class)

150000

100000

50000

0

s
e
c
n
a
t
s
n
i

f
o
r
e
b
m
u
N

SynFlood
attack

UDP
Flood
attack

Benign Slowloris

attack

Classes

Fig. 3 Number of instances of each class after SMOTE: 124,968 instances of each class

AQ1

In other hand, keeping the input features that gives a strong statistical relationship
with the output variable.

324
There are many statistical tests such as chi-squared, Pearson correlation, permu- 325
tation feature importance, ANOVA F-value test, and others. The choice of statistical 326
measures depends strongly on the types of input variables and the output variable 327
(numerical or categorical). In our dataset, the input variables are of numerical type, 328
and the output variable is of categorical type (the class), hence the interest to use 329
two statistical measures which are:
330

323

(cid:129) ANOVA F-value test that estimates the degree of linear dependence between 331
an input variable and an output variable while giving a high score to strongly 332
correlated features and a low score to weakly correlated features.
333
(cid:129) Mutual Information (MI) that measures the reduction in uncertainty between 334
each input variable and the target variable. The features in the set are classiﬁed 335

N. Ben Rabah and H. Idoudi

according to their MI value. A feature with a low MI value implies that it does not 336
have much effect on the classiﬁcation. Therefore, features with a low MI value 337
can be discarded without affecting the performance of the models [31].
338

3.3 Third Phase: Standalone and Ensemble Learning

Techniques

To validate our framework, we use two types of ML algorithms:

339

340

341

Standalone algorithms such as multilayer perceptron (MLP), decision tree (DT), 342
and k-nearest neighbors (kNN).
343
(cid:129) Ensemble algorithms such as random forest (RF), extra tree (ET), and bagging 344
345

(BAG).

We used the Scikit-learn library implementation of these algorithms [32]. The 346
choice of these algorithms’ hyperparameters has an impact on their performance. 347
For this study, we have used the default values speciﬁed by Scikit-learn as they 348
work reasonably well. It should be noted that the hyperparameters may be set using 349
grid search or randomized search, but these methods are slow and costly.
350

4 Experimental Results

351

This section presents two strategies to check the results obtained by our proposed 352
framework. First, we evaluate the performance of ML algorithms presented above 353
before and after using the SMOTE method. Then, we outline the most relevant 354
features according to the two ﬁlter-based feature selection methods: the ANOVA 355
F-value test and the Mutual Information. All experiments were performed using 356
ten-fold cross-validation.
357

4.1 Performance Metrics

358

To measure the performance of ML models, we used different metrics, such 359
accuracy and F-measure. These metrics are calculated from four basic measures 360
assessed for each class:
361

– True positive of the class Ci (TPi)
– True negative of the class Ci(TNi)
– False negative of the class Ci(FNi)
– False positive of the class Ci(FPi)

362

363

364

365

(cid:129)
A Machine Learning Framework for Intrusion Detection in VANET Communications

Table 2 Multi-class confusion matrix to illustrate TPBenign, TNBenign, FNBenign, FPBenign, and
MSBenign

Benign

SynFlood

UDP Flood

Slowloris

Benign

Slowloris

SynFlood

SynFlood

with i ∈ {Benign, SYN ﬂood attack, UDP ﬂood attack, and Slowloris attack}

366
In the following, we present these metrics calculated according to these out- 367
368

comes:

(cid:129) Accuracy represents the ratio of correctly recognized records to the entire test 369
370

dataset. It is measured as follows:

Accuracy =

(cid:2)l

i=1

T Pi +T N i
T Pi +F N i +F P i +T N i

l

(2)

371

(cid:129) F-score (Eq. 3) is used to measure precision (Eq. 4) and recall (Eq. 5) at the 372
same time. The F-score is the harmonic mean of precision and recall values and 373
reaches its best value at 1 and worst value at 0. It is calculated as follows:
374

F − score = 2 ∗ Recall ∗ Precision
Recall + Precision
(cid:2)l

Precision =

i=1

T Pi
T Pi +F P i
l

Recall =

(cid:2)l

i=1

T Pi
T Pi +F N
l

(3)

375

(4)

376

(5)

l is the number of classes.

377
We also propose to use the confusion matrix (CM) as it is representing perfor- 378
mance results in an intuitive way for non-experts in ML. Each column of the matrix 379
represents the instances in a predicted class, while each row represents the instances 380
in a real class. For example, we present in Table 2 a multi-class confusion matrix to 381
illustrate TPBenign, TNBenign, FNBenign, and FPBenign. TPBenign refers to the normal 382
instances that are correctly classiﬁed, TNBenign means attack instances (SYN ﬂood, 383

N. Ben Rabah and H. Idoudi

UDP ﬂood, and Slowloris) that are correctly predicted, FNBenign refers to the normal 384
instances that are classiﬁed as attacks (i.e., false alarms that are triggered without 385
a real attack), and FPBenign means attack instances that are predicted as normal 386
trafﬁc. The diagonal of the matrix represents the well-classiﬁed instances (TPBenign 387
and TNBenign). MSBenign means attacks that are classiﬁed as other attacks.
388

4.2 Evaluation of ML Models Before and After SMOTE

389

Tables 3 and 4 show the detection performance of the standalone and ensemble 390
models before and after oversampling with the SMOTE method, respectively. 391
Looking at these results, we can see that, whatever the used algorithm, the accuracy 392
is high in the original dataset. It exceeds 98% for all models. For kNN and MLP, the 393
accuracy of the original dataset is even higher than that of SMOTE. Therefore, these 394
results are incorrect because when the classes are not balanced, the minor classes 395
have a negative effect on the accuracy. Therefore, the F-score is the best metric 396
when working with an unbalanced dataset.
397
By analyzing these tables, we can see also that F-score values of DT, MLP, BAG, 398
RF, and ET models are improved after oversampling by the SMOTE method. On 399
the other hand, the F-score value of the kNN model decreased after oversampling 400
by the SMOTE method, and this shows that the algorithm is not inﬂuenced by the 401
class distribution. The model gave better results on the unbalanced dataset. Further 402
observations show that DT, BAG, ET, and RF have the best accuracy using SMOTE 403
(no signiﬁcant difference). That’s why we focus on those classiﬁers in the following. 404
To help non-experts in ML understand the performance of models after using 405
SMOTE method, we present in Tables 5, 6, 7, and 8 the confusion matrices of the 406
DT, BAG, RF, and ET models, respectively.
407

Table 3 Evaluation of standalone models before and after SMOTE

Standalone models
DT
Accuracy
99.998
99.998

F-score
0.99960
0.99998

Methods
None
SMOTE

kNN
Accuracy
99.814
99.672

F-score
0.99704
0.99672

MLP
Accuracy
98.113
94.176

F-score
0.72191
0.94150

Table 4 Evaluation of ensemble models before and after SMOTE

Ensemble models
BAG
Accuracy
99.998
99.998

F-score
0.99978
0.99998

Methods
None
SMOTE

RF
Accuracy
99.991
99.991

F-score
0.99985
0.99991

ET
Accuracy
99.999
99.999

F-score
0.99977
0.99999

t6.1

t6.2

t9.1

t9.2

A Machine Learning Framework for Intrusion Detection in VANET Communications

Table 5 Multi-class confusion matrix after SMOTE for DT

Benign

Slowloris

SynFlood

SynFlood

Benign
124963

0

2

0

SynFlood

UDP Flood

Slowloris

2
124968

0

0

2

0
124966

0

1

0

0
124968

Table 6 Multi-class confusion matrix after SMOTE for BAG

Benign

Slowloris

SynFlood

SynFlood

Benign
124965

0

4

1

SynFlood

UDP Flood

Slowloris

1
124968

0

0

2

0
124964

0

0

0

0
124967

Table 7 Multi-class confusion matrix after SMOTE for RF

Benign

Slowloris

SynFlood

SynFlood

Benign
124933

0

5

0

SynFlood

UDP Flood

Slowloris

0
124968

0

0

35

0
124963

0

0

0

0
124968

Table 8 Multi-class confusion matrix after SMOTE for ET

Benign

Slowloris

SynFlood

SynFlood

Benign
124933

0

5

0

SynFlood

UDP Flood

Slowloris

0
124968

0

0

35

0
124963

0

0

0

0
124968

These confusion matrices show that the different models globally correctly 408
classify “Benign” instances and instances of different attacks. In other words, BAG 409
and ET contain less false alarms than DT and RF (see orange columns). We get 3 410
false alarms for BAG and ET, 5 false alarms for DT and 35 for RF. We thus observe 411
that the models classify very well Slowloris and SYN ﬂood attacks but less for SYN 412
ﬂood attacks.
413

N. Ben Rabah and H. Idoudi

4.3 Feature Selection and Analysis

414

In Table 9, we present the performance of the different ML models incorporating 415
the two feature selection methods, ANOVA F-value and Mutual Information, while 416
varying the number of selected features.
417

The results analysis can be concluded in the following points:

418

comparatively the better performing.

– Of the two feature selection methods implemented, mutual information is 419
420
– Among the 4 classiﬁers implemented, RF and ET give the best accuracies by 421
422
– Feature selection method using Mutual Information identiﬁes features that have 423
the strongest impact on the prediction. As an example, we can see in Table 10, 424
10, 12, and 25 features selected by Mutual Information.
425

varying the number of features from 10 to 45.

5 Conclusion

426

VANETs suffer from several vulnerabilities due to the inherent characteristics of 427
vehicles and the open radio environment. Security of VANET communications is 428
hence a critical issue due to the diversity of VANET applications, architectures, 429
and characteristics. Many works have been done to study security attacks and 430
countermeasures that can tackle VANET vulnerabilities. Intrusion detection systems 431
(IDS) are an efﬁcient way to detect and prevent malicious activities; hence, they are 432
necessary before triggering the appropriate countermeasure. The use of machine 433

Table 9 Comparison between the performance of ANOVA F-value and Mutual Information

Feature selection method
ANOVA F-value

Mutual Information

Number of features
10
12
25
30
35
40
45
10
12
25
30
35
40
45

Accuracy
DT
95.746
95.743
99.612
99.709
99.717
99.708
99.709
99.979
99.992
99.993
99.993
99.994
99.998
99.998

BAG
95.757
95.820
99.618
99.728
99.729
99.728
99.734
99.986
99.993
99.993
99.994
99.995
99.997
99.998

RF
95.757
95.760
99.497
99.612
99.597
99.600
99.584
99.990
99.994
99.995
99.996
99.996
99.998
99.993

ET
96.969
96.970
99.604
99.723
99.710
99.715
99.709
99.988
99.995
99.996
99.995
99.996
99.998
99.999

t12.1

t12.2

A Machine Learning Framework for Intrusion Detection in VANET Communications

Table 10 The selected features using mutual information

Number of
features
10

12

25

Features
Flow Duration, Flow Pkts/s, Flow IAT Mean, Flow IAT Max, Flow IAT Min,
Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s,
Timestamp_hour
Flow Duration, Flow Pkts/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max,
Flow IAT Min, Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s,
Init Bwd Win Byts, Timestamp_hour
Protocol, Flow Duration, Flow Pkts/s, Flow IAT Mean, Flow IAT Std, Flow
IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Max, Fwd
IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Fwd
Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s, SYN Flag Cnt, Init
Bwd Win Byts, Idle Mean, Idle Max, Idle Min, Timestamp_hour

t15.1

t15.2

t15.3

learning techniques is particularly interesting to tackle unknown and zero-day 434
attacks.
435
In our work, we introduced a novel comprehensive framework to design VANET 436
IDS. Furthermore, unlike most existing works, we use a very recent dataset to 437
evaluate and compare both ensemble learning and standalone learning techniques 438
to detect various types of DOS and DDOS attacks.
439
For data preprocessing phase, and after data cleaning, normalization, and trans- 440
formation, we adopted the Synthetic Minority Oversampling Technique (SMOTE) 441
for class balancing; then, we used ANOVA F and Mutual Information for selecting 442
the most relevant features. Afterward, we applied several standalone ML techniques 443
and ensemble ML techniques.
444
Experiments showed that using SMOTE improves F-score for both standalone 445
and ensemble ML methods. When comparing the two considered feature selection 446
methods, ANOVA F-value and Mutual Information, while varying the number of 447
selected features, we noticed that Mutual Information performs better and is able to 448
identify features that have the strongest impact on the prediction. Moreover, among 449
the four classiﬁers implemented, RF and ET give the best accuracies by varying the 450
number of features from 10 to 45.
451
Incorporating ML techniques when designing IDS is undoubtedly efﬁcient, but 452
due to the numerous opportunities that ML techniques offer, more works are still 453
needed to investigate the design of the best ML framework for VANET IDS. For 454
instance, federated learning is a promising approach that can adapt better to the 455
distributed nature of VANET communication by alleviating the vehicle from a big 456
amount of data processing. We intend in future work to investigate this direction.

457

Acknowledgment We would like to thank the research team of the Networks and Systems 458
Laboratory-LRS, Department of Computer Science, Badji Mokhtar University, Annaba, Algeria, 459
for sharing with us their work on the VDOS-LR security dataset.
460

AQ2

References

N. Ben Rabah and H. Idoudi

461

107093, ISSN:1389-1286 (2020)

tions: a survey. Comput. Netw. 151, 52–67 (2019)

detection in vehicle systems. Wirel. Eng. Technol. 9(4), 79–94 (2018)

1. A. Ghosal, M. Conti, Security issues and challenges in V2X: a survey. Comput. Netw. 169, 462
463
2. A. Alnasser, H. Sun, J. Jiang, Cyber security challenges and solutions for V2X communica- 464
465
3. N.A. Alsulaim, R. Abdullah Alolaqi, R.Y. Alhumaidan, Proposed solutions to detect and 466
prevent DoS attacks on VANETs system, in 3rd International Conference on Computer 467
Applications & Information Security (ICCAIS), (2020), pp. 1–6
468
4. K. St˛epie´n, A. Poniszewska-Mara´nda, Security methods against Black Hole attacks in Vehic- 469
ular Ad-Hoc Network, in IEEE 19th International Symposium on Network Computing and 470
Applications (NCA), (2020), pp. 1–4
471
5. J. Montenegro, C. Iza, M.A. Igartua, Detection of position falsiﬁcation attacks in VANETs 472
applying trust model and machine learning, in PE-WASUN ’20: Proceedings of the 17th ACM 473
Symposium on Performance Evaluation of Wireless Ad Hoc, Sensor, & Ubiquitous Networks, 474
(2020), pp. 9–16
475
6. A. Alshammari, M.A. Zohdy, D. Debnath, G. Corser, Classiﬁcation approach for intrusion 476
477
7. M.A. Al-Shareeda, M. Anbar, S. Manickam, A. Khalil, I.H. Hasbullah, Security and privacy 478
schemes in vehicular Ad-Hoc network with identity-based cryptography approach: a survey. 479
IEEE Access 9, 121522–121531 (2021)
480
8. D. Koo, Y. Shin, J. Yun, J. Hur, An online data-oriented authentication based on Merkle tree 481
with improved reliability, in 2017 IEEE International Conference on Web Services (ICWS), 482
(2017), pp. 840–843
483
9. R. Barskar, M. Ahirwar, R. Vishwakarma, Secure key management in vehicular ad-hoc 484
network: a review, in International Conference on Signal Processing, Communication, Power 485
and Embedded System (SCOPES), (2016), pp. 1688–1694
486
10. D. Manivannan, S.S. Moni, S. Zeadally, Secure authentication and privacy-preserving tech- 487
niques in Vehicular Ad-hoc NETworks (VANETs). Veh. Commun. 25, 100247 (2020) 488
ISSN:2214-2096
489
11. N. Parikh, M.L. Das, Privacy-preserving services in VANET with misbehavior detection, 490
in IEEE International Conference on Advanced Networks and Telecommunications Systems 491
(ANTS), (2017), pp. 1–6
492
12. L. Chen, S. Ng, G. Wang, Threshold anonymous announcement in VANETs. IEEE J. Sel. Areas 493
494
(Wiley). 495
496
14. F.A. Ghaleb, A. Zainal, M.A. Rassam, F. Mohammed, An effective misbehavior detection 497
model using artiﬁcial neural network for vehicular Ad hoc network applications, in IEEE 498
Conference on Application, Information and Network Security (AINS), (2017), pp. 13–18
499
15. P.K. Singh, R.R. Gupta, S.K. Nandi, S. Nandi, Machine learning based approach to detect 500
wormhole attack in VANETs, in Workshops of the International Conference on Advanced 501
Information Networking and Applications, (Springer, 2019), pp. 651–661
502
16. D. Tian, Y. Li, Y. Wang, X. Duan, C. Wang, W. Wang, R. Hui, P. Guo, An intrusion detection 503
system based on machine learning for can-bus, in International Conference on Industrial 504
Networks and Intelligent Systems, (Springer, 2017), pp. 285–294
505
17. S.C. Kalkan, O.K. Sahingoz, In-vehicle intrusion detection system on controller area network 506
with machine learning models, in 11th International Conference on Computing, Communica- 507
tion and Networking Technologies (ICCCNT), (2020), pp. 1–6
508
18. T.P. Vuong, G. Loukas, D. Gan, A. Bezemskij, Decision tree-based detection of denial of 509
service and command injection attacks on robotic vehicles, in IEEE International Workshop 510
on Information Forensics and Security (WIFS), (2015), pp. 1–6
511

Commun. 29, 605–615 (2011)
TLS

and DTLS Protocols, Network

ISBN:9781848217584

13. S.S.L. André

Security

Perez,

A Machine Learning Framework for Intrusion Detection in VANET Communications

in internet of vehicles, in IEEE Global Communications Conference (GLOBECOM), (2019)

dataset for intelligent transportation systems. Wirel. Pers. Commun. 115, 1415–1444 (2020)

19. T.P. Vuong, G. Loukas, D. Gan, Performance evaluation of cyber-physical intrusion detection 512
on a robotic vehicle,
in IEEE International Conference on Computer and Information 513
Technology; Ubiquitous Computing and Communications; Dependable, Autonomic and Secure 514
Computing; Pervasive Intelligence and Computing, (2015)
515
20. L. Yang, A. Moubayed, I. Hamieh, A. Shami, Tree-based intelligent intrusion detection system 516
517
21. S. Iranmanesh, F. S. Abkenar, A. Jamalipour and R. Raad. A heuristic distributed scheme to 518
detect falsiﬁcation of mobility patterns in internet of vehicles.. IEEE Internet Things J., 2021.
519
22. A.R. Gad, A.A. Nashat, T.M. Barkat, Intrusion detection system using machine learning for 520
vehicular Ad hoc networks based on ToN-IoT dataset. IEEE Access 9, 142206–142217 (2021)
521
23. D.M. Kang, S.H. Yoon, D.K. Shin, Y. Yoon, H.M. Kim, S.H. Jang, A study on attack 522
pattern generation and hybrid MR-IDS for in-vehicle network, in International Conference 523
on Artiﬁcial Intelligence in Information and Communication (ICAIIC), (2021), pp. 291–294
524
24. D. Swessi, H. Idoudi, A comparative review of security threats datasets for vehicular networks, 525
in International Conference on Innovation and Intelligence for Informatics, Computing, and 526
Technologies (3ICT), (2021), pp. 746–751
527
25. R. Rahal, A. Amara Korba, N. Ghoualmi-Zine, Towards the development of realistic DoS 528
529
26. A. Habibi Lashkari, CICFlowMeter (formerly known as ISCXFlowMeter): a network trafﬁc 530
Bi-ﬂow generator and analyzer for anomaly detection 2018. https://github.com/ahlashkari/ 531
CICFlowMeter
532
27. P.D. Gutiérrez, M. Lastra, J.M. Benítez, F. Herrera, Smote-gpu: big data preprocessing on 533
534
28. N.V. Chawla, K.W. Bowyer, L.O. Hall, W.P. Kegelmeyer, SMOTE: synthetic minority over- 535
536
29. R. Alshamy, M. Ghurab, S. Othman, F. Alshami, Intrusion detection model for imbalanced 537
dataset using SMOTE and random forest algorithm, in International Conference on Advances 538
in Cyber Security, (Springer, Singapore, 2021), pp. 361–378
539
30. J. Cai, J. Luo, S. Wang, S. Yang, Feature selection in machine learning: a new perspective. 540
541
31. A. Thakkar, R. Lohiya, Attack classiﬁcation using feature selection techniques: a comparative 542
543
32. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, E. Duchesnay, 544
545
33. U. Coruh, O. Bayat, ESAR: enhanced secure authentication and revocation scheme for 546
547
34. H. Bangui, M. Ge, B. Buhnova, A hybrid machine learning model for intrusion detection in 548
549

commodity hardware for imbalanced classiﬁcation. Prog. Artif. Intell. 6(4), 347–354 (2017)

Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 12, 2825–2830 (2011)

vehicular Ad Hoc networks. J. Inf. Secur. Appl. 64 (2022). Elsevier

study. J. Ambient Intell. Human. Comput. 1, 1249–1266 (2021)

sampling technique. J. Artif. Intell. Res. 16, 321–357 (2002)

Neurocomputing 300, 70–79 (2018)

VANET. Computing, Springer (2021)

AUTHOR QUERIES

AQ1. Please check sentence starting “In other hand...” for clarity.
AQ2. Please check the sentence “We intend in future...” for clarity.

