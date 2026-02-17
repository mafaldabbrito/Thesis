Here's a comprehensive overview of OBU devices used and projected for use in vehicles, their hardware constraints, and why IDS models must be deployed directly on them:

---

## 1. What is an OBU (On-Board Unit)?

An OBU is a device installed in vehicles for Vehicle-to-Everything (V2X) communication — exchanging real-time data with other vehicles (V2V), infrastructure (V2I), pedestrians (V2P), and networks (V2N). OBUs are the vehicle-side endpoint in Dedicated Short-Range Communications (DSRC) or Cellular V2X (C-V2X) systems.

- [Wikipedia — Vehicle-to-everything](https://en.wikipedia.org/wiki/Vehicle-to-everything)
- [Wikipedia — Dedicated Short-Range Communications](https://en.wikipedia.org/wiki/Dedicated_short-range_communications)
- [Wikipedia — Vehicular Communication Systems](https://en.wikipedia.org/wiki/Vehicular_communication_systems)

---

## 2. Current OBU Hardware Used in Vehicles

### Cohda Wireless MK5 OBU
The MK5 is one of the most widely used OBUs in V2X research and trials (and is the one used in the Sousa et al. dataset referenced in your thesis). Key specs:

- **Radio**: Dual IEEE 802.11p radios
- **Chipset**: NXP RoadLINK (SAF5100 baseband + TEF5100 RF transceiver + NXP i.MX 6DualLite application processor)
  - i.MX 6DualLite: **Dual-core ARM Cortex-A9** @ up to 1 GHz, 64-bit DDR3 support
- **Connectivity**: Ethernet 100 Base-T, USB
- **Security**: Integrated hardware security, tamper-proof key storage, ECDSA acceleration
- **GNSS**: Lane-level accuracy positioning
- **OS**: Linux-based

**Sources:**
- [Cohda Wireless MK5 OBU product page](https://cohdawireless.com/solutions/hardware/mk5-obu/)
- [NXP SAF5100 V2X baseband processor](https://www.nxp.com/products/wireless-connectivity/dsrc-safety-modem/software-defined-radio-processor-for-v2x-communication:SAF5100)
- [NXP i.MX 6DualLite processor (used as host processor in MK5)](https://www.nxp.com/products/i.MX6DL)

### NXP RoadLINK SAF5400 (next-generation single-chip modem)
- **Single-chip** DSRC V2X modem (replaces SAF5100+TEF5100 combo)
- **ECDSA verification**: Up to 2000 messages/sec
- **AEC-Q100 Grade 2** automotive qualified
- **Host interfaces**: SDIO, SPI
- Designed to pair with NXP **i.MX 6** or **i.MX 8** family application processors (ARM Cortex-A9, A35, A53, or A72 depending on grade)

**Sources:**
- [NXP SAF5400 product page](https://www.nxp.com/products/wireless-connectivity/dsrc-safety-modem/single-chip-v2x-modem-for-dsrc-and-c-v2x:SAF5400)
- [NXP SAF5400 Fact Sheet (PDF)](https://www.nxp.com/docs/en/fact-sheet/SAF5400V2XFSA4.pdf)

### Other Notable OBU Chipset Vendors
| Vendor | Chipset/Product | Technology | Notes |
|---|---|---|---|
| **Qualcomm** | 9150 C-V2X Chipset | C-V2X (3GPP Rel-14) | Cellular-based V2X, supports PC5 direct link |
| **Autotalks** | CRATON2 / Tekton | DSRC + C-V2X dual-mode | First global dual-mode V2X chipset |
| **Alps Alpine** | UMPZ module | IEEE 802.11p (based on NXP SAF5400) | Commercial-grade module |

**Sources:**
- [Qualcomm 9150 C-V2X Chipset](https://www.qualcomm.com/products/automotive/c-v2x/qualcomm-9150-c-v2x-chipset)
- [Autotalks V2X technology](https://www.autotalks.com/technology/)

---

## 3. Typical OBU Hardware Requirements & Constraints

| Parameter | Typical Range | Notes |
|---|---|---|
| **Processor** | ARM Cortex-A7 to Cortex-A72 | Embedded-class, low-power SoCs |
| **RAM** | 256 MB – 2 GB DDR3/DDR4 | Severely constrained vs. desktop |
| **Storage** | 256 MB – 4 GB eMMC/Flash | Minimal persistent storage |
| **Power** | 5–15W total system | Vehicle 12V supply, but low thermal envelope |
| **Temperature** | -40°C to +85°C (AEC-Q100) | Automotive-grade qualification required |
| **Latency** | < 100 ms end-to-end for safety | V2V safety messages require near-real-time |
| **Communication** | 5.9 GHz, 10 MHz channels, ~300m range | IEEE 802.11p / C-V2X |
| **Security** | HSM, ECDSA, tamper-proof key storage | Message authentication is mandatory |
| **OS** | Linux (embedded), QNX, AUTOSAR | Real-time or near-real-time OS |

**Source for latency and timing requirements:**
- [IEEE 802.11p standard](https://en.wikipedia.org/wiki/IEEE_802.11p) — predicts delays of at most tens of milliseconds for high-priority traffic
- [NXP White Paper: IEEE802.11p ahead of LTE-V2V for safety applications (PDF)](https://www.nxp.com/docs/en/white-paper/LTE-V2V-WP.pdf)
- [NHTSA: Vehicle-to-Vehicle Communications Readiness Report (PDF)](https://www.nhtsa.gov/staticfiles/rulemaking/pdf/V2V/Readiness-of-V2V-Technology-for-Application-812014.pdf)

---

## 4. Can OBUs Be Used for IDS? — Yes, With Lightweight Models

### Why IDS on OBUs is Both Needed and Feasible

**The Need:**
- Flooding attacks (SYN, UDP, ACK, HTTP) are particularly dangerous in resource-constrained vehicular environments — even moderate attack volumes can exhaust resources ([as stated in your thesis, Section 2.2.1](Lightweight_Machine_Learning_Models_for_Intrusion.md)).
- Cloud-based IDS introduces unacceptable latency for safety-critical V2V decisions.
- On-device inference preserves privacy and reduces dependency on connectivity.

**The Constraints:**
- OBUs generally have **ARM Cortex-A9 class processors** (dual-core, ~1 GHz), **~512 MB–1 GB RAM**, and **no GPU**.
- This is comparable to a **Raspberry Pi 3/4** — which your preliminary experiment in Chapter 6 already validated as capable of running quantized ML inference via LiteRT.

**What makes it feasible:**
1. **TinyML frameworks** (LiteRT/TensorFlow Lite, ExecuTorch) are designed for exactly this class of hardware — as covered in your Chapter 4.
2. **Quantized models** (INT8 via PTQ or QAT) dramatically reduce model size and inference time while preserving acceptable accuracy. Your Chapter 6 MNIST experiment demonstrated this on a Raspberry Pi 4.
3. **Lightweight architectures** (pruned CNNs, small MLPs, decision trees, random forests) can run within the OBU's memory and timing budget.
4. **NXP i.MX 6DualLite** (the host processor in the Cohda MK5) supports Linux and has NEON SIMD extensions for accelerated ML inference — LiteRT can take advantage of this. ^10c9c0

**Sources:**
- [NXP i.MX6DL Processor specs (ARM Cortex-A9, NEON SIMD)](https://www.nxp.com/products/i.MX6DL)
- [TensorFlow Lite / LiteRT for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [TinyML: Machine Learning with TensorFlow Lite (book overview)](https://www.oreilly.com/library/view/tinyml/9781492052036/)
- [NXP eIQ Machine Learning Software for i.MX processors](https://www.nxp.com/design/design-center/software/eiq-ml-development-environment:EIQ)

### Key Comparison: OBU vs. Raspberry Pi 4

| Feature | Cohda MK5 OBU (i.MX6DL) | Raspberry Pi 4 |
|---|---|---|
| CPU | 2× ARM Cortex-A9 @ 1 GHz | 4× ARM Cortex-A72 @ 1.5 GHz |
| RAM | ~512 MB – 1 GB DDR3 | 1–8 GB LPDDR4 |
| GPU | Vivante GC880/GC320 (not for ML) | VideoCore VI (not for ML) |
| SIMD | NEON | NEON |
| LiteRT support | Yes (via Linux + eIQ) | Yes (via TFLite runtime) |
| Power | ~5W | ~5–7W |

The Raspberry Pi 4 is **more powerful** than the MK5's application processor. If your model runs well on a Pi 4, **it will need further optimization** (smaller model, more aggressive quantization/pruning) to fit the MK5, but the same framework (LiteRT) applies.

**Source for NXP eIQ ML environment:**
- [NXP eIQ ML Development Environment](https://www.nxp.com/design/design-center/software/eiq-ml-development-environment:EIQ) — NXP's official ML inference framework for i.MX processors, supports TFLite, ONNX, and more.

---

## 5. Future/Projected OBU Platforms

Next-generation OBUs are expected to use significantly more powerful processors:

| Platform | Processor | Expected Use |
|---|---|---|
| **NXP S32G Vehicle Network Processor** | ARM Cortex-A53 + Cortex-M7 | V2X gateway, service-oriented gateway |
| **NXP i.MX 8M** | ARM Cortex-A53 (quad) + Cortex-M4 | Listed as complementary processor for SAF5400 |
| **NXP i.MX 8X** | ARM Cortex-A35, with HW error correction | Next-gen automotive compute |
| **Qualcomm Snapdragon Ride** | ARM Cortex-A78 class | Advanced ADAS + V2X integrated platform |

These future OBUs will have substantially more ML inference capability, making IDS deployment even more practical.

**Sources:**
- [NXP SAF5400 related processors listing (shows i.MX8 family as complementary)](https://www.nxp.com/products/wireless-connectivity/dsrc-safety-modem/single-chip-v2x-modem-for-dsrc-and-c-v2x:SAF5400)
- [NXP i.MX 8M Family](https://www.nxp.com/products/i.MX8M)
- [NXP OrangeBox Automotive Connectivity Domain Controller](https://www.nxp.com/design/design-center/development-boards-and-designs/ORANGEBOX)

---

## Summary

OBUs are real, resource-constrained embedded devices (ARM Cortex-A class, hundreds of MB RAM, no GPU) that **can** run lightweight IDS models — particularly with quantization and TinyML frameworks. The Cohda MK5 used in your thesis's dataset is based on an NXP i.MX6DL (dual Cortex-A9 @ 1 GHz), which is less powerful than a Raspberry Pi 4 but still supports LiteRT inference via NXP's eIQ framework. Your preliminary experiment with LiteRT on Raspberry Pi 4 validates the approach; the production OBU deployment will require further model compression to fit the tighter constraints.

---

## 6. Why IDS Models Must Be Deployed on OBUs — Not Offloaded to Powerful Remote Computers

A natural question arises: if OBUs are resource-constrained, why not run IDS models on more powerful computers — either a cloud server, a roadside edge server, or a powerful in-vehicle ADAS computer — and simply send OBU traffic there for analysis? There are several fundamental reasons why on-OBU deployment is necessary:

### 6.1 Latency Requirements for Safety-Critical Decisions

V2V safety messages (Basic Safety Messages / CAMs) arrive every 100 ms and require near-instant classification. Offloading to a remote server introduces network round-trip latency (typically 20–100+ ms over cellular, more under congestion), which can exceed the safety deadline before a response even arrives. On-device inference eliminates this network hop entirely, enabling sub-millisecond classification.

- [NHTSA: Vehicle-to-Vehicle Communications Readiness Report (PDF)](https://www.nhtsa.gov/staticfiles/rulemaking/pdf/V2V/Readiness-of-V2V-Technology-for-Application-812014.pdf)
- [IEEE 802.11p standard](https://en.wikipedia.org/wiki/IEEE_802.11p)

### 6.2 Connectivity Cannot Be Guaranteed

Vehicles operate in tunnels, rural areas, parking garages, and regions with poor or no cellular coverage. A cloud-dependent IDS fails silently exactly when the vehicle may be most vulnerable. An on-OBU IDS operates independently of network availability, ensuring continuous protection.

### 6.3 Privacy and Data Sovereignty

V2V messages contain location, speed, heading, and vehicle identity. Streaming this data to external servers raises serious privacy concerns under regulations like GDPR (EU) and CCPA (California). On-device inference keeps sensitive data local, avoiding regulatory and ethical issues.

- [GDPR and Connected Vehicles — European Data Protection Board](https://edpb.europa.eu/)
- [CCPA — California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa)

### 6.4 Scalability and Bandwidth Constraints

With millions of vehicles each generating 10 messages/second, a centralized IDS would need to process billions of messages per second. The bandwidth required to transmit all V2V traffic to remote servers would overwhelm both cellular networks and server infrastructure. Distributing IDS to each OBU scales naturally — each vehicle handles only its own traffic.

### 6.5 Cost of Infrastructure

Deploying and maintaining centralized IDS infrastructure (cloud servers, edge servers at every intersection, dedicated network links) adds enormous capital and operational expense. An on-OBU model has zero recurring infrastructure cost — the computation is embedded in hardware that is already deployed.

### 6.6 Attack Surface and Single Point of Failure

A centralized IDS server is itself a high-value attack target. Compromising it could blind the entire network's intrusion detection. Distributed on-OBU IDS eliminates this single point of failure — an attacker would need to compromise each vehicle individually.

### 6.7 ADAS Computers Are Not Available for V2X IDS

While modern vehicles increasingly ship with powerful ADAS processors (e.g., Mobileye EyeQ6 at ~34 TOPS, Nvidia DRIVE Orin at 254 TOPS, Qualcomm Snapdragon Ride), these are dedicated to safety-critical autonomous driving functions (camera processing, sensor fusion, path planning). They operate under strict functional safety standards (ISO 26262 ASIL-D) and cannot be shared with non-certified V2X IDS software without re-certification of the entire safety case. In practice, the OBU and the ADAS computer are separate, isolated subsystems.

- [ISO 26262 — Functional Safety for Road Vehicles](https://en.wikipedia.org/wiki/ISO_26262)
- [NXP S32G Vehicle Network Processors](https://www.nxp.com/products/processors-and-microcontrollers/s32-automotive-platform/s32g-vehicle-network-processors:S32G)

### 6.8 Universal Deployment Across All Vehicle Classes

V2X only delivers safety benefits if a critical mass of vehicles participate — including budget cars, motorcycles, commercial trucks, and legacy retrofits. Not all vehicles have powerful ADAS computers; many only have the OBU itself. IDS models must run on the lowest common denominator hardware to ensure network-wide coverage.

### 6.9 The Industry Trajectory Does Not Eliminate the Need

The automotive industry is converging toward centralized vehicle computers that merge ADAS, V2X, infotainment, and networking on shared platforms. However:
- This transition will take **10–20 years** given vehicle fleet turnover cycles (~15 years average vehicle lifespan).
- Even on shared platforms, IDS will compete for compute with many other functions — lightweight models remain advantageous.
- Moore's Law enables more efficient chips, but the efficiency gains are consumed by increasing V2X message rates, more complex attack patterns, and additional vehicle functions.

---

## 7. The Case for Lightweight Models — Efficiency as a Design Principle

The justification for lightweight IDS models on OBUs is not simply that OBUs are weak — it is that **efficiency is a fundamental requirement** in safety-critical, mass-deployment systems:

| Principle | Why It Matters |
|---|---|
| **Real-time latency** | A lightweight model classifying in 1 ms is *better* than a heavy model classifying in 50 ms, even if both fit in memory. V2V safety decisions are time-critical. |
| **Resource sharing** | Even powerful future OBUs will run V2X stack, routing, certificate management, and other functions concurrently. A lightweight IDS using 5% of CPU leaves 95% for everything else. |
| **Energy efficiency at scale** | Millions of vehicles × 24/7 operation = enormous aggregate energy cost. In EVs, every watt directly reduces driving range. A model using 0.5 W vs 5 W is a real engineering advantage. |
| **Universal coverage** | Lightweight models run on the cheapest viable hardware, enabling IDS on budget vehicles, motorcycles, and retrofit units — maximizing V2X network safety. |
| **Fleet transition period** | A solution deployed today must work on 2015-era hardware AND 2035-era hardware. Lightweight models bridge this 20-year gap. |

### Thesis Framing

> *While automotive computing power is increasing, lightweight ML models for V2V intrusion detection remain essential because (a) universal V2X deployment requires cost-effective, low-power solutions across all vehicle classes, (b) real-time safety constraints demand minimal inference latency regardless of available compute, (c) IDS must coexist with other vehicle functions in a shared-resource environment, and (d) on-device deployment is required due to latency, privacy, scalability, and reliability constraints that preclude offloading to remote servers.*

### Key Contributions

| Contribution | Why It Matters |
|---|---|
| Achieving high detection accuracy with small models | Proves heavy models are not *needed* |
| Measuring inference latency on edge hardware | Demonstrates real-time feasibility on OBU-class devices |
| Comparing lightweight vs. full-size models | Quantifies the efficiency–accuracy tradeoff |
| Deploying on representative hardware (e.g., Raspberry Pi 4) | Shows practical deployability on current OBU-class processors |

---

### Bottom Line

The research question is not *"Can we make IDS work on weak hardware?"* but rather: **"What is the minimum computational cost to achieve reliable intrusion detection in V2V networks, and how close can we get to full-model accuracy with orders-of-magnitude fewer resources?"** This question is relevant whether the target runs on a Cortex-A9 or a Cortex-A78 — because efficiency, latency, cost, energy, privacy, and reliability always matter in safety-critical, mass-deployment systems.