# Copilot Instructions for Thesis Repository

## High-Level Details

This repository contains research documentation for a Master's thesis on **"Lightweight Deep Learning Models for Intrusion Detection Systems in On-Board Units (OBUs)"** in vehicular networks.

### Repository Summary
- **Purpose**: Academic thesis research repository focusing on efficiency-driven machine learning models for V2V/V2X intrusion detection
- **Type**: Research/Documentation repository
- **Size**: Small repository primarily containing markdown documentation
- **Languages**: Markdown (documentation)
- **Primary Focus**: Lightweight ML models, OBU intrusion detection, V2V security, model compression techniques

### Key Research Context
The thesis argues that **efficiency is a design principle, not a compromise forced by weak hardware**. The research explores:
1. Lightweight deep learning architectures (CNNs, RNNs, LSTMs, BiLSTMs, Transformers)
2. Model compression techniques (quantization, pruning, knowledge distillation)
3. Real-time inference under V2V timing constraints (BSM interval: 100ms)
4. Deployment on resource-constrained OBU hardware
5. Efficiency-accuracy tradeoffs for universal V2X deployment

## Project Layout

### Repository Structure
```
/
├── .git/                           # Git version control
├── .github/                        # GitHub configuration (including this file)
│   └── copilot-instructions.md    # This file
├── OBU-RESTRAINS.md               # OBU device specifications and constraints
├── README.md                       # Repository introduction
└── Research-agent.md               # Research assistant system prompt and methodology
```

### Key Files

1. **README.md**: Brief repository title/introduction
2. **Research-agent.md**: Comprehensive system prompt for academic research assistant, including:
   - Thesis framing and research context
   - Research focus areas and constraints
   - Methodology for extracting and analyzing academic papers
   - Output formats and validation procedures
   - Critical metrics to extract from literature

3. **OBU-RESTRAINS.md**: Technical documentation on:
   - OBU hardware specifications (Cohda MK5, NXP RoadLINK, etc.)
   - Resource constraints and requirements
   - Comparison with automotive ADAS platforms (Tesla FSD, Nvidia DRIVE Orin)
   - Justification for lightweight model research

## Build, Test, and Validation

### Current State
- **No build process**: This is a documentation-only repository
- **No automated tests**: Research documentation repository
- **No CI/CD pipelines**: Not applicable for this repository type
- **No linting configured**: Documentation repository without code linting requirements

### Validation Steps
When making changes to this repository:

1. **Documentation Quality**:
   - Ensure markdown files are properly formatted
   - Verify all links (URLs, DOIs) are valid and accessible
   - Maintain consistent formatting across all documentation files
   - Preserve the structured format in Research-agent.md (XML tags, tables)

2. **Content Accuracy**:
   - All citations must include valid DOIs or URLs
   - Hardware specifications must be sourced from official documentation
   - Maintain the thesis framing consistency (efficiency as design principle)
   - Verify technical claims against authoritative sources

3. **File Integrity**:
   - Do not modify the core research methodology in Research-agent.md unless explicitly requested
   - Preserve existing table structures and formatting
   - Maintain the XML output format specifications

## Key Facts and Constraints

### Research Constraints
**Never** (from Research-agent.md constraints):
- Include papers without verified DOIs
- Frame OBU limitations as purely about "weak hardware"
- Mix training metrics with inference metrics
- Make assumptions about hardware specifications if not stated
- Omit resource constraint information when analyzing OBU deployments
- State facts without trusted source citations

### Critical Metrics Priority
When adding research content, prioritize in this order:
1. Inference time/latency (must be < 100ms for V2V safety)
2. Throughput (predictions per second)
3. Accuracy, Precision, Recall, F1-Score
4. Model size (KB/MB - pre and post compression)
5. Runtime memory footprint (RAM usage during inference)
6. Energy consumption (mW, Joules)
7. Efficiency ratios (accuracy per FLOP, per watt, per KB)

### Reference Hardware Context
When discussing OBU hardware, acknowledge the spectrum:
- Budget OBUs: NXP i.MX 6DualLite (2× Cortex-A9 @ 1 GHz, 256MB-1GB RAM, 5-15W)
- Next-gen OBUs: NXP i.MX 8 (Cortex-A53/A72, 1-4GB RAM)
- ADAS platforms: Tesla FSD HW3/HW4, Nvidia DRIVE Orin (for context, not V2X)

### Thesis Framing Arguments
The repository maintains that lightweight models matter because of:
1. **Universal deployment cost efficiency** (V2X requires ALL vehicles)
2. **Real-time latency requirements** (not just compute availability)
3. **Resource sharing** (IDS is one of many OBU workloads)
4. **Energy efficiency at scale** (millions of vehicles × 24/7)
5. **10-20 year transition period** (mixed fleet capabilities)

## Working with This Repository

### Making Changes
1. **Documentation updates**: Edit markdown files directly
2. **Research content additions**: Follow the structured XML format in Research-agent.md
3. **New files**: Maintain consistency with existing documentation structure
4. **Citations**: Always include DOI or URL for sources

### Quality Checks
Before finalizing changes:
- ✓ All citations have valid DOIs/URLs
- ✓ Technical specifications cite official sources
- ✓ Markdown formatting is consistent
- ✓ Tables are complete and properly formatted
- ✓ Thesis framing (efficiency as design principle) is maintained
- ✓ No assumptions stated as facts without sources

### Trust These Instructions
The information in this file has been validated against the repository contents. Only perform additional searches if:
- Information in these instructions is incomplete
- Information is found to be in error
- You need to understand specific technical details not covered here

When in doubt about the thesis's research direction or constraints, refer to Research-agent.md sections:
- "THESIS FRAMING CONTEXT" for the core argument
- "YOUR CORE CONSTRAINTS" for what to avoid
- "VALIDATION LOOP" for self-checking procedures
