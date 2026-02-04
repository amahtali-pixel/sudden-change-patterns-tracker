# Beyond the Black Box: A Non-Neural, Evidence-Based Framework with Complete Decision Transparency

**Apache 2.0 License Implementation Code** | **Patent-Protected Methodology**

**📢 Important Notice:** This repository has been renamed and expanded from "Rotary Pattern Extraction Network (RPENet)" to reflect the broader **Radial Encoding Framework** methodology presented in our 2026 publication.

## 🔬 Research Overview

This repository contains the official implementation of the **Radial Encoding Framework**, a novel **non-neural, geometry-based approach** that achieves **99.5% accuracy** on MNIST while providing **complete decision transparency**. Our work challenges the entrenched belief that high accuracy necessitates opaque, black-box models.

## 🏆 Key Contributions

### **Geometric Foundation**
- **Radial Encoding Methodology**: Deterministic geometric operations bridging continuous spatial reality with discrete computation
- **Three Complementary Strategies**: Local (LRE), Hierarchical (HRE), and Extended (ERE) radial encoding
- **Contrapositive Geometric Validation**: Objects must prove identity through both presence of class-specific patterns *and* absence of disqualifying patterns

### **Systematic Optimization**
- **Optimal Tolerances**: $\tau_c^* = 0.12$ (clustering), $\tau_v^* = 0.22$ (validation) through comprehensive analysis
- **Digit Complexity Metric**: Quantifies geometric complexity (e.g., digit 7 requires 75% more prototypes than digit 1)
- **Proper Confidence Calibration**: Errors show **slightly lower confidence** than correct predictions (0.98× ratio)

### **Competitive Performance with Full Transparency**
- **99.5% accuracy** on full MNIST test set (9,950/10,000)
- **Within 0.295%** of state-of-the-art neural networks (99.795%)
- **Complete traceability**: Every decision references specific geometric patterns, spatial locations, and training examples

## 📊 Three Architecture Variants

### 1. **Three-Stage Validation Model** (Primary - 99.5% accuracy)
- **Stage 1**: Local Radial Encoding (LRE) classification
- **Stage 2**: Extended Radial Encoding (ERE) arbitration for ambiguous cases
- **Stage 3**: Pixel density analysis for rare persistent ambiguities

### 2. **One-Stage Model with Logarithmic Weighting** (96.74% accuracy)
- Addresses systematic voting biases through log(frequency + 1) weighting
- Eliminates cluster distribution imbalances (8:1 ratio reduced from 3.15× to 1.10×)

### 3. **Single-Stage Hierarchical Encoding** (99.2% accuracy on digits 0,1,7)
- Demonstrates HRE's standalone descriptive power
- Minimalist architecture maintaining complete interpretability

## ⚖️ Legal & Licensing Information

### 📄 Code License
The implementation code in this repository is released under the **Apache License 2.0**, permitting:

✅ Commercial and non-commercial use  
✅ Modification and distribution  
✅ Express patent grant for users of this implementation  
✅ **Requirement**: Attribution and license preservation

### 🔒 Patent Protection
The underlying **Radial Encoding Framework** methodology represents novel research and is protected under applicable patent laws.

**Protected Methodology**: Radial Encoding Framework  
**Patent Application**: Algerian Patent Application DZ/P/2025/1546  
**Filing Date**: November 5, 2025 (Pending Examination)  
**Rights**: All commercial rights reserved. © 2026

**For commercial licensing inquiries**, please contact the author.

## 🏗️ Architectural Highlights

### **Knowledge Base Construction**
- **7.5 million geometric patterns** library
- **Complete metadata integration**: Spatial coordinates, zone assignments, training references
- **Interpretable clusters**: Each represents a distinct geometric prototype with quantitative descriptors

### **Systematic Tolerance Optimization**
Comprehensive exploration across three orders of magnitude:
- **Precision regime** ($\tau_c < 0.12$): Excessive differentiation, limited by hardware
- **Optimal regime** ($0.10 < \tau_c < 0.15$): Balanced trade-off (8.5–10.5× compression, 88–90% info preservation)
- **Generalization regime** ($\tau_c > 0.20$): Rapid information loss with marginal gains

### **Computational Efficiency**
- **80% data reduction** through sudden change detection (edge-based processing)
- **Linear scaling** with contour pixels (not quadratic with image size)
- **94.3% cases resolved in Stage 1**, only 5.7% require arbitration

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/amahtali-pixel/radial-encoding-framework
cd radial-encoding-framework
