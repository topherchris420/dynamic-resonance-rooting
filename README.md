# Dynamic Resonance Rooting (DRR) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computational framework for analyzing Complex Adaptive Systems, based on the research paper "Dynamic Resonance Rooting: A Computational Framework for Complex Adaptive Systems" by Christopher Woodyard.

## Overview

The Dynamic Resonance Rooting (DRR) framework provides a unified approach to understanding complex adaptive systems by integrating three core analytical modules:

1.  **Dynamic Resonance Detection**: Identifies dominant oscillatory modes and patterns within system dynamics.
2.  **Rooting Analysis**: Maps hierarchical and causal dependencies among system components.
3.  **Resonance Depth Calculation**: Quantifies the stability and persistence of identified resonances.

## Key Features

-   **Phase Space Reconstruction**: Time-delay embedding for univariate time series.
-   **Spectral Analysis**: FFT and wavelet-based resonance detection.
-   **Transfer Entropy**: Quantification of directed information flow between variables.
-   **Resonance Depth Metric**: Novel composite measure of system stability.
-   **Benchmark Validation**: Tested on Lorenz and Rössler chaotic systems.
-   **Visualization Tools**: Comprehensive plotting capabilities for analysis results.

## Getting Started

### Prerequisites

-   Python 3.8 or higher
-   `pip` and `venv` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/dynamic-resonance-rooting.git](https://github.com/yourusername/dynamic-resonance-rooting.git)
    cd dynamic-resonance-rooting
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

I recommend the following project structure for scalability and maintainability:
