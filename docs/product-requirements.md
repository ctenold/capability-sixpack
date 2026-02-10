# Product Requirements Document: Python Minitab Analysis Sixpack

## 1. Introduction

This Product Requirements Document (PRD) outlines the specifications for developing a Python-based recreation of the Minitab Analysis Sixpack. The Minitab Analysis Sixpack is a comprehensive set of statistical tools used in quality and process improvement, particularly within Six Sigma methodologies, to assess process stability and capability. This Python implementation aims to provide similar functionality, enabling users to perform in-depth process analysis using open-source tools and programming paradigms.

## 2. Goals and Objectives

The primary goal is to replicate the core analytical and visualization capabilities of the Minitab Analysis Sixpack in Python. Key objectives include:

*   **Accurate Statistical Replication**: Ensure that all statistical calculations, including control limits, capability indices (Cp, Cpk, Pp, Ppk, Cpm), and normality tests (Anderson-Darling), are accurately reproduced according to established statistical methodologies and Minitab's documented formulas.
*   **Comprehensive Visualization**: Generate a set of six interconnected plots that visually represent process stability and capability, mirroring the layout and information presented in Minitab's output.
*   **User-Friendly Interface (API)**: Develop a clear and intuitive Python API that allows users to easily input data, specify parameters, and generate the Analysis Sixpack report.
*   **Extensibility**: Design the architecture to be modular, allowing for future enhancements such as support for non-normal distributions, additional control chart types, and integration with other data analysis workflows.

## 3. Key Features and Components

The Python Minitab Analysis Sixpack will consist of a statistical engine and a visualization module, working in conjunction to produce a comprehensive report. The core components are detailed below:

### 3.1. Statistical Engine

The statistical engine will handle data processing, calculation of control limits, process capability indices, and normality testing. It will support both individual observations and subgrouped data.

#### 3.1.1. Data Handling

*   **Input Data**: Accept tabular data, typically in a pandas DataFrame format, with columns for measurements and optional subgroup identifiers.
*   **Subgrouping**: Automatically detect or allow user specification of subgroup sizes for appropriate statistical calculations.

#### 3.1.2. Variation Estimation

Accurate estimation of process variation is crucial for capability analysis. The system will implement methods for both within-subgroup and overall standard deviation [1].

*   **Within-Subgroup Standard Deviation (σ_within)**: This estimates the short-term variation within a process. Supported methods will include:
    *   **Pooled Standard Deviation**: Calculated from the sum of squared deviations within each subgroup.
    *   **Average of Subgroup Ranges (R-bar)**: Used when subgroup sizes are small (typically 2-8). Requires unbiasing constants ($d_2$).
    *   **Average of Subgroup Standard Deviations (S-bar)**: Used when subgroup sizes are larger (typically 9 or more). Requires unbiasing constants ($c_4$).
*   **Overall Standard Deviation (σ_overall)**: This estimates the long-term variation of the entire process. It is typically calculated as the sample standard deviation of all observations, with an optional unbiasing constant ($c_4$) [1].

#### 3.1.3. Normality Testing

*   **Anderson-Darling Test**: This statistical test will be used to assess whether the process data follows a normal distribution. The output will include the Anderson-Darling statistic and the p-value, crucial for validating the assumptions of normal capability analysis [2].

#### 3.1.4. Process Capability Indices

These indices quantify the ability of a process to meet specified requirements. The following indices will be calculated:

*   **Potential Capability (Cp, Cpk)**: These indices measure the potential capability of the process, assuming the process is perfectly centered and stable. They are based on the within-subgroup standard deviation [3].
    *   **Cp (Process Capability)**: Measures the potential process spread relative to the specification spread.
    *   **Cpk (Process Capability Index)**: Accounts for the process mean's location relative to the specification limits, indicating how close the process is to the nearest specification limit.
*   **Overall Capability (Pp, Ppk, Cpm)**: These indices measure the actual performance of the process over time, considering both within-subgroup and between-subgroup variation. They are based on the overall standard deviation [4].
    *   **Pp (Process Performance)**: Similar to Cp but uses the overall standard deviation.
    *   **Ppk (Process Performance Index)**: Similar to Cpk but uses the overall standard deviation.
    *   **Cpm (Taguchi Capability Index)**: Measures process capability relative to a target value, penalizing for deviation from the target.

### 3.2. Visualization Module

The visualization module will generate a 2x3 grid of plots, replicating the Minitab Analysis Sixpack display. This will primarily leverage `matplotlib` for static reports, with potential for `plotly` integration for interactive web-based dashboards.

*   **Control Chart (Location)**: Either an Xbar chart (for subgrouped data) or an I (Individual) chart (for individual observations). These charts monitor the process mean over time, displaying control limits (UCL, CL, LCL) at ±3 standard deviations [5].
*   **Control Chart (Variation)**: Depending on subgroup size, this will be an R chart (for subgroup ranges, small subgroups), an S chart (for subgroup standard deviations, larger subgroups), or a Moving Range (MR) chart (for individual observations). These charts monitor process variation [5].
*   **Last 25 Subgroups/Observations Plot**: Displays the data points for the most recent subgroups or observations, along with the overall process mean, to assess short-term trends and stability [5].
*   **Capability Histogram**: A histogram of the process data, overlaid with normal distribution curves (one based on within-subgroup variation and another on overall variation), and showing the Lower Specification Limit (LSL) and Upper Specification Limit (USL) [5].
*   **Normal Probability Plot**: A quantile-quantile (Q-Q) plot that visually assesses the normality of the data. It includes a fitted line and confidence bounds, along with the Anderson-Darling test results [5].
*   **Capability Plot**: A visual representation comparing the potential process spread (within interval), actual process spread (overall interval), and the specification interval (LSL to USL). It also indicates the process center and target (if applicable) [5].

## 4. Technical Stack

*   **Programming Language**: Python 3.x
*   **Data Manipulation**: `pandas`, `numpy`
*   **Statistical Functions**: `scipy.stats`, `statsmodels`
*   **Plotting**: `matplotlib` (for static plots), `plotly` (for interactive plots, optional)
*   **Reporting**: Markdown for documentation, potentially `fpdf2` or `weasyprint` for PDF report generation.

## 5. Future Considerations

*   **Non-Normal Capability Analysis**: Extend the functionality to handle non-normal distributions using transformations (e.g., Box-Cox, Johnson) or non-parametric methods.
*   **Interactive Dashboards**: Develop interactive web-based dashboards using `plotly` and `dash` for dynamic visualization and exploration.
*   **Integration with Data Pipelines**: Provide utilities for seamless integration with common data ingestion and processing pipelines.
*   **Advanced Control Charts**: Implement additional control chart types (e.g., C chart, U chart, P chart, NP chart) for attribute data.

## 6. References

[1] Minitab Support. (n.d.). *Methods and formulas for methods used in Normal Capability Analysis*. Retrieved from https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-analysis/normal-capability-analysis/methods-and-formulas/methods/

[2] Minitab Support. (n.d.). *The Anderson-Darling statistic*. Retrieved from https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/basic-statistics/supporting-topics/normality/the-anderson-darling-statistic/

[3] Minitab Support. (n.d.). *Methods and formulas for potential capability measures in Normal Capability Analysis*. Retrieved from https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-analysis/normal-capability-analysis/methods-and-formulas/potential-capability/

[4] Minitab Support. (n.d.). *Methods and formulas for overall capability measures in Normal Capability Analysis*. Retrieved from https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-analysis/normal-capability-analysis/methods-and-formulas/overall-capability/

[5] Minitab Support. (n.d.). *Graphs for Normal Capability Sixpack*. Retrieved from https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-sixpack/normal-capability-sixpack/interpret-the-results/all-statistics-and-graphs/graphs/
