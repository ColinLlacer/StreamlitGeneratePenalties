# Penalty Function Generator

## Overview
This Streamlit application generates and visualizes penalty functions based on the Exponential Power distribution (Generalized Normal Distribution). Users can adjust various parameters to customize the penalty function and export the results to Excel.

## Features
- Interactive parameter adjustment for the distribution (mean, scale, beta)
- Customizable penalty scaling
- Definition of forbidden hour ranges with higher penalties
- Visualization of the penalty function curve
- Export of penalty values to Excel

## Mathematical Model
The application uses the Exponential Power distribution with the following parameters:
- μ (mu): Location parameter (mean)
- σ (sigma): Scale parameter (standard deviation)
- β (beta): Shape parameter controlling the tail behavior

The probability density function (PDF) is defined as:

f(x; μ, σ, β) = (β / (2σΓ(1/β))) * exp(-(|x-μ|/σ)^β)

Where:
- Γ is the gamma function
- β > 0 controls the shape (higher values create thinner tails than the normal distribution)
- When β = 2, this becomes the normal distribution
- When β = 1, this becomes the Laplace distribution

## Penalty Calculation
The penalty function is derived from the PDF as follows:
1. Calculate the raw penalty as `1 - PDF(x)`
2. Standardize the penalties
3. Shift to ensure minimum value is zero
4. Apply scaling factor to adjust the magnitude
5. Apply maximum penalty to any values within forbidden ranges

## Usage
1. Adjust the distribution parameters (mean, scale, beta)
2. Set the penalty scale factor
3. Define minimum and maximum hours to consider
4. Specify forbidden ranges as a list of tuples (e.g., `[(40, 60), (70, 75)]`)
5. Click "Generate Penalties" to calculate and visualize results
6. Download the results as an Excel file if needed

## Code Structure
The application uses an object-oriented approach with three main classes:
- `PenaltyDistribution`: Handles the mathematical model
- `PenaltyCalculator`: Processes the distribution to calculate penalties
- `PenaltyApp`: Manages the Streamlit interface and user interactions

## Requirements
- Python 3.6+
- Streamlit
- NumPy
- Pandas
- Plotly
- SciPy
