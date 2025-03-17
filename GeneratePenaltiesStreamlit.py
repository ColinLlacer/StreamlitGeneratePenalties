"""
Penalty Function Generator Streamlit Application

This application allows users to generate and visualize penalty functions based on 
the Exponential Power distribution. Users can adjust parameters to customize the 
penalty function and export the results to Excel.

The application uses an object-oriented approach to organize the code into logical components:
- PenaltyDistribution: Handles the mathematical model of the penalty function
- PenaltyCalculator: Processes the distribution to calculate actual penalties
- PenaltyApp: Manages the Streamlit interface and user interactions
"""

import logging
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.special import gamma


# Configuration parameters
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_FORBIDDEN_RANGE = [(40, 60)]

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format=LOGGING_FORMAT
)


class PenaltyDistribution:
    """
    Represents the mathematical distribution used to model penalty functions.
    Currently implements the Exponential Power distribution (Generalized Normal Distribution).
    """

    def __init__(self, mu, sigma, beta):
        """
        Initializes the distribution with given parameters.

        Args:
            mu (float): Location parameter (mean).
            sigma (float): Scale parameter (standard deviation).
            beta (float): Shape parameter controlling the tail behavior.
        """
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

    def pdf(self, x):
        """
        Calculates the probability density function (PDF) at given point(s).

        Args:
            x (float or np.ndarray): Point(s) at which to evaluate the PDF.

        Returns:
            float or np.ndarray: PDF value(s) at x.
        """
        z = np.abs(x - self.mu) / self.sigma
        return (self.beta / (2 * self.sigma * gamma(1 / self.beta))) * np.exp(
            -(z**self.beta)
        )


class PenaltyCalculator:
    """
    Handles the calculation of penalties based on a distribution and parameters.
    """
    
    def __init__(self, distribution, penalty_scale, forbidden_ranges):
        """
        Initializes the calculator with a distribution and parameters.
        
        Args:
            distribution (PenaltyDistribution): The distribution to use for calculations
            penalty_scale (float): Scaling factor for penalties
            forbidden_ranges (list): List of (start, end) tuples representing forbidden hour ranges
        """
        self.distribution = distribution
        self.penalty_scale = penalty_scale
        self.forbidden_ranges = forbidden_ranges
    
    def calculate_penalties(self, min_hours, max_hours, step=0.5):
        """
        Calculates penalties for a range of hours.
        
        Args:
            min_hours (float): Minimum hours to consider
            max_hours (float): Maximum hours to consider
            step (float): Increment between hour values
            
        Returns:
            tuple: (DataFrame with hours and penalties, Plotly figure)
        """
        try:
            # Generate smooth curve for plotting
            x_plot = np.linspace(0, 60, 1000)
            plot_data = self._process_values(x_plot)
            
            # Create visualization
            fig = self._create_plot(x_plot, plot_data)
            
            # Calculate penalties for specific hour increments
            hours = np.arange(min_hours, max_hours + step, step)
            hour_data = self._process_values(hours)
            
            # Create DataFrame with results
            result_df = pd.DataFrame({"Hours": hours, "Penalties": hour_data})
            
            logging.info("Penalties calculated successfully.")
            return result_df, fig
            
        except Exception as e:
            logging.error(f"Error calculating penalties: {e}")
            return pd.DataFrame(), None
    
    def _process_values(self, values):
        """
        Process values through the distribution and apply scaling and forbidden ranges.
        
        Args:
            values (np.ndarray): Array of values to process
            
        Returns:
            np.ndarray: Processed penalty values
        """
        # Calculate raw penalties
        penalties = 1 - self.distribution.pdf(values)
        
        # Standardize and scale
        penalties_standardized = (penalties - np.mean(penalties)) / np.std(penalties)
        penalties_min_zero = penalties_standardized - np.min(penalties_standardized)
        penalties_scaled = penalties_min_zero * self.penalty_scale
        
        # Apply forbidden ranges
        max_penalty = np.max(penalties_scaled)
        for start, end in self.forbidden_ranges:
            penalties_scaled = np.where(
                (values >= start) & (values <= end), max_penalty, penalties_scaled
            )
            
        return penalties_scaled
    
    def _create_plot(self, x_values, y_values):
        """
        Creates a plot visualization of the penalty function.
        
        Args:
            x_values (np.ndarray): X-axis values (hours)
            y_values (np.ndarray): Y-axis values (penalties)
            
        Returns:
            plotly.graph_objects.Figure: The plot figure
        """
        fig = px.line(
            x=x_values,
            y=y_values,
            title="Penalty Function",
            labels={"x": "Hours", "y": "Penalty"},
            width=400,
            height=300
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=self.distribution.mu, line_dash="dash", line_color="gray")
        return fig


class PenaltyApp:
    """
    Manages the Streamlit interface and user interactions for the penalty function generator.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the Streamlit user interface."""
        st.title("Penalty Function Generator")
        
        # Define columns for layout
        self.input_col, self.output_col = st.columns(2, gap="large")
        
        with self.input_col:
            self._setup_input_controls()
    
    def _setup_input_controls(self):
        """Set up the input controls in the sidebar."""
        st.header("Input Parameters")
        
        # Distribution parameters
        self.mean = st.slider("Mean (mu)", min_value=-10.0, max_value=60.0, value=30.0, step=0.5)
        self.scale = st.slider("Scale (sigma)", min_value=0.10, max_value=50.0, value=10.0, step=0.1)
        self.beta = st.slider("Beta (> 2 for thinner tails than Gaussian)", min_value=0.10, max_value=20.0, value=4.0, step=0.1)
        
        # Penalty parameters
        self.penalty_scale = st.slider("Penalty Scale", min_value=0, max_value=20000, value=10000, step=100)
        self.min_hours = st.number_input("Minimum Hours", min_value=0.0, max_value=80.0, value=0.0, step=0.5)
        self.max_hours = st.number_input("Maximum Hours", min_value=0.0, max_value=80.0, value=60.0, step=0.5)
        
        # Forbidden ranges input with validation
        forbidden_ranges_input = st.text_input(
            "Forbidden Ranges (as list of tuples, e.g., [(40, 60), (70, 75)])", 
            value="[(40, 60)]"
        )

        try:
            self.forbidden_ranges = eval(forbidden_ranges_input)
            if not isinstance(self.forbidden_ranges, list) or not all(isinstance(r, tuple) and len(r) == 2 for r in self.forbidden_ranges):
                self.forbidden_ranges = DEFAULT_FORBIDDEN_RANGE
                st.warning(f"Invalid forbidden ranges format. Using default {DEFAULT_FORBIDDEN_RANGE}. Please use a list of tuples.")
        except:
            self.forbidden_ranges = DEFAULT_FORBIDDEN_RANGE
            st.warning(f"Invalid forbidden ranges format. Using default {DEFAULT_FORBIDDEN_RANGE}. Please use a list of tuples.")

        self.generate_button = st.button("Generate Penalties")
        
        if self.generate_button:
            self.generate_penalties()
    
    def generate_penalties(self):
        """Generate penalties based on user inputs and display results."""
        try:
            # Create distribution and calculator
            distribution = PenaltyDistribution(self.mean, self.scale, self.beta)
            calculator = PenaltyCalculator(distribution, self.penalty_scale, self.forbidden_ranges)
            
            # Calculate penalties
            penalty_df, penalty_fig = calculator.calculate_penalties(self.min_hours, self.max_hours)
            
            if not penalty_df.empty and penalty_fig is not None:
                self._display_results(penalty_df, penalty_fig)
                
        except Exception as e:
            logging.error(f"Error generating penalties: {e}")
            st.error(f"Error generating penalties: {e}")
    
    def _display_results(self, penalty_df, penalty_fig):
        """
        Display results in the output column.
        
        Args:
            penalty_df (pd.DataFrame): DataFrame with penalty data
            penalty_fig (plotly.graph_objects.Figure): Plot figure
        """
        with self.output_col:
            # Display results
            st.plotly_chart(penalty_fig)
            st.dataframe(penalty_df)
            
            # Export to Excel
            self._setup_excel_export(penalty_df)
    
    def _setup_excel_export(self, df):
        """
        Set up Excel export functionality.
        
        Args:
            df (pd.DataFrame): DataFrame to export
        """
        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Penalties', index=False)
            excel_buffer.seek(0)

            st.download_button(
                label="Download Excel File",
                data=excel_buffer,
                file_name="penalties.xlsx",
                mime="application/vnd.ms-excel",
            )
        except Exception as e:
            logging.error(f"Error exporting to Excel: {e}")
            st.error(f"Error exporting to Excel: {e}")


# Run the application
if __name__ == "__main__":
    app = PenaltyApp()
