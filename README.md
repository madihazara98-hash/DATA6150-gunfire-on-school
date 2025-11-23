# Gunfire on School Grounds in the United States (1966–2025)

This repository contains the code, data, notebook, and report for a data science project analyzing gunfire incidents on or near K–12 school grounds in the United States using the K–12 School Shooting Database (SSDB).

The project includes a full data science workflow: data loading, cleaning, exploratory analysis, temporal modeling using scikit-learn, spatial aggregation, and visualization.

---

## Project Overview

This project addresses two primary research questions:

1. **Temporal Trends:**  
   How has the frequency of school gunfire incidents changed over time in the United States?

2. **Spatial Patterns:**  
   Which states have experienced the highest number of incidents, and how do these trends differ regionally?

To answer these questions, the analysis includes:

- Annual aggregation of incident counts (1966–2025)
- Fitting a **Linear Regression** model using scikit-learn to quantify long-term temporal trends
- Aggregation of incident counts at the **state level**
- Visualizations for both temporal and geographic results

---

## Repository Structure

DATA6150-gunfire-on-school/

├─ Datasets/
│ ├─ 3b_Dataset_Incident.xlsx # Original dataset (Incident sheet only)/
│ ├─ yearly_incidents_sklearn_from_notebook.csv # Yearly incidents + model predictions (from notebook)/
│ └─ state_incidents_sklearn_from_notebook.csv # State-level incident counts (from notebook)/


├─ Codes/
│ ├─ incident_analysis.py # Standalone Python script (user chooses file paths)/
│ └─ 0a_incident_analysis_optionA.ipynb # Full Jupyter analysis notebook/

├─ Pictures/
│ ├─ yearly_trend.png # Trend plot (actual vs fitted)/
│ └─ state_top15_incidents.png # Bar chart of top 15 states/

├─ Report/
│ ├─ IncidentAnalysis_Report.docx # Full written report (Word)/
│ └─ IncidentAnalysis_Report.pdf # Final PDF report/

└─ README.md


---

## Methods Used

### **Data Processing**
- Loading Excel files: **pandas**
- Cleaning and preparing date fields
- Extracting year and grouping by time and state

### **Model Development**
- Yearly incident trend model using:
  - `sklearn.linear_model.LinearRegression`
  - Metric: **R²**

### **Visualizations**
- Line plot of incidents per year + fitted regression line
- Horizontal bar chart of top 15 states by total incident count  
- Library: **matplotlib**

### **Outputs Generated**
The script and notebook generate:

- `yearly_incidents_sklearn.csv`  
- `state_incidents_sklearn.csv`  
- `yearly_trend.png`  
- `state_top_incidents.png`

---

## Running the Standalone Script (`incident_analysis.py`)

The script now supports **user-specified input and output paths**.

1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn

2. Run the script:
	python Codes/incident_analysis.py

3. When prompted:
	Enter full path to 3b_Dataset_Incident.xlsx
	Enter an output directory where CSVs and plots should be saved

Example execution:
Enter full path to the Incident Excel file:
D:\MLProject\__Individualsheet_dataset\0a_Incident_Analysis\3b_Dataset_Incident.xlsx

Enter output folder path for CSVs and plots:
D:\MLProject\__Individualsheet_dataset\0a_Incident_Analysis\

Running the Jupyter Notebook

Open the notebook: Codes/0a_incident_analysis_optionA.ipynb

Inside the notebook:
	Update excel_path and output_dir variables (top cell)
	Run all cells to:
		Load the data
		Perform analysis
		Train the regression model
		Generate CSV files
		Generate visualizations
		Display results and interpretations
		

Data Source: K–12 School Shooting Database (SSDB), Public v4.1.
The dataset documents any incident in which a gun is fired, brandished, or a bullet hits school property.

Author
	Madiha Zara
	DATA 6150 – Data Science Project
	Wentworth Institute of Technology
	Fall 2025
