# Classroom Flu Simulation

## Overview
This project simulates the spread of flu in an elementary school classroom with 61 kids, starting with one infected child. The simulation analyzes infection patterns both without immunization and with 50% immunization probability, providing statistical analysis and visualizations of the results.

## Contents
- `simulation_code.py`: Main Python script containing simulation logic and analysis
- `viz/`: Directory containing all generated visualizations
- `requirements.txt`: List of required Python packages

## Requirements
The simulation requires Python and the following packages:
- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Setup and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/k-rudee/classroom-flu-simulation.git
cd classroom-flu-simulation
```

### 2. Create a Virtual Environment
```bash
python -m venv sim
```

### 3. Activate the Virtual Environment
Windows:
```bash
sim\Scripts\activate
```

macOS/Linux:
```bash
source sim/bin/activate
```

### 4. Install Required Packages
```bash
pip install -r requirements.txt
```

### 5. Run the Simulation
```bash
python simulation_code.py
```

### 6. View Outputs
- Console Output: Statistical findings will be printed to the terminal
- Visualizations: All plots are saved in the `viz/` directory

## Simulation Description

The simulation analyzes several scenarios:

### Part A
- Calculates the distribution of initial infections from patient zero (Tommy) on Day 1
- Visualizes results using Binomial distribution

### Part B
- Computes expected number of infections from Tommy on Day 1

### Part C
- Estimates total expected infections by Day 2, including Tommy

### Part D
- Simulates epidemic progression without immunization
- Tracks daily new infections and epidemic duration

### Part E
- Simulates epidemic with 50% immunization probability
- Compares results with non-immunized scenario

## Visualizations

The `viz/` folder contains the following plots:
- `day1_infection_distribution.png`: Distribution of Day 1 infections
- `expected_infections_by_day.png`: Daily new infections without immunization
- `epidemic_durations.png`: Epidemic duration distribution without immunization
- `expected_infections_by_day_immunization.png`: Daily new infections with immunization
- `epidemic_durations_immunization.png`: Epidemic duration distribution with immunization
- `comparison_total_infections.png`: Total infections comparison
- `comparison_epidemic_durations.png`: Epidemic duration comparison

## Project Structure
```
classroom-flu-simulation/
│
├── simulation_code.py      # Main simulation script
├── requirements.txt        # Required Python packages
├── README.md              # Project documentation
└── viz/                   # Visualization directory
    ├── day1_infection_distribution.png
    ├── expected_infections_by_day.png
    ├── epidemic_durations.png
    ├── expected_infections_by_day_immunization.png
    ├── epidemic_durations_immunization.png
    ├── comparison_total_infections.png
    └── comparison_epidemic_durations.png
