# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
from scipy.stats import ttest_ind, ks_2samp

# Set random seed for reproducibility
np.random.seed(42)

# Create viz directory for visulziations to be stored 
os.makedirs('viz', exist_ok=True)

# Parameters
num_kids = 61  # Total num of kids including Tommy
num_susceptible_kids = 60  # Num of kids excluding Tommy
infection_probability = 0.01  # Prob of infection per interaction per day
days_infectious = 3  # Num of days a kid is infectious
num_simulations = 1000  # Num of sims to run
immunization_probability = 0.5  # Prob that a kid is immunized

# ----------
# Part a
# ----------

# The num of kids Tommy infects on Day 1 follows a Binomial dist
n = num_susceptible_kids  
p = infection_probability  

# Generate the prob mass function of the binomial distribution
k = np.arange(0, n+1)
pmf = binom.pmf(k, n, p)

# Calculate mean and variance
mean_infections = n * p
var_infections = n * p * (1 - p)

print(f"The expected number of infections (mean) is: {mean_infections:.2f}")
print(f"The variance of the number of infections is: {var_infections:.2f}")

# Plot the distribution
plt.figure(figsize=(10,6))
plt.bar(k, pmf, color='skyblue', edgecolor='black')
plt.title('Distribution of the Number of Kids Tommy Infects on Day 1')
plt.xlabel('Number of Kids Infected')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('viz/day1_infection_distribution.png')
plt.close()

# ---------
# Part b
# ---------

expected_infections_day1 = n * p

# ---------
# Part c
# ---------

def simulate_up_to_day2(num_kids, infection_probability, days_infectious):
    """Simulate the spread of infection up to Day 2."""
    infected_days = np.zeros(num_kids)
    infected_days[0] = days_infectious  
    total_infected = np.zeros(num_kids)
    total_infected[0] = 1 

    # Day 1: Tommy infects others
    for kid in range(1, num_kids):
        if np.random.rand() < infection_probability:
            infected_days[kid] = days_infectious
            total_infected[kid] = 1

    infected_days = np.clip(infected_days - 1, 0, None)  # Decrease infection days

    # Day 2: Newly infected kids from Day 1 infect others
    new_infections_day2 = np.zeros(num_kids)
    for kid in range(num_kids):
        if infected_days[kid] > 0:  # Infectious on Day 2
            for other_kid in range(num_kids):
                if kid != other_kid and infected_days[other_kid] == 0:
                    if np.random.rand() < infection_probability:
                        new_infections_day2[other_kid] = days_infectious
                        total_infected[other_kid] = 1
    infected_days += new_infections_day2
    infected_days = np.clip(infected_days - 1, 0, None)

    total_infected_count = np.sum(total_infected)
    return total_infected_count

# Sim 1000 times 
num_simulations_c = 10000
total_infected_counts = []

for _ in range(num_simulations_c):
    total_infected_count = simulate_up_to_day2(num_kids, infection_probability, days_infectious)
    total_infected_counts.append(total_infected_count)

expected_infected_by_day2 = np.mean(total_infected_counts)
std_infected_by_day2 = np.std(total_infected_counts)

print(f"Expected number of kids infected by Day 2 (including Tommy): {expected_infected_by_day2:.2f}")
print(f"Standard deviation: {std_infected_by_day2:.2f}")

# ---------
# Part d
# ---------

def simulate_epidemic(num_kids, infection_probability, days_infectious):
    """Simulate the epidemic over time until no more infections."""
    infected_days = np.zeros(num_kids)
    infected_days[0] = days_infectious  # Tommy is initially infected
    total_infected = np.zeros(num_kids)
    total_infected[0] = 1
    daily_new_infections = []
    day = 0

    while np.any(infected_days > 0):
        new_infections = np.zeros(num_kids)
        day_new_infections = 0

        for kid in range(num_kids):
            if infected_days[kid] > 0:
                for other_kid in range(num_kids):
                    if kid != other_kid and infected_days[other_kid] == 0:
                        if np.random.rand() < infection_probability:
                            new_infections[other_kid] = days_infectious
                            total_infected[other_kid] = 1
                            day_new_infections += 1

        infected_days += new_infections
        infected_days = np.clip(infected_days - 1, 0, None)
        daily_new_infections.append(day_new_infections)
        day += 1

    epidemic_duration = day
    cumulative_infections = np.cumsum(daily_new_infections)
    total_infected_count = np.sum(total_infected)
    return daily_new_infections, cumulative_infections, epidemic_duration, total_infected_count

# Sim over 1000 runs
num_simulations_d = 1000
all_daily_new_infections = []
epidemic_durations = []
total_infected_counts = []

for _ in range(num_simulations_d):
    daily_new_infections, cumulative_infections, epidemic_duration, total_infected_count = simulate_epidemic(
        num_kids, infection_probability, days_infectious)
    all_daily_new_infections.append(daily_new_infections)
    epidemic_durations.append(epidemic_duration)
    total_infected_counts.append(total_infected_count)

# Compute expected number of infections by day i
max_length = max(len(infections) for infections in all_daily_new_infections)
daily_infections_array = np.zeros((num_simulations_d, max_length))

for i, infections in enumerate(all_daily_new_infections):
    daily_infections_array[i, :len(infections)] = infections

expected_infections_by_day = np.mean(daily_infections_array, axis=0)

print("\nExpected number of new infections by day:")
for day, infections in enumerate(expected_infections_by_day, start=1):
    print(f"Day {day}: {infections:.2f}")

# Plot expected infections by day
plt.figure(figsize=(10,6))
plt.plot(range(1, len(expected_infections_by_day)+1), expected_infections_by_day, marker='o')
plt.title('Expected Number of New Infections by Day')
plt.xlabel('Day')
plt.ylabel('Expected Number of New Infections')
plt.grid(True)
plt.savefig('viz/expected_infections_by_day.png')
plt.close()

# Produce histogram of epidemic durations
plt.figure(figsize=(10,6))
plt.hist(epidemic_durations, bins=range(1, max(epidemic_durations)+2), align='left',
         color='skyblue', edgecolor='black')
plt.title('Histogram of Epidemic Durations')
plt.xlabel('Epidemic Duration (Days)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('viz/epidemic_durations.png')
plt.close()

# Calculate average epidemic duration
average_duration = np.mean(epidemic_durations)
median_duration = np.median(epidemic_durations)
print(f"\nAverage epidemic duration: {average_duration:.2f} days")
print(f"Median epidemic duration: {median_duration} days")

# ---------
# Part e
# ---------

def simulate_epidemic_with_immunization(num_kids, infection_probability, days_infectious, immunization_probability):
    """Simulate the epidemic considering immunization."""
    immunized = np.random.rand(num_kids) < immunization_probability
    immunized[0] = False  # Tommy is not immunized
    infected_days = np.zeros(num_kids)
    infected_days[0] = days_infectious
    total_infected = np.zeros(num_kids)
    total_infected[0] = 1
    daily_new_infections = []
    day = 0

    while np.any(infected_days > 0):
        new_infections = np.zeros(num_kids)
        day_new_infections = 0

        for kid in range(num_kids):
            if infected_days[kid] > 0:
                for other_kid in range(num_kids):
                    if (kid != other_kid and infected_days[other_kid] == 0
                            and not immunized[other_kid]):
                        if np.random.rand() < infection_probability:
                            new_infections[other_kid] = days_infectious
                            total_infected[other_kid] = 1
                            day_new_infections += 1

        infected_days += new_infections
        infected_days = np.clip(infected_days - 1, 0, None)
        daily_new_infections.append(day_new_infections)
        day += 1

    epidemic_duration = day
    cumulative_infections = np.cumsum(daily_new_infections)
    total_infected_count = np.sum(total_infected)
    return daily_new_infections, cumulative_infections, epidemic_duration, total_infected_count

# Simulate 1000 runs with immunization
num_simulations_e = 1000
all_daily_new_infections_immunized = []
epidemic_durations_immunized = []
total_infected_counts_immunized = []

for _ in range(num_simulations_e):
    daily_new_infections, cumulative_infections, epidemic_duration, total_infected_count = simulate_epidemic_with_immunization(
        num_kids, infection_probability, days_infectious, immunization_probability)
    all_daily_new_infections_immunized.append(daily_new_infections)
    epidemic_durations_immunized.append(epidemic_duration)
    total_infected_counts_immunized.append(total_infected_count)

# Compute expected number of infections by day i with immunization
max_length_immunized = max(len(infections) for infections in all_daily_new_infections_immunized)
daily_infections_array_immunized = np.zeros((num_simulations_e, max_length_immunized))

for i, infections in enumerate(all_daily_new_infections_immunized):
    daily_infections_array_immunized[i, :len(infections)] = infections

expected_infections_by_day_immunized = np.mean(daily_infections_array_immunized, axis=0)

print("\nExpected number of new infections by day with immunization:")
for day, infections in enumerate(expected_infections_by_day_immunized, start=1):
    print(f"Day {day}: {infections:.2f}")

# Plot expected infections by day with immunization
plt.figure(figsize=(10,6))
plt.plot(range(1, len(expected_infections_by_day_immunized)+1), expected_infections_by_day_immunized,
         marker='o', color='green')
plt.title('Expected Number of New Infections by Day (With Immunization)')
plt.xlabel('Day')
plt.ylabel('Expected Number of New Infections')
plt.grid(True)
plt.savefig('viz/expected_infections_by_day_immunization.png')
plt.close()

# Produce hist of epidemic durations with immunization
plt.figure(figsize=(10,6))
plt.hist(epidemic_durations_immunized, bins=range(1, max(epidemic_durations_immunized)+2),
         align='left', color='lightgreen', edgecolor='black')
plt.title('Histogram of Epidemic Durations (With Immunization)')
plt.xlabel('Epidemic Duration (Days)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('viz/epidemic_durations_immunization.png')
plt.close()

# Calc avg epidemic duration with immunization
average_duration_immunized = np.mean(epidemic_durations_immunized)
median_duration_immunized = np.median(epidemic_durations_immunized)
print(f"\nAverage epidemic duration with immunization: {average_duration_immunized:.2f} days")
print(f"Median epidemic duration with immunization: {median_duration_immunized} days")

# ------------------------
# Comparison and Results
# ------------------------

print("\nComparison and Results")

# Compare total number of infections with and without immunization
mean_total_infected = np.mean(total_infected_counts)
std_total_infected = np.std(total_infected_counts)
mean_total_infected_immunized = np.mean(total_infected_counts_immunized)
std_total_infected_immunized = np.std(total_infected_counts_immunized)

print(f"Average total number of kids infected without immunization: {mean_total_infected:.2f} ± {std_total_infected:.2f}")
print(f"Average total number of kids infected with immunization: {mean_total_infected_immunized:.2f} ± {std_total_infected_immunized:.2f}")

# Plot comparison of total infections
plt.figure(figsize=(10,6))
plt.hist(total_infected_counts, bins=range(1, num_kids+2), alpha=0.7,
         label='Without Immunization', color='skyblue', edgecolor='black')
plt.hist(total_infected_counts_immunized, bins=range(1, num_kids+2),
         alpha=0.7, label='With Immunization', color='lightgreen', edgecolor='black')
plt.title('Comparison of Total Infections')
plt.xlabel('Total Number of Kids Infected')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('viz/comparison_total_infections.png')
plt.close()

# Plot comparison of epidemic durations
plt.figure(figsize=(10,6))
plt.hist(epidemic_durations, bins=range(1, max(epidemic_durations)+2), alpha=0.7,
         label='Without Immunization', color='skyblue', edgecolor='black')
plt.hist(epidemic_durations_immunized, bins=range(1, max(epidemic_durations_immunized)+2),
         alpha=0.7, label='With Immunization', color='lightgreen', edgecolor='black')
plt.title('Comparison of Epidemic Durations')
plt.xlabel('Epidemic Duration (Days)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('viz/comparison_epidemic_durations.png')
plt.close()

# Additional Findings
print("\nAdditional Findings:")

# Calc reduction in average total infections due to immunization
reduction_infections = mean_total_infected - mean_total_infected_immunized
percent_reduction_infections = (reduction_infections / mean_total_infected) * 100

print(f"Reduction in average total infections due to immunization: {reduction_infections:.2f}")
print(f"Percentage reduction in total infections: {percent_reduction_infections:.2f}%")

# Calc reduction in average epidemic duration due to immunization
reduction_duration = average_duration - average_duration_immunized
percent_reduction_duration = (reduction_duration / average_duration) * 100

print(f"Reduction in average epidemic duration due to immunization: {reduction_duration:.2f} days")
print(f"Percentage reduction in epidemic duration: {percent_reduction_duration:.2f}%")



# -----------------------
# Statistical Validation
# ------------------------

print("\nStatistical Validation")

# Perform a t-test to compare means of total infections with and without immunization
t_stat, p_value_t = ttest_ind(total_infected_counts, total_infected_counts_immunized, equal_var=False)
print(f"T-Test for Total Infections:")
print(f"  t-statistic = {t_stat:.2f}, p-value = {p_value_t:.4f}")
if p_value_t < 0.05:
    print("  Result: Statistically significant difference in total infections (p < 0.05)")
else:
    print("  Result: No statistically significant difference in total infections (p >= 0.05)")

# Perform a t-test to compare means of epidemic durations with and without immunization
t_stat_duration, p_value_t_duration = ttest_ind(epidemic_durations, epidemic_durations_immunized, equal_var=False)
print(f"T-Test for Epidemic Durations:")
print(f"  t-statistic = {t_stat_duration:.2f}, p-value = {p_value_t_duration:.4f}")
if p_value_t_duration < 0.05:
    print("  Result: Statistically significant difference in epidemic durations (p < 0.05)")
else:
    print("  Result: No statistically significant difference in epidemic durations (p >= 0.05)")

# Perform KS test to compare distributions of total infections with and without immunization
ks_stat, p_value_ks = ks_2samp(total_infected_counts, total_infected_counts_immunized)
print(f"KS-Test for Total Infections:")
print(f"  KS-statistic = {ks_stat:.2f}, p-value = {p_value_ks:.4f}")
if p_value_ks < 0.05:
    print("  Result: Statistically significant difference in infection distributions (p < 0.05)")
else:
    print("  Result: No statistically significant difference in infection distributions (p >= 0.05)")

# Perform KS test to compare distributions of epidemic durations with and without immunization
ks_stat_duration, p_value_ks_duration = ks_2samp(epidemic_durations, epidemic_durations_immunized)
print(f"KS-Test for Epidemic Durations:")
print(f"  KS-statistic = {ks_stat_duration:.2f}, p-value = {p_value_ks_duration:.4f}")
if p_value_ks_duration < 0.05:
    print("  Result: Statistically significant difference in duration distributions (p < 0.05)")
else:
    print("  Result: No statistically significant difference in duration distributions (p >= 0.05)")

# -------------------------------
# Summary of Statistical Results
# -------------------------------

print("\nStatistical Validation Summary:")
print(f"  Total Infections T-Test: p-value = {p_value_t:.4f}")
print(f"  Total Infections KS-Test: p-value = {p_value_ks:.4f}")
print(f"  Epidemic Durations T-Test: p-value = {p_value_t_duration:.4f}")
print(f"  Epidemic Durations KS-Test: p-value = {p_value_ks_duration:.4f}")
