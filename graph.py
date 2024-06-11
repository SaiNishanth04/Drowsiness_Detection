import matplotlib.pyplot as plt

# Data
years = [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011]
drowsy_drivers = [1221, 1319, 1332, 1275, 1306, 1234, 1221, 1173]
percentage_of_fatal_crashes = [2.4, 2.5, 2.5, 2.6, 2.9, 2.8, 2.4, 2.7]
fatalities_drowsy_driving = [785, 697, 803, 824, 851, 801, 835, 810]

# Create a figure and three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

# Plot 1: Drivers involved in fatal crashes who were drowsy
ax1.plot(years, drowsy_drivers, marker='o', color='b', label='Drowsy Drivers')
ax1.set_ylabel('Drivers Involved')
ax1.set_title('Drowsy Driving Statistics (2011-2018)')
ax1.legend()

# Plot 2: Percentage of all drivers involved in fatal crashes
ax2.plot(years, percentage_of_fatal_crashes, marker='o', color='g', label='Percentage of Fatal Crashes')
ax2.set_ylabel('Percentage (%)')
ax2.legend()

# Plot 3: Fatalities involving drowsy driving
ax3.plot(years, fatalities_drowsy_driving, marker='o', color='r', label='Fatalities')
ax3.set_xlabel('Year')
ax3.set_ylabel('Fatalities')
ax3.legend()

# Adjust layout and display the chart
plt.tight_layout()
plt.show()
