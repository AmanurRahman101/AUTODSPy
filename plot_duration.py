import matplotlib.pyplot as plt

episodes = list(range(1, 31))
durations = [
    34.98,26.53,285.71,16.78,7.58,5.73,23.31,60.95,19.96,16.44,
    12.57,19.37,19.18,17.12,13.78,15.64,16.17,14.71,17.17,16.68,
    9.39,21.18,13.62,18.14,15.00,25.07,6.34,11.08,11.36,10.84
]

plt.figure(figsize=(10,5))
plt.plot(episodes, durations, marker='o')
plt.xlabel("Episode")
plt.ylabel("Duration (seconds)")
plt.title("Static K=4 GRPO: Episode Duration Across 30 Episodes")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_duration_trajectory.png", dpi=300)
plt.close()

print("Saved: baseline_duration_trajectory.png")
