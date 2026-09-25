import matplotlib.pyplot as plt

episodes = list(range(1, 31))
means = [
    0.950,0.750,0.750,0.000,0.500,0.000,0.000,0.450,0.500,0.000,
    0.750,0.000,0.375,0.875,1.000,0.000,0.000,0.000,0.875,0.000,
    0.625,0.000,0.000,0.725,0.750,0.811,1.000,1.000,1.000,1.000
]

plt.figure(figsize=(10,5))
plt.plot(episodes, means, marker='o')
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Static K=4 GRPO: Mean Reward Across 30 Episodes")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_reward_trajectory.png", dpi=300)
plt.close()

print("Saved: baseline_reward_trajectory.png")
