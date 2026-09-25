import matplotlib.pyplot as plt

labels = ["Mixed / Active", "All-Fail", "All-Success"]
values = [14, 11, 5]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.ylabel("Number of Episodes")
plt.title("Static K=4 GRPO: Episode Outcome Types")
plt.tight_layout()
plt.savefig("baseline_outcome_types.png", dpi=300)
plt.close()

print("Saved: baseline_outcome_types.png")
