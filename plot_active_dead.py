import matplotlib.pyplot as plt

labels = ["Active Episodes", "Dead Episodes"]
values = [14, 16]

plt.figure(figsize=(7,5))
plt.bar(labels, values)
plt.ylabel("Count")
plt.title("Static K=4 GRPO: Active vs Dead Episodes (30 Episodes)")
plt.tight_layout()
plt.savefig("baseline_active_vs_dead.png", dpi=300)
plt.close()

print("Saved: baseline_active_vs_dead.png")
