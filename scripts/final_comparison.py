import matplotlib.pyplot as plt

models = ["Cowrie Model", "UWF (Raw)", "UWF (Balanced)"]
accuracy = [1.00, 0.93, 0.76]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")

plt.savefig("final_comparison.png")

print("✅ Final comparison chart saved")
