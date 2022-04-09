import numpy as np
import matplotlib.pyplot as plt

labels, scores = np.loadtxt("score.txt").T

target_score = []
nontarget_score = []

for idx,i in enumerate(labels):
    if i == 0:
        nontarget_score.append(scores[idx])
    else:
        target_score.append(scores[idx])

print(scores.shape)
print(labels.shape)

plt.hist(target_score, bins=100, label="target score")
plt.hist(nontarget_score, bins=100, label="nontarget score")
plt.legend()
plt.tight_layout()
plt.savefig("test.png")
