import json

import matplotlib.pyplot as plt

lossesa = []

with open('result_B32_001.json', 'r') as f:
    lossesa = json.load(f)
f.close()

plt.plot(
    [x for x in range(len(lossesa))],
    lossesa
)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()