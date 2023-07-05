import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/oneclass_carton/TEST_objects.json", 'r') as f:
#     data1 = json.load(f)
# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/VOCdevkit/TEST_objects.json", "r") as f:
#     data2 = json.load(f)
# for x in data1:
#     data2.append(x)
# print(len(data2))
# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/TEST_objects.json", 'w') as f:
#     json.dump(data2,f)

# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/oneclass_carton/TEST_images.json", 'r') as f:
#     data1 = json.load(f)

# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/VOCdevkit/TEST_images.json", 'r') as f:
#     data2 = json.load(f)
# for x in data1[:1000]:
#     data2.append(x)

# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/TEST_images.json", 'w') as f:
#     json.dump(data2, f)

# with open("/home/famousdeer/Desktop/Praca magisterska/Program/data/TEST_images.json", 'r') as f:
#     data = json.load(f)
# print(len(data), len(data2))

with open('/home/famousdeer/Desktop/Praca magisterska/Program/data/loss.json', 'r') as f:
    loss = json.load(f)
plt.plot(loss['loss'])
plt.axvline(x = 80, color = 'g', label = 'lr = 1e-4', linestyle='dashed')
plt.axvline(x = 93, color = 'r', label = 'lr = 1e-5', linestyle='dashed')
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()