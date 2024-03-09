import torch
from PIL import Image
import cv2
import torchvision
from matplotlib import pyplot as plt
from net import Net

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

# imported = Image.open('sedmica.png')
# grayscale = imported.convert('L')
# bw = grayscale.point(lambda x: 0 if x < 100 else 255, '1')
# bw.save('sedmica_bw.png')

file = '7.png'
img_arr = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img_arr = cv2.bitwise_not(img_arr)
plt.imshow(img_arr, cmap=plt.cm.binary)
plt.show()

img_size = 28
new_arr = cv2.resize(img_arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
plt.imshow(new_arr, cmap='gray')
plt.show()

normalized = transform(new_arr)

model = Net().to('mps')
model.load_state_dict(torch.load('mnist_classification.pth'))
model.eval()
with torch.no_grad():
    output = model(normalized.to('mps'))
    pred = output.argmax(dim=1, keepdim=True)
    print(f"Predicted: {pred.item()}")  