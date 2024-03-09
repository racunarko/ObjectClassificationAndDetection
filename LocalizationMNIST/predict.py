import torch
from PIL import Image
import cv2
import torchvision

from matplotlib import pyplot as plt
import numpy as np
from net import Net

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

# imported = Image.open('sedmica.png')
# grayscale = imported.convert('L')
# bw = grayscale.point(lambda x: 0 if x < 100 else 255, '1')
# bw.save('sedmica_bw.png')

file = 'test_photos/7/test_02.png'
img_arr = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img_arr = cv2.bitwise_not(img_arr)
plt.imshow(img_arr, cmap=plt.cm.binary)
plt.show()

img_size = 64
new_arr = cv2.resize(img_arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
plt.imshow(new_arr, cmap='gray')
plt.show()

normalized = transform(new_arr)
print(normalized.shape)
model = Net().to('mps')
model.load_state_dict(torch.load('mnist_localization.pth'))
model.eval()
normalized = normalized.unsqueeze(0)
print(normalized.shape)
with torch.no_grad():
    output = model(normalized.to('mps'))
    predicted_class = output[0]
    pred = predicted_class.argmax(dim=1, keepdim=True)
    predicted_bbox = output[1:]
    print(f"Predicted bbox: {predicted_bbox}")
    print(f"Predicted class: {pred.item()}")  
    
    x1, y1, x2, y2 = [int(coord.cpu().numpy()) for coord in predicted_bbox]

    # Draw the bounding box on the image
    # Make sure new_arr is in the right format for cv2.rectangle, needs to be 3 channel RGB
    if len(new_arr.shape) == 2:  # If it's grayscale, convert to RGB
        new_arr_rgb = cv2.cvtColor(new_arr, cv2.COLOR_GRAY2RGB)
    else:
        new_arr_rgb = new_arr

    # Draw rectangle on the image
    # Note: OpenCV expects start and end coordinates in (x, y) format
    cv2.rectangle(new_arr_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Use matplotlib to display the image
    plt.imshow(new_arr_rgb, cmap='gray')
    plt.title(f"Predicted Class: {pred.item()}")
    plt.show()