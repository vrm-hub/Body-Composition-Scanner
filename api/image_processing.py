import cv2
import io
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

# Generate masked images
deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# resize images
def preprocess(f_name):
    img_orig = cv2.imread(f_name, 1)
    k = min(1.0, 1024 / max(img_orig.shape[0], img_orig.shape[1]))
    img = cv2.resize(img_orig, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    return img


def apply_deeplab(deeplab, img, device):
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return (output_predictions == 15)


# Buff to store image temp
def create_buff(mask):
    buf = io.BytesIO()
    plt.imsave(buf, mask, cmap="gray", format="png")
    buf.seek(0)
    return buf


# Access the image stored in buff
def access_stored_image(buffer):
    # Open the image from the BytesIO buffer
    image = Image.open(buffer)
    return image
