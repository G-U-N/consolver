import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from torchvision.transforms.functional import resize

model = torch.hub.load('facebookresearch/dinov2:qasfb-patch-3', 'dinov2_vitl14')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = T.Compose([
    T.Resize((572, 572)),
    T.CenterCrop((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = 'example.png' 
img_pil = Image.open(img_path).convert('RGB')
input_tensor = transform(img_pil).unsqueeze(0).to(device)

feature_maps = None
def hook_fn(module, input, output):
    global feature_maps
    feature_maps = output[:, 1:, :].detach()  # [1, N, 1024]

handle = model.blocks[-1].register_forward_hook(hook_fn)
with torch.no_grad():
    _ = model(input_tensor)
handle.remove()

B, N, D = feature_maps.shape
patch_size = 14
H = W = 518 // patch_size  # 37
feat = feature_maps.view(B, H, W, D).cpu().numpy()[0]  # [37, 37, 1024]

feat_flat = feat.reshape(-1, D)  # [37*37, 1024]
pca = PCA(n_components=3)
pca_feat = pca.fit_transform(feat_flat)
pca_feat = (pca_feat - pca_feat.min(axis=0)) / (pca_feat.max(axis=0) - pca_feat.min(axis=0) + 1e-8)
pca_feat = (pca_feat * 255).astype(np.uint8)
rgb_map = pca_feat.reshape(H, W, 3)  # [37, 37, 3]

rgb_tensor = torch.from_numpy(rgb_map).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,37,37]
rgb_up = resize(rgb_tensor, size=(518, 518), interpolation=T.InterpolationMode.BICUBIC)
rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # [518,518,3]

Image.fromarray(rgb_up).save('dinov2_pca_visualization.jpg')