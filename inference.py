import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.models.backbone import Backbone, Joiner
from src.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from src.models.transformer import Transformer
from src.models.detr import DETR, PostProcess

def build_transformer():
    return Transformer(
        d_model=256
    )

def build_backbone():
    position_embedding = PositionEmbeddingSine(256//2, normalize=True)
    backbone = Backbone("resnet50", True, False, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

confidence_threshold = 0.7

backbone = build_backbone()

transformer = build_transformer()

model = DETR(
    backbone=backbone,
    transformer=transformer,
    num_classes=91,
    num_queries=100,
    aux_loss=False
)

ckp = torch.load('weights/detr-r50-e632da11.pth', map_location=torch.device("cpu"))
model.load_state_dict(ckp['model'])

img_pil = Image.open("people.jpeg").convert('RGB').resize((800, 800))

width, height = img_pil.size

print(height, width)

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = tfms(img_pil).unsqueeze(0).float()

out = model(img)

postprocesser = PostProcess()

results = postprocesser(out, target_sizes=torch.tensor([[height, width]]))

for result in results:
    keep = torch.gt(result['scores'], confidence_threshold)
    scores = result['scores'][keep]
    boxes = result['boxes'][keep]
    labels = result['labels'][keep]

print("Scores: ", scores)
print("Boxes: ", boxes)
print("Labels: ", labels)

with open('coco-labels-91.txt') as f:
    lines = f.readlines()

classes = [line.strip('\n') for line in lines]

print(classes)

draw = ImageDraw.Draw(img_pil)
for label, box in zip(labels, boxes):
    draw.text([int(box[0]), int(box[1]) - 5], text=f"{classes[label - 1]}", fill="blue")
    draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])], outline='red')

plt.imshow(img_pil)
plt.show()