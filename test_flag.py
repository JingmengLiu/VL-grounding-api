from torchvision import transforms, models, datasets
import torch
from PIL import Image
import json

with open('weights/flag_classes.txt') as f:
    data = f.readlines()
# print(data)
flag_classes = []
for i, x in enumerate(data):
    index, name = x.split('. ')
    assert int(index) == i+1
    name_list = name.split('-')
    chinese_name = name_list[-1]
    chinese_name = chinese_name.strip()
    english_name = '-'.join(name_list[:-1])
    flag_classes.append((english_name, chinese_name))
# json.dump(flag_classes, open('weights/flag_classes.json', 'w'))
# flag_classes = json.load(open('weights/flag_classes.json'))
# name_list, idx_to_class = flag_classes['class_names'], flag_classes['idx_mapping']
valid_directory = '../fine-grained-detection/flag/data/val'
prdict=datasets.ImageFolder(root=valid_directory)
print(prdict.class_to_idx)
idx_to_class = {v: int(k)-1 for k, v in prdict.class_to_idx.items()}
flag_classes = {'class_names': flag_classes, 'idx_mapping': idx_to_class}
json.dump(flag_classes, open('weights/flag_classes.json', 'w'))

# num_classes=224
# device = torch.device('cuda:5')
# image_transforms =transforms.Compose([
#             transforms.Resize(size=256),
#             transforms.CenterCrop(size=224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])])

# model = torch.load('weights/flag_model_best.pt')
# model = model.to(device)
# image = Image.open('1.png')
# inputs = image_transforms(image).unsqueeze(0).to(device)
# with torch.no_grad():
#     model.eval()
#     outputs = model(inputs).softmax(dim=-1)
#     print(outputs.shape)
#     ret, predictions = torch.max(outputs.data, 1)

# print(ret, predictions, idx_to_class[str(int(predictions))], name_list[idx_to_class[str(int(predictions))]])