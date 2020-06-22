#!/home/relkochta/anaconda3/bin/python

# Check to ensure correct argument usage
from sys import argv, exit
if len(argv) != 2:
    print("Usage: ./predict.py image.jpg")
    exit(1)

# Import stuff we will need
import torch
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import torchvision.models as models
import glob


# Initialize torchvision transformer
normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
val_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

# Load image into memory
def image_load(path):
    ret = Image.open(path).convert('RGB')
    ret = val_transformer(ret).float()
    ret = ret.clone().detach()
    ret = ret.unsqueeze(0)
    if torch.cuda.is_available():
        ret = ret.cuda()
    return ret
image = image_load(argv[1])

# Load model
model = models.densenet169(pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()
pt_net = torch.load('../baseline methods/Self-Trans/Self-Trans.pt')
model.load_state_dict(pt_net)
model.eval()

# Run model on provided image
output = model(image)

# Print result
print(output.argmax(dim=1, keepdim=True).item())

#sys.exit(0)

# Some more testing (delete me)
covid = glob.glob("../Images-processed/CT_COVID/*.*g")
TP = 0
FN = 0
for im in covid:
    image = image_load(im)
    output = model(image)
    output = int(output.argmax(dim=1, keepdim=True).item())
    if output == 0:
        FN += 1
    if output == 1:
        TP += 1

print("TP:", TP, "FN:", FN)
print("DONE")
