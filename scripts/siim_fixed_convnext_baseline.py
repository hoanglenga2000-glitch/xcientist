"""Fixed SIIM reference: ImageNet ConvNeXt-Tiny, three epochs, no tuning."""
import random
import numpy as np
import torch
import torchvision
from PIL import Image

WEIGHTS='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_model_cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth'

class Images(torch.utils.data.Dataset):
    def __init__(self,frame,labels=None,training=False):
        self.paths=frame.image_path.tolist()
        self.labels=None if labels is None else np.asarray(labels,dtype=np.float32)
        steps=[torchvision.transforms.Resize((224,224))]
        if training:steps += [torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip()]
        steps += [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=torchvision.transforms.Compose(steps)
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        with Image.open(self.paths[index]) as opened:image=self.transform(opened.convert('RGB'))
        return image if self.labels is None else (image,self.labels[index])

class Predictor:
    def __init__(self,model):self.model=model;self.classes_=np.asarray([0,1])
    def predict_proba(self,frame):
        loader=torch.utils.data.DataLoader(Images(frame),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
        self.model.eval();values=[]
        with torch.inference_mode():
            for images in loader:
                logits=self.model(images.to('cuda',non_blocking=True)).reshape(-1)
                values.append(torch.sigmoid(logits).cpu().numpy())
        p=np.concatenate(values).astype(np.float64)
        return np.column_stack([1-p,p])

def fit_model(x_train,y_train,x_valid,y_valid,seed):
    random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    model=torchvision.models.convnext_tiny(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    model.classifier[2]=torch.nn.Linear(model.classifier[2].in_features,1)
    model.to('cuda')
    loader=torch.utils.data.DataLoader(Images(x_train,y_train,training=True),batch_size=64,shuffle=True,num_workers=4,pin_memory=True)
    positive=float(np.asarray(y_train).sum());negative=float(len(y_train)-positive)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,device='cuda'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
    for epoch in range(3):
        model.train()
        for images,labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(images.to('cuda',non_blocking=True)).reshape(-1)
            loss=loss_fn(logits,labels.to('cuda',non_blocking=True));loss.backward();optimizer.step()
    return Predictor(model)
