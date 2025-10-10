import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from clip_model import CLIPModel
from text_encoder import TextEncoderConfig
from vision_transformer import VisionTransformerConfig
from util import gpt_tokenize, get_padding_batch_input, strip_state_prefix

MODEL_FILE = "./clip_data/clip_pretrain.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel(
   text_config=TextEncoderConfig(), 
   vision_config=VisionTransformerConfig(),
   hidden_dim=512
)

state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
model.load_state_dict(strip_state_prefix(state_dict))
model.eval()

model = model.to(device)

def test_model(test_dataset, test_model, device):
    dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    class_names = test_dataset.classes
    text_prompts = [f"a photo of a {c}" for c in class_names]
    text_tokens = [gpt_tokenize(text) for text in text_prompts]
    input_ids, attention_masks = get_padding_batch_input(text_tokens)

    input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # images batch has the same size as the class numebr
            images = images.to(device)

            text_embds, vision_embds, scaler = test_model(input_ids, attention_masks, images)
        
            logits = vision_embds @ text_embds.transpose(0, 1) * scaler
            predicts = logits.argmax(dim=1).cpu()
        
            correct += (predicts == labels).sum().item()
            total += labels.size(0)

    return correct / total


# testing on CIFAR-10
dataset_cifar10 = datasets.CIFAR10(
    root="./clip_data", 
    download=True, 
    train=False,
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),           # convert to tensor [C, H, W] in [0,1]
        transforms.Normalize(            # normalize with mean/std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
)

# testing on CIFAR-100
dataset_cifar100 = datasets.CIFAR100(
    root="./clip_data", 
    download=True, 
    train=False,
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),           # convert to tensor [C, H, W] in [0,1]
        transforms.Normalize(            # normalize with mean/std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
)

cifar10_accuracy = test_model(dataset_cifar10, model, device)
cifar100_accuracy = test_model(dataset_cifar100, model, device)

print(f"Zero shot CIFAR-10 accuracy: {cifar10_accuracy}")
print(f"Zero shot CIFAR-100 accuracy: {cifar100_accuracy}")