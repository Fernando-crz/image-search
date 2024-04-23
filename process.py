from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
from interface import *
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

images_folder = "./coco"
image_paths = [os.path.join(images_folder, f) for f in listdir(images_folder) if isfile(join(images_folder, f))]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

gallery_path = "./images"
""" embeddings_path = "./images/embeddings.pth"
path_to_embeddings = {}
with torch.no_grad():
    for i, image_path in tqdm(enumerate(image_paths)):
        path = os.path.join(gallery_path, f"{i}.jpg")
        image = Image.open(image_path)
        image_embedding = get_image_embedding(model, processor, image).to("cpu")
        path_to_embeddings[path] = image_embedding """

for i, image_path in tqdm(enumerate(image_paths)):
    path = os.path.join(gallery_path, f"{i}.jpg")
    image = Image.open(image_path)
    image.save(path)

""" save_embeddings(path_to_embeddings, embeddings_path) """