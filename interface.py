import gradio as gr
import torch
import numpy as np
import hnswlib
import os
from transformers import CLIPProcessor, CLIPModel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CLIP_DIM = 768
MAX_NUM_ELEMENTS = 20000
print(f"device being used: {device}")

def get_image_embedding(model, processor, image):
    with torch.no_grad():
        processed_image = processor(images=image, return_tensors="pt").to(device)
        return model.get_image_features(**processed_image)[0].to(device)

def get_text_embedding(model, processor, text):
    with torch.no_grad():
        processed_text = processor(text=[text], return_tensors="pt", padding=True).to(device)
        return model.get_text_features(**processed_text)[0].to(device)

def load_embeddings(embeddings_path):
    return torch.load(embeddings_path)

def save_embeddings(gallery_dataset, gallery_dataset_path):
    gallery_dataset.save_index(gallery_dataset_path)

def add_image_to_gallery(model, processor, image, gallery_dataset, gallery_dataset_path, gallery_path):
    image_embedding = get_image_embedding(model, processor, image)
    image_id = gallery_dataset.element_count
    new_image_path = os.path.join(gallery_path, f"{image_id}.jpg")
    image.save(new_image_path)
    gallery_dataset.add_items(np.array(image_embedding.cpu()), np.array([image_id]))
    save_embeddings(gallery_dataset, gallery_dataset_path)

def get_gallery_dataset(gallery_dataset_path):
    gallery_dataset = hnswlib.Index(space='cosine', dim=CLIP_DIM)
    gallery_dataset.load_index(gallery_dataset_path, max_elements=MAX_NUM_ELEMENTS)
    return gallery_dataset

def get_images_from_text(model, processor, text, gallery_dataset, gallery_path, num_images_preselected=3,similarity_threshold=0):
    gallery_dataset.set_ef(50)
    text_embedding = get_text_embedding(model, processor, text)
    print(text_embedding.size())

    selected_images, _ = gallery_dataset.knn_query(np.array(text_embedding.cpu()), k=num_images_preselected)
    print(selected_images)
    selected_image_paths = [os.path.join(gallery_path, f"{selected_image}.jpg") for selected_image in selected_images[0]]
    return selected_image_paths


def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    gallery_path = "./images"
    gallery_dataset_path = os.path.join(gallery_path, "embeddings.pth")
    gallery_dataset = get_gallery_dataset(gallery_dataset_path)

    image_search = gr.Interface(
        fn=lambda text: get_images_from_text(model, processor, text, gallery_dataset, gallery_path),
        inputs=gr.Textbox(),
        outputs=gr.Gallery()
    )

    add_image_gallery = gr.Interface(
        fn=lambda image: add_image_to_gallery(model, processor, image, gallery_dataset, gallery_dataset_path, gallery_path),
        inputs=gr.Image(type="pil"),
        outputs=None
    )

    gr.TabbedInterface([image_search, add_image_gallery], ["Image search", "Add image to gallery",]).launch(share=True)

if __name__ == "__main__":
    main()

