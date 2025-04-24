from datasets import load_dataset
import matplotlib.pyplot as plt
import clip
import torch

# Load the dataset
ds = load_dataset('ceyda/fashion-products-small')
# print(ds)

entry = ds['train'][0]

image = entry['image']
dataset = ds['train']

# OpenAI CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", jit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


import matplotlib.pyplot as plt

# Select indices for three example images
indices = [0, 2, 10]

# Get the list of possible subcategories from the dataset
subcategories = list(set(example['subCategory'] for example in dataset))

# Preprocess the text descriptions for each subcategory
text_inputs = torch.cat([clip.tokenize(f"{c}") for c in subcategories]).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in subcategories]).to(device)
# text_inputs = torch.cat([clip.tokenize(f"more likely a photo of {c}") for c in subcategories]).to(device)
# text_inputs = torch.cat([clip.tokenize(f"less likely a photo of {c}") for c in subcategories]).to(device)

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the indices and process each image
for i, idx in enumerate(indices):
    # Select an example image from the dataset
    example = dataset[idx]
    image = example['image']
    subcategory = example['subCategory']

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Calculate image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity between image and text features
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    print(f"Predicted: {subcategories[indices[0]]}, Actual: {subcategory}")

    # Display the image in the subplot    
    axes[i].imshow(image)
    axes[i].set_title(f"Predicted: {subcategories[indices[0]]}, Actual: {subcategory}")
    axes[i].axis('off')

# Show the plot
plt.tight_layout()
plt.show()
# Save the figure
plt.savefig("zero_shot_predictions.png", bbox_inches='tight')

