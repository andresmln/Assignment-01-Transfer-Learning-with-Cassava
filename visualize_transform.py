import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import random

def visualize_transform_compare(dataset, transform, n=6):
    """
    Show original vs transformed images side by side for n random samples.
    """
    indices = random.sample(range(len(dataset)), n)
    
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 4*n))
    
    for i, idx in enumerate(indices):
        img_path = os.path.join(dataset.root_dir, dataset.df.iloc[idx]['image_id'])
        img = Image.open(img_path).convert('RGB')
        
        transformed_img = transform(img)
        
        # Convert tensors to numpy
        if isinstance(transformed_img, torch.Tensor):
            transformed_img = transformed_img.permute(1, 2, 0).numpy()
            transformed_img = (transformed_img - transformed_img.min()) / (transformed_img.max() - transformed_img.min())
        
        img_np = np.array(img) / 255.0  # normalize original for plotting
        
        # Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # Transformed
        axes[i, 1].imshow(transformed_img)
        axes[i, 1].set_title(f"Transformed: {transform.__class__.__name__}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()




train_df = cassava_df[cassava_df['set'] == 'train'].reset_index(drop=True)
temp_dataset = CassavaDataset(train_df, root_dir=IMAGE_ROOT_DIR)

# Example: RandomResizedCrop
single_transform = transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.8, 1.2))
visualize_transform_compare(temp_dataset, single_transform, n=6)

# Example: ColorJitter
single_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
visualize_transform_compare(temp_dataset, single_transform, n=6)

# Example: RandomHorizontalFlip
single_transform = transforms.RandomHorizontalFlip(p=0.5)
visualize_transform_compare(temp_dataset, single_transform, n=6)
