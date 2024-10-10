import os
import random
import shutil

def copy_corresponding_images(src_folder_a, src_folder_b, dest_folder_a, dest_folder_b, num_images=1000, copies=5):
    if not os.path.exists(dest_folder_a):
        os.makedirs(dest_folder_a)
    if not os.path.exists(dest_folder_b):
        os.makedirs(dest_folder_b)
        
    all_images_a = sorted(os.listdir(src_folder_a))
    all_images_b = sorted(os.listdir(src_folder_b))
    
    if len(all_images_a) != len(all_images_b):
        raise ValueError("dismatch count")
    
    selected_indices = random.sample(range(len(all_images_a)), num_images)
    
    count = 1
    
    for idx in selected_indices:
        img_a = all_images_a[idx]
        img_b = all_images_b[idx]
        
        src_image_path_a = os.path.join(src_folder_a, img_a)
        src_image_path_b = os.path.join(src_folder_b, img_b)
        
        for i in range(copies):
            dest_image_path_a = os.path.join(dest_folder_a, f"{count}.jpg")
            dest_image_path_b = os.path.join(dest_folder_b, f"{count}.png")

            shutil.copy(src_image_path_a, dest_image_path_a)
            shutil.copy(src_image_path_b, dest_image_path_b)

            print(f"Copied {src_image_path_a} to {dest_image_path_a}")
            print(f"Copied {src_image_path_b} to {dest_image_path_b}")

            count += 1
            
def remove_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  
        print(f"Deleted existing folder: {folder_path}")
    
    os.makedirs(folder_path)
    print(f"Created new folder: {folder_path}")