import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import os
import numpy as np

transform_cnt = 5

class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        
        self.preprocess()
        
        if mode == True:
            self.num_images = len(self.train_dataset) 
        else:
            self.num_images = len(self.test_dataset) 

    def preprocess(self):
        
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            label_path = os.path.join(self.label_path, str(i)+'.png')
            print (img_path, label_path) 
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])
            
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        
        trans_idx = index % 5 - 1
        
        # print("trans count ", len(self.transform_img))
        img_path, label_path = dataset[index]
        
        image = Image.open(img_path)
        label = Image.open(label_path)
        
        transform_img_func = self.transform_img[trans_idx]
        trans_img = transform_img_func(image)
        
        transform_label_func = self.transform_label[trans_idx]
        trans_label = transform_label_func(label)   
        
        return trans_img, trans_label

    def __len__(self):
        """Return the number of images."""
        return self.num_images 
    
def add_gaussian_noise(img, mean=0, std=0.1):
    img_tensor = transforms.ToTensor()(img)  
    noise = torch.randn(img_tensor.size()) * std + mean  
    noisy_img_tensor = img_tensor + noise  
    noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)  
    return transforms.ToPILImage()(noisy_img_tensor) 

def add_blur(img, kernel_size=3, sigma=(0.1, 2.0)):
    img_tensor = transforms.ToTensor()(img)  
    blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blurred_img_tensor = blur_transform(img_tensor)  
    return transforms.ToPILImage()(blurred_img_tensor)

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode
        
        self.noise = torch.randn(self.imsize) * 0.1
        
    def transform_img(self, resize, totensor, normalize, rotate, flip, noise, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
            options.append(transforms.Lambda(lambda img: add_blur(img, kernel_size=3, sigma=(0.15, 2.0))))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if rotate:
            options.append(transforms.RandomRotation((130, 130)))
            options.append(transforms.Lambda(lambda img: add_blur(img, kernel_size=5, sigma=(0.1, 1.0))))
        if flip:
            options.append(transforms.RandomHorizontalFlip(p=1.0))
            options.append(transforms.RandomVerticalFlip(p=1.0))
        if noise:
            # options.append(transforms.Lambda(lambda img: F.adjust_brightness(img, 1.2)))
            # options.append(transforms.Lambda(lambda img: F.adjust_contrast(img, 0.8)))
            # options.append(transforms.Lambda(lambda img: F.adjust_saturation(img, 1.1)))
            # options.append(transforms.Lambda(lambda img: F.adjust_hue(img, 0.05)))
            options.append(transforms.Lambda(lambda img: add_gaussian_noise(img, mean=0, std=0.1)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, rotate, flip, noise, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if rotate:
            options.append(transforms.RandomRotation((130, 130)))
        if flip:
            options.append(transforms.RandomHorizontalFlip(p=1.0))
            options.append(transforms.RandomVerticalFlip(p=1.0))
        if noise:
            pass
            # options.append(transforms.Lambda(lambda img: F.adjust_brightness(img, 1.2)))
            # options.append(transforms.Lambda(lambda img: F.adjust_contrast(img, 0.8)))
            # options.append(transforms.Lambda(lambda img: F.adjust_saturation(img, 1.1)))
            # options.append(transforms.Lambda(lambda img: F.adjust_hue(img, 0.05)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        transform_img1 = self.transform_img(True, True, True, False, False, False, True) 
        transform_img2 = self.transform_img(True, True, True, True, False, False, False) 
        transform_img3 = self.transform_img(True, True, True, False, True, False, False) 
        transform_img4 = self.transform_img(True, True, True, True, True, False, True) 
        transform_img5 = self.transform_img(True, True, True, False, False, True, False)
        
        transform_img = [transform_img1, transform_img2, transform_img3, transform_img4, transform_img5]

        transform_label1 = self.transform_label(True, True, False, False, False, False, False) 
        transform_label2 = self.transform_label(True, True, False, True, False, False, False) 
        transform_label3 = self.transform_label(True, True, False, False, True, False, False) 
        transform_label4 = self.transform_label(True, True, False, True, True, False, True) 
        transform_label5 = self.transform_label(True, True, False, False, False, False, False)
        transform_label = [transform_label1, transform_label2, transform_label3, transform_label4, transform_label5]
        
        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=1,
                                             drop_last=False)
        return loader

