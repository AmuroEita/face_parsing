from parameter import *
from trainer import Trainer
from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder

from file_process import *

def main(config):
    # For fast training
    cudnn.benchmark = True

    if config.train:

    # Create directories if not exist
        make_folder(config.model_save_path, config.version)
        make_folder(config.sample_path, config.version)
        make_folder(config.log_path, config.version)
        
        remove_and_create_folder(config.img_path)
        remove_and_create_folder(config.label_path)
        copy_corresponding_images("Data_preprocessing/train_img_all", "Data_preprocessing/train_label_all", 
                                  config.img_path, config.label_path, num_images=1000, copies=5)
        
        data_loader = Data_Loader(config.img_path, config.label_path, config.imsize,
                             config.batch_size, config.train)
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
