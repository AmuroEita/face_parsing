# Face parsing
A Pytorch implementation face parsing model using lightweight UNet3+ 

## Dependencies
* Pytorch 0.4.1
* numpy
* Python3
* Pillow
* opencv-python
* tenseorboardX
* tqdm
* scikit_learn

## Files 
```
.
├── data_loader.py ""
├── evaluate.py "evalute to get mIOU and F1-score"
├── file_process.py "copy the train image and some folder operation"
├── main.py "main"
├── model_utils.py "model utils, like DSConv, unetConv2 implementation"
├── parameter.py "default parameters"
├── README.md
├── run.sh "start this to start training"
├── run_test.sh "start this to start testing"
├── tester.py 
├── trainer.py
├── unet3Plus.py "the unet3Plus model, for training and testing"
├── unet.py "the original unet of CelebAMask, just for reference"
└── utils.py
```

## Well-trained model
The well-trained model is "unet3plus.pth", you can find in the models/ 

## Preprocessing
Put your images and labels under the folder Data_preprocessing, and rename the images folder as train_img_all, the labels folder as train_lable_all

## Training
* Run `bash run.sh #GPU_num`

## Testing & Color visualization
* Run `bash run_test.sh #GPU_num`
* Results will be saved in `./test_results`

## Evalute
* RUn `python evalute.py` and get the mIOU and F1-score
