import argparse
import os
import sys
import time

import numpy as np
import cv2
import torch
from PIL import Image
from PIL import ImageFilter

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from sr_transformer import SR_8x_TransformerNet, SR_4x_TransformerNet
from vgg import Vgg16
import cv2

rand_seed = 1050
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

is_cuda = torch.cuda.is_available()

if is_cuda:
    torch.cuda.manual_seed(rand_seed)

def train(args):
    # --------------------------------------------
    # Compose the image transforms
    # First, for the reference images
    center_crop_size = 288
    transform = transforms.Compose([
        transforms.CenterCrop(center_crop_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Second, for the downsampled images
    transform_input = transforms.Compose([
        transforms.CenterCrop(center_crop_size),
        transforms.Lambda(lambda x: x.filter(ImageFilter.GaussianBlur(radius=1.0))),
        transforms.Scale(int(center_crop_size/args.downsample_scale), Image.BICUBIC),
        transforms.ToTensor()
    ])

    # --------------------------------------------
    # Load the training datasets
    dataset_dirname = args.dataset
    print("Training on images from: " + dataset_dirname)

    train_dataset = datasets.ImageFolder(dataset_dirname, transform)
    train_loader = DataLoader(train_dataset, batch_size=4)

    input_dataset = datasets.ImageFolder(dataset_dirname, transform_input)
    input_loader = DataLoader(input_dataset, batch_size=4)

    # --------------------------------------------
    # Select the appropriate Image Transformation Network
    learning_rate = args.lr
    if args.downsample_scale == 8:
        sr_transformer = SR_8x_TransformerNet()
    else:
        sr_transformer = SR_4x_TransformerNet()
    
    # Set optimizer to Adam optimizer, with learning_rate
    optimizer = Adam(sr_transformer.parameters(), learning_rate)
    mse_loss = torch.nn.MSELoss()

    # Load vgg as well
    vgg = Vgg16(requires_grad=False)

    if is_cuda:
        sr_transformer.cuda()
        vgg.cuda()

    # --------------------------------------------
    num_epochs = args.epochs
    log_interval = args.log_interval
    checkpoint_interval = args.checkpoint_interval
    checkpoint_save_dir = args.checkpoint_model_dir
    model_save_dir = args.save_model_dir

    # Begin the training regime!
    for e in range(num_epochs):
        sr_transformer.train()
        agg_content_loss = 0.
        count = 0
        # Load both the reference input image and the downsampled image
        for batch_id, ((x, _), (x_in, _)) in enumerate(zip(train_loader, input_loader)):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            x_in = Variable(x_in)
            if is_cuda:
                x = x.cuda()
                x_in = x_in.cuda()

            # Pass the downsampled image through the Image Transformation Network
            y = sr_transformer(x_in)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            
            # Pass the reference and the previous output through vgg
            features_y = vgg(y)
            features_x = vgg(x)

            # Calculate the content loss at layer relu2_2
            content_loss = mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Optimize the Image Transformation Network
            content_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}".format(
                    time.ctime(), e, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_save_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                sr_transformer.eval()
                if is_cuda:
                    sr_transformer.cpu()

                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(checkpoint_save_dir, ckpt_model_filename)
                torch.save(sr_transformer.state_dict(), ckpt_model_path)
                if is_cuda:
                    sr_transformer.cuda()
                
                sr_transformer.train()

    sr_transformer.eval()
    if is_cuda:
        sr_transformer.cpu()

    # Training is done, save the model
    save_model_filename = "coco_epoch_" + str(num_epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    save_model_path = os.path.join(model_save_dir, save_model_filename)
    torch.save(sr_transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)       

def stylize(args):
    
    # Load the model
    saved_sr_model = args.model
    print("Using %s as the model for Super Resolution.." %saved_sr_model)
    cap = cv2.VideoCapture(0)
    n_frame = 0

    while True:
        start = time.time()
        # Capture frame from webcam
        ret_val, img = cap.read()

        # Downsample the image
        img = cv2.resize(img, None, fx = (1/args.downsample_scale), fy = (1/args.downsample_scale))
        # Upsample it for presentation
        img_up = cv2.resize(img, None, fx = args.downsample_scale, fy = args.downsample_scale, interpolation = cv2.INTER_NEAREST)

        content_image = utils.load_frame(img)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0)
        
        if is_cuda:
            content_image = content_image.cuda()
        content_image = Variable(content_image, volatile=True)

        # Select the appropriate image transformation network
        if args.downsample_scale == 8:
            sr_model = SR_8x_TransformerNet()
        else:
            sr_model = SR_4x_TransformerNet()

        sr_model.load_state_dict(torch.load(saved_sr_model))
        # Pass the image through the model and obtain the output
        if is_cuda:
            sr_model.cuda()
        output = sr_model(content_image)
        if is_cuda:
            output = output.cpu()
        output_data = output.data[0]
        
        end = time.time()
        print("Processed frame %d" %n_frame)
        print("FPS = %f" %(1/(end - start)))
        n_frame = n_frame + 1
        
        frame = output_data.clone().clamp(0, 255).numpy()
        frame = frame.transpose(1, 2, 0).astype("uint8")

        # Crop the images for presentation
        img_up = img_up[100:400, 250:550]
        frame = frame[100:400, 250:550]
        show = np.hstack((img_up, frame))
        cv2.namedWindow("output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("output",show)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

def main():
    pass
    main_arg_parser = argparse.ArgumentParser(description="parser for real-time-super-resolution")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, default="data/",
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--downsample-scale", type=int, default=8,
                                  help="amount that you wish to downsample by. Default = 8")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for parser for evaluation super resolution model")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for super resolution")
    eval_arg_parser.add_argument("--downsample-scale", type=int, default=8, required=True,
                                  help="amount that you wish to downsample by. Default = 8")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: cuda is not available, please install CUDA!")
        sys.exit(1)

    if args.subcommand == "train":
        try:
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
                os.makedirs(args.checkpoint_model_dir)
        except OSError as e:
            print(e)
            sys.exit(1)
        if not os.path.exists(args.dataset):
            print("ERROR: you need to download the datasets! Run download_dataset.sh")
            sys.exit(1)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
