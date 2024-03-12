import cv2
import torch
from PIL import Image
import argparse

def main():

    parser = argparse.ArgumentParser(description="YOLOv5 Inference Script")
    parser.add_argument("--model_path", type=str, default="/usr/src/inference/best.pt", help="Path to the YOLOv5 model checkpoint")
    parser.add_argument("--image_path", type=str, default="/usr/src/inference/123.png", help="Path to the input image")
    args = parser.parse_args()


    model = torch.hub.load('/usr/src/app', 'custom', path=args.model_path, source='local')
    model.conf = 0.20


    img = Image.open(args.image_path).resize((640, 640))  


    results = model(img, size=640) 


    results.print()
    #results.show()


    if len(results.xyxy[0]) > 0:
        print('***************************************')
        print('\ncoordinates: ', results.xyxy[0])
        print('***************************************')
    else:
        print('***************************************')
        print("No objects detected in the image.")
        print('***************************************')

if __name__ == "__main__":
    main()
