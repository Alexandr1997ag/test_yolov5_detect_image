import cv2
import torch
from PIL import Image
import argparse
import time
def main():

    parser = argparse.ArgumentParser(description="YOLOv5 Inference Script")
    parser.add_argument("--model_path", type=str, default="/usr/src/inference/best.pt", help="Path to the YOLOv5 model checkpoint")
    parser.add_argument("--image_path", type=str, default="/usr/src/inference/123.png", help="Path to the input image")
    args = parser.parse_args()


    model = torch.hub.load('/usr/src/app', 'custom', path=args.model_path, source='local')
    model.conf = 0.20


    img = Image.open(args.image_path)#.resize((640, 640))  
    start_time = time.time()

    # results = model(img, size=640) 
    results = model(img, size=640) 
    end_time = time.time()
    elapsed_time = end_time - start_time

    results.print()
    #results.show()
    coords = []
    for i in range(len(results.xyxy[0])):
        x1 = results.xyxy[0][i][0].item()
        y1 = results.xyxy[0][i][1].item()
        x2 = results.xyxy[0][i][2].item()
        y2 = results.xyxy[0][i][3].item()
        w = x2 - x1
        h = y2 - y1
        coords.append([int(x1), int(y1), int(w), int(h)])

    if len(results.xyxy[0])>0:
        for i in range(len(coords)):
            print('***************************************')
            print(f"\n{i}", coords[i])
            print('***************************************')
            print(f"Inference time: {elapsed_time:.4f} seconds")
            print('***************************************')
    else:
        print('***************************************')
        print("No objects detected in the image.")
        print('***************************************')

if __name__ == "__main__":
    main()
