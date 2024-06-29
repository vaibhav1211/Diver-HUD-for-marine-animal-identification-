import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms                                              
import cv2
import time
from PIL import Image 

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


train_transform = transforms.Compose([transforms.Resize((150,150)),
                                     transforms.ToTensor(),normalize,transforms.RandomHorizontalFlip()])

test_transform = transforms.Compose([transforms.Resize((150,150)),
                                     transforms.ToTensor(),normalize])
images_path = torchvision.datasets.ImageFolder("/Users/vaibhavmk/Downloads/marinefull",transform = train_transform)
len(images_path)
model = torchvision.models.resnet50(pretrained = False)
lassLayer = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(lassLayer,256),
                         nn.ReLU(inplace = True),
                         
                         
                         
                         nn.Linear(256,len(images_path.classes)))

                                   
model.load_state_dict(torch.load("/Users/vaibhavmk/Downloads/animalClassfiction1.pth"))                                   

def capture_and_save_frame(video_capture,output_path):
    frame_number = 0
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        cv2.imwrite(output_path, frame)
         

        test_paths =['/Users/vaibhavmk/Downloads/campics/test1.jpg']
        img_testing = [Image.open(a) for a in test_paths]

        stacked = torch.stack([test_transform(g) for g in img_testing])





        with torch.no_grad():
            model.eval()
            output = model(stacked)
            output = nn.functional.softmax(output,dim = 1)
    
        class_names = images_path.classes    

        cv2.putText(frame, class_names[np.argmax(output[0])], (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
 
        cv2.imshow('Quantum Aquatica', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
        frame_number += 1

        time.sleep(0.01)


video_capture = cv2.VideoCapture(0)

output_path = '/Users/vaibhavmk/Downloads/campics/test1.jpg'


capture_and_save_frame(video_capture,output_path)
