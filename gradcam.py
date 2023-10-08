import torch
import matplotlib.pyplot as plt
from model import fe
from dataloading import MRI_Dataset
from transformation import super_transformation
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import math
import cv2
import nibabel as nb
import collections

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:",device)

transformation = super_transformation()
train_data = MRI_Dataset(fold = 0 , stage= 'original_train',transform = transformation)

final_test_loader = DataLoader(dataset=train_data ,
                                batch_size=1, #64,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=3)



class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        #self.feature_extractor = fe(trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
        #                   fc_num_ch=16, kernel_size=3, conv_act='LeakyReLU',
        #                   fe_arch= 'fe1', dropout=0.2,
        #                   fc_dropout = 0.2, batch_size = 1).to(device)

        #self.classifier = nn.Sequential(
        #                    nn.Linear(256, 128),
        #                    nn.ReLU(),
        #                    nn.Linear(128, 16),
        #                    nn.ReLU(),
        #                    nn.Linear(16, 2),
        #                    ).to(device)
        self.feature_extractor = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=1)).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2),
            nn.Sigmoid()
        ).to(device)
        
        model = torch.load('npz_model.pth')
        encode_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' not in k])
        linear_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' in k])		
        
        self.feature_extractor.load_state_dict(encode_dict)
        #self.feature_extractor.load_state_dict(torch.load('fe_weights.pt'))
        #self.feature_extractor.load_state_dict(torch.load('fe_plain_weights.pt'))          
        
        top_layers = list(self.feature_extractor.children())[:3] 
        last_conv = list(self.feature_extractor.children())[3][0]
        #top_layers = list(list(list(self.feature_extractor.children())[0].children())[0].children())[:3]
        #last_conv = list(list(list(self.feature_extractor.children())[0].children())[0].children())[3][0]
        #linear = list(list(list(self.feature_extractor.children())[0].children())[0].children())[5]           

        self.features_conv = nn.Sequential(*top_layers,nn.Sequential(last_conv,nn.ReLU())).to(device)
        self.max_pool = nn.Sequential(nn.MaxPool3d(kernel_size=2), nn.Flatten(start_dim=1)).to(device) #add linear when running other model
        
        self.classifier.load_state_dict(linear_dict)
        #self.classifier.load_state_dict(torch.load('class_weights.pt'))
        #self.classifier.load_state_dict(torch.load('class_plain_weights.pt'))

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
    
def photo_dump():
    avg_mci = torch.zeros(1, 1, 64, 64, 64).to(device).float()
    avg_hiv = torch.zeros(1, 1, 64, 64, 64).to(device).float()
    avg_hand = torch.zeros(1, 1, 64, 64, 64).to(device).float()

    num_mci = num_hiv = num_hand = 0

    #initialize the model
<<<<<<< HEAD
    network = net()
    network.eval() 
    for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in final_test_loader:
        pred = network(images.view(1, 1, 64, 64, 64).to(device).float())[0]
        #print(pred)
        if pred[0] > 0.5 and pred[1] < 0.5: #MCI
            avg_mci += images.view(1, 1, 64, 64, 64).to(device).float()
            num_mci += 1
        elif pred[0] < 0.5 and pred[1] > 0.5: #HIV
            avg_hiv += images.view(1, 1, 64, 64, 64).to(device).float()
            num_hiv += 1
        elif pred[0] > 0.5 and pred[1] > 0.5: #HAND
=======
    net = net()
    net.eval() 
    for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in final_test_loader:
        pred = net(images.view(1, 1, 64, 64, 64).to(device).float())
        if pred[0] > 0 and pred[1] < 0: #MCI
            avg_mci += images.view(1, 1, 64, 64, 64).to(device).float()
            num_mci += 1
        elif pred[0] < 0 and pred[1] > 0: #HIV
            avg_hiv += images.view(1, 1, 64, 64, 64).to(device).float()
            num_hiv += 1
        elif pred[0] > 0 and pred[1] > 0: #HAND
>>>>>>> d2b392e4fe95be8404aee32f8e99eb4111a29bb6
            avg_hand += images.view(1, 1, 64, 64, 64).to(device).float()
            num_hand += 1
        else:
            pass
    # images, labels, actual_labels, datasets, ids, ages, genders,npzs =next(iter(final_test_loader))

    labels = ['mci', 'hiv', 'hand']
    imgs = [avg_mci/num_mci,avg_hiv/num_hiv,avg_hand/num_hand]

    for avg_img, label in zip(imgs, labels):
        #network = net()
        #network.eval()

        pred = network(avg_img)
        print(pred)
        pred_cd = pred[:,0].unsqueeze(1)
        pred_hiv = pred[:,1].unsqueeze(1)
        pred_cd.backward(retain_graph=True)
        pred_hiv.backward()
        # get the gradient of the output with respect to the parameters of the model
        #if label == 'mci':
        #    pred_cd.backward()
        #elif label == 'hiv':
        #    pred_hiv.backward()
        #else:
        #    pred.sum().backward()

        gradients = network.get_activations_gradient()
        #print(gradients)

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])
        
        # get the activations of the last convolutional layer
        activations = network.get_activations(avg_img).detach()

        # weight the channels by corresponding gradients
        for i in range(32):
            activations[:, i, :, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        #plt.imsave('heat.png',np.array(heatmap[:,:,4]))
        heatmap = cv2.resize(np.array(heatmap[:,4,:]), (138,138))
        heatmap = np.uint8(255 * heatmap)
             
        #threshold and rescale
        #flat = np.matrix.flatten(heatmap) 
        #max_heat = np.max(flat)
        #flat = flat/(flat.max()/255.0)
        #thresh = 0.8*max_heat
        #dimension = np.shape(heatmap)
        #flat[flat<thresh] = 0 
        #heatmap=np.reshape(flat, dimension)
        
        heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
        plt.gray()
        template = nb.load('/home/groups/kpohl/t1_data/hand/template.nii.gz')
        img=template.get_fdata()[:,int(176/2),:]
        plt.imsave('template_slice.png',img)

        img = cv2.imread('template_slice.png')
        img = cv2.resize(img, (138,138))      
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sensitivity = 15
        lower_white = np.array([0,0,0])
        upper_white = np.array([180,255,40])
        mask = 255-cv2.inRange(hsv, lower_white, upper_white)
        plt.imsave('mask.jpg',mask)
        superimposed_img = heatmap*0.4 + img
        superimposed_img = cv2.bitwise_and(superimposed_img,superimposed_img, mask= mask)        
        superimposed_img = cv2.rotate(superimposed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite('./map' + label +'.jpg', superimposed_img)          
   # draw the heatmap
    #plt.matshow(heatmap.squeeze()[3])
    #plt.savefig('heat.png')

photo_dump()
#     import cv2
# img = cv2.imread('./data/Elephant/data/05fig34.jpg')
# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img
# cv2.imwrite('./map.jpg', superimposed_img)

