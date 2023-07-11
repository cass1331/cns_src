from torch.utils.data import Dataset, DataLoader
from dataloading import MRI_Dataset

class Paired_UCSF_Dataset(MRI_Dataset):
    def __init__(self, data,  transform=None):
        self.transform = transform
        self.data = [[d[0][0],d[1][0]] for d in data]
        self.dx = [[d[0][1],d[1][1]]  for d in data]
        self.actual_dx = [[d[0][2],d[1][2]]  for d in data]
        self.dataset = [[d[0][3],d[1][3]]  for d in data]
        self.ids = [[d[0][4],d[1][4]]  for d in data]
        self.ages = [[d[0][5],d[1][5]]  for d in data]
        self.genders = [[d[0][6],d[1][6]]  for d in data]
        self.subject_num = len(self.data)
        # print('original_train_data num ', self.data.shape[0])
        # print('actual label ', self.actual_dx.shape)

    def __getitem__(self, idx):
        #print('idx:'+ str(idx))
        # if self.transform != None:
        #   image = self.transform(self.data)
        # else:
        #   image = self.data
        #
        # return image, self.dx,self.actual_dx,self.dataset,self.ids,self.ages,self.genders

        if self.transform != None:
          images = [self.transform(self.data[idx][0]), self.transform(self.data[idx][1])]
        else:
          images = [self.data[idx][0],self.data[idx][1]]

        return images, self.dx[idx],self.actual_dx[idx],self.dataset[idx],self.ids[idx],self.ages[idx],self.genders[idx]


class Specific_MRI_Dataset(MRI_Dataset):
    def __init__(self, data,  transform=None):
        self.transform = transform
        self.data = [d[0] for d in data]
        self.dx = [d[1] for d in data]
        self.actual_dx = [d[2] for d in data]
        self.dataset = [d[3] for d in data]

        self.ids = [d[4] for d in data]
        self.ages = [d[5] for d in data]
        self.genders = [d[6] for d in data]
        self.subject_num = len(self.data)
      # print('original_train_data num ', self.data.shape[0])
      # print('actual label ', self.actual_dx.shape)

    def __getitem__(self, idx):
        #print('idx:'+ str(idx))
        # if self.transform != None:
        #   image = self.transform(self.data)
        # else:
        #   image = self.data
        #
        # return image, self.dx,self.actual_dx,self.dataset,self.ids,self.ages,self.genders

        if self.transform != None:
          image = self.transform(self.data[idx])
        else:
          image = self.data[idx]

        return image, self.dx[idx],self.actual_dx[idx],self.dataset[idx],self.ids[idx],self.ages[idx],self.genders[idx]
