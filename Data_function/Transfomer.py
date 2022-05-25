from torchvision import transforms as T

class Transform:
    def __init__(self, resize, mean,  std):
        self.tranformers = {
            'train': T.Compose([
                T.RandomResizedCrop(resize),
                T.RandomHorizontalFlip(),
                # T.RandomRotation((10, 10)),
                # T.Resize(resize),
                # T.ToTensor(),
                # T.Normalize(mean, std)
                # T.RandomResizedCrop(32),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
            ]),
            'val': T.Compose([
                T.Resize(256),
                T.CenterCrop(resize),
                # T.ToTensor(),
                # T.Normalize(mean, std)
                # T.Resize(32),
                # T.CenterCrop(32),
                T.ToTensor(),
                T.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
            ])
        }
        
    def __call__(self, img, phase = 'train'):
        return self.tranformers[phase](img)
 