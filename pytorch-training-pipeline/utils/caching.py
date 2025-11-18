from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, runtime_transforms: v2.Transform):
        # This operation caches all transformations from the wrapped dataset. Stores the results as a Tuple
        # instead of list, decreasing memory usage. Tuples also have faster indexing, even though it is negligible.
        self.dataset = tuple([x for x in dataset])

        # These are the runtime transformations that can't be cached. Usually, they involve randomness which is a
        # form of regularization for the network, and caching the randomness usually results in overfitting.
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        image = image.clone()
        
        if self.runtime_transforms is not None:
            return self.runtime_transforms(image), label
    
        return image, label