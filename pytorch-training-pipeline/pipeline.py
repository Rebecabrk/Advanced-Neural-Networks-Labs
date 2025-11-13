import torch
import timm
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.optim as optim
import torchvision
import muon

from datasets.caching import CachedDataset

def setup_pipeline(config):
	device = torch.device('cuda' if torch.cuda.is_available() and config['global']['device'] == 'auto' else 'cpu')

    # transforms
	test_transforms = [
		v2.ToTensor(), 
		v2.ToDtype(torch.float32, scale=True)
	]
	cacheable_transforms = [v2.ToTensor(), v2.ToDtype(torch.float32, scale=True)]
	runtime_transforms = []
	normalization_cfg = None
	
	if config['data'].get('data_augmentation', False):
		t_cfg = config['data'].get('transforms', {})
		if t_cfg.get('normalization', {}).get('enabled', False):
			normalization_cfg = t_cfg['normalization']
		if t_cfg.get('random_horizontal_flip', {}).get('enabled', False):
			p = t_cfg['random_horizontal_flip'].get('probability', 0.5)
			runtime_transforms.append(v2.RandomHorizontalFlip(p=p))
		if t_cfg.get('random_vertical_flip', {}).get('enabled', False):
			p = t_cfg['random_vertical_flip'].get('probability', 0.5)
			runtime_transforms.append(v2.RandomVerticalFlip(p=p))
		if t_cfg.get('random_rotation', {}).get('enabled', False):
			degrees = t_cfg['random_rotation'].get('degrees', 15)
			runtime_transforms.append(v2.RandomRotation(degrees=degrees))
		if t_cfg.get('random_crop', {}).get('enabled', False):
			size = t_cfg['random_crop'].get('size', 32)
			padding = t_cfg['random_crop'].get('padding', 4)
			runtime_transforms.append(v2.RandomCrop(size=size, padding=padding))
		if t_cfg.get('color_jitter', {}).get('enabled', False):
			cj = t_cfg['color_jitter']
			runtime_transforms.append(v2.ColorJitter(
				brightness=cj.get('brightness', 0.2),
				contrast=cj.get('contrast', 0.2),
				saturation=cj.get('saturation', 0.2),
				hue=cj.get('hue', 0.1)
			))
		if t_cfg.get('random_grayscale', {}).get('enabled', False):
			p = t_cfg['random_grayscale'].get('probability', 0.1)
			runtime_transforms.append(v2.RandomGrayscale(p=p))
		if t_cfg.get('random_erasing', {}).get('enabled', False):
			p = t_cfg['random_erasing'].get('probability', 0.1)
			runtime_transforms.append(v2.RandomErasing(p=p))

	if normalization_cfg is not None:
		mean = normalization_cfg.get('mean', [0.5, 0.5, 0.5])
		std = normalization_cfg.get('std', [0.5, 0.5, 0.5])
		test_transforms.append(v2.Normalize(mean=mean, std=std))
		if len(runtime_transforms) > 0:
			runtime_transforms.append(v2.Normalize(mean=mean, std=std))
		else:
			cacheable_transforms.append(v2.Normalize(mean=mean, std=std))

	test_transforms = v2.Compose(test_transforms)
	cacheable_transform = v2.Compose(cacheable_transforms)
	runtime_transform = v2.Compose(runtime_transforms) if runtime_transforms else None
  
	# dataset
	dataset_name = config['data']['dataset_name']
	if dataset_name == 'CIFAR-100':
		num_classes = 100
		trainset_base = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=cacheable_transform)
		testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transforms)
	elif dataset_name == 'CIFAR-10':
		num_classes = 10
		trainset_base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cacheable_transform)
		testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
	elif dataset_name == 'MNIST':
		num_classes = 10
		trainset_base = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=cacheable_transform)
		testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transforms)
	elif dataset_name == 'OxfordIIITPet':
		num_classes = 37
		trainset_base = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=cacheable_transform)
		testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=test_transforms)
	else:
		raise ValueError(f"Dataset {dataset_name} not supported.")

	trainset = CachedDataset(trainset_base, runtime_transform)

	# data loaders
	batch_size = config['training'].get('batch_size', 128)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=config['global'].get('num_workers', 4))
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=config['global'].get('num_workers', 4))

	# model
	model_name = config['model']['name']
	if model_name == 'MLP':
		mlp_params = config['model'].get('mlp_params', {})
		sample_batch = next(iter(trainloader))
		images, _ = sample_batch
		input_size = images[0].numel()  # Flattened size per image
		hidden_layers = mlp_params.get('hidden_layers', [1024, 512, 256])
		activations = mlp_params.get('activations', ['ReLU', 'ReLU', 'Sigmoid'])
		dropout = mlp_params.get('dropout', 0.5)
		layers = []
		in_features = input_size
		for h, act in zip(hidden_layers, activations):
			layers.append(nn.Linear(in_features, h))
			if act == 'ReLU':
				layers.append(nn.ReLU())
			elif act == 'Tanh':
				layers.append(nn.Tanh())
			elif act == 'Sigmoid':
				layers.append(nn.Sigmoid())
			elif act == 'LeakyReLU':
				layers.append(nn.LeakyReLU())
			if dropout:
				layers.append(nn.Dropout(dropout))
			in_features = h
		layers.append(nn.Linear(in_features, num_classes))
		model = nn.Sequential(*layers)
	else:
		pretrained = config['model'].get('pretrained', False)
		model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

	# optimizer
	optimizer_name = config['optimizer']['name']
	lr = config['optimizer']['lr']
	weight_decay = config['optimizer'].get('weight_decay', 0.0)
	if optimizer_name == 'AdamW':
		optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == 'SGD':
		momentum = config['optimizer'].get('sgd_params', {}).get('momentum', 0.9)
		optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
	elif optimizer_name == 'Muon':
		muon_params = config['optimizer'].get('muon_params', {})
		optimizer = muon.Muon(model.parameters(), lr=lr, weight_decay=weight_decay, **muon_params)
	else:
		raise ValueError(f"Optimizer {optimizer_name} not supported.")

	# scheduler
	scheduler = None
	if config['lr_scheduler'].get('enabled', False):
		scheduler_name = config['lr_scheduler']['name']
		if scheduler_name == 'StepLR':
			step_size = config['lr_scheduler'].get('step_size', 15)
			gamma = config['lr_scheduler'].get('gamma', 0.1)
			scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
		elif scheduler_name == 'ReduceLROnPlateau':
			factor = config['lr_scheduler'].get('factor', 0.1)
			patience = config['lr_scheduler'].get('patience', 10)
			scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

	return {
		'device': device,
		'trainloader': trainloader,
		'testloader': testloader,
		'model': model,
		'optimizer': optimizer,
		'scheduler': scheduler
	}
