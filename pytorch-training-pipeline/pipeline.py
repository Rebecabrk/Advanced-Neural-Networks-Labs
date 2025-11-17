import torch
import timm
from torch.cuda.amp import GradScaler
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.optim as optim
import torchvision
import muon
from tqdm import tqdm

from datasets.caching import CachedDataset

def setup_pipeline(config):
	print("Setting up the training pipeline...")
	device = torch.device('cuda' if torch.cuda.is_available() and config['global']['device'] == 'auto' else 'cpu')
	enable_half = device.type != "cpu"
	scaler = GradScaler(device, enabled=enable_half)

	print("Grad scaler is enabled:", enable_half)
	if device.type == "cuda":
    # This flag tells pytorch to use the cudnn auto-tuner to find the most efficient convolution algorithm for
    # This training.
		torch.backends.cudnn.benchmark = True
		torch.set_float32_matmul_precision('high')
		print("Benchmark set to true with high float32 precision")
	else:
		print("Cuda is not used")

    # transforms
	test_transforms = [
		v2.ToTensor(), 
		v2.ToDtype(torch.float32, scale=True)
	]
	cacheable_transforms = [
		v2.ToTensor(), 
		v2.ToDtype(torch.float32, scale=True)]
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

	# trainset = CachedDataset(trainset_base, runtime_transform)

	# data loaders
	batch_size = config['training'].get('batch_size', 128)
	num_workers = config['global'].get('num_workers', 4)
	trainloader = torch.utils.data.DataLoader(trainset_base, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
		for param in model.parameters():
			param.requires_grad = False
		for param in model.get_classifier().parameters():
			param.requires_grad = True

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
		'scheduler': scheduler,
		'scaler': scaler,
		'enable_half': enable_half
	}


def train(model, train_loader, device, criterion, optimizer, scaler, enable_half):
    model.train()
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        # inputs, targets = cutmix_or_mixup(inputs, targets)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        inputs = inputs.view(inputs.size(0), -1) 
		# if targets.ndim > 1:
        #     # We do this when cutmix or mixup was used, transforming the hard labels into soft labels
        #     targets = targets.argmax(1)
            
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total

def run_pipeline(config):
	setup = setup_pipeline(config)
	device = setup['device']
	trainloader = setup['trainloader']
	testloader = setup['testloader']
	model = setup['model'].to(device)
	optimizer = setup['optimizer']
	scheduler = setup['scheduler']
	scaler = setup['scaler']
	enable_half = setup['enable_half']
	# criterion = setup['criterion']
	criterion = nn.CrossEntropyLoss()

	model = model.to(device)
	# model = torch.jit.script(model) # compilation

	best = 0.0
	best_epoch = 0
	epochs = list(range(config['training'].get('epochs', 50)))

	with tqdm(epochs) as tbar:
		for epoch in tbar:
			train_acc = train(
				model, trainloader, 
				device, criterion, 
				optimizer, scaler, 
				enable_half)
			if config['lr_scheduler']['name'] == 'ReduceLROnPlateau':
				scheduler.step(train_acc)
			elif scheduler is not None:
				scheduler.step()
			
			if train_acc > best:
				best = train_acc
				best_epoch = epoch
				
			tbar.set_description(f"Train: {train_acc:.2f}, Best: {best:.2f} at epoch {best_epoch}")
