import torch
import timm
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
from sam import SAM
# import ipdb

from utils.caching import CachedDataset

def setup_pipeline(config):
	print("Setting up the training pipeline...")
	device = torch.device('cuda' if torch.cuda.is_available() and config['global']['device'] == 'auto' else 'cpu')

	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
		torch.set_float32_matmul_precision('high')
		print("Benchmark set to true with high float32 precision")
	else:
		print("Cuda is not used")

	# transforms
	test_transforms = []
	cacheable_transforms = []
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

	# Add default resize for OxfordIIITPet if no resizing present
	dataset_name = config['data']['dataset_name']
	has_resize = any(isinstance(t, v2.Resize) for t in runtime_transforms)
	if dataset_name == 'OxfordIIITPet' and not has_resize:
		# 224x224 is a common size for pretrained models
		runtime_transforms.insert(0, v2.Resize((224, 224)))

	# Always add ToImage and ToDtype after augmentations but before normalization
	test_transforms.extend(runtime_transforms)
	test_transforms.append(v2.ToImage())
	test_transforms.append(v2.ToDtype(torch.float32, scale=True))
	cacheable_transforms.extend(runtime_transforms)
	cacheable_transforms.append(v2.ToImage())
	cacheable_transforms.append(v2.ToDtype(torch.float32, scale=True))

	if normalization_cfg is not None:
		mean = normalization_cfg.get('mean', [0.5, 0.5, 0.5])
		std = normalization_cfg.get('std', [0.5, 0.5, 0.5])
		test_transforms.append(v2.Normalize(mean=mean, std=std))
		cacheable_transforms.append(v2.Normalize(mean=mean, std=std))

	test_transforms = v2.Compose(test_transforms)
	cacheable_transform = v2.Compose(cacheable_transforms)
	runtime_transform = None  # runtime_transforms are now included above

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

	# Split trainset into train and validation
	val_split = config['training'].get('val_split', 0.2)  # default 20% validation
	train_len = int((1 - val_split) * len(trainset))
	val_len = len(trainset) - train_len
	train_subset, val_subset = torch.utils.data.random_split(trainset, [train_len, val_len])

	# data loaders
	batch_size = config['training'].get('batch_size', 128)
	num_workers = config['global'].get('num_workers', 4)
	trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
		
		optimizer_name = config['optimizer']['name']
		use_bias = not (optimizer_name == 'Muon')
		print(f"Using bias in MLP layers: {use_bias}")

		for h, act in zip(hidden_layers, activations):
			layers.append(nn.Linear(in_features, h, bias=use_bias))
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
		layers.append(nn.Linear(in_features, num_classes, bias=use_bias))
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
		optimizer = optim.Muon(model.parameters(), lr=lr, weight_decay=weight_decay, **muon_params)
	elif optimizer_name == 'SAM':
		sam_params = config['optimizer'].get('sam_params', {})
		base_optimizer_name = sam_params.get('base_optimizer', 'SGD')
		rho = sam_params.get('rho', 0.05)
		base_optimizer_params = {}
		if base_optimizer_name == 'SGD':
			base_optimizer = optim.SGD
		elif base_optimizer_name == 'Adam':
			base_optimizer = optim.Adam
		elif base_optimizer_name == 'AdamW':
			base_optimizer = optim.AdamW
		else:
			raise ValueError(f"SAM base optimizer {base_optimizer_name} not supported.")
		optimizer = SAM(model.parameters(), base_optimizer, rho=rho)
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
		'valloader': valloader,
		'testloader': testloader,
		'model': model,
		'optimizer': optimizer,
		'scheduler': scheduler
	}

def evaluate(model, val_loader, device, criterion):
	model.eval()
	correct = 0
	total = 0
	val_loss = 0.0
	with torch.no_grad():
		for inputs, targets in val_loader:
			inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
			if isinstance(model, nn.Sequential):
				inputs = inputs.view(inputs.size(0), -1)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			val_loss += loss.item() * targets.size(0)
			predicted = outputs.argmax(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	avg_loss = val_loss / total
	accuracy = 100.0 * correct / total
	return accuracy, avg_loss

def train(model, train_loader, device, criterion, optimizer):
	model.train()
	correct = 0
	total = 0

	is_sam_optimizer = isinstance(optimizer, SAM)

	for inputs, targets in train_loader:
		inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

		if isinstance(model, nn.Sequential): # or isinstance(optimizer, optim.Muon):
			inputs = inputs.view(inputs.size(0), -1)

		if is_sam_optimizer:
			# --- SAM Training Logic (Two-Step Update) ---
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step(lambda: criterion(model(inputs), targets))
		else:
			# --- Standard Optimizer Logic (One-Step Update) ---
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		if is_sam_optimizer:
			with torch.no_grad():
				outputs = model(inputs)

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
	criterion = nn.CrossEntropyLoss()

	model = model.to(device)

	best = 0.0
	best_epoch = 0
	epochs = list(range(config['training'].get('epochs', 50)))
	
	wandb.init(project=config['logging'].get('project_name', "Default_Project"), 
			config=config, name=config['logging'].get('run_name', "Default_Run"))
	writer = SummaryWriter(log_dir="runs/" + config['logging'].get('run_name', "Default_Run"))

	batch_scheduler_cfg = config['training'].get('batch_scheduler', {})
	batch_scheduler_enabled = batch_scheduler_cfg.get('enabled', False)
	schedule_epochs = batch_scheduler_cfg.get('schedule_epochs', [])
	batch_size_increments = batch_scheduler_cfg.get('batch_size_increments', [])

	with tqdm(epochs) as tbar:
		for epoch in tbar:
			# Batch size scheduler logic
			if batch_scheduler_enabled and epoch in schedule_epochs:
				idx = schedule_epochs.index(epoch)
				if idx < len(batch_size_increments):
					new_batch_size = batch_size_increments[idx]
					trainloader = torch.utils.data.DataLoader(
						setup['trainloader'].dataset,
						batch_size=new_batch_size,
						shuffle=True,
						num_workers=config['global'].get('num_workers', 4)
					)
					print(f"[BatchScheduler] Epoch {epoch}: Updated train batch size to {new_batch_size}")

			print(f"Epoch {epoch}: Current batch size: {trainloader.batch_size}")
			train_acc = train(
				model, trainloader, 
				device, criterion, 
				optimizer)
			val_acc, val_loss = evaluate(
				model, setup['valloader'], 
				device, criterion)
			# Wandb logging
			wandb.log({
			  "train_acc": train_acc,
			  "val_acc": val_acc,
			  "val_loss": val_loss,
			  "epoch": epoch,
			  "learning_rate": optimizer.param_groups[0]['lr']
			})
			# TensorBoard logging
			writer.add_scalar("Accuracy/train", train_acc, epoch)
			writer.add_scalar("Accuracy/val", val_acc, epoch)
			writer.add_scalar("Loss/val", val_loss, epoch)
			writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

			if config['lr_scheduler']['name'] == 'ReduceLROnPlateau':
				scheduler.step(val_loss)
			elif scheduler is not None:
				scheduler.step()

			if train_acc > best:
				best = train_acc
				best_epoch = epoch

			tbar.set_description(f"Train: {train_acc:.2f}, Best: {best:.2f} at epoch {best_epoch}")

	writer.close()
