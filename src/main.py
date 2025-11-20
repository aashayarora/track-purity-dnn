import yaml
import torch
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from model import TrackPurityDNN
from dataset import TrackDataset, MinMaxScaler
from trainer import LightningTrainer, create_trainer

def load_config(config_path='../config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path='config.yaml'):
    config = load_config(config_path)
    
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    callback_config = config['callbacks']
    logger_config = config['logger']
    trainer_config = config['trainer']
    
    model = TrackPurityDNN(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        dropout=model_config['dropout']
    )
    
    dataset = TrackDataset(
        data_config['input_files'], 
        transform=MinMaxScaler,
        n_workers=data_config.get('n_workers', None),
        batch_size=data_config.get('batch_size', 100),
        lazy_load=data_config.get('lazy_load', True)
    )
    
    if data_config.get('lazy_load', True):
        print("Computing class weights from first batch...")
        first_batch = torch.load(dataset.batch_cache_files[0], weights_only=False)
        sample_labels = first_batch['labels'].numpy().flatten()
        n_fake = (sample_labels == 0).sum()
        n_real = (sample_labels == 1).sum()
        # Estimate total from sample
        sample_ratio = len(sample_labels) / len(dataset)
        estimated_fake = int(n_fake / sample_ratio)
        estimated_real = int(n_real / sample_ratio)
        pos_weight = estimated_fake / estimated_real
        print(f"Estimated class distribution - Real: {estimated_real:,}, Fake: {estimated_fake:,}")
        print(f"Estimated class ratio (Real:Fake): {estimated_real/estimated_fake:.2f}:1")
    else:
        labels = dataset.labels.numpy().flatten()
        n_fake = (labels == 0).sum()
        n_real = (labels == 1).sum()
        pos_weight = n_fake / n_real
        print(f"Class distribution - Real: {n_real:,}, Fake: {n_fake:,}")
        print(f"Class ratio (Real:Fake): {n_real/n_fake:.2f}:1")
    
    print(f"Using pos_weight: {pos_weight:.4f}")
    
    lightning_model = LightningTrainer(
        model, 
        learning_rate=training_config['learning_rate'],
        scheduler_config=training_config.get('scheduler'),
        pos_weight=pos_weight
    )
    
    train_size = int(data_config['train_split'] * len(dataset))
    val_size = int(data_config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(data_config['random_seed'])
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=True, 
        num_workers=training_config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        num_workers=training_config['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        num_workers=training_config['num_workers']
    )
    
    trainer = create_trainer(
        training_config=training_config,
        callback_config=callback_config,
        logger_config=logger_config,
        trainer_config=trainer_config
    )
    
    trainer.logger.log_hyperparams({
        'input_dim': model_config['input_dim'],
        'hidden_dims': model_config['hidden_dims'],
        'dropout': model_config['dropout'],
        'learning_rate': training_config['learning_rate'],
        'batch_size': training_config['batch_size'],
        'max_epochs': training_config['max_epochs'],
        'train_split': data_config['train_split'],
        'val_split': data_config['val_split'],
        'random_seed': data_config['random_seed'],
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'scheduler_type': training_config.get('scheduler', {}).get('type', 'None'),
        'optimizer': 'Adam',
    })
    
    trainer.fit(lightning_model, train_loader, val_loader)
    trainer.test(lightning_model, test_loader)

    if trainer.global_rank == 0:
        onnx_path = training_config.get('onnx_path', trainer.logger.log_dir + '/final_model.onnx')
        dummy_input = torch.randn(1, model_config['input_dim'])
        torch.onnx.export(model, dummy_input, onnx_path)

if __name__ == '__main__':    
    parser = ArgumentParser(description='Train track purity DNN')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    
    main(config_path=args.config)
