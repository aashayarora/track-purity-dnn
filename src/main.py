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

    lightning_model = LightningTrainer(
        model, 
        learning_rate=training_config['learning_rate'],
        scheduler_config=training_config.get('scheduler')
    )
    
    dataset = TrackDataset(data_config['input_files'], transform=MinMaxScaler)
    
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
