import argparse
from utils.train_utils import prepare_dataloaders, prepare_objectives, prepare_models
import yaml
import torch
from trainer import train_and_evaluate
from utils.helper import get_args_parser
from utils.preprocess_data import preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Metadata training script", parents=[get_args_parser()])
    
    parser.add_argument('--path',default= '/home/toannn/PythonCode/ISIC2024/dataset/data2024', type=str, required=True,
                        help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    data, meta_handler, meta_features = preprocess(config)
    input_dim = data['x_train'].shape[1]

    training_loader, validation_loader = prepare_dataloaders(config, data)
    
    model = prepare_models(input_dim, config)

    criterion, optimizer, scheduler = prepare_objectives(config, model)

    if config['continue_training']:
        model.load_state_dict(torch.load(
            config['trained_weights'])['model_state_dict'])
        epoch = torch.load(config['trained_weights'])['epoch']
    else:
        epoch = 0

    train_and_evaluate(config, training_loader, validation_loader,
                       model, criterion, optimizer, scheduler, epoch)
