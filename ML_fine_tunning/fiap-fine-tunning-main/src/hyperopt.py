import sys

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.train import main


def optimize(params):
    # Update configuration with current hyperparameters
    config = {
        'data_dir': params['data_dir'],
        'output_dir': params['output_dir'],
        'model_dir': params['model_dir'],
        'pretrained': params['pretrained'],
        'pretrained_generator': params['pretrained_generator'],
        'pretrained_discriminator': params['pretrained_discriminator'],
        'batch_size': int(params['batch_size']),
        'image_size': params['image_size'],
        'nz': int(params['nz']),
        'ngf': int(params['ngf']),
        'ndf': int(params['ndf']),
        'nc': params['nc'],
        'lr': params['lr'],
        'beta1': params['beta1'],
        'num_epochs': params['num_epochs'],
        'log_interval': params['log_interval']
    }

    # Run the training script with the current hyperparameters
    # You can also import and call the main function directly
    # Here, we'll call the main function directly
    try:
        main(config)
        # After training, retrieve the final Loss_G as the metric to minimize
        # This is a placeholder; you should adjust based on your actual logging
        # For simplicity, we'll assume lower Loss_G is better
        # In practice, you might want to retrieve metrics from MLflow
        # Here, we return STATUS_OK without actual metric
        return {'loss': 0, 'status': STATUS_OK}
    except Exception as e:
        print(f"Error during training: {e}")
        return {'loss': sys.maxsize, 'status': STATUS_OK}

def main_hyperopt():
    # Define the hyperparameter search space
    space = {
        'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-3)),
        'beta1': hp.uniform('beta1', 0.4, 0.6),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'ngf': hp.choice('ngf', [64, 128]),
        'ndf': hp.choice('ndf', [64, 128]),
        'num_epochs': hp.choice('num_epochs', [5, 10]),
        # Add or adjust hyperparameters as needed
        # Static parameters
        'data_dir': '../data/celeba/img_align_celeba',
        'output_dir': '../output',
        'model_dir': '../models',
        'pretrained': True,
        'pretrained_generator': '../models/generator.pth',
        'pretrained_discriminator': '../models/discriminator.pth',
        'image_size': 64,
        'nz': 100,
        'nc': 3,
        'log_interval': 100
    }

    trials = Trials()
    best = fmin(
        fn=optimize,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    print("Best hyperparameters:", best)


if __name__ == "__main__":
    import numpy as np
    main_hyperopt()
