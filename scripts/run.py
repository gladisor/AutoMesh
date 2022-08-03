from optuna.trial import FixedTrial
from main import heatmap_regressor

if __name__ == '__main__':
    ## For evaluating a fixed trial
    trial = FixedTrial({
        ## model
        'model': 'ParamGCN',
        'conv_layer': 'GATv2Conv',
        'act': 'LeakyReLU',
        'negative_slope': 0.04876889,
        'dropout': 0.1,
        'in_channels': 3,
        'hidden_channels': 128,
        'num_layers': 7,
        'out_channels': 8,
        'norm': 'GraphNorm',
        'heads': 3,
        'add_self_loops': True,

        'loss_func': 'FocalTverskyLoss',
        'alpha_t': 0.03401540046,
        'gamma_ft': 1.5437116,

        ## optimizer
        'opt': 'Adam',
        'lr': 0.000751094,
        'weight_decay': 4e-05
    })

    heatmap_regressor(trial)