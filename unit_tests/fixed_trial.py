from optuna.trial import FixedTrial

if __name__ == '__main__':
    trial = FixedTrial({
        'conv_layer': 'SAGEConv',
        'normalize': True,
        'act': 'GELU',
        'hidden_channels': 100,
        'lr': 0.001,
        'num_layers': 3,
        'norm': 'GraphNorm',
        'loss_func': 'FocalLoss',
        'alpha_f': 0.7,
        'gamma_f': 2.1,
        'opt': 'Adam',
        'weight_decay': 0.001
        })

    heatmap_regressor(trial)