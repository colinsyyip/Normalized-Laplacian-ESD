# Static file with realworld dataset hyperparameters for all filters

jacobi_rw_params = {
    'cora': {
        "a": 1, "alpha": 1, "b": 1.25, "dpb": 0.1, "dpt": 0.5,
        "lr1": 0.05, "lr2": 0.001, "lr3": 0.05,
        "wd1": 0.001, "wd2": 0.0001, "wd3": 0.00005,
        "beta": {"none":1, "uniform": 0.4, "mu_f": None}
    },
    'citeseer': {
        "a": -0.5, "alpha": 0.5, "b": -0.5, "dpb": 0.9, "dpt": 0.8,
        "lr1": 0.05, "lr2": 0.001, "lr3": 0.01,
        "wd1": 5e-05, "wd2": 0.0, "wd3": 0.001,
        "beta": {"none":1, "uniform": 0.3, "mu_f": None}
    },
    'pubmed': {
        "a": 1.5, "alpha": 0.5, "b": 0.25, "dpb": 0.0, "dpt": 0.5,
        "lr1": 0.05, "lr2": 0.05, "lr3": 0.05,
        "wd1": 0.0005, "wd2": 0.0005, "wd3": 0.0,
        "beta": {"none":1, "uniform": 0.5, "mu_f": None}
    },
    'computers': {
        "a": 1.75, "alpha": 1.5, "b": -0.5, "dpb": 0.8, "dpt": 0.2,
        "lr1": 0.05, "lr2": 0.05, "lr3": 0.05,
        "wd1": 0.0001, "wd2": 0.0, "wd3": 0.0,
        "beta": {"none":1, "uniform": 0.39, "mu_f": None}
    },
    'photo': {
        "a": 1.0, "alpha": 1.5, "b": 0.25, "dpb": 0.3, "dpt": 0.3,
        "lr1": 0.05, "lr2": 0.0005, "lr3": 0.05,
        "wd1": 5e-05, "wd2": 0.0, "wd3": 0.0,
        "beta": {"none":1, "uniform": 0.47, "mu_f": None}
    },
    'chameleon': {
        "a": 0.0, "alpha": 2.0, "b": 0.0, "dpb": 0.6, "dpt": 0.5,
        "lr1": 0.05, "lr2": 0.01, "lr3": 0.05,
        "wd1": 0.0, "wd2": 0.0001, "wd3": 0.0005,
        "beta": {"none":1, "uniform": 0.29, "mu_f": None}
    },
    'actor': {
        "a": -1.0, "alpha": 1.0, "b": 0.5, "dpb": 0.9, "dpt": 0.7,
        "lr1": 0.05, "lr2": 0.05, "lr3": 0.01,
        "wd1": 0.001, "wd2": 0.0005, "wd3": 0.001,
        "beta": {"none":1, "uniform": 0.26, "mu_f": None}
    },
    'squirrel': {
        "a": 0.5, "alpha": 2.0, "b": 0.25, "dpb": 0.4, "dpt": 0.1,
        "lr1": 0.01, "lr2": 0.01, "lr3": 0.05,
        "wd1": 5e-05, "wd2": 0.0, "wd3": 0.0,
        "beta": {"none":1, "uniform": 0.76, "mu_f": None}
    },
    "texas": {
        "a": -0.5, "alpha": 0.5, "b": 0.0, "dpb": 0.8, "dpt": 0.7,
        "lr1": 0.05, "lr2": 0.005, "lr3": 0.01,
        "wd1": 0.001, "wd2": 0.0005, "wd3": 0.0005,
        "beta": {"none":1, "uniform": 0.40, "mu_f": None}
    },
    "cornell": {
        "a": -0.75, "alpha": 0.5, "b": 0.25, "dpb": 0.4, "dpt": 0.7,
        "lr1": 0.05, "lr2": 0.005, "lr3": 0.001,
        "wd1": 0.0005, "wd2": 0.0005, "wd3": 0.0001,
        "beta": {"none":1, "uniform": 0.7, "mu_f": None}
    }
}

gprgnn_rw_params = {
    "cornell": {
        "lr": 0.05,
        "dprate": 0.5,
        "alpha": 0.9,
        "beta": {"none":1, "uniform": 0.77, "mu_f": None}
    },
    "texas": {
        "lr": 0.05,
        "dprate": 0.5,
        "alpha": 1.0,
        "beta": {"none":1, "uniform": 0.91, "mu_f": None}
    },
    "actor": {
        "lr": 0.01,
        "dprate": 0.9,
        "alpha": 1.0,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.90, "mu_f": None}
    },
    "chameleon": {
        "lr": 0.05,
        "dprate": 0.7,
        "alpha": 1.0,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.09, "mu_f": None}
    },
    "squirrel": {
        "lr": 0.05,
        "dprate": 0.7,
        "alpha": 0.0,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.23, "mu_f": None}
    },
    "cora": {
        "alpha": 0.8,
        "dprate": 0.8,
        "lr": 0.05,
        "beta": {"none":1, "uniform": 0.9, "mu_f": None}
    },
    "citeseer": {
        "alpha": 0,
        "dprate": 0.3,
        "lr": 0.01,
        "beta": {"none":1, "uniform": 0.25, "mu_f": None}
    },
    "pubmed": {
        "alpha": 0.9,
        "dprate": 0.7,
        "lr": 0.05,
        "beta": {"none":1, "uniform": 0.75, "mu_f": None}
    },
    "computers": {
        "alpha": 0.6,
        "dprate": 0.4,
        "lr": 0.05,
        "beta": {"none":1, "uniform": 0.15, "mu_f": None}
    },
    "photo": {
        "alpha": 0.7,
        "dprate": 0.2,
        "lr": 0.05,
        "beta": {"none":1, "uniform": 0.05, "mu_f": None}
    }
}

bernnet_rw_params = {
    "squirrel": {
        "lr": 0.05,
        "Bern_lr": 0.01,
        "dprate": 0.6,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.44, "mu_f": None}
    },
    "chameleon": {
        "lr": 0.05,
        "Bern_lr": 0.01,
        "dprate": 0.7,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.33, "mu_f": None}
    },
    "actor": {
        "lr": 0.05,
        "Bern_lr": 0.01,
        "dprate": 0.9,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.86, "mu_f": None}
    },
    "cora": {
        "Bern_lr": 0.01,
        "dprate": 0.0,
        "beta": {"none":1, "uniform": 0.63, "mu_f": None}
    },
    "citeseer": {
        "Bern_lr": 0.01,
        "dprate": 0.5,
        "beta": {"none":1, "uniform": 0.78, "mu_f": None}
    },
    "computers": {
        "Bern_lr": 0.05,
        "dprate": 0.6,
        "beta": {"none":1, "uniform": 0.12, "mu_f": None}
    },
    "photo": {
        "Bern_lr": 0.01,
        "dprate": 0.5,
        "beta": {"none":1, "uniform": 0.05, "mu_f": None}
    },
    "texas": {
        "lr": 0.05,
        "Bern_lr": 0.002,
        "dprate": 0.5,
        "beta": {"none":1, "uniform": 0.86, "mu_f": None}
    },
    "cornell": {
        "lr": 0.05,
        "Bern_lr": 0.001,
        "dprate": 0.5,
        "beta": {"none":1, "uniform": 0.18, "mu_f": None}
    },
    "pubmed": {
        "Bern_lr": 0.01,
        "dprate": 0.0,
        "weight_decay": 0.0,
        "beta": {"none":1, "uniform": 0.05, "mu_f": None}
    }
}