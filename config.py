class Config:
    lr = 0.05
    dropout = 0.
    max_len = 150
    char_max_len = 16
    epochs = 30
    batch_size = 30
    char_dim = 100
    l2_weight = 1e-6

    patience = 5
    k_fold = 0
    categories_num = 3
    period = 100

    need_punct = True
    need_shuffle = True
    use_char_level = True
    load_best_model = True

    model_dir = './models/CQAModel'
    log_dir = './models/CQAModel'
    glove_filename = 'glove.6B.300d.txt'

    train_list = []
    dev_list = []
    test_list = []
