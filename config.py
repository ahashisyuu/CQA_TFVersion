class Config:
    lr = 0.001
    dropout = 0.
    qs_max_len = 20
    qb_max_len = 95
    ct_max_len = 150
    char_max_len = 16
    epochs = 30
    batch_size = 60
    char_dim = 15
    l2_weight = 0

    patience = 5
    k_fold = 0
    categories_num = 3
    period = 100

    need_punct = True
    need_shuffle = True
    use_char_level = False
    load_best_model = False

    model_dir = './models/CQAModel'
    log_dir = './models/CQAModel'
    glove_filename = 'glove.6B.300d.txt'

    train_list = []
    dev_list = []
    test_list = []
