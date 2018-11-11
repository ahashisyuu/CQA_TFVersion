class Config:
    lr = 1e-4
    dropout = 0.2
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
    categories_num = 2
    period = 50

    need_punct = False
    wipe_num = 0

    word_trainable = False
    concat_q = False
    need_shuffle = True
    use_char_level = False
    load_best_model = True

    model_dir = './models/CQAModel'
    log_dir = './models/CQAModel'
    glove_filename = 'glove.6B.300d.txt'

    train_list = []
    dev_list = []
    test_list = []
