class Config:
    lr = 1e-4
    dropout = 0.2
    qs_max_len = 20
    qb_max_len = 95
    ct_max_len = 150
    char_max_len = 16
    epochs = 50
    batch_size = 30
    char_dim = 15
    l2_weight = 0

    patience = 5
    k_fold = 0
    categories_num = 2
    period = 50

    need_punct = False
    wipe_num = 0

    word_trainable = False
    concat_q = True
    need_shuffle = True
    use_char_level = True
    load_best_model = False

    model_dir = './models/CQAModel'
    log_dir = './models/CQAModel'
    glove_filename = 'word2vec_dim200_domain_specific.pkl'

    train_list = []
    dev_list = []
    test_list = []
