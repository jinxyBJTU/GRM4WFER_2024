class Config(object):
    def __init__(self):
        self.batch_size = 30 # case:98 ceap:30
        self.lr = 0.0001
        self.train_epochs = 50 # 50
        self.early_stop_steps = 10
        self.valid_best = True

        # False True
        self.use_uncertaiy = True
        self.use_dis_constraint = True
        self.use_crossTrail_gcn = True

        self.repeat_times = 10
        self.drop_ratio = 0.2
        self.dis_type = 'JS' # JS KL Elu
        self.graph_type = 'TA_adaptive' # TA_adaptive adaptive weak