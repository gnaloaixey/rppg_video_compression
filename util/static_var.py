

class StaticVar:
    config  = None
    config_hash = None
    non_dnn_method_list = None

    train_generator = None
    test_generator = None

    train_reader = None
    test_reader = None

    model_map = dict()
    @staticmethod
    def clear_var():
        StaticVar.train_generator = None
        StaticVar.test_generator = None
        StaticVar.train_reader = None
        StaticVar.test_reader = None
        StaticVar.model_map = dict()

        StaticVar.config  = None
        StaticVar.non_dnn_method_list = None
        StaticVar.config_hash = None
