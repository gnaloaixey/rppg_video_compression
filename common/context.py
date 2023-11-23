

class Context:
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
        Context.train_generator = None
        Context.test_generator = None
        Context.train_reader = None
        Context.test_reader = None
        Context.model_map = dict()

        Context.config  = None
        Context.non_dnn_method_list = None
        Context.config_hash = None
