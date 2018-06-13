def init_model(**kwargs):
    model = {}
    for k,v in kwargs.items():
        if type(v) == str:
            model[k] = v.lower()
        else:
            model[k] = v
    return model