
def dict_from_str(dict_str):
    while True:
        try:
            dict_ = eval(dict_str)
        except NameError as e:
            key = e.message.split("'")[1]
            dict_str = dict_str.replace(key, "'{}'".format(key))
        else:
            return file