import logging


def get_logger(name):
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(name)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger




