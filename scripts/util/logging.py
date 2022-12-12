import logging
import logging.handlers

def get_logger():
    logger = logging.getLogger('Logger')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    logStreamHandler = logging.StreamHandler()
    logStreamHandler.setLevel(logging.INFO)
    logStreamHandler.setFormatter(formatter)
    
    logger.addHandler(logStreamHandler)

    return logger  

logger = get_logger()