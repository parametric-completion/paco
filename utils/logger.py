import logging

import torch.distributed as dist


# Dictionary to track which loggers have been initialized
logger_initialized = {}


def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """
    Get root logger and add a keyword filter to it.
    
    The logger will be initialized if it hasn't been initialized yet. By default, a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is typically the top-level package name.
    
    Parameters
    ----------
    log_file : str, optional
        File path of log. If None, no file handler will be added. Default: None.
    log_level : int, optional
        The level of logger (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: logging.INFO.
    name : str, optional
        The name of the root logger, also used as a filter keyword. Default: 'main'.
        
    Returns
    -------
    logging.Logger
        The configured root logger with appropriate handlers and filters.
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # Add a logging filter to only include messages with the logger name
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """
    Initialize and get a logger by name.
    
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    
    Parameters
    ----------
    name : str
        Logger name, which will serve as a unique identifier.
    log_file : str or None, optional
        The log filename. If specified, a FileHandler will be added to the logger
        on the main process (rank 0). Default: None.
    log_level : int, optional
        The logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Note that only 
        the process of rank 0 will use this level, other processes will set the 
        level to "ERROR" to be silent most of the time. Default: logging.INFO.
    file_mode : str, optional
        The file mode used in opening log file ('w' for write, 'a' for append).
        Default: 'w'.
        
    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
        
    # Handle hierarchical names
    # e.g., if logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # Fix for PyTorch DDP's duplicate logging issue:
    # Starting in PyTorch 1.8.0, DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating unwanted output duplication.
    # To fix this issue, we set the root logger's StreamHandler to ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    # Create a stream handler for console output
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # Determine the current process rank for distributed training
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Only the main process (rank 0) will add a FileHandler to avoid
    # multiple processes writing to the same log file simultaneously
    if rank == 0 and log_file is not None:
        # Create a file handler with the specified mode (write or append)
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    # Configure the format and level for all handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # Set the logger level based on the process rank
    if rank == 0:
        logger.setLevel(log_level)
    else:
        # Non-main processes only show errors to avoid log duplication
        logger.setLevel(logging.ERROR)

    # Mark this logger as initialized
    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """
    Print a log message to the specified logger or console.
    
    This function provides a unified interface for logging messages
    through different channels based on the logger parameter.
    
    Parameters
    ----------
    msg : str
        The message to be logged.
    logger : logging.Logger, str, or None, optional
        The logger to use. Special values include:
        - None: The `print()` function will be used to output the message.
        - "silent": No message will be printed.
        - Other str: The logger obtained with `get_logger(logger)` will be used.
        - logging.Logger: The provided logger object will be used directly.
        Default: None.
    level : int, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Only applicable when `logger` is a Logger object or a string.
        Default: logging.INFO.
        
    Raises
    ------
    TypeError
        If the logger is not a logging.Logger object, str, "silent", or None.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass  # No output when logger is set to 'silent'
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
