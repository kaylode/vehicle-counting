import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Logger():
    """
    Logger for Tensorboard visualization
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        
        if self.log_dir is None:
            self.log_dir = os.path.join('./loggers/runs',datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.datetime = os.path.basename(self.log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def write(self, tags, values, step):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """
        if not isinstance(tags, list):
            tags = list(tags)
        if not isinstance(values, list):
            values = list(values)

        for i, (tag, value) in enumerate(zip(tags,values)):
            self.writer.add_scalar(tag, value, step)

    def write_image(self, tag, image, step):
        """
        Write a matplotlib fig to tensorboard
        :param tags: (str) tag for log
        :param image: (image) image to log
        :param step: (int) logging step
        """

        self.writer.add_figure(tag, image, global_step=step)

