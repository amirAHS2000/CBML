# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
from cbml_benchmark.utils.model_serialization import load_state_dict


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        criterion=None,
        proxy_optimizer=None,
    ):
        self.criterion = criterion
        self.proxy_optimizer = proxy_optimizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **metadata):
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        data = dict(metadata)
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()

        if self.criterion is not None:
            data["criterion"] = self.criterion.state_dict()
        if self.proxy_optimizer is not None:
            data["proxy_optimizer"] = self.proxy_optimizer.state_dict()

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        if name != "best_model":
            self.tag_last_checkpoint(os.path.abspath(save_file))

    def load(self, f=None, expected_classes=None, require_proxy_state=False):
        if not f and self.has_checkpoint():
            # Use the recorded checkpoint when no explicit path is provided.
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        if expected_classes is not None and checkpoint.get("class_ids") != expected_classes:
            raise ValueError("Checkpoint class IDs do not match the training dataset.")
        if require_proxy_state:
            required = ("criterion", "proxy_optimizer", "optimizer", "scheduler", "iteration")
            if any(key not in checkpoint for key in required):
                raise ValueError("Incomplete proxy training checkpoint.")
        if require_proxy_state:
            self.model.load_state_dict(checkpoint.pop("model"), strict=True)
        else:
            self._load_model(checkpoint)
        if self.criterion is not None and "criterion" in checkpoint:
            self.criterion.load_state_dict(checkpoint.pop("criterion"))
        if self.proxy_optimizer is not None and "proxy_optimizer" in checkpoint:
            self.proxy_optimizer.load_state_dict(checkpoint.pop("proxy_optimizer"))
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))
