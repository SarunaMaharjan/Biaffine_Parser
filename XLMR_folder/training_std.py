from tempfile import NamedTemporaryFile
import os
import torch
from tqdm import tqdm

import utils

# ==========================================================================================
# CREATE_TRAINER
# ==========================================================================================
def create_trainer(model, **kwargs):
    """
    CHANGED: Switched to AdamW optimizer and implemented differential learning rates.
    """
    lr = kwargs.pop("lr", 2e-5)
    
    # --- NEW: Set up parameter groups for differential learning rates ---
    # Give a larger learning rate to new components (the CNN)
    # and a smaller one to the pretrained transformer base.
    
    # Note: You need to know the names of your new CNN layers. 
    # I will assume they contain "cnn" in their name.
    transformer_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "transformer" in name: # Assumes your encoder is named 'transformer' in the TransformerEncoder class
                transformer_params.append(param)
            else:
                # This group includes your CNN layers and the Biaffine MLPs
                new_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': transformer_params, 'lr': lr}, # Small LR for the base
        {'params': new_params, 'lr': lr * 10}      # 10x larger LR for the new parts
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    # --- END OF NEW CODE ---
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.75 ** (step / 5000))
    kwargs.setdefault("max_grad_norm", 5.0)
    kwargs.setdefault("step", forward)
    
    trainer = utils.training.Trainer(model, (optimizer, scheduler), **kwargs)
    trainer.add_metric("loss", "head_loss", "head_accuracy", "deprel_loss", "deprel_accuracy")
    trainer.add_callback(ProgressCallback())
    return trainer

# ==========================================================================================
# FORWARD FUNCTION
# ==========================================================================================
def forward(model, batch):
    return model(batch)


# ==========================================================================================
# CALLBACKS
# ==========================================================================================

class PrintCallback(utils.training.Callback):
    def __init__(self, printer=print):
        self.printer = printer

    def on_epoch_start(self, context, **kwargs):
        pass

    def on_epoch_end(self, context, **kwargs):
        pass

    def on_loop_end(self, context, metrics):
        if not context.train:
            self.printer(f"[eval] epoch {context.epoch}")

class ProgressCallback(utils.training.ProgressCallback):
    def on_epoch_start(self, context, **kwargs):
        pass

    def on_epoch_end(self, context, **kwargs):
        pass

    def on_step_end(self, context, **kwargs):
        super().on_step_end(context, **kwargs)
        if context.train:
            loss = kwargs["output"]["loss"].item()
            correct, total = kwargs["output"]["head_accuracy"]
            accuracy = correct / total if total > 0 else float("nan")
            pbar_dict = {
                "epoch": context.epoch,
                "loss": f"{loss:.4f}",
                "accuracy": f"{accuracy:.4f}",
            }
            self.training_pbar.set_postfix(pbar_dict)

class SaveCallback(utils.training.Callback):
    def __init__(self, save_dir, monitor="eval/UAS", mode="max"):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_value = -float('inf') if mode == "max" else float('inf')
        self.printer = tqdm.write
        self.trainer = None 

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_start(self, context, **kwargs):
        pass

    def on_epoch_end(self, context, **kwargs):
        pass
    
    def on_loop_end(self, context, metrics):
        if context.train:
            return
            
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return

        improved = (self.mode == "max" and current_value > self.best_value) or \
                   (self.mode == "min" and current_value < self.best_value)

        if improved:
            self.printer(f"Metric {self.monitor} improved from {self.best_value:.4f} to {current_value:.4f}. Saving model...")
            self.best_value = current_value
            
            checkpoint_path = os.path.join(self.save_dir, f"step-{context.step}.ckpt")
            
            checkpoint = {
                'model': context.model.state_dict(),
                'optimizer': self.trainer.optimizers[0].state_dict(),
                'scheduler': self.trainer.optimizers[1].state_dict(),
                'step': context.step,
                'epoch': context.epoch,
                'best_value': self.best_value
            }
            
            torch.save(checkpoint, checkpoint_path)

class EvaluateCallback(utils.training.Callback):
    printer = tqdm.write

    def __init__(self, gold_file, deprel_map, verbose=False):
        self.gold_file = gold_file
        self.deprel_map = deprel_map
        self.verbose = verbose
        self.result = {}
        self._outputs = []

    def on_epoch_start(self, context, **kwargs):
        pass

    def on_epoch_end(self, context, **kwargs):
        pass

    def on_step_end(self, context, output):
        if context.train:
            return
        
        # --- THE FIX IS HERE ---
        # Changed ["lengths"] to output["lengths"] to get the actual data.
        heads, deprels, lengths = output["heads"], output["deprels"], output["lengths"]
        
        assert heads.size(0) == len(lengths)
        heads = (idxs[:n] for idxs, n in zip(heads.tolist(), lengths))
        if deprels is not None:
            deprel_map = self.deprel_map
            deprels = (
                [deprel_map[idx] for idx in idxs[:n]] for idxs, n in zip(deprels.tolist(), lengths)
            )
        else:
            deprels = ([None] * n for n in lengths)
        self._outputs.extend(zip(heads, deprels))

    def on_loop_end(self, context, metrics):
        if context.train:
            metrics.update({"train/UAS": float("nan"), "train/LAS": float("nan")})
            return
        with NamedTemporaryFile(mode="w") as f:
            utils.conll.dump_conll(self._yield_prediction(), f)
            self.result.update(utils.conll.evaluate(self.gold_file, f.name, self.verbose))
        metrics.update({"eval/UAS": self.result["UAS"], "eval/LAS": self.result["LAS"]})
        
        uas = self.result.get("UAS", 0.0)
        las = self.result.get("LAS", 0.0)
        self.printer(f"\n--- Evaluation Results ---")
        self.printer(f"UAS: {uas:.2f}%")
        self.printer(f"LAS: {las:.2f}%")
        self.printer(f"--------------------------\n")

        self._outputs.clear()

    def on_evaluate_end(self, context, metrics):
        metrics.update({"eval/UAS": self.result["UAS"], "eval/LAS": self.result["LAS"]})
        if self.verbose:
            self.printer(self.result["raw"])

    def _yield_prediction(self):
        for tokens, (heads, deprels) in zip(utils.conll.read_conll(self.gold_file), self._outputs):
            if len(heads) != len(tokens):
                raise ValueError("heads must be aligned with tokens")
            for token, head, deprel in zip(tokens, heads, deprels):
                token.update(head=head, deprel=deprel)
            yield tokens
