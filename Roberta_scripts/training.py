from tempfile import NamedTemporaryFile

import torch
from tqdm import tqdm

import utils

# ==========================================================================================
# MODIFIED CREATE_TRAINER
# ==========================================================================================
def create_trainer(model, **kwargs):
    """
    CHANGED: Switched to AdamW optimizer, which is standard for transformer-based models.
    The default learning rate is also adjusted.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs.pop("lr", 2e-5),  # Common learning rate for fine-tuning transformers
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # This scheduler is fine, but linear warmup-decay is also common.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.75 ** (step / 5000))
    kwargs.setdefault("max_grad_norm", 5.0)
    
    # CHANGED: The 'step' function is now the simplified 'forward' function below
    kwargs.setdefault("step", forward)
    
    trainer = utils.training.Trainer(model, (optimizer, scheduler), **kwargs)
    trainer.add_metric("loss", "head_loss", "head_accuracy", "deprel_loss", "deprel_accuracy")
    trainer.add_callback(ProgressCallback())
    return trainer

# ==========================================================================================
# SIMPLIFIED FORWARD FUNCTION
# ==========================================================================================
def forward(model, batch):
    """
    This function is now extremely simple.
    It just passes the dictionary batch directly to the model's forward pass.
    The model now handles all the internal logic.
    """
    # The model's forward pass now returns a dictionary with all results
    return model(batch)


# ==========================================================================================
# NO CHANGES NEEDED FOR CALLBACKS
# The callbacks work with the dictionary ('output') returned by the forward function,
# so they do not need to be changed.
# ==========================================================================================
class ProgressCallback(utils.training.ProgressCallback):
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


# In training.py

class EvaluateCallback(utils.training.Callback):
    printer = tqdm.write

    def __init__(self, gold_file, deprel_map, verbose=False):
        self.gold_file = gold_file
        self.deprel_map = deprel_map
        self.verbose = verbose
        self.result = {}
        self._outputs = []

    def on_step_end(self, context, output):
        if context.train:
            return
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
        
        # --- NEW CODE ADDED HERE ---
        # This will print the final scores to the console.
        uas = self.result.get("UAS", 0.0)
        las = self.result.get("LAS", 0.0)
        self.printer(f"\n--- Evaluation Results ---")
        self.printer(f"UAS: {uas:.2f}%")
        self.printer(f"LAS: {las:.2f}%")
        self.printer(f"--------------------------\n")
        # --- END OF NEW CODE ---

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