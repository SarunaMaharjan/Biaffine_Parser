import torch
import utils
from tempfile import NamedTemporaryFile
import os

def create_trainer(model, **kwargs):
    callbacks = kwargs.pop("callbacks", [])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs.pop("lr", 2e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.75 ** (step / 5000))
    
    kwargs.setdefault("step", forward)
    kwargs.setdefault("max_grad_norm", 5.0)
    
    trainer = utils.training.Trainer(model, (optimizer, scheduler), **kwargs)
    trainer.add_metric("loss", "head_accuracy")
    trainer.add_callback(ProgressCallback())
    for cb in callbacks:
        trainer.add_callback(cb)
    return trainer

def forward(model, batch):
    """The model's forward pass returns all results, including decoded predictions during eval."""
    return model(batch)

class ProgressCallback(utils.training.ProgressCallback):
    """This callback correctly calculates and saves the epoch's average metrics."""
    def on_loop_start(self, context, **kwargs):
        super().on_loop_start(context, **kwargs)
        if context.train:
            self.total_loss = 0.0
            self.total_correct = 0
            self.total_count = 0

    def on_step_end(self, context, **kwargs):
        super().on_step_end(context, **kwargs)
        if context.train:
            output = kwargs["output"]
            self.total_loss += output["loss"].item()
            correct, total = output["head_accuracy"]
            self.total_correct += correct
            self.total_count += total
            
            avg_loss = self.total_loss / context.step
            avg_uas = self.total_correct / self.total_count if self.total_count > 0 else 0.0
            
            self.training_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "UAS": f"{avg_uas:.4f}"})

    def on_loop_end(self, context, metrics):
        if context.train:
            avg_loss = self.total_loss / context.max_steps
            avg_uas = self.total_correct / self.total_count if self.total_count > 0 else 0.0
            metrics.update({"train/loss": avg_loss, "train/UAS": avg_uas})
        
        # --- THE FIX IS HERE ---
        # The parent method only needs 'context'.
        super().on_loop_end(context)
        # -----------------------


class EvaluateCallback(utils.training.Callback):
    """This callback correctly calculates UAS and LAS on the evaluation set."""
    def __init__(self, gold_file, deprel_map):
        self.gold_file = gold_file
        self.deprel_map = deprel_map
        self._outputs = []

    def on_step_end(self, context, output):
        if context.train:
            return
            
        lengths = output["lengths"].cpu().tolist()
        heads_list = output["heads"].cpu().tolist()
        deprels_list = output["deprels"].cpu().tolist()
        
        for i in range(len(lengths)):
            # Slicing from 1 to length to remove the <root> token
            num_words = lengths[i] - 1
            self._outputs.append((heads_list[i][1:lengths[i]], deprels_list[i][1:lengths[i]]))

    def on_loop_end(self, context, metrics):
        if context.train:
            return
            
        with NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            utils.conll.dump_conll(self._yield_prediction(), f)
            fname = f.name
        
        result = utils.conll.evaluate(self.gold_file, fname)
        os.remove(fname)
        
        metrics.update({"eval/UAS": result["UAS"], "eval/LAS": result["LAS"]})
        self._outputs.clear()

    def _yield_prediction(self):
        for i, tokens in enumerate(utils.conll.read_conll(self.gold_file)):
            if i >= len(self._outputs): break
            heads, deprels = self._outputs[i]
            for j, token in enumerate(tokens):
                if j < len(heads) and j < len(deprels):
                    token.update(head=heads[j], deprel=self.deprel_map.get(deprels[j], '_'))
            yield tokens