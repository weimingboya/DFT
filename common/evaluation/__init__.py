from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice import Spice
from .tokenizer import PTBTokenizer
from .eval import COCOEvalCap

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    # metrics = (Bleu(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
