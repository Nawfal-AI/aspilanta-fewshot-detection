"""Cross-family baselines: OWL-ViT, TFA, and a prototypical network.

Each represents a different established answer to the few-shot detection
problem, so the comparison shows the proposed pipeline is not just better than
nothing but better than the obvious alternatives. The TFA baseline lives in
``aspilanta.models.train`` because it is a training recipe rather than a
separate model.
"""

__all__ = ["OwlVitBaseline", "collect_exemplars", "evaluate_owlvit_variant",
           "ProtoTrainer", "ranking_metrics", "compare_embeddings"]


def __getattr__(name):
    if name in ("OwlVitBaseline", "collect_exemplars", "evaluate_owlvit_variant"):
        from . import owlvit
        return getattr(owlvit, name)
    if name in ("ProtoTrainer", "ProtoEmbedNet", "ranking_metrics",
                "compare_embeddings"):
        from . import protonet
        return getattr(protonet, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
