"""KVA: training-free approximate late-layer prefill (reference implementation).

The first `split` layers run exactly for every prompt token. For the prefix rows (all but the
last `tail` tokens) a closed-form ridge projector maps the layer-`split` boundary state to each
later layer's predicted input, and each later layer only fills its cache from that prediction:
attention layers write K/V, Gated DeltaNet layers advance their recurrent and conv state. The
tail and decoding run exactly against those caches.
"""
