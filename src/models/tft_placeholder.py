"""
Temporal Fusion Transformer placeholder.

To use:
1) pip install torch pytorch-lightning pytorch-forecasting
2) Convert weekly panel into time-series format expected by PyTorch Forecasting:
   - group_id: "market"
   - time_idx: integer weekly index
   - target: next-week excess return (regression) or direction (classification)
3) Configure encoder/decoder lengths (e.g., 24/1) and static/covariate features.
4) Train with early stopping and loggers; export predictions aligned to dates.
"""
