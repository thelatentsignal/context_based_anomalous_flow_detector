dataset_id = "unsw_nb15"
version = "v1"

from context_based_anomalous_flow_detector.training.trainer_handler import (
    TrainerHandler,
    TrainingParams,
)

trainer = TrainerHandler()
params = TrainingParams(seed=42, batch_size=64, num_epochs=5, d_model=128, n_heads=4, num_layers=4)
model_dir = trainer.run(dataset_id, version, params=params)

print("Trained model written to:", model_dir)
