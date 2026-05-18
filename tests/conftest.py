import os

# Force single-threaded OpenMP to prevent a PyTorch ↔ XGBoost OpenMP segfault on Apple Silicon.
# Both libraries bundle their own libomp; parallel execution causes a double-initialization crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
