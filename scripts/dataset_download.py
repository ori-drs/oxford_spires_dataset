from huggingface_hub import snapshot_download

hf_repo_id = "ori-drs/oxford_spires_dataset"

# example_pattern = "sequences/*" # download all sequences
example_pattern = "sequences/2024-03-12-keble-college-02/*"  # download all files in a particular sequence
# example_pattern = "reconstruction_benchmark/* # download the whole reconstruction benchmark
# example_pattern = "novel_view_synthesis_benchmark" # download the whole novel view synthesis benchmark
# example_pattern = "ground_truth_map/*" # download all ground truth maps


local_dir = "oxford_spires_dataset_download"

snapshot_download(
    repo_id=hf_repo_id,
    allow_patterns=example_pattern,
    local_dir=local_dir,
    repo_type="dataset",
    use_auth_token=False,
)
