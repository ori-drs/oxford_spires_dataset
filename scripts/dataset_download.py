from huggingface_hub import snapshot_download

hf_repo_id = "ori-drs/oxford_spires_dataset"

############## 1. Download a specific sequence or benchmark ##############

# example_pattern = "sequences/*" # download all sequences
example_pattern = "sequences/2024-03-12-keble-college-02/*"  # download all files in a particular sequence
# example_pattern = "reconstruction_benchmark/* # download the whole reconstruction benchmark
# example_pattern = "novel_view_synthesis_benchmark" # download the whole novel view synthesis benchmark
# example_pattern = "ground_truth_map/*" # download all ground truth maps


local_dir = "download"

snapshot_download(
    repo_id=hf_repo_id,
    allow_patterns=example_pattern,
    local_dir=local_dir,
    repo_type="dataset",
    use_auth_token=False,
)

############## 2. Download the core sequences ##############
core_sequences = [
    "sequences/2024-03-12-keble-college-02/*",
    "sequences/2024-03-12-keble-college-03/*",
    "sequences/2024-03-12-keble-college-04/*",
    "sequences/2024-03-12-keble-college-05/*",
    "sequences/2024-03-13-observatory-quarter-01/*",
    "sequences/2024-03-13-observatory-quarter-02/*",
    "sequences/2024-03-14-blenheim-palace-01/*",
    "sequences/2024-03-14-blenheim-palace-02/*",
    "sequences/2024-03-14-blenheim-palace-05/*",
    "sequences/2024-03-18-christ-church-01/*",
    "sequences/2024-03-18-christ-church-02/*",
    "sequences/2024-03-18-christ-church-03/*",
    "sequences/2024-03-18-christ-church-05/*",
    "sequences/2024-05-20-bodleian-library-02/*",
]


for sequence in core_sequences:
    example_pattern = sequence
    print(f"Downloading sequence: {example_pattern}")
    snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=example_pattern,
        local_dir=local_dir,
        repo_type="dataset",
        use_auth_token=False,
    )
