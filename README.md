# Smol Models 🤏

Welcome to Smol Models, a family of efficient and lightweight AI models from Hugging Face. Our mission is to create powerful yet compact models, for text and vision, that can run effectively on-device while maintaining strong performance.

**News 📰**
- **Introducing [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), the best public math pretraining dataset 🚀**
- Added continual pretraining code for Llama 3.2 3B on FineMath & FineWeb-Edu with `nanotron`

## 💬 SmolLM2 (Language Model)
[SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) is our family of compact language models available in three sizes:
- **SmolLM2-135M**: Ultra-lightweight model for basic text tasks
- **SmolLM2-360M**: Balanced model for general use
- **SmolLM2-1.7B**: Our most capable language model, available at **🤏 SmolLM2-1.7B-Instruct** [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

All models have instruction-tuned versions optimized for assistant-like interactions. Find them in our [SmolLM2 collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

## 👁️ SmolVLM (Vision Language Model)
[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) is our compact multimodal model that can:
- Process both images and text and perform tasks like visual QA, image description, and visual storytelling
- Handle multiple images in a single conversation
- Run efficiently on-device

## Repository Structure
```
smollm/
├── text/               # SmolLM2 related code and resources
├── vision/            # SmolVLM related code and resources
└── tools/             # Shared utilities and inference tools
    ├── smol_tools/    # Lightweight AI-powered tools
    ├── smollm_local_inference/
    └── smolvlm_local_inference/
```

## Getting Started

### SmolLM2
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

messages = [{"role": "user", "content": "Write a 100-word article on 'Benefits of Open-Source in AI research"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
```

### SmolVLM
```python
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What's in this image?"}
        ]
    }
]
```

## Ecosystem
<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/RvHjdlRT5gGQt5mJuhXH9.png" width="700"/>
</div>

## Resources

### Documentation
- [SmolLM2 Documentation](text/README.md)
- [SmolVLM Documentation](vision/README.md)
- [Local Inference Guide](tools/README.md)

### Pretrained Models
- [SmolLM2 Models Collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)
- [SmolVLM Model](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

### Datasets
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - Our instruction-tuning dataset
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) - Mathematics pretraining dataset
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Educational content pretraining dataset

---

## EE-628: Getting Started

Quick step-by-step guide on how to pretrain models (assuming you are working on `$HOME`).
1. Clone this repo on RCP.
   ```
   git clone git@github.com:AleHD/smollm.git
   ```
1. Clone nanotron
   ```
   git clone git@github.com:AleHD/nanotron.git
   ```
1. Launch the job
   ```
   runai submit-dist pytorch \
   	--name nanotron \
   	--image registry.rcp.epfl.ch/<your_username>/ee628 \
   	--workers 1 \
   	--node-pool h200 \
   	--gpu 8 \
   	--backoff-limit 1 \
   	--large-shm \
   	--environment HF_TOKEN=... \
   	--environment WANDB_API_KEY=... \
	--environment SUBMIT_TIME=$(date '+%Y-%m-%d_%H:%M:%S') \
   	--existing-pvc claimname=course-ee-628-scratch,path=/scratch \
   	--existing-pvc claimname=home,path=/home \
   	--annotation k8s.v1.cni.cncf.io/networks=kube-system/roce \
   	--extended-resource rdma/rdma=1 \
   	--command -- bash -c "cd && bash smollm/text/pretraining/entrypoint.sh"
   ```
   You can also run `pretraining/entrypoint.sh` directly in an interactive node if you don't wish to do multi-job training.
   Note that the total node count is the amount of workers+1 (the master node), so the above example runs in two nodes.
   You can also use the following submission command as an example for non-interactive single job training:
   ```
   runai submit \
	--name nanotron \
	--image registry.rcp.epfl.ch/alhernan/ee628 \
	--node-pool h200 \
	--gpu 8 \
	--large-shm \
	--environment HF_TOKEN=... \
	--environment WANDB_API_KEY=... \
	--environment SUBMIT_TIME=$(date '+%Y-%m-%d_%H:%M:%S') \
	--existing-pvc claimname=course-ee-628-scratch,path=/scratch \
	--existing-pvc claimname=home,path=/home \
	--command -- bash -c "cd && bash smollm/text/pretraining/entrypoint.sh"
   ```
   It is recommended to pass the `SUBMIT_TIME` as specified so the logfiles are more easy to manage.

## EE-628: Extending the Pretraining Effort

In this section I will give general guidelines on the workings of `entrypoint.sh` and how to extend to your particular needs.
The `entrypoint.sh` serves as a generic training function that will automatically handle single or multi-job training and a flexible number of nanotron configuration files.
This makes it (relatively) easy to extend the script with experimental features.
The script does the following:
1. It has a number of environment variables to customize the general behaviour (e.g. change `NANOTRON_DIR` if your experimental feature is being developed on a separate nanotron branch).
1. In the prelude, it creates the `TORCHRUN_ARGS` to be used depending on the number of nodes and GPUs availabel.
1. Then, it runs `text/pretraining/unify_config.py` to use the base configuration file in `BASE_CONFIG`, and add the configuration from a number of different files `NEW_CONFIGS`.
   Currently, the used `BASE_CONFIG` sets up the general configurations that most training runs will share (fixed random seed, clip gradient value, sequence lenght, etc).
   The two new configurations currently specified (`arch_smollm2_135M.yaml` and `data_slimpajama.yaml`) provide the arguments needed to specify the SmolLM2 architecture and the use of SlimPajama6BT dataset.
   If you want to experiment with a new architecture, you might create your own `arch_custom.yaml` and use that path instead of the current default `arch_smollm2_135M.yaml` for instance.
   If you want to experiment with a new optimizer, you can create a new file `opt_scion.yaml` and add that path to the list.
   Lastly, the `OVERRIDE_CONFIG_KEYS` serves in case you want to make minor changes to the final unified configuration file.
   For instance, to set a custom run name (as in the current example) or other small configuration changes that don't necessarily need a file of their own.
1. It runs the training command, and saves the stdout and stderr to the log files in the directory `LOG_ROOT/run_${SUBMIT_TIME}_$JOB_UUID` so you can access the logs even after the job gets deleted.
