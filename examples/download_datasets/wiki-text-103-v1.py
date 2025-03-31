from datasets import load_dataset
import os
import shutil

from tqdm import tqdm

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1" )

output_dir = "wikitext-103"

os.makedirs(output_dir, exist_ok=True)

for split in ["train", "validation", "test"]:
    filename = f"wiki.{split[:5]}.tokens"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in tqdm(ds[split]["text"]):
            f.write(line+"\n")

shutil.make_archive(output_dir, "zip", output_dir)
print("zip done")


# TEXT=examples/download_datasets/wikitext-103
# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/wiki.train.tokens \
#     --validpref $TEXT/wiki.valid.tokens \
#     --testpref $TEXT/wiki.test.tokens \
#     --destdir data-bin/wikitext-103 \
#     --workers 20


# $TEXT = "examples\download_datasets\wikitext-103"^C
# fairseq-preprocess `
#     --only-source `
#     --trainpref $TEXT\wiki.train.tokens `
#     --validpref $TEXT\wiki.valid.tokens `
#     --testpref $TEXT\wiki.test.tokens `
#     --destdir data-bin\wikitext-103 `
#     --workers 20


# python train.py --task language_modeling data-bin/wikitext-103 --save-dir wt103/ --arch transformer_fast_lm_wiki103 --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 --criterion adaptive_loss --max-tokens 4608 --update-freq 1 --tokens-per-sample 512 --seed 1 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp --fp16 --required-batch-size-multiple 1 --wandb-project log-bias-lm