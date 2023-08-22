import re
import argparse
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    MBartForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)
from tqdm.auto import tqdm
from metrics import compute_metric, ALL_METRICS


def normalize_table_string(s):
    s = s.lower()  # Convert the text to lowercase
    s = s.replace("\n", ", ")  # Replace new line with comma
    s = re.sub(" +", " ", s)  # Replace multiple spaces with a single space
    return s


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_sample_path",
        type=str,
        default="/content/drive/MyDrive/Achatbot/data/dashboard/train_samples.jsonl",
    )
    
    parser.add_argument(
        "--train_description_path",
        type=str,
        default="/content/drive/MyDrive/Achatbot/data/dashboard/train_descriptions.txt",
    )

    parser.add_argument(
        "--val_sample_path",
        type=str,
        default="/content/drive/MyDrive/Achatbot/data/dashboard/val_samples.jsonl",
    )
    parser.add_argument(
        "--val_description_path",
        type=str,
        default="/content/drive/MyDrive/Achatbot/data/dashboard/val_descriptions.txt",
    )

    parser.add_argument(
        "--test_sample_path",
        type=str,
        default="/content/drive/MyDrive/Achatbot/data/dashboard/test_samples.jsonl",
    )
    parser.add_argument(
        "--test_description_path",
        type=str,
        default="/content/drive/MyDrive/Achatbot/data/dashboard/test_descriptions.txt",
    )

    parser.add_argument("--model_name", type=str, default="vinai/bartpho-syllable")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--metric", type=str, default="BLEU", choices=["BLEU", "ROUGE", "GLEU", "ALL"]
    )
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--wandb_project", type=str, default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_test_loss", action="store_true")

    Config = parser.parse_args()

    print(Config)
    print("#" * 100)
    print()

    return Config


def wandb_init(args):
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=args)


def wandb_log(args, metrics, values):
    if wandb.run is None:
        return
    wandb.log({metric: value for metric, value in zip(metrics, values)})


def get_tokenizer(args):
    model_name = args.model_name
    print(f"Loading tokenizer {model_name}")
    if "mbart" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, src_lang="vi_VN", tgt_lang="vi_VN"
        )
        # tokenizer.src_lang = "vi_VN"
        # tokenizer.tgt_lang = "vi_VN"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def get_model(args):
    print(f"Using model {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(args.device)

    if args.load_model_path:
        print(f"Loading model from {args.load_model_path}")
        model.load_state_dict(
            torch.load(args.load_model_path, map_location=torch.device(args.device))
        )

    # print the number of parameters, in the form 1,234,567
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")

    return model


def prepare_model_inputs(batch, tokenizer, is_train, args):
    inputs = tokenizer(
        batch["src"],
        text_target=batch["tgt"] if is_train else None,
        padding="longest",
        max_length=args.max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    for k, v in inputs.items():
        inputs[k] = v.to(args.device)

    return inputs


def train_epoch(
    epoch, model, train_dataloader, optimizer, scheduler, tokenizer, Config
):
    losses = []
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_idx, batch in enumerate(pbar):
        # forward
        inputs = prepare_model_inputs(batch, tokenizer, is_train=True, args=Config)
        outputs = model(**inputs)
        loss = outputs.loss

        # gradient accumulation with loss normalization
        loss = loss / Config.gradient_accumulation_steps

        # if Config.max_grad_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # backward
        loss.backward()

        # update parameters
        if (batch_idx + 1) % Config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # log
        losses.append(loss.item())

        wandb_log(
            Config,
            ["train_loss", "avg_train_loss"],
            [loss.item(), sum(losses) / len(losses)],
        )
        pbar.set_description(
            f"Epoch: {epoch}, Train Loss: {loss.item():.4f}, Avg Train Loss: {sum(losses) / len(losses):.4f}"
        )

        # debug
        if batch_idx == 3 and Config.debug:
            break

    # log
    avg_epoch_loss = sum(losses) / len(losses)
    return avg_epoch_loss


@torch.no_grad()
def evaluate_epoch(epoch, model, dataloader, tokenizer, Config):
    model.eval()
    losses = []
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch_idx, batch in enumerate(pbar):
        inputs = prepare_model_inputs(batch, tokenizer, is_train=True, args=Config)
        outputs = model(**inputs)
        loss = outputs.loss

        losses.append(loss.item())
        pbar.set_description(
            f"Epoch: {epoch}, Eval Loss: {loss.item():.4f}, Avg Eval Loss: {sum(losses) / len(losses):.4f}"
        )

        if batch_idx == 1 and Config.debug:
            break

    avg_loss = sum(losses) / len(losses)
    return avg_loss


@torch.no_grad()
def get_scores(dataloader, model, tokenizer, args, epoch=0):
    model.eval()
    scores = []
    all_target_sentences = []
    all_output_sentences = []

    pbar = tqdm(dataloader, total=len(dataloader))
    for batch_idx, batch in enumerate(pbar):
        input_sentences = batch["src"]
        target_sentences = batch["tgt"]

        # print(target_sentences)
        inputs = prepare_model_inputs(batch, tokenizer, is_train=False, args=args)
        if "mbart" in args.model_name:
            inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["vi_VN"]

        outputs = model.generate(
            **inputs,
            max_length=args.max_target_length,
            num_beams=args.beam_size,
            early_stopping=True,
        )

        output_sentences = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if args.debug:
            print(f"Generated sentences: {output_sentences}")
            print(f"Target sentences: {target_sentences}")

        score = compute_metric(args, target_sentences, output_sentences)

        scores.append(score)
        pbar.set_description(
            f"Epoch: {epoch}, {args.metric}: {score:.4f}, Avg {args.metric}: {sum(scores) / len(scores):.4f}"
        )

        all_target_sentences.extend(target_sentences)
        all_output_sentences.extend(output_sentences)

        if batch_idx == 1 and args.debug:
            break

    # print(f"Generated sentences: {output_sentences}")
    for input_sentence, output_sentence, target_sentence in zip(
        input_sentences, output_sentences, target_sentences
    ):
        print(f"Input: {input_sentence}")
        print(f"Output: {output_sentence}")
        print(f"Target: {target_sentence}")
        print()

    # avg_score = sum(scores) / len(scores)
    # return avg_score
    final_score = compute_metric(args, all_target_sentences, all_output_sentences)
    return final_score


@torch.no_grad()
def get_output_sentences(dataloader, model, tokenizer, args):
    model.eval()
    scores = []
    all_target_sentences = []
    all_output_sentences = []

    pbar = tqdm(dataloader, total=len(dataloader))
    for batch_idx, batch in enumerate(pbar):
        input_sentences = batch["src"]
        target_sentences = batch["tgt"]

        # print(target_sentences)
        inputs = prepare_model_inputs(batch, tokenizer, is_train=False, args=args)
        if "mbart" in args.model_name:
            inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["vi_VN"]

        outputs = model.generate(
            **inputs,
            max_length=args.max_target_length,
            num_beams=args.beam_size,
            early_stopping=True,
        )

        output_sentences = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # print(output_sentences)

        pbar.set_description(f"Generating {batch_idx}/{len(dataloader)} batches")

        all_target_sentences.extend(target_sentences)
        all_output_sentences.extend(output_sentences)

        if batch_idx == 1 and args.debug:
            break

    # print last batch
    for input_sentence, output_sentence, target_sentence in zip(
        input_sentences, output_sentences, target_sentences
    ):
        print(f"Input: {input_sentence}")
        print(f"Output: {output_sentence}")
        print(f"Target: {target_sentence}")
        print()

    return all_output_sentences, all_target_sentences


def compute_all_metrics(dataloader, model, tokenizer, Config):
    all_output_sentences, all_target_sentences = get_output_sentences(
        dataloader, model, tokenizer, Config
    )

    result = {}
    for metric in ALL_METRICS:
        Config.metric = metric
        avg_test_score = compute_metric(
            Config, all_target_sentences, all_output_sentences
        )
        result[metric] = avg_test_score
    Config.metric = "all"

    return result