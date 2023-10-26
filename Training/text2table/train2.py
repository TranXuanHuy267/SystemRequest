import torch
import os
from transformers import (
    get_linear_schedule_with_warmup,
)

from dataset import ViettelReportDataset

from utils import (
    get_args,
    wandb_init,
    wandb_log,
    get_tokenizer,
    get_model,
    get_tokenizer,
    train_epoch,
    evaluate_epoch,
    get_scores,
)
from metrics import compute_metric
from utils import get_output_sentences
from metrics import ALL_METRICS


if __name__ == "__main__":
    Config = get_args()

    wandb_init(Config)
    os.makedirs(os.path.join(Config.output_dir, Config.model_name), exist_ok=True)

    tokenizer = get_tokenizer(Config)
    # tokenizer.save_pretrained(f"./tokenizer/{Config.model_name}")

    train_dataset = ViettelReportDataset(
        sample_path=Config.train_sample_path,
        description_path=Config.train_description_path,
        # tokenizer=tokenizer,
        max_src_len=Config.max_source_length,
        max_tgt_len=Config.max_target_length,
        device=Config.device,
    )

    val_dataset = ViettelReportDataset(
        sample_path=Config.val_sample_path,
        description_path=Config.val_description_path,
        # tokenizer=tokenizer,
        max_src_len=Config.max_source_length,
        max_tgt_len=Config.max_target_length,
        device=Config.device,
    )

    test_dataset = ViettelReportDataset(
        sample_path=Config.test_sample_path,
        description_path=Config.test_description_path,
        # tokenizer=tokenizer,
        max_src_len=Config.max_source_length,
        max_tgt_len=Config.max_target_length,
        device=Config.device,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False
    )

    model = get_model(Config)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.warmup_steps,
        num_training_steps=len(train_dataloader) * Config.epochs,
    )

    train_losses = []
    val_losses = []
    best_loss, best_score = 1e9, 0
    for epoch in range(Config.epochs):
        print(f"Epoch: {epoch}")

        # train loss
        avg_train_loss = train_epoch(
            epoch, model, train_dataloader, optimizer, scheduler, tokenizer, Config
        )
        train_losses.append(avg_train_loss)
        print(f"Epoch: {epoch}, Avg Train Loss: {avg_train_loss:.4f}")

        # val loss
        avg_val_loss = evaluate_epoch(epoch, model, val_dataloader, tokenizer, Config)
        val_losses.append(avg_val_loss)
        wandb_log(Config, ["val_loss"], [avg_val_loss])
        print(f"Epoch: {epoch}, Avg Val Loss: {avg_val_loss:.4f}")

        # Save best loss model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(Config.output_dir, Config.model_name, f"best_loss.bin"),
            )
            print(f"Best loss model saved to {Config.output_dir}/{Config.model_name}")

        # val score (bleu, rouge, meteor)
        avg_val_score = get_scores(val_dataloader, model, tokenizer, Config, epoch)
        print(f"Epoch: {epoch}, Avg {Config.metric}: {avg_val_score:.4f}")
        wandb_log(Config, [f"avg_val_{Config.metric}"], [avg_val_score])

        # Save best score model
        if avg_val_score > best_score:
            best_score = avg_val_score
            torch.save(
                model.state_dict(),
                os.path.join(Config.output_dir, Config.model_name, "best_score.bin"),
            )
            print(
                f"Best {Config.metric} model saved to {Config.output_dir}/{Config.model_name}"
            )

        print(f"End of Epoch: {epoch}")
        print(f"Avg Train Loss: {avg_train_loss:.4f}")
        print(f"Avg Val Loss: {avg_val_loss:.4f}")
        print(f"Avg Val {Config.metric}: {avg_val_score:.4f}")
        print("#" * 50)
        print()

# test loss
model.load_state_dict(
    torch.load(os.path.join(Config.output_dir, Config.model_name, "best_loss.bin"))
)

avg_test_loss = evaluate_epoch(epoch, model, test_dataloader, tokenizer, Config)
print(f"Avg Test Loss: {avg_test_loss:.4f}")
wandb_log(Config, ["avg_test_loss"], [avg_test_loss])

if Config.metric != "all":
    # test score (bleu, rouge, meteor)
    avg_test_score = get_scores(test_dataloader, model, tokenizer, Config, "test")
    print(f"Avg Test {Config.metric}: {avg_test_score:.4f}")
    wandb_log(Config, [f"avg_test_{Config.metric}"], [avg_test_score])
else:
    all_output_sentences, all_target_sentences = get_output_sentences(
        test_dataloader, model, tokenizer, Config
    )

    for metric in ALL_METRICS:
        Config.metric = metric
        avg_test_score = compute_metric(
            Config, all_target_sentences, all_output_sentences
        )

        print(f"Avg Test {metric}: {avg_test_score:.4f}")
        wandb_log(Config, [f"test_{metric}"], [avg_test_score])