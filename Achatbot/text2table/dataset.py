import torch

from utils import normalize_table_string


class ViettelReportDataset2(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_path,
        description_path,
        tokenizer=None,
        max_src_len=256,
        max_tgt_len=256,
        device="cpu",
    ):
        with open(sample_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip()[1:-1] for line in f.readlines()]
        with open(description_path, "r", encoding="utf-8") as f:
            self.descriptions = [line.strip() for line in f.readlines()]

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src = self.samples[idx]
        tgt = self.descriptions[idx]

        if self.tokenizer is None:
            return {"src": src, "tgt": tgt}

        encoded_src = self.tokenizer.encode_plus(
            src,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_tgt = self.tokenizer.encode_plus(
            tgt,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_src["input_ids"].squeeze(0),
            "input_mask": encoded_src["attention_mask"].squeeze(0),
            "target_ids": encoded_tgt["input_ids"].squeeze(0),
            "target_mask": encoded_tgt["attention_mask"].squeeze(0),
        }

class ViettelReportDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_path,
        description_path,
        tokenizer=None,
        max_src_len=256,
        max_tgt_len=256,
        device="cpu",
    ):
        with open(sample_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip()[1:-1] for line in f.readlines()]
        with open(description_path, "r", encoding="utf-8") as f:
            self.descriptions = [line.strip() for line in f.readlines()]

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tgt = self.samples[idx]
        src = self.descriptions[idx]

        if self.tokenizer is None:
            return {"src": src, "tgt": tgt}

        encoded_src = self.tokenizer.encode_plus(
            src,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_tgt = self.tokenizer.encode_plus(
            tgt,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_src["input_ids"].squeeze(0),
            "input_mask": encoded_src["attention_mask"].squeeze(0),
            "target_ids": encoded_tgt["input_ids"].squeeze(0),
            "target_mask": encoded_tgt["attention_mask"].squeeze(0),
        }


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        description,
        df,
        target_sentences,
        tokenizer,
        src_max_len=64,
        tgt_max_len=128,
    ):
        self.description = description
        self.df = df
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = self.target_sentences[idx]

        table_str = (
            self.description.strip()
            + self.tokenizer.bos_token
            + normalize_table_string(row.to_string())
        )

        encoded_table = self.tokenizer.encode_plus(
            table_str,
            max_length=self.src_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_table["input_ids"]
        input_mask = encoded_table["attention_mask"]

        target = self.tokenizer.encode_plus(
            target,
            max_length=self.tgt_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_ids = target["input_ids"]
        target_mask = target["attention_mask"]

        return {
            "input_ids": input_ids.squeeze(0),
            "input_mask": input_mask.squeeze(0),
            "target_ids": target_ids.squeeze(0),
            "target_mask": target_mask.squeeze(0),
        }


if __name__ == "__main__":
    import pandas as pd
    from transformers import BartTokenizer
    import matplotlib.pyplot as plt
    import json

    # df = pd.read_csv("./table.csv")
    # target_sentences = []
    # with open("./target.txt", "r") as f:
    #     for line in f:
    #         target_sentences.append(line.strip())
    # dataset = Seq2SeqDataset(
    #     description="This is a table about the products in the store.",
    #     df=df,
    #     target_sentences=target_sentences,
    #     tokenizer=tokenizer,
    # )

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    dataset = ViettelReportDataset(
        sample_path="/content/drive/MyDrive/Achatbot/text2table/data/viettel/final/train_samples.jsonl",
        description_path="/content/drive/MyDrive/Achatbot/text2table/data/viettel/final/train_descriptions.txt",
        tokenizer=None,
        max_src_len=256,
        max_tgt_len=256,
        device="cpu",
    )

    sample = dataset[0]
    print(sample)
    print("#" * 50)
    # print(tokenizer.decode(sample["input_ids"]))
    # print(tokenizer.decode(sample["target_ids"]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v)
            print()
        break

    # src_lens = []
    # with open("./data/viettel/final/train_samples.jsonl", "r") as f:
    #     samples = [line.strip()[1:-1] for line in f.readlines()]
    # for sample in samples:
    #     src_lens.append(
    #         len(
    #             tokenizer.encode_plus(sample, return_tensors="pt")["input_ids"].squeeze(
    #                 0
    #             )
    #         )
    #     )

    # tgt_lens = []
    # with open("./data/viettel/final/train_descriptions.txt", "r") as f:
    #     descriptions = [line.strip() for line in f.readlines()]
    # for description in descriptions:
    #     tgt_lens.append(
    #         len(
    #             tokenizer.encode_plus(description, return_tensors="pt")[
    #                 "input_ids"
    #             ].squeeze(0)
    #         )
    #     )

    # # draw 75% quantile
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].hist(src_lens, bins=20)
    # ax[0].axvline(x=256, color="red")
    # ax[0].set_title("Source Length")
    # ax[1].hist(tgt_lens, bins=20)
    # ax[1].axvline(x=384, color="red")
    # ax[1].set_title("Target Length")
    # plt.show()