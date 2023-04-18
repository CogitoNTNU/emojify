import json

import emoji  # type: ignore
import pandas as pd
import stweet as st  # type: ignore
from tqdm import tqdm  # type: ignore


def json_dump_tweet(tweet):
    return json.loads(tweet.raw_value)


def get_tweet_by_status(status: str) -> list[str] | None:
    try:
        task_id = st.TweetsByIdTask(status)
        output = st.CollectorRawOutput()
        # output_print = st.PrintRawOutput()
        st.TweetsByIdRunner(tweets_by_id_task=task_id, raw_data_outputs=[output]).run()
        tweets = output.get_raw_list()
        # ["raw_value"]["legacy"]["full_text"]
        some = [json_dump_tweet(tweet) for tweet in tweets]  # type: ignore
        if len(some) > 0:
            return [data["legacy"]["full_text"] for data in some]
        return None
    except Exception:
        return None


def extract_emojis(s):
    return [c for c in s if c in emoji.UNICODE_EMOJI["en"]]


def get_tweets():
    dataset = pd.DataFrame()
    tweets1 = pd.read_csv(
        ".data/full_test_plaintext.txt", sep="\t", names=["text", "label"], header=0
    )
    tweets2 = pd.read_csv(
        ".data/full_train_plaintext.txt", sep="\t", names=["text", "label"], header=0
    )
    tweets3 = pd.read_csv(
        ".data/full_valid_plaintext.txt", sep="\t", names=["text", "label"], header=0
    )
    tweeets = pd.concat([tweets1, tweets2, tweets3])
    strings = []
    for i, row in tqdm(tweeets.iterrows(), total=len(tweeets)):
        tid = str(tweeets.iloc[i]["text"])
        tweeets_ret = get_tweet_by_status(tid)
        if tweeets_ret is not None:
            strings.extend(tweeets_ret)
        if i % 100 == 0:
            dataset["text"] = strings
            dataset.to_csv(".data/tweets.tsv", sep="\t")
    return dataset


def create_dataset():
    dataset = get_tweets()
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        ems = extract_emojis(row["text"])
        label1 = ems[0] if len(ems) > 0 else None
        label2 = ems[1] if len(ems) > 1 else None
        dataset.at[i, "label"] = label1
        dataset.at[i, "label2"] = label2
        # remove emojis
        if label1 is not None:
            dataset.at[i, "text"] = row["text"].replace(label1, "")
        if label2 is not None:
            dataset.at[i, "text"] = row["text"].replace(label2, "")
    dataset.to_csv(".data/dataset.tsv", sep="\t")


def main_old():
    no = 743464860458061824
    # act = 742796925729157122
    act = 747643690521341953
    no_tweet = get_tweet_by_status(str(no))
    act_tweet = get_tweet_by_status(str(act))
    print("----new get twee ----")
    print(no_tweet)
    print(act_tweet)


def main():
    create_dataset()


if __name__ == "__main__":
    main()
