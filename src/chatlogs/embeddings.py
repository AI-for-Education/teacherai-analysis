from pathlib import Path

import numpy as np
from openai import OpenAI
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tai_constants import DATADIR

CHATDATA = DATADIR / "chat_data_anon_threaded_Algorithm1_gpt-4o-mini-2024-07-18.csv"
CHATDATA = DATADIR / "threaded_data_gpt_4o.csv"

FNAMES = ["embed_combined", "embed_avg", "embed_prompt", "embed_response"]


MODEL = "text-embedding-3-small"


client = OpenAI()


def create_all_embeddings(threaded_data_df, combined_df, model=MODEL):
    embed_keys = {"prompt": "UserQuestion", "response": "ModelAnswer"}

    simple_embeddings = {
        key: get_embeddings(threaded_data_df[column].to_list(), model=model)
        for key, column in embed_keys.items()
    }

    thread_embeddings = {
        key: np.vstack(
            [
                embs[df.index].mean(axis=0)
                for _, df in threaded_data_df.groupby(
                    # ["UniqueUserReference", "ThreadID.Algorithm1"]
                    ["UniqueUserReference", "thread_id"]
                )
            ]
        )
        for key, embs in simple_embeddings.items()
    }

    embed_thread = np.mean(list(thread_embeddings.values()), axis=0)

    np.save(DATADIR / f"embed_avg_{model}.npy", embed_thread)

    for key, embsarr in simple_embeddings.items():
        np.save(DATADIR / f"embed_{key}_{model}.npy", embsarr)

    # Combined Embedding:
    embed_combo = get_embeddings(combined_df["combined"].to_list(), model=model)

    # saving embedding values to .npy file
    np.save(DATADIR / f"embed_combined_{model}.npy", embed_combo)


# function to get embedding for text
def get_embeddings(text, model=MODEL, batchsize=2048):
    if not isinstance(text, list):
        text = [text]
    embslist = []
    for batchidx in range(0, len(text), batchsize):
        batchtext = text[batchidx : batchidx + batchsize]
        embsobjs = client.embeddings.create(input=batchtext, model=model).data
        embslist.extend([embs.embedding for embs in embsobjs])
    embsarr = np.vstack(embslist)
    return embsarr


# function to calculate avg. embedding over same threadID
def average_embedding(embeddings):
    return np.mean(embeddings, axis=0)


def load_embeddings(
    DATADIR=DATADIR,
    fnames=FNAMES,
    model=MODEL,
    return_tsne=False,
    use_tsne_cache=True,
):
    out_embs = {}
    out_tsne = {}
    for fname in fnames:
        fpath = DATADIR / f"{fname}_{model}.npy"
        embs = np.load(fpath)
        if return_tsne:
            fpath_tsne = fpath.parent / f"{fpath.stem}_tsne.npy"
            if fpath_tsne.exists() and use_tsne_cache:
                tsne = np.load(fpath_tsne)
            else:
                tsne = TSNE(random_state=42).fit_transform(embs)
                np.save(fpath_tsne, tsne)
            out_tsne[fname] = tsne
        out_embs[fname] = embs
    if return_tsne:
        return out_embs, out_tsne
    else:
        return out_embs


def load_chatdata(chatdata=CHATDATA):
    threaded_data_df = pd.read_csv(chatdata)
    threaded_data_df["QuestionDate"] = pd.to_datetime(threaded_data_df["QuestionDate"])
    threaded_data_df["combined"] = (
        "user: "
        # "Prompt: "
        + threaded_data_df["UserQuestion"].str.strip()
        + "\n\nassistant: "
        # + "; Response: "
        + threaded_data_df["ModelAnswer"].str.strip()
    )

    combined_df = (
        # threaded_data_df.groupby(["UniqueUserReference", "ThreadID.Algorithm1"])
        threaded_data_df.groupby(["UniqueUserReference", "thread_id"])
        .agg(
            {
                "combined": "\n\n".join,  # combine texts
                "ResearchGroup": "first",  # grab ResearchGroup
            }
        )
        .reset_index()
    )
    combined_df["QuestionDate"] = (
        # threaded_data_df.groupby(["UniqueUserReference", "ThreadID.Algorithm1"])
        threaded_data_df.groupby(["UniqueUserReference", "thread_id"])
        .agg({"QuestionDate": "first"})
        .reset_index(drop=True)
    )

    return threaded_data_df, combined_df


def plot_clusters(
    embeddings,
    labels,
    pre_transformed=False,
    ax=None,
    scores=None,
    scale=1,
    normalize_scores=True,
    legend=True,
    **sct_kwargs,
):
    if pre_transformed:
        tsne_vecs = embeddings
    else:
        tsne_vecs = TSNE(random_state=42).fit_transform(embeddings)

    if ax is None:
        fig, ax = plt.subplots()

    if scores is not None:
        scores = scores - np.min(scores)
        if normalize_scores:
            size = scale * (scores / np.max(scores))
        else:
            size = scores * scale
    else:
        size = scale
    sct = ax.scatter(*tsne_vecs.T, c=labels, s=size, **sct_kwargs)
    if legend:
        ax.legend(*sct.legend_elements(), loc="lower left", title="Cluster")

    return ax, tsne_vecs
