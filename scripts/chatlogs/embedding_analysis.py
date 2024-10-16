# %% Import Statements:
#! %matplotlib qt

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import product

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE

from chatlogs.embeddings import (
    load_embeddings,
    load_chatdata,
    plot_clusters,
    get_embeddings,
    create_all_embeddings,
)
from tai_constants import DATADIR

HERE = Path(__file__).resolve().parent

FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

startdate = datetime(year=2024, month=2, day=18)

# %% Retrive the Files:
models = ["text-embedding-ada-002", "text-embedding-3-large", "text-embedding-3-small"]
# read the csv file
threaded_data_df, combined_df = load_chatdata()

embeddings = {}
embeddings_tsne = {}
for model in tqdm(models):
    try:
        embs, tsne = load_embeddings(model=model, return_tsne=True)
    except:
        create_all_embeddings(threaded_data_df, combined_df, model)
        embs, tsne = load_embeddings(model=model, return_tsne=True)
    embeddings[model] = embs
    embeddings_tsne[model] = tsne

valid_index_combined = np.array(combined_df["QuestionDate"] > startdate)

# %% Clustering:
k = 12

# cluster class
Clusterer = KMeans
cluster_kwargs = dict(
    n_clusters=k,
    init="k-means++",
    algorithm="elkan",
    n_init=40,
    random_state=67256235,
)

# dict to store clustering results
cluster_results = {}
cluster_scores = {}

# prob can just make this a function, and then have k be a parameter so we can adjust it easily?
# also so that you can choose which embedding array you want?
# f'n would return cluster_df
# iterate through every model's embeddings
for model, embs in embeddings.items():
    # if model != models[1]:
    #     continue
    # feature scaling
    # scalar = StandardScaler()
    # embeddings_scaled = scalar.fit_transform(
    #     embs["embed_combined"][valid_index_combined]
    # )
    embeddings_scaled = embs["embed_combined"][valid_index_combined]

    # clustering w/ K-Means
    km = Clusterer(**cluster_kwargs)
    nreps = 1
    best_score = -np.inf
    for _ in range(nreps):
        labels_ = km.fit_predict(embeddings_scaled)
        score = silhouette_score(embeddings_scaled, labels_)
        if score > best_score:
            labels = labels_
    scores = silhouette_samples(embeddings_scaled, labels)

    # reorder labels by score
    unqlabels = np.sort(np.unique(labels))
    group_scores = [np.percentile(scores[labels == label], 75) for label in unqlabels]
    relabels = np.argsort(group_scores)[::-1]
    newlabels = np.zeros_like(labels)
    for i, relabel in enumerate(relabels):
        newlabels[labels == unqlabels[relabel]] = i
    labels = newlabels

    # store results
    cluster_results[model] = labels
    cluster_scores[model] = scores

# new df for clusters
cluster_df = combined_df.loc[
    valid_index_combined,
    # ["UniqueUserReference", "ThreadID.Algorithm1", "ResearchGroup", "combined"],
    ["UniqueUserReference", "thread_id", "ResearchGroup", "combined"],
].copy()

# show the cluster distribution
for model_name, labels in cluster_results.items():
    print(f"Model: {model_name}, Cluster distribution: {np.bincount(labels)}")

    # insert cluster labels in df
    cluster_df[f"{model_name}_cluster"] = labels
    cluster_df[f"{model_name}_sillhouette_score"] = cluster_scores[model_name]

# plot the clusters
fig, axs = plt.subplots(ncols=len(cluster_results), figsize=(16, 6))
if not isinstance(axs, np.ndarray):
    axs = [axs]
fig.tight_layout()
for (model_name, labels), ax in zip(cluster_results.items(), axs):
    ax.set_aspect("equal")
    # ax.set_title("Blah")
    plot_clusters(
        embeddings_tsne[model_name]["embed_combined"][valid_index_combined],
        labels,
        pre_transformed=True,
        scores=np.clip(cluster_scores[model_name], a_min=0, a_max=None),
        scale=20,
        ax=ax,
        cmap="tab20",
    )
# fig.suptitle("Blah")
# save figure
fig.savefig(FIGDIR / f"cluster_plot_k{k :03d}.png")

# %% Subject Classification:
# function to calculate sim between string embs and thread embs
def calculate_similarities(strings_embeddings, thread_embeddings):
    strings_embs = np.array(list(strings_embeddings.values()))
    sim_matrix = strings_embs @ thread_embeddings.T
    # nearest_threads = np.argsort(sim_matrix, axis=1)[:, ::-1]
    return sim_matrix


# function to count occurancess for each str between groups
def count_group_occurrences(
    strings_list, sim_matrix, threshold, combined_df, verbose=0
):
    tags_df = pd.DataFrame(
        index=combined_df.index,
        columns=strings_list,
        data=((sim_matrix > threshold[0]) & (sim_matrix <= threshold[1])).T,
    )

    if verbose > 1:
        for i, string in enumerate(strings_list):
            print(f"\n\n{string.upper()}\n\n")
            tmp, *_ = np.where(
                (sim_matrix[i] > threshold[0]) & (sim_matrix[i] <= threshold[1])
            )
            for sim, convo, group in zip(
                sim_matrix[i, tmp],
                combined_df.iloc[tmp]["combined"].to_list(),
                combined_df.iloc[tmp]["ResearchGroup"].to_list(),
            ):
                print("#" * 50)
                print(f"Similarity: {sim}")
                print(convo)
                print("#" * 50)

    # get count of all
    other_idx = (sim_matrix <= threshold[0]).all(axis=0)
    tags_df["other"] = other_idx

    group_counts = {
        group: tags_df.loc[subdf.index].mean(axis=0)
        for group, subdf in combined_df.groupby("ResearchGroup")
    }
    group_counts_df = pd.DataFrame(group_counts)

    return group_counts_df, tags_df


# function to plot clusters for each str in the list
def plot_clusters_for_strings(
    strings_list,
    sim_matrix,
    thread_tsne,
    string_tsne,
    cluster_results,
    cluster_scores,
    valid_index_combined,
    threshold,
    cmap="tab20",
):
    close_to_strings = (sim_matrix > threshold[0]) & (sim_matrix < threshold[1])
    nrows = int(np.floor(len(strings_list) ** 0.5))
    ncols = len(strings_list) // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.tight_layout()

    for (i, string), ax in zip(enumerate(strings_list), axs.ravel()):
        ax.set_aspect("equal")
        plot_clusters(
            thread_tsne[valid_index_combined],
            cluster_results,
            pre_transformed=True,
            scores=(close_to_strings * sim_matrix)[i, valid_index_combined] ** 2,
            scale=20,
            ax=ax,
            cmap=cmap,
            legend=False,
        )
        ax.scatter(*string_tsne[i].T, marker="*", color=[1, 0, 0])
        ax.annotate(string, (string_tsne[i, 0], string_tsne[i, 1]), fontsize=8)
        ax.set_xticks(ax.get_xticks(), [""] * len(ax.get_xticks()))
        ax.set_yticks(ax.get_yticks(), [""] * len(ax.get_yticks()))

    fig.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    plot_clusters(
        thread_tsne[valid_index_combined],
        cluster_results,
        pre_transformed=True,
        scores=np.clip(cluster_scores, 0, None),
        scale=20,
        ax=ax,
        cmap=cmap,
    )
    for i, string in enumerate(strings_list):
        ax.scatter(*string_tsne[i].T, marker="*", color=[1, 0, 0])
        ax.annotate(string, (string_tsne[i, 0], string_tsne[i, 1]), fontsize=8)
        ax.set_xticks(ax.get_xticks(), [""] * len(ax.get_xticks()))
        ax.set_yticks(ax.get_yticks(), [""] * len(ax.get_yticks()))


# main function
def classify_and_plot_strings(
    strings_list,
    thread_embeddings,
    combined_df,
    cluster_results,
    cluster_scores,
    valid_index_combined,
    threshold=(0.3, np.inf),
    model="text-embedding-3-large",
    verbose=0,
):
    # get embeddings for the list of strings
    strings_embeddings = {
        string: emb
        for string, emb in zip(strings_list, get_embeddings(strings_list, model=model))
    }

    # calculate similarities
    sim_matrix = calculate_similarities(strings_embeddings, thread_embeddings)

    # count group occurrences
    group_counts_df, tags_df = count_group_occurrences(
        strings_list, sim_matrix, threshold, combined_df, verbose=verbose
    )

    if verbose > 0:
        # combine embeddings and make t-SNE
        strings_embs = np.array(list(strings_embeddings.values()))
        subj_thread_embs = np.vstack([strings_embs, thread_embeddings])
        tsne_results = TSNE(random_state=42).fit_transform(subj_thread_embs)
        strings_tsne = tsne_results[: strings_embs.shape[0]]
        thread_tsne = tsne_results[strings_embs.shape[0] :]

        # plot the results
        plot_clusters_for_strings(
            strings_list,
            sim_matrix,
            thread_tsne,
            strings_tsne,
            cluster_results,
            cluster_scores,
            valid_index_combined,
            threshold,
        )

    return group_counts_df, tags_df.loc[valid_index_combined]


# current model
use_model = "text-embedding-3-large"

# list of subjects
subject_list = [
    "Assembly and Registration",
    # "RME",
    "Religious and Moral Education",
    "Mathematics",
    # "PHE",
    "Physical Health Education",
    "Oral Written Composition",
    "General Science",
    # "PVS",
    "Pre-vocational Studies",
    "Reading",
    # "ESP & S",
    "English for a specific purpose",
    "Social Studies/Civics",
    "Spellings and Dictation",
    "Language Art",
    "Number Work",
    "Word Building",
    "Environmental Studies",
    "Quantitative Aptitude",
    "Verbal Aptitude",
    "Rhymes, Stories and Songs",
    # "CPA",
    "Creative Practical Arts",
    "Home Economics",
    "Agriculture",
    "Poetry",
    "English",
    "Quant",
]

other_list = [
    "Student behaviour",
    "Lesson planning",
    "Assessment",
    "Training",
    "Professional development",
    "Family",
    "Home environment",
    "Working environment",
    "Government",
    # "Primary education",
    # "Secondary education",
    "Education system",
    "Health",
    "Medicine",
    "Healthcare",
    #
    # "Administration",
    # "Checking attendance",
    # "Providing direct, whole-class instruction",
    # "Working with students individually",
    # "Facilitating group work",
    # "Children working or reading together from the blackboard",
    # "Addressing student disciplinary issues or other disruptions",
    #
    # "Literacy",
    # "Numeracy",
]

groupings = {
    "Language": [
        "Oral Written Composition",
        "Reading",
        "English for a specific purpose",
        "Spellings and Dictation",
        "Language Art",
        "Word Building",
        "Verbal Aptitude",
        "English",
        "Rhymes, Stories and Songs",
        "Poetry",
    ],
    "Mathematics": [
        "Mathematics",
        "Number Work",
        "Quantitative Aptitude",
        "Quant",
    ],
}

full_groupings = {}
from itertools import chain

grouped_subjs = list(chain.from_iterable(groupings.values()))
for subj in subject_list + other_list:
    if subj not in grouped_subjs:
        full_groupings[subj] = []
full_groupings = {**groupings, **full_groupings}


# calling the main function
group_counts_df, tags_df = classify_and_plot_strings(
    strings_list=subject_list + other_list,
    thread_embeddings=embeddings[use_model]["embed_combined"],
    combined_df=combined_df,
    cluster_results=cluster_results[use_model],
    cluster_scores=cluster_scores[use_model],
    valid_index_combined=valid_index_combined,
    threshold=(0.3, np.inf),
    verbose=0,
    model=use_model,
)

# recount the tags based on tag groupings
grouped_tags = {}
for groupstr, strlist in full_groupings.items():
    if strlist:
        grouped_tags[groupstr] = np.array(tags_df[strlist].any(axis=1))
    else:
        grouped_tags[groupstr] = np.array(tags_df[groupstr])

grouped_tags_df = pd.DataFrame(index=tags_df.index, data=grouped_tags)
grouped_tags_df["other"] = tags_df["other"]

# only the subject tags
subj_tags = list(set(grouped_tags) - set(other_list))

# logical index of all threasds not tagged by any subject
other = ~grouped_tags_df[subj_tags].any(axis=1)

##### print summary of counts
# % of non-subject threads
print(other.mean() * 100)

## subject-tagged threads
print("Subject tagged conversations:".upper())
# % of each subject
print("\nSubjects:".upper())
print(
    grouped_tags_df.loc[~other].mean().loc[subj_tags].sort_values(ascending=False) * 100
)
# % of each non-subject
print("\nNon-Subjects:".upper())
print(
    grouped_tags_df.loc[~other].mean().loc[other_list].sort_values(ascending=False)
    * 100
)
## non-subject-tagged threads
print("\n\nNot subject tagged conversations:".upper())
# % of each non-subject
print("\nNon-Subjects:".upper())
print(
    grouped_tags_df.loc[other].mean().loc[other_list].sort_values(ascending=False) * 100
)

#### separate counts by time (i.e. in school hours or not)
##### print summary of counts
school_start = pd.to_datetime("08:00").time()
school_end = pd.to_datetime("14:00").time()
thread_time = combined_df.loc[valid_index_combined]["QuestionDate"].dt.time

time_ranges = {
    "in_school_hours": [school_start, school_end],
    "outside_school_hours": [school_end, school_start],
}

summary_dfs_exc_other = defaultdict(lambda: defaultdict(dict))
summary_dfs_overall = defaultdict(lambda: defaultdict(dict))

for time_label, (start_time, end_time) in time_ranges.items():
    print("#" * 80)
    print(start_time, end_time)
    if start_time < end_time:
        time_filter = (thread_time > start_time) & (thread_time <= end_time)
    else:
        time_filter = (thread_time > start_time) | (thread_time <= end_time)
    print(time_filter.mean())
    # % of non-subject threads
    print(other.loc[time_filter].mean() * 100)

    ## subject-tagged threads
    # % of each subject
    # as % of non-other
    summary_dfs_exc_other["subject"]["subject"][time_label] = (
        grouped_tags_df.loc[~other & time_filter]
        .mean()
        .loc[subj_tags]
        .sort_values(ascending=False)
        * 100
    )
    # as % overall
    summary_dfs_overall["subject"]["subject"][time_label] = (
        grouped_tags_df.loc[time_filter]
        .mean()
        .loc[subj_tags]
        .sort_values(ascending=False)
        * 100
    )
    # % of each non-subject
    summary_dfs_exc_other["subject"]["nonsubject"][time_label] = (
        grouped_tags_df.loc[~other & time_filter]
        .mean()
        .loc[other_list]
        .sort_values(ascending=False)
        * 100
    )
    # as % overall
    summary_dfs_overall["subject"]["nonsubject"][time_label] = (
        grouped_tags_df.loc[time_filter]
        .mean()
        .loc[other_list]
        .sort_values(ascending=False)
        * 100
    )
    ## non-subject-tagged threads
    # % of each non-subject
    summary_dfs_exc_other["nonsubject"]["nonsubject"][time_label] = (
        grouped_tags_df.loc[other & time_filter]
        .mean()
        .loc[other_list]
        .sort_values(ascending=False)
        * 100
    )
    # as % overall
    summary_dfs_overall["nonsubject"]["nonsubject"][time_label] = (
        grouped_tags_df.loc[time_filter]
        .mean()
        .loc[other_list]
        .sort_values(ascending=False)
        * 100
    )

print("#" * 80)
for subjl1, subjl2 in product(["subject", "nonsubject"], repeat=2):
    for summary_dfs, sdflab in zip(
        [summary_dfs_exc_other, summary_dfs_overall], ["Excluding Other", "Overall"]
    ):
        time_dfs = summary_dfs[subjl1][subjl2]
        if not time_dfs:
            continue
        print(f"Subset: {subjl1.upper()}")
        print(f"Topic: {subjl2.upper()}")
        print(sdflab)
        summary_time_combo = pd.DataFrame(time_dfs)
        print(summary_time_combo.sort_values(by="in_school_hours", ascending=False))
        print("\n")

# %%
## upper / lower primary split
class_data_file = DATADIR / "class_data_anon.csv"
class_data = pd.read_csv(class_data_file, index_col=0)

for user, subdf in combined_df.loc[valid_index_combined].groupby("UniqueUserReference"):
    if user not in class_data.index:
        grouped_tags_df.loc[subdf.index, "lower_pri"] = pd.NA
    else:
        lower_pri = np.any(class_data.loc[user, [f"class{i}_EL" for i in [1, 2, 3]]])
        grouped_tags_df.loc[subdf.index, "lower_pri"] = lower_pri

grouped_tags_df.groupby("lower_pri")[list(grouped_tags) + ["other"]].agg(
    "mean"
).T.rename(columns={True: "Lower_Pri", False: "Upper_Pri"}).sort_values(
    by="Lower_Pri", ascending=False
) * 100

datapath = (
    Path.home()
    / "Fab Inc Dropbox/Fab Inc Projects/Fab Data Projects/TeacherAI Gates/Data/Interim"
)
group_file = datapath / "combined_group_data_20240723.csv"
group_data = pd.read_csv(group_file)

