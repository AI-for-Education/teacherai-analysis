# %%
from pathlib import Path
from itertools import combinations
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(override=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ttest_ind, pearsonr
import pandas as pd

from tai_config import load_config
from tai_constants import DATADIR

HERE = Path(__file__).resolve().parent

# %%
cfg = load_config()
model = cfg["thread_params"]["model"]
threadlabel = cfg["thread_params"]["threadlabel"]
# thread_file = DATADIR / f"chat_data_anon_threaded_{threadlabel}_{model}.csv"
thread_file = DATADIR / "threaded_data_gpt_4o.csv"

# read the csv file
threaded_data_df = pd.read_csv(thread_file)
threaded_data_df["QuestionDate"] = pd.to_datetime(threaded_data_df["QuestionDate"])

# training date was February 17th 2024. For most analyses we should exclude this day
training_end_date = datetime(2024, 2, 17, 23, 59)
post_training_filter = threaded_data_df["QuestionDate"] > training_end_date

# %% Average Number of Messages per Group

# creating a df with total # of messages per user
total_count_df = (
    threaded_data_df.loc[post_training_filter]
    .groupby(["ResearchGroup", "UniqueUserReference"])
    .size()
    .reset_index(name="TotalCount")
)

# initalize dictionaries
group_counts = {0: 0, 1: 0, 2: 0, 3: 0}
group_users = {0: 0, 1: 0, 2: 0, 3: 0}

# find the sum of users and messages across all groups
for i, row in total_count_df.iterrows():
    if row["ResearchGroup"] == "BMGF Educaid Group 1":
        group_counts[0] += row["TotalCount"]
        group_users[0] += 1
    elif row["ResearchGroup"] == "BMGF Educaid Group 2":
        group_counts[1] += row["TotalCount"]
        group_users[1] += 1
    elif row["ResearchGroup"] == "BMGF Educaid Group 3":
        group_counts[2] += row["TotalCount"]
        group_users[2] += 1
    else:
        group_counts[3] += row["TotalCount"]
        group_users[3] += 1

# observed counts
f_obs2 = np.array([group_counts[0], group_counts[1], group_counts[2], group_counts[3]])

# total count
f_tot2 = f_obs2.sum()

# ratio of number of samples (number of users times number of days)
days = 137  # 137 days inbetween feb. 18th and July 4th (inclusive)
g4_days = 51  # 51 days inbetween may 14th and July 4th (inclusive)
denom = np.array(
    [
        (group_users[0] * days),
        (group_users[1] * days),
        (group_users[2] * days),
        (group_users[3] * g4_days),
    ]
).astype(float)

print(f"Average messages per week: {7 * f_obs2 / denom}")

f_ratio2 = denom / denom.sum()

# null hypothesis is equal count is each group, standardized by number of samples
f_exp2 = f_tot2 * f_ratio2

# chi square test stat
res2 = chisquare(f_obs=f_obs2, f_exp=f_exp2)

print(f"Across all Groups:")
print(res2)
print(f"f_oberseved: {f_obs2}")
print(f"f_expected: {f_exp2}")
print("-------------------------\n")

# comparing two groups at a time: iterating over every combination
for i in range(3):
    for j in range(i + 1, 4):
        print(f"Group {i+1} v.s. Group {j+1}")
        obs = np.array([group_counts[i], group_counts[j]])
        tot = obs.sum()
        if j != 4:
            ratio = np.array(
                [
                    (group_users[i] * days),
                    (group_users[j] * days),
                ]
            ).astype(float)
            ratio /= ratio.sum()
        else:
            ratio = np.array(
                [
                    (group_users[i] * days),
                    (group_users[j] * g4_days),
                ]
            ).astype(float)
            ratio /= ratio.sum()

        exp = tot * ratio

        res = chisquare(f_obs=obs, f_exp=exp)

        print(res)
        print(f"f_oberseved: {obs}")
        print(f"f_expected: {exp}\n")
    print("-------------------------")

# %% Average Thread Length:
# get thread lengths for each user
thread_len_df = threaded_data_df.loc[post_training_filter].groupby(
    ["ResearchGroup", "UniqueUserReference", "thread_id"]
)
thread_len_df = thread_len_df.agg(len).iloc[:, 0].reset_index(name="ThreadLength")

# filter out thread lengths less than 1
thread_len_grt_df = thread_len_df[thread_len_df["ThreadLength"] > 0]

# display the histogram
plt.figure(figsize=(10, 6))
groups = thread_len_grt_df["ResearchGroup"].unique()

# Create a list of arrays for thread lengths for each group
data = [
    thread_len_grt_df[thread_len_grt_df["ResearchGroup"] == group]["ThreadLength"]
    for group in groups
]

fig, ax = plt.subplots()
ax.hist(
    data,
    bins=range(
        min(thread_len_grt_df["ThreadLength"]),
        max(thread_len_grt_df["ThreadLength"]) + 1,
    ),
    stacked=True,
    density=False,
    label=groups,
    edgecolor="black",
)

ax.set_title("Histogram of Thread Lengths by Research Group")
ax.set_xlabel("Thread Length")
ax.set_ylabel("Frequency")
ax.legend(title="Research Group")

### t-test on thread lengths between groups
ttres = pd.DataFrame(index=groups, columns=groups)
for (lab1, g1), (lab2, g2) in combinations(
    thread_len_df.loc[post_training_filter].groupby("ResearchGroup")["ThreadLength"], 2
):
    tt = ttest_ind(g1, g2)
    print(
        f"{lab1}: {g1.mean()}, {lab2}: {g2.mean()}; t: {tt.statistic}",
        f"p: {tt.pvalue}",
    )
    ttres.loc[lab1, lab2] = f"{tt.statistic} (p={tt.pvalue})"

# %% How many Messages were sent during School Hours?
# Sierra Leone School Hours: 08:45 - 14:00 (8:45am - 2:00pm)
# accumulate for every thread start time that falls between school hours range

# declare start and end times
school_start = pd.to_datetime("08:00").time()
school_end = pd.to_datetime("14:50").time()

school_start_hour = school_start.hour + school_start.minute / 60
school_end_hour = school_end.hour + school_end.minute / 60

question_time = threaded_data_df["QuestionDate"]
# question_time = thread_dates_df["StartDate"]
question_time = question_time[question_time > datetime(2024, 2, 18)]
question_time = question_time[question_time.dt.day_of_week.isin(list(range(1, 6)))]

# count messages during school hours
# count = ((thread_dates_df["StartDate"].dt.time >= school_start) &
#          (thread_dates_df["StartDate"].dt.time <= school_end)).sum()

count = (
    (question_time.dt.time >= school_start) & (question_time.dt.time <= school_end)
).sum()

print(
    f"Number of occurrences between 08:45 and 14:00: {count} out of {len(question_time)}"
)

fig, ax = plt.subplots(figsize=(12, 8))
counts, bins, im = ax.hist(
    question_time.dt.hour,
    bins=np.arange(0, 25),
    color="cornflowerblue",
    rwidth=1,
    linewidth=0.1,
    edgecolor=[0, 0, 0, 0.5],
)
ax.set_xticks(bins)
ax.axvspan(
    xmin=school_start_hour,
    xmax=school_end_hour,
    color="tab:red",
    alpha=0.2,
    ymin=0,
    ymax=1,
)
ax.set_xlabel("Hour of day", fontsize=14)
ax.set_ylabel("Number of messages", fontsize=14)
ax.annotate(
    xy=(school_start_hour + (school_end_hour - school_start_hour) * 0.3, 0.97),
    text="School Hours",
    xycoords=("data", "axes fraction"),
    fontsize=14,
)
fig.suptitle("Distribution of messages over the day", fontsize=18)
fig.tight_layout()

fig.savefig("messages_by_time.png")
fig.savefig("messages_by_time.svg")
