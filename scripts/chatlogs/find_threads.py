# %%
from pathlib import Path
import os
import json
from fdllm.sysutils import register_models
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
from dotenv import load_dotenv

load_dotenv(override=True)

import pandas as pd

from chatlogs.threads import extract_threads_df, convo_from_df, format_convo
from tai_config import load_config
from tai_download import download_chatdata
from tai_constants import DATADIR

# %%
new_thread_key = "continuation"
new_thread_value = False
include_response = True

threadfields = {
    new_thread_key: "boolean, true only if the user is directly continuing a conversation"
    " thread with their most recent message, false if the user is starting a new thread."
    "\n\nMost user messages are not thread continuations."
    " Assume the answer is false without good evidence otherwise."
    " The threshold for deciding that continuation is true is if the "
    " user message (and assistant reply) would not make any sense without seeing"
    " the preceeding sequence first."
    " Imagine reading the user message and reply by itself, and then image reading it in the"
    " context of the preceeding sequence."
    "Do the message and reply only make sense in the context of the"
    " preceeding sequence? If so, return true."
    " Do the message and reply make any sense at all"
    " without the preceeding sequence? If so, return false."
}

threadtem = (
    "Following is a sequence of back-and-forth messages between an AI assistant and a user,"
    " along with the user message and the assistant response which came immediately after that sequence."
    "\n\nInitial sequence:\n{thread}"
    "\n\nSubsequent user message and response:\n{nextmsg}"
    "\n\nI want you to try to determine the following information about"
    " the state of the dialogue based on the the subsequent user message and response:"
    "\n{{{fields}}}."
    "\n\nOnly return the raw JSON formatted exactly as above."
)

# %%
cfg = load_config()
chatfile = cfg["chatdata"]["path"] / cfg["chatdata"]["file"]

if not chatfile.exists():
    try: 
        download_chatdata()
    except:
        raise IOError(f"{chatfile} doesn't exist")

# load the registration data
if chatfile.suffix == ".csv":
    chatdata = pd.read_csv(chatfile)
elif chatfile.suffix == ".xlsx":
    chatdata = pd.read_excel(chatfile)

model = cfg["thread_params"]["model"]
threadlabel = cfg["thread_params"]["threadlabel"]

outfile = DATADIR / f"chat_data_anon_threaded_{threadlabel}_{model}.csv"

# %%
chatdatathreaded = {}

if outfile.exists():
    chatdatathreadeddf = pd.read_csv(outfile)
    for user, df_ in chatdatathreadeddf.groupby("UniqueUserReference"):
        chatdatathreaded[user] = df_.reset_index(drop=True)

# %%
# we do a manual groupby loop like this inside its own cell,
# along with checking if the user has already been processed,
# so that if there are any exceptions (which can be unpredictable
# with openai) we can resume by re-running the cell without
# rerunning all of the previous users.

# alternative would be a groupby apply like this:
# chatdatathreaded = (
#     chatdata.groupby("UniqueUserReference")
#     .apply(extract_threads_df, model=os.getenv("OPENAI_MODEL"), verbose=1)
# )

for user, df in tqdm(chatdata.groupby("UniqueUserReference")):
    if user not in chatdatathreaded:
        df_ = extract_threads_df(
            df,
            model=model,
            verbose=1,
            threadlabel=threadlabel,
            fields=threadfields,
            msgtem=threadtem,
            include_response=include_response,
            new_thread_key=new_thread_key,
            new_thread_value=new_thread_value,
        )
        chatdatathreaded[user] = df_

# %%
chatdatathreadeddf = pd.concat(chatdatathreaded.values(), axis=0).reset_index(drop=True)

# %%
chatdatathreadeddf.to_csv(outfile, index=False)

# %%
## check output
chatdatathreadeddf = pd.read_csv(outfile)

for threadid, threaddf in chatdatathreadeddf.groupby(
    ["UniqueUserReference", f"ThreadID.{threadlabel}"]
):
    print(threadid)
    print(json.dumps(format_convo(convo_from_df(threaddf)), indent=4))
# %%
