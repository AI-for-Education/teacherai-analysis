# %%
import json

from openai import BadRequestError
import pandas as pd
from tqdm import tqdm
from fdllm.llmtypes import LLMMessage
from fdllm import get_caller

THREADFIELDS = {
    "new_thread": "boolean, true if the user is starting a new dialogue thread"
    " with their most recent message."
    " A message that starts a new thread breaks the 'flow' of the dialogue."
    " It both changes the specific topic and it lacks any reference to the"
    " previous messages in the thread, including clarifications or corrections."
    " It is not enough to continue on the same general topic to be considered"
    " as the same thread."
}

THREADTEM = (
    "Following is a thread from a dialogue between an AI assistant and a user,"
    " and the subsequent message from that dialogue."
    " I want you to try to determine the following information about"
    " the state of the dialogue after the subsequent message:"
    "\n{{{fields}}}."
    " Only return the raw JSON formatted exactly as above."
    "\nThread follows:\n{thread}"
    "\nSubsequent message follows:\n{nextmsg}"
)


def gen_msg(
    msgtem, thread, fields, caller, min_new_token_window=500, include_response=False
):
    def msg(thread):
        if include_response:
            nextmsgthread = thread[-2:]
            historythread = thread[:-2]
        else:
            nextmsgthread = thread[-1:]
            historythread = thread[:-1]
        return LLMMessage(
            Role="user",
            Message=msgtem.format(
                thread=caller.format_messagelist(historythread),
                nextmsg=caller.format_messagelist(nextmsgthread),
                fields=fields,
            ),
        )

    msg_ = msg(thread)
    usethread = thread
    while caller.Token_Window - len(caller.tokenize([msg_])) < min_new_token_window:
        usethread = usethread[1:]
        msg_ = msg(usethread)
    return [msg_]


def extract_threads_df(df, *args, threadlabel="Algorithm0", **kwargs):
    convo = convo_from_df(df)

    _, threadidxs = extract_threads_convo(convo, *args, **kwargs)

    threadidxseries = pd.concat(
        [
            pd.Series(data=threadid, index=df.index[threadidx])
            for threadid, threadidx in enumerate(threadidxs)
        ]
    ).astype(int)

    df = df.copy()
    df[f"ThreadID.{threadlabel}"] = threadidxseries

    return df


def extract_threads_convo(
    convo: list[LLMMessage],
    model: str = "gpt-4o-mini-2024-07-18",
    fields: dict[str, str] = THREADFIELDS,
    msgtem: str = THREADTEM,
    include_response: bool = False,
    verbose: int = 0,
    new_thread_key: str = "new_thread",
    new_thread_value: bool = True,
):
    caller = get_caller(model)

    min_new_token_window = 200
    threads, threadidxs = [], []
    thread, threadidx = [], []

    convorange = range(1, len(convo), 2)
    if verbose > 0:
        convorange = tqdm(convorange)
    lastmsgbad = False
    for pairi, i in enumerate(convorange):
        if lastmsgbad:
            thread.extend(convo[max(i - 1, 0) : i])
        else:
            thread.extend(convo[max(i - 2, 0) : i])
        threadidx.append(pairi)
        if verbose > 1:
            print(json.dumps(caller.format_messagelist(thread), indent=4))
        if include_response:
            usethread = thread + [convo[i]]
        else:
            usethread = thread
        msg = gen_msg(
            msgtem,
            usethread,
            fields,
            caller,
            min_new_token_window,
            include_response=include_response,
        )
        try:
            response = json.loads(
                caller.call(
                    msg,
                    max_tokens=None,
                    temperature=0,
                    response_format={"type": "json_object"},
                ).Message
            )
            lastmsgbad = False
        except Exception as e:
            if isinstance(e, BadRequestError):
                thread.pop(-1)
                threadidx.pop(-1)
                lastmsgbad = True
                continue
        if response[new_thread_key] == new_thread_value:
            oldthread = thread[:-1]
            oldthreadidx = threadidx[:-1]
            if oldthread:
                threads.append(oldthread)
                threadidxs.append(oldthreadidx)
            thread = [thread[-1]]
            threadidx = [threadidx[-1]]
        if verbose > 1:
            print(response)
    threads.append(thread)
    threadidxs.append(threadidx)

    return threads, threadidxs


def convo_from_df(df):
    convouser = [LLMMessage(Role="user", Message=txt) for txt in df["UserQuestion"]]
    convoAI = [LLMMessage(Role="assistant", Message=txt) for txt in df["ModelAnswer"]]
    return [it for cucai in zip(convouser, convoAI) for it in cucai]


def format_convo(convo, caller=get_caller("gpt-3.5-turbo")):
    return [caller.format_message(msg) for msg in convo]


# %%
