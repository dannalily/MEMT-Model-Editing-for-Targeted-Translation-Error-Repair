import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import numpy, os
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import math

LABELS = [
        "First idiom token",
        "Middle idiom tokens",
        "Last idiom token",
        "First subsequent token",
        "Further tokens",
        "Last token",
    ]

font_files = [
    "fonts/NotoSans-Regular.ttf",
    "fonts/NotoSansArabic-Regular.ttf",     
    "fonts/NotoSansJP-Regular.ttf",    
    "fonts/NotoSansSC-Regular.ttf",  
    "fonts/NotoSansTC-Regular.ttf",          
]
font_names = []
for f in font_files:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
        font_names.append(fm.FontProperties(fname=f).get_name())

if font_names:
    plt.rcParams["font.family"] = font_names

plt.rcParams["axes.unicode_minus"] = False

def make_inputs(prompts, device="cpu"):
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )



class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        return numpy.concatenate(self.d).mean(axis=0)

    def std(self):
        return numpy.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)


def plot_AIE(
    differences,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
    archname="GPT2-XL",
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()
    answer = "AIE"

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "mlp": "Greens", "self_attn": "Reds"}[kind],
        vmin=low_score,
        vmax=high_score,
    )
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1])])
    ax.set_xticklabels(list(range(0, differences.shape[1])))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel(f"Layer number in {archname}")
    cb = plt.colorbar(h)

    if answer:
        cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
    else:
        plt.show()

def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    correct_answer = str(result["answer"])
    answer_t = make_inputs([correct_answer])
    A = answer_t["input_ids"].shape[1]
    
    differences = result["scores"]/A
    low_score = result["low_score"]/A
    answer = result["answer"]
    start_index = result["user_start_index"]

    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"
    labels= labels[start_index:]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "self_attn": "Reds"}[
            kind
        ],
        vmin=low_score,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] , 1)])
    ax.set_xticklabels(list(range(0, differences.shape[1] , 1)))
    ax.set_yticklabels(labels)
    if not modelname:
        modelname = "GPT"
    if not kind:
        ax.set_title("Impact of restoring state after corrupted input")
        # ax.set_xlabel(f"single restored layer within {modelname}")
    else:
        kindname = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(f"Impact of restoring {kindname} after corrupted input")
        # ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
    ax.set_xlabel(f"Layer number in {modelname}")

    cb = plt.colorbar(h)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif answer is not None:
        # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
        cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def read_knowlege(dirname, count=100, kind=None):
    kindcode = "" if not kind else f"_{kind}"
    (
        avg_fe,
        avg_ee,
        avg_le,
        avg_fa,
        avg_ea,
        avg_la,
        avg_hs,
        avg_ls,
        avg_fs,
        avg_fle,
        avg_fla,
    ) = [Avg() for _ in range(11)]
    for i in tqdm(range(count)):
        try:
            data = numpy.load(f"{dirname}/index_{i}{kindcode}.npz")
        except:
            continue
        # Only consider cases where the model begins with the correct prediction
        if "correct_prediction" in data and not data["correct_prediction"]:
            continue
        correct_answer = str(data["answer"])
        answer_t = make_inputs([correct_answer])
        A = answer_t["input_ids"].shape[1]

        scores = data["scores"]/A
        
        user_start_index = data['user_start_index']
        first_e, first_a = data["subject_range"]
        first_e = first_e - user_start_index
        first_a = first_a - user_start_index
        last_e = first_a - 1
        last_a = len(scores) - 1
        # original prediction
        avg_hs.add(data["high_score"]/A)
        # prediction after subject is corrupted
        avg_ls.add(data["low_score"]/A)
        avg_fs.add(scores.max())
        # some maximum computations
        avg_fle.add(scores[last_e].max())
        avg_fla.add(scores[last_a].max())
        # First subject middle, last subjet.
        avg_fe.add(scores[first_e])
        avg_ee.add_all(scores[first_e + 1 : last_e])
        avg_le.add(scores[last_e])
        # First after, middle after, last after
        avg_fa.add(scores[first_a])
        avg_ea.add_all(scores[first_a + 1 : last_a])
        avg_la.add(scores[last_a])

    result = numpy.stack(
        [
            avg_fe.avg(),
            avg_ee.avg(),
            avg_le.avg(),
            avg_fa.avg(),
            avg_ea.avg(),
            avg_la.avg(),
        ]
    )
    result_std = numpy.stack(
        [
            avg_fe.std(),
            avg_ee.std(),
            avg_le.std(),
            avg_fa.std(),
            avg_ea.std(),
            avg_la.std(),
        ]
    )
    print("Average Total Effect", avg_hs.avg() - avg_ls.avg())
    print(
        "Best average indirect effect on last subject",
        avg_le.avg().max() - avg_ls.avg(),
    )
    print(
        "Best average indirect effect on last token", avg_la.avg().max() - avg_ls.avg()
    )
    print("Average best-fixed score", avg_fs.avg())
    print("Average best-fixed on last subject token score", avg_fle.avg())
    print("Average best-fixed on last word score", avg_fla.avg())
    print("Argmax at last subject token", numpy.argmax(avg_le.avg()))
    print("Max at last subject token", numpy.max(avg_le.avg()))
    print("Argmax at last prompt token", numpy.argmax(avg_la.avg()))
    print("Max at last prompt token", numpy.max(avg_la.avg()))
    print("==============================================\n")
    return dict(
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size()
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translator", type=str, choices=['Qwen2-5-3B-Instruct', 'Qwen2-5-7B-Instruct'])
    parser.add_argument("--slang", type=str, required=True, help="source lang, en, zh")
    parser.add_argument("--tlang", type=str, required=True, help="target lang, en, zh, ja, fr, de, ar")
    args = parser.parse_args()

    the_count = 100
    slang = args.slang
    tlang =  args.tlang

    dirname = f'Causal_Tracing_Results/{args.translator}/{slang}_{tlang}'

    #plot causal tracing each sample
    for kind in [None, "mlp", "self_attn"]:
        kindcode = "" if not kind else f"_{kind}"
       
        for i in tqdm(range(the_count)):
            try:
                data = numpy.load(f"{dirname}/index_{i}{kindcode}.npz")
            except:
                continue

            pdf_name= f"heatmap/index_{i}{kindcode}.pdf"
            plot_trace_heatmap(data, savepdf=pdf_name, modelname=f"{args.translator}")


    #plot AIE heatmap
    color_order = [0, 1, 2, 4, 5, 3]
    cmap = plt.get_cmap("tab10")

 
    for j, (kind, what) in  enumerate(
        [
            (None, "Indirect Effect of $h_i^{(l)}$"),
            ("mlp", "Indirect Effect of MLP"),
            ("self_attn", "Indirect Effect of Attn"),
        ]
    ):
        kindcode = "" if not kind else f"_{kind}"
        dirname = f'Causal_Tracing_Results/{args.translator}/{slang}_{tlang}' 
        d = read_knowlege(dirname, count=the_count, kind=kind)

        differences = numpy.clip(d["result"] - d["low_score"], 0, None)
        count = d["size"]
        title = f"Avg {what} over {count} prompts"

        #plot AIE heatmap
        plot_AIE(
            differences,
            kind=kind,
            title=title,
            low_score=0.0,
            high_score=differences.max(),
            archname=f"{args.translator}",
            savepdf=f"Causal_Tracing_AVE/{args.translator}/{slang}_{tlang}/avg{kindcode}.pdf",
        )

    #plot AIE linemap
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True, dpi=200)
    for j, (kind, what) in  enumerate(
        [
            (None, "Indirect Effect of $h_i^{(l)}$"),
            ("mlp", "Indirect Effect of MLP"),
            ("self_attn", "Indirect Effect of Attn"),
        ]
    ):
        kindcode = "" if not kind else f"_{kind}"
        dirname = f'Causal_Tracing_Results/{args.translator}/{slang}_{tlang}'
        d = read_knowlege(dirname, count=the_count, kind=kind)

        differences = numpy.clip(d["result"] - d["low_score"], 0, None)
        count = d["size"]
        title = f"{slang}_{tlang}: Avg {what} over {count} prompts"

        for i, label in list(enumerate(LABELS)):
            y = d["result"][i] - d["low_score"]
            x = list(range(len(y)))
            std = d["result_std"][i]
            error = std * 1.96 / math.sqrt(count)
            axes[j].fill_between(
                x, y - error, y + error, alpha=0.3, color=cmap.colors[color_order[i]],label="_nolegend_"
            )
            axes[j].plot(x, y, label=label, color=cmap.colors[color_order[i]])

        axes[j].set_title(title)
        axes[j].set_ylabel("Average indirect effect")
        axes[j].set_xlabel("Layer number in Qwen2-5-3B-Instruct")
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        line_handles, line_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05), 
        ncol=6,                      
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"Causal_Tracing_AVE/{args.translator}/{slang}_{tlang}_line_graph.pdf",bbox_inches="tight")

                

