"""Split GravityAwareRouting_QIP.tex into a main file (statements, Algorithm 1, Figs. 1-3)
and a Supplementary Information file (all proofs, Appendices B-I).  Cross-references use
the xr package: main refers to SI labels as S-<label>, SI to main labels as M-<label>."""
import re

SRC = "GravityAwareRouting_QIP.tex"
s = open(SRC).read()
lines = s.split("\n")

def block(start_pat, end_pat, text, start=0):
    i = text.index(start_pat, start)
    j = text.index(end_pat, i) + len(end_pat)
    return i, j

# ---------------------------------------------------------------- 1. extract main-text proofs
main_proofs = {}   # label of statement -> proof text
proof_targets = [  # (unique text right before proof, statement label, SI title)
    ("\\label{eq:bracket}", "thm:bracket", "Theorem"),
    ("attained.\n\\end{theorem}", "thm:balanced", "Theorem"),
    ("For every $r<M$, $G_{\\rm tot}^{2}(r)\\ge M/\\lambda_{r+1}(C)$.\n\\end{theorem}", "thm:cancel", "Theorem"),
    ("the direction that maximizes the guaranteed threshold.\n\\end{theorem}", "thm:misaligned", "Theorem"),
    ("with $\\|\\cdot\\|$ the operator norm.\n\\end{corollary}", "cor:dk", "Corollary"),
    ("No spectral-gap condition is required.\n\\end{proposition}", "prop:cert", "Proposition"),
]
for anchor, lab, kind in proof_targets:
    a = s.index(anchor)
    i = s.index("\\begin{proof}", a)
    assert i - a < 200, lab
    j = s.index("\\end{proof}", i) + len("\\end{proof}")
    body = s[i + len("\\begin{proof}"):j - len("\\end{proof}")].strip()
    main_proofs[lab] = (kind, body)
    s = s[:i] + "\\noindent\\textit{Proof.} Supplementary Information, Sect.~\\ref{S-app:mainproofs}.\n" + s[j:]

# ---------------------------------------------------------------- 2. cut the appendices
ai = s.index("\\begin{appendices}")
aj = s.index("\\end{appendices}") + len("\\end{appendices}")
app = s[ai + len("\\begin{appendices}"):aj - len("\\end{appendices}")]
ni = app.index("\\section{Notation}")
nj = app.index("\\section{The threshold law, proved globally}")
notation = app[ni:nj]
si_body = app[nj:]
# keep the separator comment lines tidy
notation = notation.rstrip().rstrip("%-").rstrip()
s = s[:ai] + "\\begin{appendices}\n\n" + notation + "\n\n\\end{appendices}" + s[aj:]

# ---------------------------------------------------------------- 3. label bookkeeping
def labels(t):
    return set(re.findall(r"\\label\{([^}]*)\}", t))

si_labels = labels(si_body) | {"app:mainproofs"}
main_labels = labels(s)

def prefix_refs(t, labs, pre):
    def rep(m):
        cmd, lab = m.group(1), m.group(2)
        return f"\\{cmd}{{{pre}{lab}}}" if lab in labs else m.group(0)
    return re.sub(r"\\(ref|eqref)\{([^}]*)\}", rep, t)

s = prefix_refs(s, si_labels - main_labels, "S-")

# ---------------------------------------------------------------- 4. build the SI
mp = ["%% -----------------------------------------------------------------------------",
      "\\section{Proofs of statements given in the main text}\\label{app:mainproofs}", "",
      "This section collects the short proofs of the main-text statements whose proofs are not",
      "given in Sects.~\\ref{app:threshold}--\\ref{app:general}.", ""]
names = {"thm:bracket": "Theorem", "thm:balanced": "Theorem", "thm:cancel": "Theorem",
         "thm:misaligned": "Theorem", "cor:dk": "Corollary", "prop:cert": "Proposition"}
for lab, (kind, body) in main_proofs.items():
    mp.append(f"\\begin{{proof}}[Proof of {kind}~\\ref{{{lab}}}]")
    mp.append(body)
    mp.append("\\end{proof}\n")
si_body = si_body.replace("%% -----------------------------------------------------------------------------\n\\section{Proofs for Sect.~\\ref{sec:general}}",
                          "\n".join(mp) + "\n%% -----------------------------------------------------------------------------\n\\section{Proofs for Sect.~\\ref{sec:general}}")
si_body = prefix_refs(si_body, main_labels - si_labels, "M-")
open("si_body.tex", "w").write(si_body)
open(SRC, "w").write(s)
print("main labels", len(main_labels), "SI labels", len(si_labels))
print("moved proofs:", list(main_proofs))
