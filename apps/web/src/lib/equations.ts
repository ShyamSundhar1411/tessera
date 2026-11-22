import type { Node, Shape } from "tessera-engine";

// What each layer computes, written three ways: the general equation, the
// same equation with this layer's real sizes filled in, and a sentence in
// plain words. The trace view shows all three beside the values themselves.

export interface Explanation {
  /** The general rule, in TeX. */
  tex: string;
  /** The same rule with this layer's numbers, in TeX. */
  numbers?: string;
  /** What it does, in a sentence or two, using the real sizes. */
  words: string;
}

const t = (shape: Shape) => (shape.length ? `[${shape.join("\\times ")}]` : "");
const n = (value: unknown, fallback: number) => (value === undefined || value === "" ? fallback : Number(value));
const yes = (value: unknown) => value === true || value === "true";

export function explain(node: Node, inputs: Shape[], output: Shape): Explanation {
  const a = node.args;
  const x = inputs[0] ?? [];
  switch (node.op) {
    case "input":
      return {
        tex: "x \\in \\mathbb{R}^{" + x.concat(output).join("\\times ") + "}",
        words: output.length === 3 ? `The image arrives as ${output[0]} channel${output[0] > 1 ? "s" : ""} of ${output[1]}×${output[2]} numbers.` : output.length === 1 ? `${output[0]} token ids arrive, one number per token.` : "The data arrives.",
      };
    case "conv2d": {
      const k = n(a.k, 3);
      const s = n(a.s, 1);
      const p = n(a.p, Math.floor(k / 2));
      const groups = a.groups === "depthwise" ? x[0] : n(a.groups, 1);
      const [c, h] = x;
      return {
        tex: `y_{o,i,j} = b_o + \\sum_{c}\\sum_{u=0}^{k-1}\\sum_{v=0}^{k-1} W_{o,c,u,v}\\, x_{c,\\, s i+u-p,\\, s j+v-p}`,
        numbers: `${t(x)} \\xrightarrow{\\;k=${k},\\, s=${s},\\, p=${p}\\;} ${t(output)},\\quad H_{out} = \\left\\lfloor \\tfrac{${h} + 2\\cdot${p} - ${k}}{${s}} \\right\\rfloor + 1 = ${output[1]}`,
        words: `${output[0]} filters, each ${k}×${k}${groups > 1 ? (groups === c ? " and looking at one channel only (depthwise)" : `, in ${groups} groups`) : ` across all ${c} input channels`}, slide over the image ${s > 1 ? `${s} pixels at a time, halving it ` : ""}and each writes one map of how strongly its pattern appears.`,
      };
    }
    case "upconv2d": {
      const k = n(a.k, 2);
      const s = n(a.s, 2);
      return {
        tex: `y = W^{\\top} \\circledast x \\quad (H_{out} = (H-1)s - 2p + k)`,
        numbers: `${t(x)} \\to ${t(output)},\\quad (${x[1]}-1)\\cdot ${s} + ${k} = ${output[1]}`,
        words: `Learns to enlarge the maps ${s}× in each direction, the reverse of a strided convolution, so the decoder can climb back to full size.`,
      };
    }
    case "maxpool":
    case "avgpool": {
      const k = n(a.k, 2);
      return {
        tex: node.op === "maxpool" ? `y_{c,i,j} = \\max_{0\\le u,v<k} x_{c,\\, s i+u,\\, s j+v}` : `y_{c,i,j} = \\frac{1}{k^2}\\sum_{u,v} x_{c,\\, s i+u,\\, s j+v}`,
        numbers: `${t(x)} \\to ${t(output)},\\quad k=${k}`,
        words: `Each ${k}×${k} window becomes one number (its ${node.op === "maxpool" ? "largest" : "average"}), shrinking ${x[1]}×${x[2]} to ${output[1]}×${output[2]} while keeping the strongest signals.`,
      };
    }
    case "gap":
      return { tex: `y_c = \\frac{1}{HW}\\sum_{i,j} x_{c,i,j}`, numbers: `${t(x)} \\to ${t(output)}`, words: `Averages each of the ${x[0]} maps down to one number: how much of that feature the whole image contains.` };
    case "flatten":
      return { tex: `y = \\operatorname{vec}(x)`, numbers: `${t(x)} \\to [${output[0]}]`, words: `Lays the ${x.join("×")} block out as one list of ${output[0]} numbers, so dense layers can read it.` };
    case "linear":
      return {
        tex: `y = W x + b,\\quad W \\in \\mathbb{R}^{${a.out}\\times ${x[x.length - 1]}}`,
        numbers: `${t(x)} \\to ${t(output)},\\quad ${x[x.length - 1]}\\cdot${a.out}${yes(a.bias ?? true) ? ` + ${a.out}` : ""} \\text{ weights}`,
        words: `Every one of the ${x[x.length - 1]} inputs feeds every one of the ${a.out} outputs through its own weight: each output is a weighted vote over all inputs.`,
      };
    case "batchnorm":
      return { tex: `y = \\gamma\\, \\frac{x - \\mu_{\\text{batch}}}{\\sqrt{\\sigma^2_{\\text{batch}} + \\epsilon}} + \\beta`, numbers: `\\gamma, \\beta \\in \\mathbb{R}^{${x[0]}}`, words: `Re-centres and re-scales each of the ${x[0]} channels using statistics from the batch, so the next layer sees steady numbers.` };
    case "layernorm":
      return { tex: `y = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta`, numbers: `\\mu, \\sigma \\text{ over } d=${x[x.length - 1]}`, words: `Normalises each position's ${x[x.length - 1]} numbers to mean 0 and spread 1, then lets the model rescale them.` };
    case "rmsnorm":
      return { tex: `y = \\gamma \\odot \\frac{x}{\\sqrt{\\tfrac{1}{d}\\sum_i x_i^2 + \\epsilon}}`, numbers: `d = ${x[x.length - 1]}`, words: `Divides each position's ${x[x.length - 1]} numbers by their root mean square: layer norm without subtracting the mean, cheaper and just as steady.` };
    case "dropout":
      return { tex: `y = \\frac{m \\odot x}{1-p},\\; m_i \\sim \\text{Bernoulli}(1-p)`, numbers: `p = ${n(a.rate, 0.1)}`, words: `While training, randomly zeroes ${Math.round(n(a.rate, 0.1) * 100)}% of the numbers so no single path is relied on. At inference it does nothing.` };
    case "relu":
      return { tex: `y = \\max(0, x)`, words: "Keeps positive numbers, sets negatives to zero. This bend is what lets stacked layers model more than straight lines." };
    case "relu6":
      return { tex: `y = \\min(\\max(0, x), 6)`, words: "Like ReLU, but capped at 6, which keeps numbers small for low-precision phones." };
    case "gelu":
      return { tex: `y = x\\,\\Phi(x) \\approx \\tfrac{x}{2}\\left(1+\\tanh\\!\\left[\\sqrt{2/\\pi}\\,(x+0.044715x^3)\\right]\\right)`, words: "A smooth ReLU: weighs each number by how likely it is to be positive under a bell curve." };
    case "silu":
      return { tex: `y = x\\,\\sigma(x) = \\frac{x}{1+e^{-x}}`, words: "x times its own sigmoid: smooth, lets small negatives through a little." };
    case "sigmoid":
      return { tex: `y = \\frac{1}{1+e^{-x}}`, words: "Squashes every number into 0 to 1, like a dimmer switch." };
    case "tanh":
      return { tex: `y = \\tanh(x)`, words: "Squashes every number into -1 to 1." };
    case "hardswish":
      return { tex: `y = x\\,\\frac{\\operatorname{ReLU6}(x+3)}{6}`, words: "A cheap piecewise version of SiLU." };
    case "softmax":
      return { tex: `y_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}`, words: "Turns scores into probabilities that add up to 1, exaggerating the gaps." };
    case "add":
      return { tex: `y = ${inputs.map((_, i) => `x_{${i + 1}}`).join(" + ")}`, numbers: t(output), words: inputs.length === 2 ? "The skip connection: adds the block's input back to its output, so information and gradients can flow straight past the block." : "Adds the inputs element by element." };
    case "mul":
      return { tex: `y = x_1 \\odot x_2`, numbers: `${inputs.map(t).join(" \\odot ")} \\to ${t(output)}`, words: inputs[1]?.length === 3 && inputs[1][1] === 1 ? `A gate: each of the ${output[0]} channels is scaled by a number between 0 and 1 that the squeeze branch computed.` : "Multiplies the inputs element by element." };
    case "concat":
      return { tex: `y = [\\,x_1 \\,\\|\\, x_2 \\,\\|\\, \\cdots]`, numbers: `${inputs.map(t).join(" \\,\\|\\, ")} \\to ${t(output)}`, words: `Stacks ${inputs.length} inputs side by side, keeping everything each one found (${inputs.map((shape) => shape[0]).join(" + ")} = ${output[0]}).` };
    case "embedding":
      return {
        tex: `y_t = E[\\,\\text{token}_t\\,],\\quad E \\in \\mathbb{R}^{${a.vocab}\\times ${a.dim}}`,
        numbers: `[${x[0]}] \\to ${t(output)}`,
        words: `Looks up a learned list of ${a.dim} numbers for each token${yes(a.tied) ? ", sharing the table with the source embedding" : ""}. Similar tokens end up with similar lists.`,
      };
    case "posembed":
      if (yes(a.fixed)) return { tex: `y_{t} = x_t + PE_t,\\; PE_{t,2i} = \\sin\\!\\left(t / 10000^{2i/d}\\right),\\; PE_{t,2i+1} = \\cos\\!\\left(t / 10000^{2i/d}\\right)`, words: "Adds sine waves of different speeds to each position, so the model can tell first from last. No weights to learn." };
      if (yes(a.types)) return { tex: `y_t = x_t + T[\\,\\text{segment}_t\\,]`, words: "Adds which sentence each token belongs to (here, all segment 0)." };
      return { tex: `y_t = x_t + P_t,\\quad P \\in \\mathbb{R}^{${a.max ?? x[0]}\\times ${x[1]}}`, numbers: t(output), words: "Adds a learned vector for each position, so word order matters." };
    case "attention": {
      const heads = n(a.heads, 1);
      const kv = n(a.kv, heads);
      const d = x[1];
      const hd = n(a.head_dim, Math.floor(d / heads));
      const causal = a.causal === undefined || yes(a.causal);
      return {
        tex: `\\operatorname{Attention}(Q,K,V) = \\operatorname{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}${causal ? " + M" : ""}\\right)V,\\quad Q = xW_Q,\\; K = xW_K,\\; V = xW_V`,
        numbers: `${heads}\\text{ heads},\\; d_k = ${hd},\\; ${kv < heads ? `${kv}\\text{ key/value heads, each shared by } ${heads / kv}` : `${kv}\\text{ key/value heads}`},\\; ${t(x)} \\to ${t(output)}`,
        words: `Every token asks a question (query), offers a description (key) and carries content (value). Each token takes a weighted mix of the values whose keys match its query best${causal ? ", only looking at earlier tokens (the mask M hides the future)" : ", looking both ways"}. ${heads} heads do this in parallel, each free to watch for something different.`,
      };
    }
    case "cross_attention":
      return {
        tex: `\\operatorname{softmax}\\!\\left(\\frac{(x W_Q)(m W_K)^{\\top}}{\\sqrt{d_k}}\\right)(m W_V)`,
        numbers: `\\text{queries } ${t(inputs[0] ?? [])},\\; \\text{memory } ${t(inputs[1] ?? [])}`,
        words: "The decoder's tokens ask questions of the encoder's output: this is where the translation reads the source sentence.",
      };
    case "mla":
      return {
        tex: `c = x W_{DKV},\\; K = [\\,c W_{UK} \\,\\|\\, \\text{RoPE}(x W_{KR})\\,],\\; V = c W_{UV}`,
        numbers: `\\text{latent } c \\in \\mathbb{R}^{${a.kv_rank}} \\text{ instead of } ${n(a.heads, 1)}\\times 2 \\text{ full heads}`,
        words: `Keys and values are squeezed through a small shared latent of ${a.kv_rank} numbers, which is all the cache keeps: far less memory per token than ordinary attention.`,
      };
    case "mlp": {
      const act = String(a.act ?? "swiglu");
      const gated = act === "swiglu" || act === "geglu";
      return {
        tex: gated ? `y = W_{down}\\big(\\,${act === "swiglu" ? "\\operatorname{SiLU}" : "\\operatorname{GELU}"}(xW_{gate}) \\odot xW_{up}\\big)` : `y = W_2\\,${act === "relu" ? "\\operatorname{ReLU}" : "\\operatorname{GELU}"}(W_1 x + b_1) + b_2`,
        numbers: `${x[x.length - 1]} \\to ${a.ff} \\to ${x[x.length - 1]}`,
        words: `Each token on its own is widened to ${a.ff} numbers, bent by a nonlinearity${gated ? " and gated (one branch decides how much of the other passes)" : ""}, and narrowed back. Most of a transformer's knowledge lives here.`,
      };
    }
    case "moe":
      return {
        tex: `y = \\sum_{e \\in \\operatorname{TopK}(g(x))} g_e(x)\\, E_e(x),\\quad g(x) = \\operatorname{softmax}(x W_r)`,
        numbers: `${a.experts}\\text{ experts, top-}${n(a.topk, 2)}\\text{ per token}${n(a.shared, 0) ? `,\\; + ${a.shared}\\text{ shared}` : ""}`,
        words: `A router scores all ${a.experts} expert feed-forwards for each token and only the best ${n(a.topk, 2)} run: lots of knowledge, little compute per token.`,
      };
    case "lmhead":
      return {
        tex: `p(\\text{next} \\mid \\cdot) = \\operatorname{softmax}(h\\, W^{\\top})${yes(a.tied) ? ",\\; W = E" : ""}`,
        numbers: `${t(x)} \\to ${t(output)}`,
        words: `Scores every one of the ${a.vocab} tokens as the next one, for every position${yes(a.tied) ? ", reusing the embedding table backwards" : ""}.`,
      };
    case "to_sequence":
      return { tex: `[C, H, W] \\to [HW, C]`, numbers: `${t(x)} \\to ${t(output)}`, words: `Each of the ${output[0]} grid positions becomes a token of ${output[1]} numbers, so a transformer can read the image like a sentence.` };
    case "cls_token":
      return { tex: `y = [\\,\\texttt{[CLS]} \\,\\|\\, x\\,]`, numbers: `${t(x)} \\to ${t(output)}`, words: "Puts one extra learned token in front. By the end it has gathered a summary of the whole input." };
    case "take_first":
      return { tex: `y = x_0`, numbers: `${t(x)} \\to ${t(output)}`, words: "Keeps only the first token's numbers, the summary token, for the classifier." };
    case "mean_pool":
      return { tex: `y = \\frac{1}{S}\\sum_t x_t`, numbers: `${t(x)} \\to ${t(output)}`, words: "Averages all tokens into one vector for the whole input." };
    default:
      return { tex: `y = f(x)`, numbers: `${t(x)} \\to ${t(output)}`, words: "Passes the data on." };
  }
}
