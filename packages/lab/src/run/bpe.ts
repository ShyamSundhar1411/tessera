// GPT-2's byte-level BPE tokenizer: text is split into words with GPT-2's
// pattern, each word's UTF-8 bytes are mapped to printable characters, then
// pairs are merged in the order merges.txt ranks them. Same ids as the
// Python tokenizer.

const PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** GPT-2's byte-to-character table: printable bytes map to themselves, the rest to code points from 256 up. */
function byteTable(): string[] {
  const keep: number[] = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const table: string[] = new Array(256);
  let extra = 0;
  for (let b = 0; b < 256; b++) table[b] = String.fromCodePoint(keep.includes(b) ? b : 256 + extra++);
  return table;
}

export class Gpt2Tokenizer {
  private readonly ids: Map<string, number>;
  private readonly tokens: string[];
  private readonly ranks: Map<string, number>;
  private readonly toChar = byteTable();
  private readonly toByte = new Map(this.toChar.map((ch, b) => [ch, b]));
  private readonly cache = new Map<string, number[]>();

  constructor(vocab: Record<string, number>, merges: string) {
    this.ids = new Map(Object.entries(vocab));
    this.tokens = [];
    for (const [token, id] of this.ids) this.tokens[id] = token;
    this.ranks = new Map();
    merges
      .split("\n")
      .filter((line) => line && !line.startsWith("#version"))
      .forEach((line, rank) => this.ranks.set(line, rank));
  }

  get vocabSize() {
    return this.tokens.length;
  }

  encode(text: string): number[] {
    const out: number[] = [];
    const bytes = new TextEncoder();
    for (const word of text.match(PATTERN) ?? []) {
      const mapped = Array.from(bytes.encode(word), (b) => this.toChar[b]).join("");
      out.push(...this.word(mapped));
    }
    return out;
  }

  private word(word: string): number[] {
    const known = this.cache.get(word);
    if (known) return known;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      const [a, b] = [parts[best], parts[best + 1]];
      const next: string[] = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          next.push(a + b);
          i++;
        } else next.push(parts[i]);
      }
      parts = next;
    }
    const ids = parts.map((part) => this.ids.get(part) ?? 0);
    this.cache.set(word, ids);
    return ids;
  }

  decode(ids: ArrayLike<number>): string {
    const bytes: number[] = [];
    for (let i = 0; i < ids.length; i++) for (const ch of this.tokens[ids[i]] ?? "") bytes.push(this.toByte.get(ch) ?? 63);
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}
