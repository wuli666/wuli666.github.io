// 首页的 Live trace：一个很小的推理引擎模拟，画成 Nsight Systems 那样的时间线。
//
// 两个 GUI agent 循环"观察 → 思考 → 操作"，思考时向引擎发请求；另外还有随机到达的背景流量。
// 引擎做连续批处理：先 prefill 新请求，再把在跑的请求一起 decode；KV cache 共 64 个 block，
// decode 让请求变长，满了就抢占最新的请求。
//
// 开关（knobs）对应几种常见优化，打开关掉都会改变 trace：
//   tp2     两张卡张量并行，每个块之后 all-reduce，快的那张卡要等慢的
//   sp      序列并行：长 prefill 把 all-reduce 换成 reduce-scatter + all-gather
//   mtp     decode 后跑两个 MTP 草稿头，每个请求一步能多出 0~2 个 token
//   prefix  两个 agent 共享 system prompt，命中时不重算，共享的 block 单独计
//   graph   decode 用 CUDA graph 整步重放（batch 补齐到 1/2/4/8）；关掉就是 eager，每个 kernel 之间
//           GPU 要等 CPU 发下一个，kernel 车道碎成一段一段，SM 利用率出现空档
// 每次拨开关都在时间线上留一条竖线，方便对比前后。
//
// 时间全是模拟时钟，只在画面可见时前进；显示的时间按 1/20 慢放换算回引擎时间。

export type Hooks = { request(): void; prefill(): void; kernel(): void; gpu(): void };
export type Knobs = { tp2: boolean; sp: boolean; mtp: boolean; prefix: boolean; graph: boolean };

const SLOW = 20; // 慢放倍数：屏幕上 1 秒 = 引擎里 50 毫秒
const SPAN = 6500; // 画面里能看到多少毫秒（模拟时钟）
const BLOCKS = 64;
const BLOCK_TOKENS = 40;
const PREFIX = 320; // agent 共享的 system prompt + 工具说明，正好 8 个 block
const HIDDEN = 4096; // 估算通信量用
const AXIS = 20;

type RowSpec = { layer: string; name?: string; sub: string; short?: string; h: number };
const ROWS: RowSpec[] = [
  { layer: 'agent', name: 'agent', sub: 'A', h: 24 },
  { layer: 'agent', sub: 'B', h: 24 },
  { layer: 'serving', name: 'serving', sub: 'sched', h: 26 },
  { layer: 'serving', sub: 'kv', h: 30 },
  { layer: 'kernel', name: 'kernel', sub: 'rank 0', short: 'r0', h: 24 },
  { layer: 'kernel', sub: 'rank 1', short: 'r1', h: 24 },
  { layer: 'gpu', name: 'GPU', sub: 'nccl', h: 14 },
  { layer: 'gpu', sub: 'sm', h: 28 },
];
const [AGENT_A, AGENT_B, SCHED, KV, RANK0, RANK1, NCCL, SM] = ROWS.map((_, i) => i);
const HEIGHT = ROWS.reduce((s, r) => s + r.h, 0) + AXIS;
const ACTIONS = ['tap', 'swipe', 'type', 'scroll', 'back'];
const GRAPH_SIZES = [1, 2, 4, 8]; // 抓过图的 batch 大小
const KNOB_NAMES: Record<string, string> = { tp2: 'TP ×2', sp: 'SP', mtp: 'MTP', prefix: 'prefix cache', graph: 'CUDA graph' };

type Bar = { row: number; t: number; dur: number; label: string; alpha: number; info: string[]; w?: number };
type Seg = { t: number; dur: number; u: number }; // SM 利用率的一段
type Flow = { from: number; row: number; to?: number };
type Req = {
  id: number;
  prompt: number;
  out: number;
  gen: number;
  blocks: number;
  hit: boolean; // 这次 prefill 用上了共享前缀
  firstToken?: boolean; // TTFT 记过了（被抢占后重新 prefill 不再算）
  agent?: number;
  issued: number;
  flow?: Flow;
  onDone?: () => void;
};

const rand = (a: number, b: number) => a + Math.random() * (b - a);
const randi = (a: number, b: number) => Math.floor(rand(a, b + 1));
const pick = <T>(xs: T[]) => xs[(Math.random() * xs.length) | 0];
const ms = (t: number) => {
  const e = t / SLOW;
  return e < 1000 ? `${e.toFixed(e < 10 ? 1 : 0)} ms` : `${(e / 1000).toFixed(2)} s`;
};
const size = (b: number) => (b < 1e6 ? `${Math.round(b / 1024)} KB` : `${(b / 1048576).toFixed(1)} MB`);

export function createTrace(canvas: HTMLCanvasElement, tip: HTMLElement, stats: Record<string, Element>, hooks: Hooks) {
  const knobs: Knobs = { tp2: true, sp: true, mtp: true, prefix: true, graph: true };

  let now = 0;
  const timers: { at: number; fn: () => void }[] = [];
  const after = (d: number, fn: () => void) => {
    timers.push({ at: now + d, fn });
    timers.sort((a, b) => a.at - b.at);
  };

  const bars: Bar[] = [];
  const segs: Seg[] = [];
  const kv: { t: number; u: number; s: number }[] = [{ t: 0, u: 0, s: 0 }];
  const flows: Flow[] = [];
  const marks: { t: number; label: string }[] = []; // 抢占
  const switches: { t: number; label: string }[] = []; // 拨开关的时刻
  const bar = (row: number, t: number, dur: number, label: string, alpha: number, info: string[]) => {
    const b = { row, t, dur, label, alpha, info };
    bars.push(b);
    return b;
  };

  // ---------- 引擎 ----------
  const waiting: Req[] = [];
  const running: Req[] = [];
  let used = 0; // 含共享前缀
  let shared = 0; // 共享前缀占的 block，0 表示还没缓存
  let nextId = 1;
  let busy = false;
  let done = 0;
  let preempts = 0;
  let drafted = 0;
  let accepted = 0;
  let tpot = 0; // 每个输出 token 的平均耗时（指数平均），开关的效果在这上面最明显
  let ttft = 0; // 从发出请求到第一个 token（含排队），同样指数平均
  const ema = (old: number, x: number) => (old ? old * 0.85 + x * 0.15 : x);
  const tokenTimes: number[] = [];

  const own = (r: Req) => Math.ceil(((r.hit ? r.prompt - PREFIX : r.prompt) + r.gen + 1) / BLOCK_TOKENS);
  const setKv = () => kv.push({ t: now, u: used / BLOCKS, s: shared / BLOCKS });

  function submit(r: Req) {
    waiting.push(r);
    if (!busy) step();
  }

  function step() {
    busy = true;
    // 关掉前缀缓存后，等没人用了再把共享的 block 还回去
    if (!knobs.prefix && shared && !running.some((r) => r.hit)) {
      used -= shared;
      shared = 0;
      setKv();
    }
    const r = waiting[0];
    if (r) {
      r.hit = r.agent !== undefined && knobs.prefix;
      const fill = r.hit && shared === 0 ? PREFIX / BLOCK_TOKENS : 0;
      if (used + own(r) + fill <= BLOCKS) return prefill(waiting.shift()!, fill > 0);
    }
    if (running.length) return decode();
    busy = false;
  }

  // 一次前向画成两组 attn/mlp，每组之后要跨卡同步
  const PIECES = [
    { label: 'attn', unit: 0.9, alpha: 0.8 },
    { label: 'mlp', unit: 1.1, alpha: 0.42 },
    { label: 'attn', unit: 0.9, alpha: 0.8 },
    { label: 'mlp', unit: 1.1, alpha: 0.42 },
  ];

  // 把一步画到 rank / nccl / sm 车道上，返回这一步多长。
  // gap：eager 时每个 kernel 之后 GPU 空等 CPU 发射下一个的时间（占一个 unit 的比例）；CUDA graph 没有这段
  function forward(work: number, tokens: number, o: { sp: boolean; graph: boolean; mtp: number; u: number; gap: number }) {
    const tp = knobs.tp2 ? 2 : 1;
    const per = work / 4 / (tp === 2 ? 1.85 : 1); // 每个 unit 多少毫秒
    const bytes = HIDDEN * 2 * tokens; // bf16 激活
    const gap = o.graph ? 0 : per * o.gap;
    const t0 = now;
    let t = now;
    const put = (label: string, d0: number, alpha: number, what: string) => {
      const d1 = d0 * (1 + rand(-0.05, 0.09)); // 两张卡不会一样快
      if (!o.graph) {
        bar(RANK0, t, d0, label, alpha, [what, tp === 2 ? 'rank 0' : 'single GPU', ms(d0)]);
        if (tp === 2) bar(RANK1, t, d1, label, alpha, [what, 'rank 1', ms(d1)]);
      }
      const d = tp === 2 ? Math.max(d0, d1) : d0;
      segs.push({ t, dur: d, u: Math.min(0.97, o.u + rand(-0.03, 0.03)) });
      t += d + gap;
    };
    for (const p of PIECES) {
      put(p.label, per * p.unit, p.alpha, p.label);
      if (tp === 1) continue;
      // 每块结束把两张卡的部分和合起来；序列并行拆成两半，各传一半
      const colls = o.sp ? ['reduce-scatter', 'all-gather'] : ['all-reduce'];
      for (const name of colls) {
        const c = (o.sp ? 0.45 : 0.8) * (3 + bytes / 2.5e5);
        const tag = name === 'all-reduce' ? 'ar' : name === 'all-gather' ? 'ag' : 'rs';
        bar(NCCL, t, c, tag, 0.9, [name, `${size(o.sp ? bytes / 2 : bytes)} across 2 GPUs`, ms(c)]);
        segs.push({ t, dur: c, u: 0.3 });
        t += c;
      }
    }
    for (let i = 0; i < o.mtp; i++) put('mtp', per * 0.35, 0.25, `MTP draft head ${i + 1}`);
    put('sample', per * 0.2, 0.25, 'sample');
    const dur = t - t0;
    if (o.graph) {
      bar(RANK0, t0, dur, 'cuda graph', 0.72, ['CUDA graph replay', tp === 2 ? 'rank 0' : 'single GPU', ms(dur)]);
      if (tp === 2) bar(RANK1, t0, dur, 'cuda graph', 0.72, ['CUDA graph replay', 'rank 1', ms(dur)]);
    }
    return dur;
  }

  function prefill(r: Req, fill: boolean) {
    r.blocks = own(r);
    used += r.blocks;
    if (fill) {
      shared = PREFIX / BLOCK_TOKENS;
      used += shared;
    }
    setKv();
    const tokens = r.hit && !fill ? r.prompt - PREFIX : r.prompt;
    const sp = knobs.tp2 && knobs.sp && tokens > 300;
    const dur = forward(60 + tokens * 0.32, tokens, { sp, graph: false, mtp: 0, u: 0.92, gap: 0.04 });
    const who = r.agent === undefined ? 'background' : `agent ${'AB'[r.agent]}`;
    bar(SCHED, now, dur, `prefill r${r.id}${r.hit && !fill ? ' hit' : ''}`, 0.85, [
      `r${r.id} prefill, from ${who}`,
      r.hit && !fill
        ? `${r.prompt} prompt tokens, ${PREFIX} from the prefix cache`
        : `${r.prompt} prompt tokens${fill ? ', prefix now cached' : ''}`,
      knobs.tp2 ? (sp ? 'TP 2 with sequence parallel' : 'TP 2') : 'single GPU',
      `${r.blocks} KV blocks, ${ms(dur)}`,
    ]);
    if (r.flow) r.flow.to = now;
    hooks.prefill();
    after(dur * 0.15, hooks.kernel);
    after(dur * 0.3, hooks.gpu);
    after(dur, () => {
      // prefill 的最后一步就采样出第一个 token
      if (!r.firstToken) {
        r.firstToken = true;
        ttft = ema(ttft, now - r.issued);
      }
      running.push(r);
      step();
    });
  }

  function decode() {
    const batch = running.slice(0, 8);
    const k = knobs.mtp ? 2 : 0;
    // CUDA graph 只能按抓好的大小重放，batch 要补齐；eager 每个 kernel 之后都有发射空档
    const graph = knobs.graph;
    const padded = graph ? GRAPH_SIZES.find((n) => n >= batch.length)! : batch.length;
    const work = (50 + 14 * padded) * (k ? 1.15 : 1);
    const u = Math.min(0.9, 0.3 + 0.07 * padded + (k ? 0.05 : 0));
    const dur = forward(work, padded * (1 + k), { sp: false, graph, mtp: k, u, gap: 0.45 });
    const info = [
      `decode, batch ${batch.length}`,
      batch.map((r) => `r${r.id}`).join(' '),
      graph
        ? `CUDA graph replay${padded > batch.length ? `, padded to ${padded}` : ''}`
        : 'eager: GPU idles between kernel launches',
      ms(dur),
    ];
    const b = bar(SCHED, now, dur, `decode ×${batch.length}`, 0.3, info);

    after(dur, () => {
      let out = 0;
      for (const r of batch) {
        if (!running.includes(r)) continue; // 这一步里被抢占了
        // MTP：草稿 token 按顺序验收，错一个后面的都作废
        let n = 1;
        for (let i = 0; i < k && Math.random() < 0.7; i++) n++;
        drafted += k;
        accepted += n - 1;
        out += n;
        r.gen += n;
        for (let i = 0; i < n; i++) tokenTimes.push(now);
        while (running.includes(r) && own(r) > r.blocks) {
          if (used < BLOCKS) {
            r.blocks++;
            used++;
          } else preempt(r);
        }
        if (running.includes(r) && r.gen >= r.out) finish(r);
      }
      if (k) b.info.splice(2, 0, `MTP: ${(out / batch.length).toFixed(1)} tokens per request`);
      if (out) tpot = ema(tpot, dur / (out / batch.length));
      setKv();
      step();
    });
  }

  // KV 满了：把最晚进来的请求踢回等待队列，之后重新 prefill
  function preempt(asker: Req) {
    const victim = running.pop()!;
    used -= victim.blocks;
    victim.blocks = 0;
    victim.gen = 0;
    waiting.unshift(victim);
    marks.push({ t: now, label: `preempt r${victim.id}` });
    preempts++;
    if (victim !== asker && used < BLOCKS) {
      asker.blocks++;
      used++;
    }
  }

  function finish(r: Req) {
    running.splice(running.indexOf(r), 1);
    used -= r.blocks;
    done++;
    r.onDone?.();
  }

  const makeReq = (o: Partial<Req> & { prompt: number; out: number }): Req => ({
    id: nextId++,
    gen: 0,
    blocks: 0,
    hit: false,
    issued: now,
    ...o,
  });

  // ---------- 流量 ----------
  function agent(a: number) {
    const row = a === 0 ? AGENT_A : AGENT_B;
    const obs = rand(250, 600);
    bar(row, now, obs, 'observe', 0.16, [`agent ${'AB'[a]} observes the screen`, ms(obs)]);
    after(obs, () => {
      const flow: Flow = { from: now, row };
      flows.push(flow);
      const r = makeReq({ agent: a, prompt: PREFIX + randi(80, 680), out: randi(6, 18), flow });
      const think = bar(row, now, Infinity, `think r${r.id}`, 0.82, [`agent ${'AB'[a]} waits on r${r.id}`]);
      hooks.request();
      r.onDone = () => {
        think.dur = now - think.t;
        think.info.push(`${r.gen} tokens back, ${ms(think.dur)}`);
        const act = pick(ACTIONS);
        const d = rand(160, 420);
        bar(row, now, d, act, 0.38, [`agent ${'AB'[a]}: ${act}`, ms(d)]);
        after(d, () => agent(a));
      };
      submit(r);
    });
  }

  function background() {
    // 偶尔来一个生成很长的请求，把 KV 撑满、逼出抢占
    const long = Math.random() < 0.12;
    submit(makeReq({ prompt: randi(64, 600), out: long ? randi(50, 90) : randi(6, 20) }));
    after(-Math.log(1 - Math.random()) * 1300, background); // 泊松到达，平均 1.3 秒一个
  }

  agent(0);
  after(1300, () => agent(1));
  after(400, background);

  // ---------- 推进 ----------
  function tick(dt: number) {
    const end = now + dt;
    while (timers.length && timers[0].at <= end) {
      const t = timers.shift()!;
      now = t.at;
      t.fn();
    }
    now = end;
    // 扔掉已经滚出画面的
    const old = now - SPAN - 1000;
    const keep = <T>(xs: T[], alive: (x: T) => boolean) => {
      while (xs.length && !alive(xs[0])) xs.shift();
    };
    for (let i = bars.length - 1; i >= 0; i--) if (bars[i].t + bars[i].dur < old) bars.splice(i, 1);
    keep(segs, (s) => s.t + s.dur > old);
    keep(flows, (f) => f.from > old);
    keep(marks, (m) => m.t > old);
    keep(switches, (s) => s.t > old);
    while (kv.length > 2 && kv[1].t < old) kv.shift();
    keep(tokenTimes, (t) => t > now - 2000);
  }

  // ---------- 画 ----------
  const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  let theme: Record<string, string> = {};
  const readTheme = () =>
    (theme = Object.fromEntries(
      ['--agent', '--serving', '--kernel', '--gpu', '--rule', '--muted', '--ink', '--bg', '--paper', '--mono', '--sans'].map((n) => [
        n,
        css(n),
      ]),
    ));
  readTheme();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readTheme);

  canvas.style.height = `${HEIGHT}px`;
  const rowTop: number[] = [];
  ROWS.reduce((y, r, i) => ((rowTop[i] = y), y + r.h), 0);

  let pointer: { x: number; y: number } | null = null;
  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    pointer = { x: e.clientX - r.left, y: e.clientY - r.top };
  });
  canvas.addEventListener('pointerleave', () => {
    pointer = null;
    tip.hidden = true;
  });

  // 实色的阈值：高于它的 bar 用纸色写字，低于它的用墨色
  const STRONG = 0.7;
  // 主题色都是 #rrggbb，补两位 00 就是同色的全透明，渐变不会发灰
  const clear = (hex: string) => `${hex}00`;

  // 阶梯面积（KV、SM）：顶上一条细线，下面从颜色渐隐到透明
  function area(g: CanvasRenderingContext2D, row: number, x0: number, x1: number, u: number, col: string) {
    const top = rowTop[row] + 3;
    const h = ROWS[row].h - 6;
    const y = top + h * (1 - u);
    const grad = g.createLinearGradient(0, top, 0, top + h);
    grad.addColorStop(0, col);
    grad.addColorStop(1, clear(col));
    g.globalAlpha = 0.3;
    g.fillStyle = grad;
    g.fillRect(x0, y, x1 - x0, top + h - y);
    g.globalAlpha = 0.9;
    g.fillStyle = col;
    g.fillRect(x0, y, x1 - x0, 1);
    g.globalAlpha = 1;
  }

  function draw() {
    const dpr = devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = HEIGHT;
    if (canvas.width !== Math.round(w * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const narrow = w < 500;
    const gutter = narrow ? 82 : 108;
    const plot = w - gutter;
    const px = plot / SPAN;
    const tx = (t: number) => w - (now - t) * px;
    const color = (row: number) => theme[`--${ROWS[row].layer}`];
    const { '--mono': mono, '--sans': sans, '--ink': ink, '--paper': paper, '--muted': muted, '--rule': rule } = theme;

    // 只在层与层之间画分隔线；车道名和图之间一条竖线
    g.fillStyle = rule;
    ROWS.forEach((r, i) => {
      if (i && ROWS[i - 1].layer !== r.layer) g.fillRect(0, rowTop[i], w, 1);
    });
    g.fillRect(gutter, 0, 1, h - AXIS);
    g.fillRect(0, h - AXIS, w, 1);

    g.save();
    g.beginPath();
    g.rect(gutter + 1, 0, plot, h);
    g.clip();

    // 刻度：每秒一条淡虚线
    g.setLineDash([2, 4]);
    g.strokeStyle = rule;
    g.lineWidth = 1;
    g.font = `10px ${mono}`;
    for (let s = Math.floor(now / 1000); s >= 0; s--) {
      const x = Math.round(tx(s * 1000)) + 0.5;
      if (x < gutter - 60) break;
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, h - AXIS);
      g.stroke();
      // 窄屏刻度太密，隔一个标一次
      if (narrow && s % 2) continue;
      g.fillStyle = muted;
      g.fillText(ms(s * 1000), x + 5, h - 7);
    }
    g.setLineDash([]);

    // SM 利用率、KV 占用
    for (const s of segs) {
      if (now < s.t) continue;
      const x = tx(s.t);
      const x1 = x + Math.min(now - s.t, s.dur) * px;
      if (x1 >= gutter) area(g, SM, x, x1, s.u, theme['--kernel']);
    }
    for (let i = 0; i < kv.length; i++) {
      const x0 = Math.max(gutter, tx(kv[i].t));
      const x1 = i + 1 < kv.length ? tx(kv[i + 1].t) : w;
      if (x1 < gutter) continue;
      area(g, KV, x0, x1, kv[i].u, theme['--serving']);
      // 共享前缀：压在底部的一条实色带
      if (kv[i].s) {
        const hh = (ROWS[KV].h - 6) * kv[i].s;
        g.globalAlpha = 0.45;
        g.fillStyle = theme['--serving'];
        g.fillRect(x0, rowTop[KV] + ROWS[KV].h - 3 - hh, x1 - x0, hh);
        g.globalAlpha = 1;
      }
    }

    // 关联线：agent 发出请求 → 引擎开始 prefill 它
    g.strokeStyle = theme['--agent'];
    g.globalAlpha = 0.3;
    for (const f of flows) {
      if (f.to === undefined) continue;
      const x0 = tx(f.from);
      const y0 = rowTop[f.row] + ROWS[f.row].h / 2;
      const x1 = tx(f.to);
      const y1 = rowTop[SCHED] + ROWS[SCHED].h / 2;
      g.beginPath();
      g.moveTo(x0, y0);
      g.bezierCurveTo(x0 + 24, y0, x1 - 24, y1, x1, y1);
      g.stroke();
    }
    g.globalAlpha = 1;

    // bar：圆角、垂直居中；集合通信画成一条细灰线
    let hovered: { b: Bar; x: number; y: number; w: number; h: number } | null = null;
    g.font = `10px ${mono}`;
    for (const b of bars) {
      if (now < b.t) continue;
      const strong = b.alpha >= STRONG;
      const x = tx(b.t);
      const bw = Math.max(1.5, Math.min(now - b.t, b.dur) * px - (strong ? 1 : 0));
      if (x + bw < gutter) continue;
      const r = ROWS[b.row];
      const comm = b.row === NCCL;
      const bh = comm ? 5 : Math.round(r.h * 0.64);
      const y = rowTop[b.row] + (r.h - bh) / 2;
      g.globalAlpha = comm ? 0.5 : b.alpha;
      g.fillStyle = comm ? muted : color(b.row);
      g.beginPath();
      g.roundRect(x, y, bw, bh, Math.min(3, bw / 2, bh / 2));
      g.fill();
      g.globalAlpha = 1;
      b.w ??= g.measureText(b.label).width;
      if (!comm && bw > b.w + 12) {
        g.fillStyle = strong ? paper : ink;
        g.globalAlpha = strong ? 1 : 0.7;
        g.fillText(b.label, x + 6, y + bh / 2 + 3.5);
        g.globalAlpha = 1;
      }
      if (pointer && pointer.x >= x - 1 && pointer.x <= x + bw + 1 && pointer.y >= y - 2 && pointer.y <= y + bh + 2)
        hovered = { b, x, y, w: bw, h: bh };
    }
    // 悬停的那个描一圈
    if (hovered) {
      g.strokeStyle = ink;
      g.beginPath();
      g.roundRect(hovered.x - 1.5, hovered.y - 1.5, hovered.w + 3, hovered.h + 3, 4);
      g.stroke();
    }

    // 抢占：调度车道顶上一个小三角
    g.font = `9px ${mono}`;
    for (const m of marks) {
      const x = tx(m.t);
      if (x < gutter) continue;
      const y = rowTop[SCHED] + 1;
      g.fillStyle = ink;
      g.beginPath();
      g.moveTo(x - 4, y);
      g.lineTo(x + 4, y);
      g.lineTo(x, y + 6);
      g.fill();
      g.fillStyle = muted;
      g.fillText(m.label, x + 6, rowTop[KV] + 10);
    }

    // 左边缘渐隐，滚出去的东西不会硬切
    const fade = g.createLinearGradient(gutter, 0, gutter + 40, 0);
    fade.addColorStop(0, paper);
    fade.addColorStop(1, clear(paper));
    g.fillStyle = fade;
    g.fillRect(gutter + 1, 0, 40, h - AXIS);

    // 拨开关的时刻：竖线 + 小药丸标签
    g.font = `500 10px ${sans}`;
    for (const s of switches) {
      const x = Math.round(tx(s.t)) + 0.5;
      if (x < gutter) continue;
      g.globalAlpha = 0.6;
      g.strokeStyle = ink;
      g.setLineDash([3, 3]);
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, h - AXIS);
      g.stroke();
      g.setLineDash([]);
      g.globalAlpha = 1;
      const tw = g.measureText(s.label).width;
      g.fillStyle = ink;
      g.beginPath();
      g.roundRect(x - 1, 2, tw + 12, 15, 7.5);
      g.fill();
      g.fillStyle = paper;
      g.fillText(s.label, x + 5, 13);
    }

    // "现在"
    g.fillStyle = ink;
    g.fillRect(w - 1, 0, 1, h - AXIS);
    g.restore();

    // 左侧车道名
    ROWS.forEach((r, i) => {
      const mid = rowTop[i] + r.h / 2 + 3.5;
      if (r.name) {
        g.fillStyle = color(i);
        g.save();
        g.translate(5, mid - 3.5);
        g.rotate(Math.PI / 4);
        g.fillRect(-3, -3, 6, 6);
        g.restore();
        g.font = `600 11px ${sans}`;
        g.fillStyle = ink;
        g.fillText(r.name, 15, mid);
      }
      // 窄屏放不下"层名 + 子名"，有层名的那一行只留很短的子名
      const sub = narrow ? (r.short ?? r.sub) : r.sub;
      if (sub && (!narrow || !r.name || sub.length <= 2)) {
        g.font = `10px ${mono}`;
        g.fillStyle = muted;
        g.fillText(sub, gutter - 10 - g.measureText(sub).width, mid);
      }
    });

    // 悬停提示
    if (pointer) {
      const lines = hovered?.b.info ?? hoverArea(pointer, px);
      if (lines) {
        tip.hidden = false;
        tip.innerHTML = lines.map((l, i) => (i ? `<span>${l}</span>` : `<b>${l}</b>`)).join('');
        tip.style.left = `${Math.min(pointer.x + 12, w - tip.offsetWidth)}px`;
        tip.style.top = `${pointer.y + 14}px`;
      } else tip.hidden = true;
    }
  }

  // 面积车道没有 bar，悬停时报当时的数值
  function hoverArea(p: { x: number; y: number }, px: number): string[] | null {
    const t = now - (canvas.clientWidth - p.x) / px;
    const inRow = (row: number) => p.y >= rowTop[row] && p.y < rowTop[row] + ROWS[row].h;
    if (inRow(KV)) {
      const k = [...kv].reverse().find((s) => s.t <= t);
      if (!k) return null;
      return ['KV cache', `${Math.round(k.u * BLOCKS)} of ${BLOCKS} blocks`, ...(k.s ? [`${Math.round(k.s * BLOCKS)} shared prefix`] : [])];
    }
    if (inRow(SM)) {
      const s = segs.find((s) => s.t <= t && t < s.t + s.dur);
      return ['SM utilization', s ? `${Math.round(s.u * 100)}%` : 'idle'];
    }
    return null;
  }

  // 顶上的数字，一秒刷新几次就够
  let lastStats = -1;
  function updateStats() {
    if (now - lastStats < 250) return;
    lastStats = now;
    const set = (k: string, v: string) => stats[k] && (stats[k].textContent = v);
    set('done', String(done));
    set('tps', String(Math.round((tokenTimes.length / 2) * SLOW)));
    set('batch', String(Math.min(8, running.length)));
    set('kv', `${Math.round((used / BLOCKS) * 100)}%`);
    set('ttft', ttft ? `${(ttft / SLOW).toFixed(1)} ms` : '–');
    set('tpot', tpot ? `${(tpot / SLOW).toFixed(1)} ms` : '–');
    if (stats.accept) stats.accept.textContent = knobs.mtp && drafted ? `${Math.round((accepted / drafted) * 100)}%` : 'off';
  }

  return {
    knobs,
    // 换开关时清零 MTP 的验收统计，免得混着旧数据；同时在时间线上记一笔
    set(k: keyof Knobs, v: boolean) {
      knobs[k] = v;
      if (k === 'mtp') drafted = accepted = 0;
      switches.push({ t: now, label: `${KNOB_NAMES[k]} ${v ? 'on' : 'off'}` });
    },
    tick(dt: number) {
      tick(Math.min(dt, 64)); // 切回标签页时别一下子跳一大段
      updateStats();
    },
    draw,
    readTheme,
    // 给 trace.check.ts 用
    state: () => ({ used, shared, waiting: waiting.length, running: running.length, done, preempts }),
  };
}
