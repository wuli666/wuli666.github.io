// 栈的四层（从下往上），首页的爆炸视图和 /stack/[id] 页面共用

// 8x8 的格子画在每一层上：0 空，1 浅色，2 实色
const grid = (f: (r: number, c: number) => number) =>
  Array.from({ length: 64 }, (_, k) => f(Math.floor(k / 8), k % 8));
const noise = (k: number) => Math.abs(Math.sin(k * 12.9898) * 43758.5453) % 1;
const path = [70, 60, 61, 62, 52, 42, 43, 44, 34, 24, 25, 26, 16, 6, 7].map((rc) => Math.floor(rc / 10) * 8 + (rc % 10));

export type Layer = { id: string; name: string; note: string; cells: number[]; path?: number[] };

export const layers: Layer[] = [
  // GPU：一排排 SM
  { id: 'gpu', name: 'GPU', note: '', cells: grid(() => 1) },
  // kernel：GEMM 分块，A 的行条 × B 的列条 = C 的一个 tile
  {
    id: 'kernel',
    name: 'kernel',
    note: 'Triton, CUDA',
    cells: grid((r, c) => {
      const row = r >> 1 === 1;
      const col = c >> 1 === 2;
      return row && col ? 2 : row || col ? 1 : 0;
    }),
  },
  // serving：PagedAttention 的 KV block，零散地被占用
  { id: 'serving', name: 'serving', note: 'vLLM-Omni, vllm-rlt', cells: grid((r, c) => (noise(r * 8 + c) < 0.4 ? 2 : 0)) },
  // agent：一条在屏幕上走出来的轨迹
  {
    id: 'agent',
    name: 'agent',
    note: 'OpenDazi, GUI agents',
    path,
    cells: grid((r, c) => {
      const i = path.indexOf(r * 8 + c);
      return i === 0 || i === path.length - 1 ? 2 : i > 0 ? 1 : 0;
    }),
  },
];
