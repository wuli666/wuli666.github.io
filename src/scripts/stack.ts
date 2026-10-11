// 每一层对"一次推理"的反应，首页和各层页面共用。状态都从 DOM 里读，不和服务端的图案重复定义。
// 每次反应返回一个短标签（step 3、kv 17……），首页的时间线拿它当 trace 上的名字。

const cellsOf = (layer: HTMLElement) => [...layer.querySelectorAll<HTMLElement>(':scope > i')];
const cssVar = (el: Element, name: string) => getComputedStyle(el).getPropertyValue(name).trim();
const pick = <T>(xs: T[]) => xs[(Math.random() * xs.length) | 0];

const pop = (cell: HTMLElement) =>
  cell.animate([{ transform: 'scale(0.2)' }, { transform: 'scale(1.4)', offset: 0.6 }, { transform: 'scale(1)' }], {
    duration: 520,
    easing: 'ease-out',
  });

export function glow(layer: HTMLElement) {
  const c = cssVar(layer, '--c');
  layer.animate([{ boxShadow: `0 0 0 1px ${c}, 0 0 36px 4px ${c}` }, { boxShadow: '0 0 0 0 transparent' }], {
    duration: 900,
    easing: 'ease-out',
  });
}

export function reactor(layer: HTMLElement): () => string {
  const cells = cellsOf(layer);

  // agent：沿着轨迹走一步
  if (layer.dataset.layer === 'agent') {
    const path: number[] = JSON.parse(layer.dataset.path!);
    const base = (k: number) => (k === 0 || k === path.length - 1 ? '2' : '1');
    let pos = 0;
    return () => {
      cells[path[pos]].dataset.v = base(pos);
      pos = (pos + 1) % path.length;
      cells[path[pos]].dataset.v = '2';
      pop(cells[path[pos]]);
      return `step ${pos}`;
    };
  }

  // serving：给新请求分一个 KV block，同时有一个旧的被释放
  if (layer.dataset.layer === 'serving') {
    return () => {
      const freed = pick(cells.filter((c) => c.dataset.v === '2'));
      const got = pick(cells.filter((c) => c.dataset.v === '0'));
      freed.dataset.v = '0';
      got.dataset.v = '2';
      pop(got);
      return `kv ${cells.indexOf(got)}`;
    };
  }

  // kernel：输出 C 的下一个 tile，行条和列条跟着挪
  if (layer.dataset.layer === 'kernel') {
    const first = cells.findIndex((c) => c.dataset.v === '2');
    let tile = ((first >> 4) << 2) | ((first & 7) >> 1);
    return () => {
      tile = (tile + 1) % 16;
      cells.forEach((c, i) => {
        const row = i >> 4 === tile >> 2;
        const col = (i & 7) >> 1 === (tile & 3);
        c.dataset.v = row && col ? '2' : row || col ? '1' : '0';
      });
      return `gemm ${tile >> 2},${tile & 3}`;
    };
  }

  // GPU：kernel 发射到 SM 上，从一点涟漪般扩散
  return () => {
    const c = cssVar(document.documentElement, '--kernel');
    const o = (Math.random() * 64) | 0;
    cells.forEach((cell, i) =>
      cell.animate([{ background: c, borderColor: c, transform: 'scale(1.25)' }, {}], {
        duration: 420,
        delay: Math.hypot((i >> 3) - (o >> 3), (i & 7) - (o & 7)) * 90,
        easing: 'ease-out',
      }),
    );
    return `sm ${o}`;
  };
}
