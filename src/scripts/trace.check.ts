// 引擎模拟的自检：node src/scripts/trace.check.ts
// 快进 10 分钟模拟时间，期间随机拨开关；KV 记账不能越界，请求要一直完成，压力大时要出现抢占。
const noop = () => {};
const el = () => ({ style: {}, hidden: true, textContent: '', addEventListener: noop }) as unknown as HTMLElement;
Object.assign(globalThis, {
  document: { documentElement: {} },
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  matchMedia: () => ({ addEventListener: noop }),
});
const { createTrace } = await import('./trace.ts');

const stats = { done: el(), tps: el(), batch: el(), kv: el(), accept: el(), tpot: el() };
const trace = createTrace(el() as HTMLCanvasElement, el(), stats, { request: noop, prefill: noop, kernel: noop, gpu: noop });

let lastDone = 0;
for (let minute = 1; minute <= 10; minute++) {
  for (let i = 0; i < 60_000 / 16; i++) {
    trace.tick(16);
    if (i % 1000 === 0) {
      const keys = Object.keys(trace.knobs) as (keyof typeof trace.knobs)[];
      const k = keys[Math.floor(Math.random() * keys.length)];
      trace.set(k, !trace.knobs[k]);
    }
    const s = trace.state();
    if (s.shared < 0 || s.used < s.shared || s.used > 64) throw new Error(`KV 越界：${JSON.stringify(s)}`);
  }
  const s = trace.state();
  if (s.done <= lastDone) throw new Error(`第 ${minute} 分钟没有请求完成，可能卡住了：${JSON.stringify(s)}`);
  lastDone = s.done;
}
const s = trace.state();
if (s.preempts === 0) throw new Error('10 分钟里一次抢占都没有，KV 压力太小');
console.log('ok', s);
