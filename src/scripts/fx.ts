// 全站动效。html 上没有 .fx（减少动效 / 没有 JS）时什么都不做，内容照常显示。
export const motion = document.documentElement.classList.contains('fx');

// ---------- 乱码解码 ----------
const glyphs = '01<>/[]{}#*+=_-$%';
export function scramble(el: HTMLElement) {
  if (!motion || el.dataset.busy) return;
  const text = (el.dataset.text ??= el.textContent ?? '');
  el.dataset.busy = '1';
  const start = performance.now();
  const dur = 260 + text.length * 45;
  const frame = (now: number) => {
    const t = (now - start) / dur;
    el.textContent = [...text]
      .map((ch, i) => (ch === ' ' || t > 0.3 + (0.7 * i) / text.length ? ch : glyphs[(Math.random() * glyphs.length) | 0]))
      .join('');
    if (t < 1) requestAnimationFrame(frame);
    else {
      el.textContent = text;
      delete el.dataset.busy;
    }
  };
  requestAnimationFrame(frame);
}

// 联系方式：HTML 里是 base64，这里还原成可以点的链接
document.querySelectorAll<HTMLAnchorElement>('[data-contact]').forEach((a) => {
  const v = atob(a.dataset.contact!);
  a.href = a.dataset.kind === 'email' ? `mailto:${v}` : `tel:${v.replace(/\s/g, '')}`;
  a.textContent = v;
});

if (motion) {
  // 进入视口时加上 .in（CSS 据此浮现、画线），[data-scramble=enter] 顺便解码
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        io.unobserve(e.target);
        const el = e.target as HTMLElement;
        el.classList.add('in');
        if (el.dataset.scramble === 'enter') scramble(el);
      }
    },
    { threshold: 0.2 },
  );
  document.querySelectorAll('[data-scramble=enter], [data-reveal]').forEach((el) => io.observe(el));

  document.querySelectorAll<HTMLElement>('[data-scramble=hover]').forEach((el) =>
    el.addEventListener('pointerenter', () => scramble(el)),
  );

  // 背景点阵：光标附近的点亮起来
  const dots = document.querySelector<HTMLElement>('.dots');
  addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse' || !dots) return;
    dots.style.setProperty('--mx', `${e.clientX}px`);
    dots.style.setProperty('--my', `${e.clientY}px`);
  });
}
