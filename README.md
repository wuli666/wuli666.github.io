# wuli666.github.io

个人主页 + 博客，Astro 静态站点。

```sh
npm install
npm run dev      # http://localhost:4321
npm run build    # 输出到 dist/
```

## 写文章

在 `src/content/blog/` 新建一个 `.md`，文件名就是 URL（`foo.md` → `/blog/foo/`）：

```md
---
title: 标题
date: 2026-10-11
description: 一句话摘要（可选）
layer: kernel      # agent | serving | kernel，决定标记颜色（可选）
draft: true        # 草稿只在 npm run dev 里显示（可选）
---

正文，支持代码高亮和 $\LaTeX$ 公式。
```

## 部署

1. 在 GitHub 建仓库 `wuli666.github.io`，推送到 `main`
2. 仓库 Settings → Pages → Source 选 **GitHub Actions**

之后每次推送 `main` 会自动构建发布到 https://wuli666.github.io 。
