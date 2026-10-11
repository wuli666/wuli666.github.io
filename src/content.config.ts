import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    // 对应首页那一层的颜色
    layer: z.enum(['agent', 'serving', 'kernel', 'gpu']).optional(),
    lang: z.string().default('en'),
    draft: z.boolean().default(false),
  }),
});

// 首页爆炸视图里每一层的详情页，文件名就是层的 id
const stack = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/stack' }),
  schema: z.object({ summary: z.string() }),
});

export const collections = { blog, stack };
