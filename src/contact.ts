// 联系方式只以 base64 写进 HTML，由 scripts/fx.ts 在浏览器里还原成链接，挡掉不跑 JS 的爬虫
export const email = btoa('421774554@qq.com');
export const phone = btoa('+86 178 9931 4625');
