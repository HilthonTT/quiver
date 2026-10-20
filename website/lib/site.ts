export const site = {
  name: "quiver",
  tagline: "A fast, embeddable vector database written from scratch in Rust",
  repo: "https://github.com/HilthonTT/quiver",
  install: "git clone https://github.com/HilthonTT/quiver && cd quiver && cargo run --release -p quiver-server",
};

/** Prefixes public assets with the deploy base path (GitHub Pages). */
export function asset(path: string): string {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}
