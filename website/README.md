# Quiver website

The documentation site for Quiver, built with Next.js (App Router, static export), Tailwind CSS and MDX.

```sh
npm install
npm run dev        # http://localhost:3000
npm run build      # static site in out/
```

- Pages live in `app/`; documentation pages are MDX files in `app/docs/<page>/page.mdx`.
- The docs sidebar order is in `lib/docs.ts`.
- Screenshots and GIFs go in `public/media/` and are shown with the `<Media>` component. There are none yet; only add ones recorded from the real server, TUI or clients.
- Pushing to `main` deploys to GitHub Pages through `.github/workflows/website.yml`. Set `NEXT_PUBLIC_BASE_PATH` when serving from a sub-path.
