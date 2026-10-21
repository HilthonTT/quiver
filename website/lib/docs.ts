export type DocLink = { title: string; href: string };
export type DocSection = { title: string; links: DocLink[] };

/** Order of the docs sidebar and the previous/next links. */
export const docs: DocSection[] = [
  {
    title: "Getting started",
    links: [
      { title: "Introduction", href: "/docs/" },
      { title: "Installation", href: "/docs/installation/" },
      { title: "Quick start", href: "/docs/quick-start/" },
    ],
  },
  {
    title: "Concepts",
    links: [
      { title: "Collections & points", href: "/docs/collections/" },
      { title: "Indexes & metrics", href: "/docs/indexes/" },
      { title: "Filtering", href: "/docs/filtering/" },
    ],
  },
  {
    title: "Using Quiver",
    links: [
      { title: "Searching", href: "/docs/searching/" },
      { title: "Data operations", href: "/docs/data-operations/" },
      { title: "Listing points", href: "/docs/listing-points/" },
      { title: "Snapshots & backups", href: "/docs/snapshots/" },
    ],
  },
  {
    title: "Clients",
    links: [
      { title: "Python client", href: "/docs/python/" },
      { title: "Rust client", href: "/docs/rust-client/" },
      { title: "Embedding in Rust", href: "/docs/embedding/" },
      { title: "Terminal UI", href: "/docs/tui/" },
    ],
  },
  {
    title: "API reference",
    links: [
      { title: "REST API", href: "/docs/rest-api/" },
      { title: "gRPC API", href: "/docs/grpc-api/" },
    ],
  },
  {
    title: "Operating Quiver",
    links: [
      { title: "Configuration", href: "/docs/configuration/" },
      { title: "Monitoring", href: "/docs/monitoring/" },
      { title: "Storage & durability", href: "/docs/storage/" },
      { title: "Security & API keys", href: "/docs/security/" },
      { title: "Docker", href: "/docs/docker/" },
    ],
  },
  {
    title: "Contributing",
    links: [
      { title: "Architecture", href: "/docs/architecture/" },
      { title: "Development", href: "/docs/development/" },
    ],
  },
];

export const allDocs: DocLink[] = docs.flatMap((s) => s.links);
