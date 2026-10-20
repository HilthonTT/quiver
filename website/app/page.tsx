import Link from "next/link";
import {
    Archive,
    ArrowRight,
    Boxes,
    Filter,
    HardDrive,
    KeyRound,
    Network,
    PencilLine,
    Ruler,
    SquareTerminal,
} from "lucide-react";

import { CopyCommand } from "@/components/copy-command";
import { GithubIcon } from "@/components/github-icon";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { site } from "@/lib/site";

const flight = ["#2dd4bf", "#30c8dc", "#38bdf8", "#5ca4f8", "#818cf8"];

/** Soft streaks behind the hero, like arrows loosed from a quiver. */
function Flight() {
    return (
        <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[56rem] overflow-hidden [mask-image:linear-gradient(to_bottom,black_45%,transparent)]"
        >
            <div className="bg-grid absolute inset-0" />
            {flight.map((c, i) => (
                <div
                    key={c}
                    className="absolute left-1/2 h-28 w-[140%] -translate-x-1/2 rounded-[100%] blur-3xl"
                    style={{
                        top: `${6 + i * 7}rem`,
                        background: c,
                        opacity: 0.06 + (4 - i) * 0.01,
                        transform: `translateX(-50%) rotate(${-9 + i * 1.5}deg)`,
                    }}
                />
            ))}
        </div>
    );
}

function Section({
    id,
    eyebrow,
    title,
    intro,
    children,
}: {
    id?: string;
    eyebrow: string;
    title: React.ReactNode;
    intro?: string;
    children: React.ReactNode;
}) {
    return (
        <section id={id} className="mx-auto max-w-7xl scroll-mt-24 px-5 py-24">
            <p className="font-mono text-xs uppercase tracking-[0.25em] text-teal">
                {eyebrow}
            </p>
            <h2 className="mt-3 max-w-3xl text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                {title}
            </h2>
            {intro && (
                <p className="mt-4 max-w-2xl text-lg leading-relaxed text-mist-300">
                    {intro}
                </p>
            )}
            <div className="mt-12">{children}</div>
        </section>
    );
}

/** A code sample; lines starting with a comment marker are dimmed. */
function Sample({
    title,
    lang,
    code,
}: {
    title: string;
    lang: string;
    code: string;
}) {
    return (
        <div className="flex min-w-0 flex-col overflow-hidden rounded-3xl border border-white/[0.07] bg-ink-950">
            <div className="flex items-center justify-between border-b border-white/[0.07] px-6 py-3">
                <span className="text-sm font-semibold text-white">
                    {title}
                </span>
                <span className="font-mono text-xs text-mist-400">{lang}</span>
            </div>
            {/* The card keeps its padding; only the inner area scrolls, so the scrollbar stays off the rounded edge. */}
            {/* The scroll area fills the rest of the card, so every card's scrollbar sits at the same height. */}
            <div className="relative flex min-h-0 min-w-0 flex-1 flex-col px-6 pt-5 pb-4">
                <div className="min-h-0 min-w-0 flex-1 overflow-x-auto pb-2">
                    <pre className="w-max min-w-full font-mono text-[0.8rem] leading-6 text-mist-300">
                        <code>
                            {code.split("\n").map((line, i) => {
                                const comment = /^\s*(#|\/\/)/.test(line);
                                return (
                                    <span
                                        key={i}
                                        className={
                                            comment
                                                ? "text-mist-400"
                                                : undefined
                                        }
                                    >
                                        {line}
                                        {"\n"}
                                    </span>
                                );
                            })}
                        </code>
                    </pre>
                </div>
                {/* A soft fade on the right edge, the full height of the code but clear of the scrollbar. */}
                <div
                    aria-hidden
                    className="pointer-events-none absolute top-5 right-6 bottom-[calc(1rem+10px)] w-10 bg-gradient-to-l from-ink-950 to-transparent"
                />
            </div>
        </div>
    );
}

const features = [
    {
        icon: Network,
        title: "Three indexes",
        text: "HNSW for large collections, exact flat search, and a k-d tree for exact search in low dimensions.",
    },
    {
        icon: Ruler,
        title: "Five metrics",
        text: "Cosine, Euclidean, dot product, Manhattan and Hamming. Distances are always lower = closer.",
    },
    {
        icon: Filter,
        title: "Filtered search",
        text: "MongoDB-style metadata filters ($gte, $in, $or, nested fields), applied during the index traversal.",
    },
    {
        icon: PencilLine,
        title: "Data operations",
        text: "Merge or replace metadata, count and delete by filter, search by id, recommend, distance thresholds.",
    },
    {
        icon: HardDrive,
        title: "Durable storage",
        text: "A checksummed write-ahead log with checkpoints by default, RocksDB behind a feature flag, or memory.",
    },
    {
        icon: Archive,
        title: "Snapshots",
        text: "SHA-256-verified .tar.gz backups you can download, upload and restore, with retention and a schedule.",
    },
    {
        icon: KeyRound,
        title: "API keys",
        text: "REST and gRPC behind read-only and read-write keys, compared in constant time.",
    },
    {
        icon: Boxes,
        title: "Embeddable",
        text: "Run it as a server, or link the quiver-db crate and keep everything in-process.",
    },
];

const python = `# pip install ./clients/python
from quiver import Client

client = Client("http://localhost:6900")
books = client.get_or_create_collection("books", dimension=3)
books.upsert([
    ("dune", [0.9, 0.1, 0.0], {"year": 1965}),
    ("hobbit", [0.1, 0.9, 0.2], {"year": 1937}),
])
for hit in books.search([1.0, 0.0, 0.0], k=1, filter={"year": {"$gt": 1950}}):
    print(hit.id, hit.distance)  # dune 0.006116271018981934`;

const curl = `# create a collection, add a point, search
curl -X POST localhost:6900/collections \\
  -H 'content-type: application/json' \\
  -d '{"name": "places", "dimension": 2, "metric": "euclidean"}'

curl -X POST localhost:6900/collections/places/points \\
  -H 'content-type: application/json' \\
  -d '{"points": [{"id": "brussels", "vector": [4.35, 50.85]}]}'

curl -X POST localhost:6900/collections/places/search \\
  -H 'content-type: application/json' \\
  -d '{"vector": [5.5, 50.5], "k": 1}'`;

const rust = `// cargo add quiver-db --git https://github.com/HilthonTT/quiver
use quiver_db::{CollectionConfig, Database, Point};
use serde_json::json;

let db = Database::open("./quiver-data")?;
let books = db.create_collection("books", CollectionConfig::new(3))?;
books.upsert([
    Point::new("dune", vec![0.9, 0.1, 0.0])
        .with_metadata(json!({"year": 1965})),
])?;
for hit in books.search(&[1.0, 0.0, 0.0], 1)? {
    println!("{} {}", hit.id, hit.distance);
}`;

export default function Home() {
    return (
        <>
            <SiteHeader />
            <main className="relative isolate">
                <Flight />

                {/* Hero */}
                <section className="mx-auto max-w-7xl px-5 pt-20 pb-16 text-center sm:pt-28">
                    <Link
                        href="/docs/"
                        className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-1.5 text-xs text-mist-300 transition hover:border-white/25 hover:text-white"
                    >
                        <span className="size-1.5 rounded-full bg-teal" />
                        Pre-1.0 · MIT licensed · Rust 1.89+
                        <ArrowRight className="size-3" />
                    </Link>
                    <h1 className="mx-auto mt-8 max-w-4xl text-6xl font-semibold tracking-tight text-white sm:text-8xl">
                        <span className="text-gradient">Quiver</span>
                    </h1>
                    <p className="mx-auto mt-5 max-w-3xl text-2xl font-medium tracking-tight text-white sm:text-3xl">
                        {site.tagline}.
                    </p>
                    <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-mist-300">
                        Store vectors with JSON metadata, find their nearest
                        neighbours with HNSW, exact or k-d tree search, and
                        filter as you go. Run it as a server with REST and gRPC,
                        or embed it in your Rust program.
                    </p>
                    <div className="mt-10 flex flex-col items-center gap-4">
                        <CopyCommand command={site.install} />
                        <div className="flex flex-wrap justify-center gap-3">
                            <Link
                                href="/docs/quick-start/"
                                className="inline-flex items-center gap-2 rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-ink-900 transition hover:bg-mist-100"
                            >
                                Get started <ArrowRight className="size-4" />
                            </Link>
                            <a
                                href={site.repo}
                                className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-5 py-2.5 text-sm font-semibold text-white transition hover:border-white/30"
                            >
                                <GithubIcon /> Star on GitHub
                            </a>
                        </div>
                    </div>
                </section>

                {/* Features */}
                <Section
                    id="features"
                    eyebrow="Features"
                    title={
                        <>
                            A whole vector database,{" "}
                            <span className="text-gradient">
                                no dependencies to run
                            </span>
                            .
                        </>
                    }
                    intro="One binary serves REST on port 6900 and gRPC on 6901. Collections, indexes, the write-ahead log and the snapshot archives are all Quiver's own code."
                >
                    <div className="grid gap-px overflow-hidden rounded-3xl border border-white/[0.07] bg-white/[0.07] sm:grid-cols-2 lg:grid-cols-4">
                        {features.map(({ icon: Icon, title, text }) => (
                            <div key={title} className="bg-ink-850 p-7">
                                <Icon className="size-5 text-teal" />
                                <h3 className="mt-4 font-semibold text-white">
                                    {title}
                                </h3>
                                <p className="mt-1.5 text-sm leading-relaxed text-mist-300">
                                    {text}
                                </p>
                            </div>
                        ))}
                    </div>
                    <div className="mt-5 grid gap-5 sm:grid-cols-2">
                        <Link
                            href="/docs/python/"
                            className="group flex items-center gap-4 rounded-3xl border border-white/[0.07] bg-ink-850 p-7 transition hover:border-white/20"
                        >
                            <span className="inline-flex rounded-xl bg-white/5 p-2.5 text-sky ring-1 ring-white/10">
                                <Boxes className="size-5" />
                            </span>
                            <span>
                                <span className="block font-semibold text-white">
                                    Python and Rust clients
                                </span>
                                <span className="mt-1 block text-sm text-mist-300">
                                    Typed sync and async Python with numpy
                                    support, and an async Rust client.
                                </span>
                            </span>
                        </Link>
                        <Link
                            href="/docs/tui/"
                            className="group flex items-center gap-4 rounded-3xl border border-white/[0.07] bg-ink-850 p-7 transition hover:border-white/20"
                        >
                            <span className="inline-flex rounded-xl bg-white/5 p-2.5 text-sky ring-1 ring-white/10">
                                <SquareTerminal className="size-5" />
                            </span>
                            <span>
                                <span className="block font-semibold text-white">
                                    A terminal UI
                                </span>
                                <span className="mt-1 block text-sm text-mist-300">
                                    Browse collections, insert, search and
                                    manage snapshots, with optional text
                                    embeddings.
                                </span>
                            </span>
                        </Link>
                    </div>
                </Section>

                {/* Code */}
                <Section
                    id="code"
                    eyebrow="In code"
                    title="From Python, from curl, or in-process."
                    intro="The same collection, three ways in. Every client speaks the same JSON filters and returns the same lower-is-closer distances."
                >
                    <div className="grid gap-5 lg:grid-cols-3">
                        <Sample title="Python" lang="python" code={python} />
                        <Sample title="REST" lang="sh" code={curl} />
                        <Sample title="Embedded" lang="rust" code={rust} />
                    </div>
                    <Link
                        href="/docs/quick-start/"
                        className="mt-8 inline-flex items-center gap-2 font-semibold text-sky hover:underline"
                    >
                        Walk through the quick start{" "}
                        <ArrowRight className="size-4" />
                    </Link>
                </Section>

                {/* CTA */}
                <section className="mx-auto max-w-7xl px-5 pb-28">
                    <div className="relative overflow-hidden rounded-3xl border border-white/[0.07] bg-ink-850 px-8 py-16 text-center">
                        <div
                            aria-hidden
                            className="absolute inset-0 -z-0 bg-gradient-to-br from-teal/10 via-transparent to-indigo/15"
                        />
                        <h2 className="relative text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                            Clone it, run it, search.
                        </h2>
                        <p className="relative mx-auto mt-4 max-w-xl text-mist-300">
                            One command builds and starts the server with a
                            local data directory. The docs cover the rest, from
                            filters to snapshots to API keys.
                        </p>
                        <div className="relative mt-8 flex justify-center">
                            <CopyCommand command={site.install} />
                        </div>
                        <div className="relative mt-6 flex justify-center">
                            <Link
                                href="/docs/"
                                className="inline-flex items-center gap-2 rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-ink-900 transition hover:bg-mist-100"
                            >
                                Read the docs <ArrowRight className="size-4" />
                            </Link>
                        </div>
                    </div>
                </section>
            </main>
            <SiteFooter />
        </>
    );
}
