/**
 * The pixel "Q" from assets/logo.svg: its tail is an arrow, and the bright
 * cell is the query point. Each cell is [x, y, fill, opacity] on a 300 grid.
 */
const cells: [number, number, string, number][] = [
    [87, 35, "#30cecd", 0.73],
    [113, 35, "#31cbd4", 0.66],
    [139, 35, "#32c8dc", 0.86],
    [61, 61, "#30cecd", 0.63],
    [87, 61, "#31cbd4", 0.81],
    [113, 61, "#32c8dc", 0.75],
    [139, 61, "#34c6e3", 0.62],
    [165, 61, "#35c3ea", 0.8],
    [35, 87, "#30cecd", 0.61],
    [61, 87, "#31cbd4", 0.77],
    [165, 87, "#37c0f1", 0.63],
    [191, 87, "#38bdf8", 0.64],
    [35, 113, "#31cbd4", 0.77],
    [61, 113, "#32c8dc", 0.93],
    [113, 113, "#f0fdfa", 1],
    [165, 113, "#38bdf8", 0.65],
    [191, 113, "#41b7f8", 0.69],
    [35, 139, "#32c8dc", 0.85],
    [61, 139, "#34c6e3", 0.98],
    [139, 139, "#38bdf8", 1],
    [165, 139, "#41b7f8", 0.83],
    [191, 139, "#4ab1f8", 0.76],
    [61, 165, "#35c3ea", 0.99],
    [87, 165, "#37c0f1", 0.62],
    [113, 165, "#38bdf8", 0.94],
    [139, 165, "#41b7f8", 0.72],
    [165, 165, "#4ab1f8", 1],
    [87, 191, "#38bdf8", 0.66],
    [113, 191, "#41b7f8", 0.65],
    [139, 191, "#4ab1f8", 0.72],
    [191, 191, "#5ca4f8", 1],
    [243, 191, "#6f98f8", 1],
    [217, 217, "#6f98f8", 1],
    [243, 217, "#7892f8", 1],
    [191, 243, "#6f98f8", 1],
    [217, 243, "#7892f8", 1],
    [243, 243, "#818cf8", 1],
    [217, 35, "#37c0f1", 0.35],
    [243, 61, "#41b7f8", 0.23],
    [35, 243, "#38bdf8", 0.3],
    [87, 217, "#41b7f8", 0.32],
    [243, 113, "#53abf8", 0.27],
    [139, 243, "#5ca4f8", 0.3],
];

export function LogoMark({ className = "size-7" }: { className?: string }) {
    return (
        <svg viewBox="0 0 300 300" className={className} aria-hidden>
            <rect width="300" height="300" rx="64" fill="#0b1324" />
            {cells.map(([x, y, fill, opacity]) => (
                <rect
                    key={`${x}-${y}`}
                    x={x}
                    y={y}
                    width="22"
                    height="22"
                    rx="5"
                    fill={fill}
                    fillOpacity={opacity}
                />
            ))}
        </svg>
    );
}

export function Logo() {
    return (
        <span className="flex items-center gap-2.5">
            <LogoMark />
            <span className="font-mono text-lg font-semibold tracking-tight">
                quiver
            </span>
        </span>
    );
}
