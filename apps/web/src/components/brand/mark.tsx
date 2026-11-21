/** Tessera's mark: four tiles of a mosaic, one turned, in the magenta-to-orange brand gradient. */
export function Mark({ className = "size-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="tessera-mark" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0.1" stopColor="#b8005c" />
          <stop offset="1" stopColor="#f6731f" />
        </linearGradient>
      </defs>
      <rect x="3" y="3" width="12" height="12" rx="3" fill="url(#tessera-mark)" />
      <rect x="17" y="3" width="12" height="12" rx="3" fill="url(#tessera-mark)" opacity="0.55" />
      <rect x="3" y="17" width="12" height="12" rx="3" fill="url(#tessera-mark)" opacity="0.55" />
      <rect x="17.5" y="17.5" width="11" height="11" rx="3" fill="url(#tessera-mark)" transform="rotate(12 23 23)" />
    </svg>
  );
}
