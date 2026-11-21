import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";
import { DESCRIPTION, SITE_URL, TAGLINE } from "@/lib/site";

// The font variables sit on <html>, where the theme's font tokens are read.
const plex = IBM_Plex_Sans({ variable: "--font-plex", subsets: ["latin"], weight: ["400", "500", "600"] });
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: `Tessera: ${TAGLINE.toLowerCase()}`, template: "%s · Tessera" },
  description: DESCRIPTION,
  applicationName: "Tessera",
  keywords: [
    "neural network architecture",
    "compare LLM architectures",
    "visualize neural networks",
    "neural network builder",
    "transformer architecture diagram",
    "mixture of experts",
    "grouped-query attention",
    "ResNet vs EfficientNet",
    "export PyTorch code",
    "ablation study",
    "config.json",
    "ONNX viewer",
  ],
  authors: [{ name: "Shyam Sundhar", url: "https://shyamsundhar.com" }],
  creator: "Shyam Sundhar",
  openGraph: { type: "website", siteName: "Tessera", url: "/", title: `Tessera: ${TAGLINE}`, description: DESCRIPTION, images: [{ url: "/og.png", width: 1200, height: 630 }] },
  twitter: { card: "summary_large_image", title: `Tessera: ${TAGLINE}`, description: DESCRIPTION, images: ["/og.png"] },
  alternates: { canonical: "/" },
};

export const viewport: Viewport = { themeColor: "#0b0d12", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plex.variable} ${plexMono.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
