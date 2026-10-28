import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "OceanMind AI | Marine Intelligence",
  description: "AI-assisted marine pollution segmentation and monitoring."
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
