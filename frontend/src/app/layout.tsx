import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Test Maker",
  description: "Made @ UNC Charlotte",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <nav className="bg-blue-600 p-4 text-white flex gap-6">
          <Link href="/">Home</Link>
          <Link href="/test">Test</Link>
          <Link href="/results">Results</Link>
        </nav>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
