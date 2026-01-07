import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Swish Vision | AI Basketball Analytics',
  description: 'AI-powered basketball game analysis with player tracking, team classification, and tactical views',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-[#0a0a0a] text-white antialiased">
        <div className="mesh-gradient" />
        <div className="noise-overlay" />
        {children}
      </body>
    </html>
  )
}